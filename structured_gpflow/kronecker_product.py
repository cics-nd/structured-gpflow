# Kronecker product class
# Steven Atkinson
# satkinso@nd.edu
# January 30, 2018

from __future__ import absolute_import

from .linalg import jit_cholesky, jit_self_adjoint_eig

import tensorflow as tf
import numpy as np
from warnings import warn


class KroneckerProduct:
    """
    Kronecker product class

    x: the subtensors in the KP
    k: the number of subtensors
    shape: shape of the tensor
    shape_i: for an individual subtensor

    TODO non-2D Kronecker products
    """

    def __init__(self, x, shape_hint=None):
        """
        Construct a KP object from an Iterable of submatrices

        :param x: The submatrices of the KP matrix, i.e.,
            A = X[0] o X[1] o ... o X[end], where "o" is the Kronecker product
            operation.
        :type x: Iterable of tf.Variables
        :param shape_hint: If the sizes of x aren't available, then you can
            provide a "hint".  This is necessary, e.g., when doing KP-matrix
            multiplication and the size of a submatrix is unknown.
        :type shape_hint:
        """
        self.x = x
        self._shape_hint = None
        if shape_hint is not None:
            self.shape_hint = shape_hint

    def __str__(self):
        s = 'KroneckerProduct with ' + str(self.k)
        if self.k == 1:
            s = s + ' submatrix:\n'
        else:
            s = s + ' submatrices:\n'
        s = s + 'sizes:\n'
        for xi in self.x:
            s = s + ' [' + str(xi.shape[0]) + ' x ' + str(xi.shape[1]) + ']\n'
        s = s + 'data:\n'
        for xi in self.x:
            s = s + str(xi)

        return s

    def __mul__(self, other):
        """
        Schur (element-wise) product
        :param other:
        :type other: KroneckerProduct
        :return:
        """
        return KroneckerProduct([xi * yi for xi, yi in zip(self.x, other.x)])

    def __matmul__(self, other):
        """
        result = self @ other

        :param other: the tensor that self left-multiplies
        :type other: tf.Tensor or KroneckerProduct
        :return: (same type as other)
        """
        return self.matmul(other)

    @property
    def k(self):
        """
        Number of submatrices
        :return: (int)
        """
        return len(self.x)

    @property
    def ndims(self):
        """
        Dimension of the matrix (1=vector, 2=matrix, etc)
        """
        return len(self.shape_i[0])

    @property
    def shape(self):
        """
        The shape of the full matrix
        :return: (tuple)
        """
        shape = tuple([np.prod([s[j] for s in self.shape_i])
                       for j in range(2)])
        return shape

    @property
    def shape_i(self):
        """
        The shape of the submatrices
        :return: (list of tuples)
        """
        if self._shape_hint is not None:
            return self._shape_hint
        else:
            return [tuple(xi.get_shape().as_list()) for xi in self.x]
            # return [tf.shape(xi) for xi in self.x]

    @property
    def shape_hint(self):
        return self._shape_hint

    @shape_hint.setter
    def shape_hint(self, val):
        """
        Provide hints about self's shape.
        This is sometimes needed to help TensorFlow figure things out.

        :param val:
        :type val: list of tuples
        :return:
        """
        # Make sure everything is tuples:
        val = [tuple(s) for s in val]

        # make sure that the hint doesn't conflict with anything we actually
        # know:
        # TensorFlow only has .is_fully_defined, but not .is-fully_undefined, so
        # I have this bit of a kludge here where .as_list() errors out on
        # fully-undefined shapes, wherein I can replace them with all Nones.
        shape_x = []
        for x_i, val_i in zip(self.x, val):
            try:
                shape_x.append(tuple(x_i.get_shape().as_list()))
            except Exception as _:
                shape_x.append(tuple([None] * len(val_i)))

        for shape_x_i, val_i in zip(shape_x, val):
            assert all([shape_ij == shape_hint_ij or shape_ij is None
                        for shape_ij, shape_hint_ij in
                        zip(shape_x_i, val_i)]), \
                "shape_hint conflicts with known dimensions"
        self._shape_hint = val

    def matmul(self, other, shape_hint=None):
        """
        result = self @ other

        :param other: the tensor that self left-multiplies
        :type other: tf.Tensor or KroneckerProduct
        :return: (same type as other)
        """

        if isinstance(other, tf.Tensor) or isinstance(other, tf.Variable):
            return self._matmul_tensor(other, shape_hint)
        elif isinstance(other, KroneckerProduct):
            return self._matmul_kronecker_product(other, shape_hint)
        else:
            raise NotImplementedError("Unrecognized other type {}".format(
                type(other)))

    def _matmul_kronecker_product(self, other, shape_hint):
        """
        Compute the product of two KP matrices in a submatrix-wise manner

        :param other: the matrix right-multiplying self.
        :type other: KroneckerProduct
        :return: (KroneckerProduct)
        """
        return KroneckerProduct([xi @ yi for xi, yi in zip(self.x, other.x)])

    def _matmul_tensor(self, other, shape_hint):
        """
        KP times matrix extending the algorithm from  Saatci's thesis.
        Also use python row-reshapes, of course.

        :param other: the tensor x to right-multiply with self.
        :type other: tf.Tensor
        :param shape_hint: Provide the shape for "other".
        :type shape_hint: tuple of ints
        :return: (tf.Tensor)
        """

        p = other.shape[1] if shape_hint is None else shape_hint[1]
        nm_i = self.shape_i
        n_i = [nm_ii[0] for nm_ii in nm_i]
        m_i = [nm_ii[1] for nm_ii in nm_i]
        m = np.prod(m_i)
        y_cols = []
        for j in range(p):
            y_j = other[:, j]
            for i in range(self.k):
                cur_size = m * np.prod(n_i[: i], dtype=int) // \
                           np.prod(m_i[: i], dtype=int)
                next_size = m * np.prod(n_i[: i + 1], dtype=int) // \
                        np.prod(m_i[: i + 1], dtype=int)
                s = [m_i[i], cur_size // m_i[i]]
                s_next = [next_size, 1]
                x_mtx = tf.reshape(y_j, s)
                z1 = self.x[i] @ x_mtx
                z2 = tf.transpose(z1)
                y_j = tf.reshape(z2, s_next)
            y_cols.append(y_j)
        return tf.concat(y_cols, 1)

    def eval(self):
        """
        Defines the evaluation of the KP
        Beware: this can easily result in a very large matrix and is generally
        not soemthing you should be using!

        :return: the evaluated KP as a tf Variable
        """

        if self.k > 2:
            other = KroneckerProduct(self.x[1:]).eval()
        elif self.k == 2:
            other = self.x[1]
        elif self.k == 1:
            return self.x[0]

        def _eval(mat1, mat2):
            """
            Computes the Kronecker product of two matrices.

            Adapted from tf.contrib.kfac.utils.kronecker_product to allow for
            placeholder matrices where at most one dimension is unknown...
            """
            m1, n1 = [val or -1 for val in mat1.get_shape().as_list()]
            m2, n2 = [val or -1 for val in mat2.get_shape().as_list()]
            mat1_rsh = tf.reshape(mat1, [m1, 1, n1, 1])
            mat2_rsh = tf.reshape(mat2, [1, m2, 1, n2])
            # any unknown sizes affect their dimension
            m = m1 * m2 if m1 >= 0 and m2 >= 0 else -1
            n = n1 * n2 if n1 >= 0 and n2 >= 0 else -1
            return tf.reshape(mat1_rsh * mat2_rsh, [m, n])

        return _eval(self.x[0], other)

    def transpose(self):
        """
        Compute the transpose
        :return: (KroneckerProduct)
        """
        return KroneckerProduct([tf.transpose(xi) for xi in self.x])

    def diag_part(self, col=False):
        """
        Get the diag part of the KP.

        :param col: if True, returns x's as n_i x 1 (2D) tensors
        :return:
        """
        if col:
            return KroneckerProduct([tf.reshape(tf.diag_part(xi), (-1, 1))
                                     for xi in self.x])
        else:
            return KroneckerProduct([tf.diag_part(xi) for xi in self.x])

    def cholesky(self):
        """
        Compute the Cholesky decomposition of this matrix
        :return: (KroneckerProduct)
        """
        return KroneckerProduct([jit_cholesky(xi) for xi in self.x])

    def self_adjoint_eig(self, evals_1d=True):
        """
        Computes the eigenvalues and eigenvectors of the matrix
        Assumes that each submatrix of self is self-adjoint.

        The eigenvalue ordering is from low to high.
        The eigenvalue submatrices are returned as 2D, N x 1 matrices by default

        :param evals_1d: whether to return the submatrices of the eigenvalue KP
            as vectors (True) or as full matrices (False)
        :return: eigenvalues (KP), eigenvectors (KP)
        """
        uv = [jit_self_adjoint_eig(xi) for xi in self.x]
        if evals_1d:
            return KroneckerProduct([tf.reshape(uv_i[0], [-1, 1])
                                     for uv_i in uv]),\
                   KroneckerProduct([uv_i[1] for uv_i in uv])
        else:
            return KroneckerProduct([tf.diag(uv_i[0]) for uv_i in uv]), \
                   KroneckerProduct([uv_i[1] for uv_i in uv])

    def matrix_inverse(self):
        """
        Compute the inverse of this matrix
        :return: (KroneckerProduct)
        """
        return KroneckerProduct([tf.matrix_inverse(xi) for xi in self.x])

    def _matrix_triangular_solve_kronecker_product(self, other, lower):
        """
        Solve self @ x = other for x when other is a Kronecker product

        :param other: the right-hand side of the system of equations
        :type other: KroneckerProduct
        :param lower: whether self is a lower (True) or upper (False) triangular
            matrix
        :type lower: bool
        :return: (KroneckerProduct)
        """
        return KroneckerProduct([tf.matrix_triangular_solve(xi, yi, lower)
                                 for xi, yi in zip(self.x, other.x)])

    def _matrix_triangular_solve_tensor(self, other, lower):
        """
        Solve self @ x = other for x when other is a full tensor

        Matrix sizes:
        A    : n x m
        A[i] : n_i x m_i
        X    : m x p
        rhs  : n x p

        Recursive algorithm based on Bilionis et al., "Multi-output separable
        Gaussian process: Towards an efficient, fully Bayesian paradigm for
        uncertainty quantification" (2013)

        :param other: the right-hand side of the system of equations
        :type other: tf.Tensor
        :param lower: whether self is a lower (True) or upper (False) triangular
            matrix
        :type lower: bool
        :return: (KroneckerProduct)
        """
        assert lower, "upper triangular not implemented"
        if self.k == 1:
            return tf.matrix_triangular_solve(self.x[0], other, lower)
        else:
            n = self.shape[0]
            p = other.shape[1]
            n_0 = int(self.x[0].shape[0])
            n_prime = n // n_0

            a_prime = KroneckerProduct(self.x[1:])
            a_0 = self.x[0]

            x_cols = []
            for i in range(p):
                # See KP times matrix for notes about Fortran-style reshaping...
                x1i = a_prime.matrix_triangular_solve(tf.transpose(
                    tf.reshape(other[:, i], (n_0, n_prime))), lower)
                # Note: The formula has a transpose before vectorizing.
                # However, F-style reshape needs a transpose as well.
                # So, they cancel and no transpose is carried out after trtrs.
                x_cols.append(tf.reshape(
                    tf.matrix_triangular_solve(a_0, tf.transpose(x1i), lower),
                    [-1]))
            return tf.stack(x_cols, 1)

    def matrix_triangular_solve(self, other, lower=True):
        """
        Solve self @ x = other for x, assuming that self is a triangular matrix.

        :param other: the right-hand side of the system of equations
        :type other: KroneckerProduct
        :param lower: whether self is a lower (True) or upper (False) triangular
            matrix
        :type lower: bool
        :return:(KroneckerProduct)
        """
        if isinstance(other, tf.Tensor) or isinstance(other, tf.Variable):
            return self._matrix_triangular_solve_tensor(other, lower)
        elif isinstance(other, KroneckerProduct):
            return self._matrix_triangular_solve_kronecker_product(other, 
                                                                   lower)
        else:
            raise NotImplementedError("Unrecognized other type {}".format(
                type(other)))
