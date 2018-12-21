# Kronecker product class implemented in numpy
# Steven Atkinson
# satkinso@nd.edu
# March 7, 2018

from scipy.linalg import solve_triangular
import numpy as np
from warnings import warn


class KroneckerProduct:
    """
    Kronecker product class

    x: the submatrices in the KP
    k: the number of submatrices
    shape: shape of the tensor
    shape_i: for an individual subtensor

    TODO non-2D Kronecker products
    """

    def __init__(self, x):
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

    def __str__(self):
        s = 'KroneckerProduct with ' + str(self.k)
        if self.k == 1:
            s = s + ' submatrix ( '
        else:
            s = s + ' submatrices ( '
        for xi in self.x:
            s = s + '(' + str(xi.shape[0]) + ' x ' + str(xi.shape[1]) + ') '
        s = s + ')\n'

        return s

    def __mul__(self, other):
        """
        Schur product (KP o KP)
        :param other:
        :type other: KroneckerProduct
        :return: (KroneckerProduct)
        """
        return KroneckerProduct([x_i * y_i
                                 for x_i, y_i in zip(self.x, other.x)])

    def __matmul__(self, other):
        """
        result = self @ other

        :param other: the tensor that self left-multiplies
        :type other: tf.Tensor or KroneckerProduct
        :return: (same type as other)
        """
        f_dict = {
            np.ndarray: self._matmul_matrix,
            KroneckerProduct: self._matmul_kronecker_product
        }
        return f_dict[type(other)](other)

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
                       for j in range(self.ndims)])
        return shape

    @property
    def shape_i(self):
        """
        The shape of the submatrices
        :return: (list of tuples)
        """
        return [tuple(xi.shape) for xi in self.x]

    def _matmul_kronecker_product(self, other):
        """
        Compute the product of two KP matrices in a submatrix-wise manner

        :param other: the matrix right-multiplying self.
        :type other: KroneckerProduct
        :return: (KroneckerProduct)
        """
        return KroneckerProduct([xi @ yi for xi, yi in zip(self.x, other.x)])

    def _matmul_matrix(self, other):
        """
        Kronecker product, matrix-matrix product
        (A_1 o ... o A_K) @ x = y
        Follows Algorithm 15, p.137 (kron_mvprod) of Y. Saatci, "Scalable
        Inference for Structured Gaussian Process Models (2011)"

        Altered for Python-style ROW-major reshapes:
        Col: (A o B)x = y = vec_c(BXA')
        Row: (A o B)x = y = vec_r(AXB')

        :param other: the matrix x to right-multiply with self.
        :type other: np.ndarray
        :return: (np.ndarray)
        """
        y = []
        for y_j in other.transpose():  # By-column
            for x_i in self.x:
                y_j = y_j.reshape((x_i.shape[1],
                                   y_j.size // x_i.shape[1])).transpose() \
                      @ x_i.transpose()
            y.append(y_j.reshape((-1, 1)))
        return np.concatenate(y, axis=1)

    def eval(self):
        """
        Defines the evaluation of the KP
        Since this is potentially a very large matrix, this is not invoked in
        the constructor by default.

        :return: (np.ndarray) the evaluated KP
        """

        if self.k > 2:
            other = KroneckerProduct(self.x[1:]).eval()
        elif self.k == 2:
            other = self.x[1]
        elif self.k == 1:
            return self.x[0]

        def _eval(mat1, mat2):
            """
            Computes the Kronecker product two matrices.

            Adapted from tf.contrib.kfac.utils.kronecker_product to allow for
            placeholder matrices where at most one dimension is unknown...
            """
            m1, n1 = mat1.shape
            m2, n2 = mat2.shape
            mat1_rsh = mat1.reshape((m1, 1, n1, 1))
            mat2_rsh = mat2.reshape((1, m2, 1, n2))
            # any unknown sizes affect their dimension
            m = m1 * m2
            n = n1 * n2
            return (mat1_rsh * mat2_rsh).reshape((m, n))

        return _eval(self.x[0], other)

    def transpose(self):
        """
        Compute the transpose
        :return: (KroneckerProduct)
        """
        return KroneckerProduct([xi.transpose() for xi in self.x])

    def cholesky(self):
        """
        Compute the Cholesky decomposition of this matrix
        :return: (KroneckerProduct)
        """
        return KroneckerProduct([np.linalg.cholesky(xi) for xi in self.x])

    def eigh(self, evals_1d=True):
        """
        Computes the eigenvalues and eigenvectors of the matrix
        Assumes that each submatrix of self is self-adjoint.

        The eigenvalue ordering in each submatrix is from high to low.
        The eigenvalue submatrices are returned as 2D, N x 1 matrices by default

        :param evals_1d: whether to return the submatrices of the eigenvalue KP
            as vectors (True) or as full matrices (False)
        :return: eigenvectors (KP), eigenvalues (KP)
        """
        uv = [np.linalg.eigh(xi) for xi in self.x]
        if evals_1d:
            return KroneckerProduct([uv_i[0].reshape((-1, 1)) for uv_i in uv]),\
                   KroneckerProduct([uv_i[1] for uv_i in uv])
        else:
            return KroneckerProduct([np.diag(uv_i[0]) for uv_i in uv]), \
                   KroneckerProduct([uv_i[1] for uv_i in uv])

    def inv(self):
        """
        Compute the inverse of this matrix
        :return: (KroneckerProduct)
        """
        return KroneckerProduct([np.linalg.inv(xi) for xi in self.x])

    def _solve_triangular_kronecker_product(self, other, lower):
        """
        Solve self @ x = other for x when other is a Kronecker product

        :param other: the right-hand side of the system of equations
        :type other: KroneckerProduct
        :param lower: whether self is a lower (True) or upper (False) triangular
            matrix
        :type lower: bool
        :return: (KroneckerProduct)
        """
        return KroneckerProduct([solve_triangular(xi, yi, lower=lower)
                                 for xi, yi in zip(self.x, other.x)])

    def _solve_triangular_matrix(self, other, lower):
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

    def solve_triangular(self, other, lower=True):
        """
        Solve self @ x = other for x, assuming that self is a triangular matrix.

        :param other: the right-hand side of the system of equations
        :type other: KroneckerProduct
        :param lower: whether self is a lower (True) or upper (False) triangular
            matrix
        :type lower: bool
        :return:(KroneckerProduct)
        """
        f_dict = {
            np.ndarray: self._solve_triangular_matrix,
            KroneckerProduct: self._solve_triangular_kronecker_product
        }
        return f_dict[type(other)](other, lower)
