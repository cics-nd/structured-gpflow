# Steven Atkinson
# satkinso@nd.edu
# April 3, 2018

"""
Miscellaneous linear algebra operations
"""

from __future__ import absolute_import

from gpflow import settings
import tensorflow as tf

_rcond_tolerance = 1.0e-14
DEBUG_STATEMENTS = True


def _jit_op(x, op, jitter, rcond_tolerance, *args, **kwargs):
    """
    Perform some operation on x, assuming that it's positive-definite.
    If you fail, then add jitter and try again.

    Uses recursion to build up several attempts.

    :param x: The tensor to be operated on
    :param op: The TensorFlow operation
    :param jitter: How much jitter to
        (gets normalized w.r.t. the mean of x's diagonal)
    :param rcond_tolerance: tolerance for the reciprocal condition number of x.
        If it's below this, then x needs jitter.
    :return: (Whatever the op does)
    """

    alpha = 10.0  # multiplicative factor for the jitter.
    jitter0 = jitter * tf.reduce_mean(tf.diag_part(x))
    n = tf.shape(x)[0]

    def is_positive_definite(_x):
        """
        Rough test within TF's capabilities to test if a matrix is
        positive-definite.
        :return: (tf.bool)
        """
        return rcond(_x) > rcond_tolerance

    delta = jitter0 * tf.eye(n, dtype=x.dtype)

    # Alert if this will need jitter or fail:
    if DEBUG_STATEMENTS:
        summarize = 3  # Number of entries to show
        x = tf.cond(
            is_positive_definite(x),
            lambda: tf.identity(x),
            lambda: tf.cond(
                is_positive_definite(x + alpha ** 9 * delta),
                lambda: tf.Print(x, [x],
                                 "While performing {}, ".format(op.__name__) +
                                 "tensor {} is poorly-conditioned [warn]!\n".
                                 format(x.name), summarize=summarize),
                lambda: tf.Print(x, [x],
                                 "While performing {}, ".format(op.__name__) +
                                 "tensor {} is poorly-conditioned [fail]!\n".
                                 format(x.name), summarize=summarize)
            ))

    # Recursive code wasn't behaving...Sorry for this!
    # TODO recursive implementation
    result = \
        tf.cond(is_positive_definite(x),
                lambda: op(x, *args, **kwargs),
        lambda: tf.cond(is_positive_definite(x + delta),
                lambda: op(x + delta, *args, **kwargs),
        lambda: tf.cond(is_positive_definite(x + alpha * delta),
                lambda: op(x + alpha * delta, *args, **kwargs),
        lambda: tf.cond(is_positive_definite(x + alpha ** 2 * delta),
                lambda: op(x + alpha ** 2 * delta, *args, **kwargs),
        lambda: tf.cond(is_positive_definite(x + alpha ** 3 * delta),
                lambda: op(x + alpha ** 3 * delta, *args, **kwargs),
        lambda: tf.cond(is_positive_definite(x + alpha ** 4 * delta),
                lambda: op(x + alpha ** 4 * delta, *args, **kwargs),
        lambda: tf.cond(is_positive_definite(x + alpha ** 5 * delta),
                lambda: op(x + alpha ** 5 * delta, *args, **kwargs),
        lambda: tf.cond(is_positive_definite(x + alpha ** 6 * delta),
                lambda: op(x + alpha ** 6 * delta, *args, **kwargs),
        lambda: tf.cond(is_positive_definite(x + alpha ** 7 * delta),
                lambda: op(x + alpha ** 7 * delta, *args, **kwargs),
        lambda: tf.cond(is_positive_definite(x + alpha ** 8 * delta),
                lambda: op(x + alpha ** 8 * delta, *args, **kwargs),
                lambda: op(x + alpha ** 9 * delta, *args, **kwargs)
        ))))))))))  # I'm so sorry...

    return result


def jit_cholesky(x, jitter=settings.jitter,
                 rcond_tolerance=_rcond_tolerance, **kwargs):
    """
    Cholesky decomposition
    :return: (tf.Tensor)
    """
    return _jit_op(x, tf.cholesky, jitter, rcond_tolerance, **kwargs)


def jit_self_adjoint_eig(x, jitter=settings.jitter,
                         rcond_tolerance=_rcond_tolerance, **kwargs):
    """
    Eigendecomposition of a self-adjoint matrix
    :return: (tf.Tensor, tf.Tensor)
    """
    return _jit_op(x, tf.self_adjoint_eig, jitter, rcond_tolerance)


def jit_matrix_inverse(x, jitter=settings.jitter,
                         rcond_tolerance=_rcond_tolerance, **kwargs):
    """
    Eigendecomposition of a self-adjoint matrix
    :return: (tf.Tensor, tf.Tensor)
    """
    return _jit_op(x, tf.matrix_inverse, jitter, rcond_tolerance, **kwargs)


def jit_matrix_triangular_solve(x, rhs, jitter=settings.jitter,
                                rcond_tolerance=_rcond_tolerance, **kwargs):
    """
    Triangular solve inv(x) * rhs
    :param x:
    :param rhs:
    :param jitter:
    :param rcond_tolerance:
    :return:
    """
    return _jit_op(x, tf.matrix_triangular_solve, jitter, rcond_tolerance,
                   rhs, **kwargs)


def rcond(x, use_abs=True):
    """
    Compute reciprocal condition number of a matrix
    :param x:
    :return:
    """
    s = tf.svd(x, compute_uv=False)
    s_max, s_min = tf.reduce_max(s), tf.reduce_min(s)
    # Note: since I'm not taking an absolute value here, then this can be
    # negative if min(s)<0 (assume that things aren't so bad that max(s)<0
    # too.)
    r = s_min / s_max
    return tf.abs(r) if use_abs else r
