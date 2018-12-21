# Steven Atkinson
# satkinso@nd.edu
# March 30, 2018

"""
Kullback-Liebler divergence handling
"""

from __future__ import absolute_import

from .probability_distributions import ColumnGaussian, RowMatrixGaussian
from .linalg import jit_cholesky, jit_matrix_triangular_solve

from gpflow.probability_distributions import DiagonalGaussian, Gaussian
from gpflow.kullback_leiblers import gauss_kl
from gpflow import settings
import tensorflow as tf
from warnings import warn


float_type = settings.float_type


def _diagonal_gaussian_vs_diagonal_gaussian(qx, px):
    mu = qx.mu - px.mu
    nq = tf.cast(tf.size(mu), float_type)

    return 0.5 * (
            tf.reduce_sum(mu ** 2)
            + tf.reduce_sum(qx.cov - tf.log(qx.cov))
            - nq
    )


def _gpflow_gauss_kl(q_mu, q_sqrt, K):
    """
    [Note: copied out of GPflow so that I can use jit-chol]
    Compute the KL divergence KL[q || p] between

          q(x) = N(q_mu, q_sqrt^2)
    and
          p(x) = N(0, K)

    We assume N multiple independent distributions, given by the columns of
    q_mu and the last dimension of q_sqrt. Returns the sum of the divergences.

    q_mu is a matrix (M x N), each column contains a mean.

    q_sqrt can be a 3D tensor (N xM x M), each matrix within is a lower
        triangular square-root matrix of the covariance of q.
    q_sqrt can be a matrix (M x N), each column represents the diagonal of a
        square-root matrix of the covariance of q.

    K is a positive definite matrix (M x M): the covariance of p.
    If K is None, compute the KL divergence to p(x) = N(0, I) instead.
    """

    if K is None:
        white = True
        alpha = q_mu
    else:
        white = False
        Lp = jit_cholesky(tf.identity(K, name="KL-K"))
        alpha = jit_matrix_triangular_solve(Lp, q_mu, lower=True)

    if q_sqrt.get_shape().ndims == 2:
        diag = True
        num_latent = tf.shape(q_sqrt)[1]
        NM = tf.size(q_sqrt)
        Lq = Lq_diag = q_sqrt
    elif q_sqrt.get_shape().ndims == 3:
        diag = False
        num_latent = tf.shape(q_sqrt)[0]
        NM = tf.reduce_prod(tf.shape(q_sqrt)[:2])
        Lq = tf.matrix_band_part(q_sqrt, -1, 0)  # force lower triangle
        Lq_diag = tf.matrix_diag_part(Lq)
    else:  # pragma: no cover
        raise ValueError(
            "Bad dimension for q_sqrt: {}".format(q_sqrt.get_shape().ndims))

    # Mahalanobis term: μqᵀ Σp⁻¹ μq
    mahalanobis = tf.reduce_sum(tf.square(alpha))

    # Constant term: - N x M
    constant = -tf.cast(NM, settings.float_type)

    # Log-determinant of the covariance of q(x):
    logdet_qcov = tf.reduce_sum(tf.log(tf.square(Lq_diag)))

    # Trace term: tr(Σp⁻¹ Σq)
    if white:
        trace = tf.reduce_sum(tf.square(Lq))
    else:
        if diag:
            M = tf.shape(Lp)[0]
            Lp_inv = jit_matrix_triangular_solve(
                Lp, tf.eye(M, dtype=settings.float_type), lower=True)
            K_inv = jit_matrix_triangular_solve(
                tf.transpose(Lp), Lp_inv, lower=False)
            trace = tf.reduce_sum(
                tf.expand_dims(tf.matrix_diag_part(K_inv), 1) * tf.square(
                    q_sqrt))
        else:
            Lp_tiled = tf.tile(tf.expand_dims(Lp, 0), [num_latent, 1, 1])
            LpiLq = tf.matrix_triangular_solve(Lp_tiled, Lq, lower=True)
            trace = tf.reduce_sum(tf.square(LpiLq))

    twoKL = mahalanobis + constant - logdet_qcov + trace

    # Log-determinant of the covariance of p(x):
    if not white:
        log_sqdiag_Lp = tf.log(tf.square(tf.matrix_diag_part(Lp)))
        sum_log_sqdiag_Lp = tf.reduce_sum(log_sqdiag_Lp)
        prior_logdet = tf.cast(num_latent,
                               settings.float_type) * sum_log_sqdiag_Lp
        twoKL += prior_logdet

    return 0.5 * twoKL


def _column_gaussian_vs_row_matrix_gaussian(qx, px):
    mu = qx.mu - px.mu
    s = qx.cov
    # Cholesky of each cov:
    ls = tf.stack([tf.identity(jit_cholesky(s[i, :, :]),
                               name="KL-ls_{}".format(i + 1))
                   for i in range(s.shape[0])])
    return _gpflow_gauss_kl(mu, ls, px.cov)


def _full_gaussian_vs_full_gaussian(qx, px):
    # Due to a limitation in the KL computation, we need a rank-2 px.cov:
    with tf.Session() as sess:
        px_cov_rank = sess.run(tf.rank(px.cov))
    assert px_cov_rank == 2, "Need rank-2 px.cov for KL function"

    mu = tf.transpose(qx.mu - px.mu)  # Again, for _gpflow()
    if qx.chol_cov is None:
        warn("Provide chol_cov for faster KL evaluation")
        ls = tf.stack([tf.identity(jit_cholesky(qx.cov[i, :, :]),
                                   name="KL-ls_{}".format(i + 1))
                       for i in range(qx.cov.shape[0])])
    else:
        ls = qx.chol_cov

    return _gpflow_gauss_kl(mu, ls, px.cov)


def kl_divergence(qx, px):
    """

    :param qx:
    :param px:
    :return:
    """
    f_dict = {
        (DiagonalGaussian, DiagonalGaussian):
            _diagonal_gaussian_vs_diagonal_gaussian,
        (ColumnGaussian, RowMatrixGaussian):
            _column_gaussian_vs_row_matrix_gaussian,
        (Gaussian, Gaussian):
            _full_gaussian_vs_full_gaussian
    }
    types = (type(qx), type(px))
    if types in f_dict:
        return f_dict[types](qx, px)
    else:
        raise Exception("Type combination {} not found".format(types))
