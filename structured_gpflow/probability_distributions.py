# Steven Atkinson
# satkinso@nd.edu
# March 30, 2018

"""
Additional probability distributions not included in GPflow
"""

from __future__ import absolute_import

from gpflow.probability_distributions import ProbabilityDistribution, \
    DiagonalGaussian

import tensorflow as tf


class ColumnGaussian(ProbabilityDistribution):
    """
    Matrix random variable is distributed according to the product of
    multivariate Gaussians, one for each column.

    i.e. columns are independent multivariate Gaussians with their own
    (full) covariance matrices

    random variable X is N x D
    mu is N x D
    cov is N x N (D of them)
    """
    def __init__(self, mu, cov):
        self.mu = mu  # N x D
        self.cov = cov  # D x N x N

    def diag(self):
        """
        Return the "diagonalized" distribution
        :return: DiagonalGaussian
        """
        # diag_list = [tf.diag_part]
        # var = tf.reshape(tf.diag_part(),
        #                  (tf.shape(self.cov)[0], tf.shape(self.cov)[2]))
        # return DiagonalGaussian(self.mu, var)
        # How to pull out diag_part from each?
        raise NotImplementedError("TF how to pull out diags?")


class RowMatrixGaussian():
    """
    Matrix RV is matrix-variate Gaussian-distributed with identity
    column-covariance matrix

    aka same as ColumnGaussian, except there's only one, shared, covariance
    matrix (each row still has its own mean)

    random variable X is N x D
    mu is N x D
    cov is N x N  (shared)
    """

    def __init__(self, mu, cov):
        self.mu = mu  # N x D
        self.cov = cov  # D x N x N

    def diag(self):
        """
        Return the "diagonalized" distribution
        :return: DiagonalGaussian
        """
        # diag_list = [tf.diag_part]
        # var = tf.reshape(tf.diag_part(),
        #                  (tf.shape(self.cov)[0], tf.shape(self.cov)[2]))
        # return DiagonalGaussian(self.mu, var)
        # How to pull out diag_part from each?
        # tf.matrix_diag_part(self.cov)?
        raise NotImplementedError("TF how to pull out diags?")
