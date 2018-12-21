# Kronecker GP regression
# Steven Atkinson
# satkinso@nd.edu
# January 30, 2018

from __future__ import absolute_import

from .smodel import SgpModel
from ..kronecker_product import KroneckerProduct
from .. import kronecker_product_numpy as knp

from gpflow import likelihoods
from gpflow.params import DataHolder
from gpflow.decors import params_as_tensors, name_scope
import gpflow

from warnings import warn
import tensorflow as tf
import numpy as np


class Sgpr(SgpModel):
    """
    Structured Gaussian Process Regression
    """
    def __init__(self, x, y, kern, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()
        x = [DataHolder(x_i) for x_i in x]
        y = DataHolder(y)
        print(x)
        print(y)
        print(kern)
        super().__init__(x, y, kern, likelihood, mean_function, **kwargs)

    @name_scope('likelihood')
    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        # Pull out input subgrids into a list:
        x = [getattr(self, name) for name in self.input_names]
        kern_list = [getattr(self, name) for name in self.kern_names]
        k_ff_i = [k_i.K(x_i) for k_i, x_i in zip(kern_list, x)]
        k_ff = KroneckerProduct(k_ff_i)
        l_ff, q_ff = k_ff.self_adjoint_eig()
        l_ff_ev = l_ff.eval()
        q_ff_t = q_ff.transpose()
        q_ff_t.shape_hint = [(n_i, n_i) for n_i in self.n_subgrid]
        logdet_kyy = tf.reduce_sum(tf.log(l_ff_ev + self.likelihood.variance))
        y_tilde = q_ff_t.matmul(self.Y, shape_hint=(self.n, self.num_latent))

        y_kyy_y = (y_tilde ** 2) \
            / tf.reshape(l_ff_ev + self.likelihood.variance, [-1, 1])
        exp_term = tf.reduce_sum(y_kyy_y)
        n_log_2pi = self.n * tf.log(tf.constant(2.0 * np.pi,
                                                dtype=gpflow.settings.tf_float))

        return -0.5 * (n_log_2pi + logdet_kyy + exp_term)

    @name_scope('predict')
    @params_as_tensors
    def _build_predict(self, x_new, full_cov=False):
        """
        Xnew is a list of input subgrids at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        raise NotImplementedError(
            "Don't use {}.predict_y()! Use np_predict_y instead.".
                format(self.__class__.__name__))

    def numpy(self):
        """
        Convert inputs, kernels, and likelihood variance to numpy variables to
        allow imperative manipulation.
        Bit of a hack; sorry.

        :return: (x, kern, y)
        """
        x = [x_i.value for x_i in self.X]
        y = self.Y.value
        sy = self.likelihood.variance.value
        return x, y, self.kern_list, sy

    def np_predict_f(self, x_new, full_cov=False):
        """
        x_new is a list of input subgrids at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at x_new, Y are noisy observations at X.

        """
        x, y, kern_list, sy = self.numpy()

        k_fs = knp.KroneckerProduct([kern.compute_K(x_i, x_new_i)
                                     for kern, x_i, x_new_i
                                     in zip(self.kern_list, x, x_new)])
        k_ff = knp.KroneckerProduct([kern.compute_K_symm(x_i)
                                     for kern, x_i in zip(kern_list, x)])
        k_ss_diag = knp.KroneckerProduct([
            kern.compute_Kdiag(x_new_i).reshape((-1, 1))
            for kern, x_new_i in zip(kern_list, x_new)]).eval()
        l_ff, q_ff = k_ff.eigh()
        l_yy = l_ff.eval() + sy
        k_sf_q = k_fs.transpose() @ q_ff
        # Encourage efficient order of operations w/ parentheses
        fmean = k_sf_q @ (1.0 / l_yy * (q_ff.transpose() @ y))
        fvar = k_ss_diag - (k_sf_q * k_sf_q) @ (1.0 / l_yy)

        return fmean, fvar

    def np_predict_y(self, x_new, full_cov=False):
        """
        Predict outputs

        :param x_new:
        :param full_cov:
        :return:
        """

        ymean, fvar = self.np_predict_f(x_new, full_cov)
        yvar = fvar + self.likelihood.variance.value
        return ymean, yvar
