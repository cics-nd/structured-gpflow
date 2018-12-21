# Steven Atkinson
# satkinso@nd.edu
# March 30, 2018

"""
Kronecker Bayesian warped Gaussian process model

THINGS TO BE AWARE OF:
* Things that aren't what they meant for KGPLVM:
    * h_s isn't the variance of the X0 posterior.
      It's a reparameterization a la Opper & Archambault (2009).
    * X0 isn't the posterior mean.
      You need to include the t mean function to it!

***TROUBLESHOOTING / TIPS FOR SUCCESSFUL MODELING:***

***Make sure that the dynamical GP prior (t_kern) always has a unit variance.
(Or, if it's a Combination, make sure that at least one part of it keeps a unit
variance).
Otherwise, you can just push the objective lower and lower via the KL term by
rescaling the LV priors and posteriors smaller and smaller (and the LV
length scales can just shrink proportionally).
An alternative is to place priors on parameters, but this is a better solution
since using priors can be tricky ("how strong?") and can have adverse effects
elsewhere (e.g. prior on LV length scales can prevent dimensions from "turning
off").

"""

from __future__ import absolute_import


from .smodel import SgpModel
from .sgplvm import Sgplvm
from ..util import as_data_holder, debug_check
from ..probability_distributions import ColumnGaussian, RowMatrixGaussian
from ..linalg import jit_cholesky, jit_matrix_inverse

from gpflow.decors import autoflow, params_as_tensors
from gpflow.mean_functions import Zero
from gpflow.probability_distributions import DiagonalGaussian
from gpflow.transforms import Identity
from gpflow import settings
import tensorflow as tf
import numpy as np
from warnings import warn

float_type = settings.float_type
np.random.seed(42)
DEBUG = False

jitter = 10.0 * settings.numerics.jitter_level
st_jitter = 1.0e-12  # For spatiotemporal kernel


class Swgp(Sgplvm):
    """
    Structured Bayesian warped Gaussian process model

    Inputs are propagated through a GP, then are "gridded" against the
    spatiotemporal inputs in latent space as in the usual Sgplvm

    T -> X -> Y

    The following are the major differences in this model:
    Training:
        * x_prior depends on t->X prior (involves t_kern and mean_function)
        * x_0 posterior has a full data covariance matrix
    Predictions:
        * No inference for now.


    """
    def __init__(self, t, h_mu_transformed, h_s_transformed, y, t_kern,
                 kern_list, z=None, jitter=None, st_jitter=None,
                 tie_inducing_points=False, mean_function=None, x_st_test=None):
        """

        :param t: observed inputs
        :param h_mu: [reparameterized LV mean, spatiotemporal inputs]
        :param h_s_transformed: reparam'd LV covariance
        :param y: observed outputs
        :param t_kern: Kernel governing t->X0
        :param kern_list:
        :param z:
        :param mean_function: for t->X0 GP
        :param x_st_test: spatiotemporal test points for predictions
        :type x_st_test: list of np.ndarrays
        """

        super().__init__(h_mu_transformed, h_s_transformed, y, kern_list, z=z,
                         jitter=jitter, st_jitter=st_jitter,
                         tie_inducing_points=tie_inducing_points,
                         x_st_test=x_st_test)
        # self.h_s no longer has to remain positive since this is a
        # reparameterization.
        # self.h_s.transform = Identity()
        self.t = as_data_holder(t)
        self.mean_function = mean_function or Zero(output_dim=self.num_latent)
        self.t_kern = t_kern
        self.input_dim = self.t.shape[1]
        self._predict_x_samples = 0  # Don't have analytic var implemented yet

    @property
    def predict_x_samples(self):
        return self._predict_x_samples

    def infer(self):
        raise NotImplementedError("No inference for warped GP")

    def predict_y(self, t_new, n_samples=None):
        """
        Note: Name input as t_new since this model has t->X->Y
        :param t_new:
        :type t_new: np.ndarray (n* x d_t)
        :return:
        """

        if n_samples is not None:
            self._predict_x_samples = n_samples
        if self.predict_x_samples == 0:
            warn("LV uncertainty not propagated in variance!")
            x_mu, x_s = self._predict_tx(t_new, False)
        else:
            warn("LV uncertainty propagated via sampling")
            assert t_new.shape[0] == 1, "Single t for now"
            # x_mu, x_s = self._predict_tx(t_new, True)
            x_mu, x_s = self._predict_tx(t_new, False)

        f_mu, f_s0 = self._predict_xf(x_mu, x_s)
        if self.predict_x_samples == 0:
            f_s = f_s0
        else:
            f_s = np.zeros(f_s0.shape)
            # l_x_s = np.stack(npz.jit_cholesky(x_s))
            for _ in range(n_samples):
                x_mu_sample = np.random.normal(x_mu, np.sqrt(x_s))
                f_mu_sample, f_s_sample = self._predict_xf(x_mu_sample)
                f_s += f_s_sample + (f_mu_sample - f_mu) ** 2
            f_s /= n_samples

        return self._predict_fy(f_mu, f_s)

    @params_as_tensors
    def _latent_variables(self, full_cov=False):
        """
        Transform the reparameterized latent variables to their actual
        Gaussian variational posterior distribution's parameters (mean & cov)
        :return:
        """

        n_0 = self.n_subgrid[0]
        px_cov = self.t_kern.K(self.t) + self.jitter * tf.eye(self.n_subgrid[0],
                                                              dtype=float_type)
        px_cov_inv = jit_matrix_inverse(px_cov)
        # Reparameterization trick:
        # Opper and Archambault (2009)
        # mu = Kx * mu_bar + m(t)  (mean function addition is mine)
        # For each latent dimension, compute
        # S_j = inv(int(Kx) + Lambda_j).
        qx_mean = px_cov @ self.X0 + self.mean_function(self.t)
        # D x N x N
        #   N = # LVs
        #   D = Latent dimension
        qx_cov_list = [
            tf.reshape(
                jit_matrix_inverse(px_cov_inv + tf.diag(self.h_s[:, i])),
                (1, n_0, n_0))
            for i in range(self.num_latent)]
        if full_cov:
            qx_cov = tf.concat(qx_cov_list, 0)
            varcov_ret = qx_cov
        else:
            qx_var_list = [tf.reshape(tf.diag_part(tf.reshape(qci, (n_0, n_0))),
                                      (-1, 1))
                           for qci in qx_cov_list]
            qx_var = tf.concat(qx_var_list, 1)
            varcov_ret = qx_var

        return qx_mean, varcov_ret

    def _build(self):
        # Avoid the crazy buidling for inference with GP-LVMs
        super(Sgplvm, self)._build()

    @params_as_tensors
    def _build_likelihood(self):
        """
        Same as for the GP-LVM except we need to change the prior and posterior
        over x's first subspace.

        :return:
        """
        n_0 = self.n_subgrid[0]

        # Check for weird stuff.
        if DEBUG:
            # Awful kludge because Parameters don't take writes well :(
            t_var = debug_check(self.t_kern.variance, tf.is_nan)
            t_var = debug_check(t_var, tf.is_inf)
            t_ls = debug_check(self.t_kern.lengthscales,
                               tf.is_nan)
            t_ls = debug_check(t_ls, tf.is_inf)

        px_mean = self.mean_function(self.t)
        px_cov_no_jit = self.t_kern.K(self.t)
        delta = tf.reduce_mean(tf.diag_part(px_cov_no_jit))
        px_cov = tf.add(px_cov_no_jit, delta * self.jitter
                        * tf.eye(self.n_subgrid[0], dtype=float_type),
                        name="Kxx")

        if DEBUG:
            px_cov = debug_check(px_cov, tf.is_nan)
            px_cov = debug_check(px_cov, tf.is_inf)
        px = RowMatrixGaussian(px_mean, px_cov)

        px_cov_inv = tf.identity(jit_matrix_inverse(px_cov), name="KxxInv")
        if DEBUG:
            px_cov_inv = debug_check(px_cov_inv, tf.is_nan)
            px_cov_inv = debug_check(px_cov_inv, tf.is_inf)
        # Reparameterization trick:
        # Opper and Archambault (2009)
        # mu = Kx * mu_bar + m(t)  (mean function addition is mine)
        # For each latent dimension, compute
        # S_j = inv(int(Kx) + Lambda_j).
        qx_mean = px_cov @ self.X0 + self.mean_function(self.t)
        # D x N x N
        #   N = # LVs
        #   D = Latent dimension
        qx_cov_list = [
            tf.reshape(
                jit_matrix_inverse(
                    tf.add(px_cov_inv, tf.diag(self.h_s[:, i]),
                           name="SkInv_{}".format(i + 1))),
                (1, n_0, n_0), name="Sk_{}".format(i + 1))
            for i in range(self.num_latent)]
        qx_var_list = [tf.reshape(tf.diag_part(tf.reshape(qci, (n_0, n_0))),
                                  (-1, 1))
                       for qci in qx_cov_list]
        qx_cov = tf.concat(qx_cov_list, 0, name="SkCov")
        qx_var = tf.concat(qx_var_list, 1, name="SkVar")
        qx = ColumnGaussian(qx_mean, qx_cov)
        qx_diag = DiagonalGaussian(qx_mean, qx_var)

        if DEBUG:
            kludge_factor = 0.0 * (t_var + tf.reduce_mean(t_ls))
        else:
            kludge_factor = tf.constant(0.0, dtype=float_type)
        return super()._build_likelihood(qx, px, qx_diag) + kludge_factor

    @autoflow((float_type, [None, None]), (tf.bool, None))
    @params_as_tensors
    def _predict_tx(self, t_new, full_cov):
        """
        Given an input t_new, compute the predictive density on the latent
        variable.
        :param t_new:
        :return:
        """

        # (px_mean unneeded)
        px_cov = self.t_kern.K(self.t) + self.jitter * tf.eye(
            self.n_subgrid[0],
            dtype=float_type)
        # px_cov_inv = jit_matrix_inverse(px_cov)
        mu_bar = self.X0  # Again, zero mean for now!
        kxs = self.t_kern.K(self.t, t_new)
        kss = tf.cond(full_cov,
                      lambda: self.t_kern.K(t_new),
                      lambda: tf.reshape(self.t_kern.Kdiag(t_new), [-1, 1]))

        # Predict t* -> X*, one dimension at a time.
        mu_star_list = []
        s_star_list = []
        for i in range(self.num_latent):
            mu_bar_i = tf.reshape(mu_bar[:, i], [-1, 1])
            # Damianou, Titsias, and Lawrence, JMLR (2016), p.22
            kxx = px_cov + tf.diag(1.0 / self.h_s[:, i])
            lxx = jit_cholesky(kxx)
            a = tf.matrix_triangular_solve(lxx, kxs, lower=True)
            mu_star_list.append(tf.transpose(kxs) @ mu_bar_i)
            s_star_list.append(tf.cond(full_cov,
                lambda: tf.matmul(a, a, transpose_a=True),
                lambda: tf.reshape(tf.reduce_sum(a ** 2, 0), [-1, 1])))

        # Get the posterior predictive density t* -> X0*
        # Only need mean & variance...
        mu_star = tf.concat(mu_star_list, 1) + self.mean_function(t_new)
        # TODO expand kss to 3D
        # s_star = tf.cond(full_cov,
        #                  lambda: EXPAND(kss) - tf.stack(s_star_list),
        #                  lambda: kss - tf.concat(s_star_list, 1))
        s_star = kss - tf.concat(s_star_list, 1)

        return mu_star, s_star

    @autoflow((float_type, [None, None]), (float_type, [None, None]))
    @params_as_tensors
    def _af_predict_xf(self, x_mu_test, x_s_test):
        return super()._build_predict(DiagonalGaussian(x_mu_test, x_s_test))

    @autoflow((float_type, [None, None]), (float_type, [None, None]))
    @params_as_tensors
    def _predict_fy(self, f_mu, f_s):
        return self.likelihood.predict_mean_and_var(f_mu, f_s)

    def _predict_xf(self, x_mu_test, x_s_test=None):
        """
        Given a realization of the LV distribution, predict the latent outputs
        at all spatiotemporal points.

        This is just the KGPLVM step
        Wrapper allows for unspecified variance-> X* is a point

        :param x_mu_test: latent mean
        :param x_s_test: latent variance (covariance not needed!)
        :return:
        """

        x_s_test = np.zeros(x_mu_test.shape) if x_s_test is None else x_s_test
        return self._af_predict_xf(x_mu_test, x_s_test)
