# Steven Atkinson
# satkinso@nd.edu
# March 16, 2018

"""
Kronecker Bayesian GPLVM

This model assumes that the observed data can be assigned inputs that possess
grid structure so that the relevant covariance matrices inherit Kronecker
structure.

The convention used hereis that the FIRST subgrid ("subgrid 0") is reserved for
the "true" latent variables (i.e. that we are interested in learning), whereas
the other subgrids are reserved for information that is "known" (such as
spatiotemporal coordinates for the data).

In this way, the "interesting" mathematics (similar to the original Bayesian
GP-LVM) occurs on the first subgrid, whereas rather trivial results come out for
the other subgrids.

TODOs:
# MULTICHANNEL: for when we assumed that there was only self.output_dim=1
# IPSPATIAL: for when we assumed that the spatial inducing points were exactly
             the training data's spatial inputs.
"""

from __future__ import absolute_import

from . import SgpModel
from ..kronecker_product import KroneckerProduct
from ..util import as_data_holder, nearest_neighbor, debug_check
from ..kullback_liebler import kl_divergence
from ..linalg import jit_cholesky, jit_self_adjoint_eig, rcond
from ..probability_distributions import ColumnGaussian

from gpflow import settings
from gpflow import likelihoods
from gpflow import transforms
import gpflow.kernels
from gpflow import features
from gpflow.core.compilable import Build
from gpflow.models import Model
from gpflow.params import Parameter
from gpflow.decors import autoflow, params_as_tensors
from gpflow.mean_functions import Zero
from gpflow.expectations import expectation
from gpflow import probability_distributions
from gpflow.training import ScipyOptimizer

import tensorflow as tf
from time import time
import numpy as np
from warnings import warn

DEBUG = True
DEBUG_PREDICT = False
DEBUG_INFER = False
float_type = settings.float_type
np.random.seed(42)


class Sgplvm(SgpModel):
    """
    Bayesian Gaussian process latent variable model exploiting
    Kronecker-structured kernel matrices.

    The inputs are taken as a Cartesian product; the first subspace ("0") is
    taken to correspond to the usual latent variable inputs that must be
    learned.

    The subsequent subspaces correspond to spatiotemporal ("st") inputs that are
    known about the training data.

    You can do (1) uplifting and (2) inference with the trained model.
    """

    def __init__(self, h_mu, h_s, y, kern_list, m_0=None, z=None,
                 x_prior=None, jitter=None, st_jitter=None,
                 tie_inducing_points=False, x_st_test=None, x_st_infer=None,
                 train_kl=True):
        """
        Initialize Kronecker Bayesian GPLVM object. This method only works with
        a Gaussian likelihood.
        :param h_mu: initial latent positions
        :type h_mu: list of np.ndarrays
        :param h_s: variance of latent positions--ONLY FOR THE FIRST SUBGRID
        :type h_s: np.ndarray
        :param y: data matrix (n x d_out)
            (NOTE: provide as "many data, few dimensions")
        :param kern_list: kernels for each subgrid
        :param m_0: number of inducing points (subgrid 0 only)
        :param z: matrix of inducing points, size (m x q).
            Default is a random permutation of x_mean.
        :param x_prior: prior on learned LVs; default is a standard normal
        :type x_prior: TODO
        :param jitter: For inducing points kernel matrix
        :param st_jitter: For spatiotemporal kernel matrices
        :param tie_inducing_points: If true, fixes IPs to follow the LV means
        :param x_st_test: spatiotemporal test points (sorry that this is defined
            at construction...)
        :type x_st_test: list of np.ndarrays
        :param x_st_infer: spatiotemporal points on which observations for
            inference with partially-collapsed bound are observed (sorry that
            this is defined at construction...)
        :type x_st_infer: list of np.ndarrays
        :param train_kl: if False, omit the KL divergence from the training
            objective.  This can be used if you're not allowing the latent
            variables to be optimized over, but isn't recommended otherwise.
        """
        super().__init__(h_mu, y, kern_list,
                         likelihood=likelihoods.Gaussian(),
                         mean_function=Zero())
        del self.X0  # in GPLVM this is a Param, not DataHolder
        self.X0 = Parameter(h_mu[0])
        self.X[0] = self.X0
        self._d_xi = self.X0.shape[1]
        assert h_s.ndim == 2
        self.h_s = Parameter(h_s, transform=transforms.Exp(lower=1.0e-16))

        self.num_latent = h_mu[0].shape[1]  # Latent dimensionality
        self.output_dim = y.shape[1]

        assert np.all(h_mu[0].shape == h_s.shape), \
            "Latent means and variances (1st subgrid) must be the same size"
        assert np.prod([h_mu_i.shape[0] for h_mu_i in h_mu]) == y.shape[0], \
            "Number of inputs ({}) does not match number of outputs ({}).".\
            format([h_mu_i.shape[0] for h_mu_i in h_mu], y.shape[0])

        # inducing points
        self._tie_inducing_points = tie_inducing_points
        if self.tie_inducing_points:
            z = h_mu[0].copy()
            m_0 = len(z)
        else:
            if z is None:
                assert m_0 is not None, "Must provide either m or z"
                # By default we initialize by subset of initial latent points
                lv_means, _ = self._latent_variables()
                z = np.random.permutation(h_mu[0].copy())[:m_0]
            elif m_0 is None:
                m_0 = len(z)

        self.feature = features.InducingPoints(z)
        if self.tie_inducing_points:
            self.feature.trainable = False
        self.m_subgrid = [len(self.feature)] + self.n_subgrid[1:]
        n_st = np.prod(self.n_subgrid[1:])

        assert len(self.feature) == m_0
        assert h_mu[0].shape[1] == self.num_latent

        # deal with parameters for the prior mean variance of X
        if x_prior is None:
            x_prior = probability_distributions.DiagonalGaussian(
                np.zeros(h_mu[0].shape), np.ones(h_s.shape))
        self.x_prior = x_prior

        assert np.all(self.x_prior.mu.shape == self.X0.shape) and \
               np.all(self.x_prior.cov.shape == self.X0.shape), \
            "LV prior size must match first subgrid of inputs"

        # For inferring LV of a test realization
        self._n_0_test = None  # Observations...
        # Test parameters have to not be protected or else they don't get picked
        # up as trainable.
        # Initialize to something so that compile works...
        self.y_test = as_data_holder(np.zeros((n_st, self.output_dim)))
        self.h_mu_test = Parameter(np.zeros((1, self.X0.shape[1])))
        self.h_s_test = Parameter(np.ones((1, self.X0.shape[1])),
                                  transform=transforms.positive)
        self._train_objective = None
        self._infer_objective = None

        # Precomputed quantities:

        # Train time:
        self._train_kl = train_kl
        self._tr_yyt = None

        # Infer time:
        self._tr_yyt_test = None
        self._l_kuu_0 = None
        self._l_kff_st = None
        self._psi0_train = None
        self._li_psi1t_0 = None
        self._c_0_train = None
        self._lambda_c_st = None
        self._q_c_st = None
        self._kl_train = None
        self._precomputed_infer = False
        # 0=no (train), 1=same spatials, 2=partially collapsed (any spatials)
        self._infer_mode = 0

        # For inference with partially-collapsed bound:
        self.sigma2_infer = Parameter(self.likelihood.variance.value,
                                      transform=transforms.positive)
        self._train_bound = None
        self._k_psi_inv_psi1t_y = None
        self._d_diag_inv = None
        self._lt_inv_q_0 = None
        self._tr_c_infer_st = None
        self._qlplq_st_diag = None
        self._precomputed_infer_pcb = False
        # Inference spatial points
        # Stored as Parameters with trainable=False because DataHolders don't
        # work if you don't know how many rows there are.
        self.input_st_infer_names = [name + "_infer"
                                     for name in self.input_names[1:]]
        if x_st_infer is None:
            x_st_infer = [x_i.value for x_i in self.X[1:].copy()]
        [setattr(self, name, Parameter(x_i, trainable=False))
         for name, x_i in zip(self.input_st_infer_names, x_st_infer)]
        self._precompute_train()

        # Forward predict time:
        # Stored as Parameters with trainable=False because DataHolders don't
        # work if you don't know how many rows there are.
        if x_st_test is None:
            x_st_test = [x_i.value for x_i in self.X[1:].copy()]
        [setattr(self, name, Parameter(x_i, trainable=False))
         for name, x_i in zip(self.input_st_test_names, x_st_test)]
        self._n_test_subgrid = [getattr(self, name).value.shape[0]
                                for name in self.input_st_test_names]

        # Forward predict (full cov) time:
        self._kss_st = None
        self._g_st = None
        self._e_xi = None
        self._d_tilde_sqrt_mtx = None
        self._precomputed_predict_full_cov = False

        # Numerics...
        self._jitter = jitter or 10.0 * settings.numerics.jitter_level
        self._st_jitter = st_jitter or 1.0e-12 # For spatiotemporal kernel

    @property
    def d_xi(self):
        return self._d_xi

    @property
    def infer_mode(self):
        return self._infer_mode

    @infer_mode.setter
    def infer_mode(self, val):
        if not val == self.infer_mode:
            self._infer_mode = val
            self.clear()

    @property
    def jitter(self):
        return self._jitter

    @property
    def precomputed_infer(self):
        return self._precomputed_infer

    @property
    def precomputed_infer_pcb(self):
        return self._precomputed_infer_pcb

    @property
    def n_xi(self):
        return self.latent_variables()[0].shape[0]

    @property
    def n_test_subgrid(self):
        return self._n_test_subgrid

    @property
    def st_jitter(self):
        return self._st_jitter

    @property
    def tie_inducing_points(self):
        return self._tie_inducing_points

    def infer(self, y_test, h_mu_test_init=None, h_s_test_init=None,
              init_strategy="nearest_neighbor", opt=None, maxiter=None,
              replace_objective=False):
        """
        Infer the latent variable of a test observation via optimization
        TODO more than one at a time

        :param y_test: The observation(s) to be inferred
        :type y_test: np.ndarray of size (n_xi* x n_st) ("Many dimensions")
            (Reshape to "many data, few dimensions" happens here.)
        :param h_mu_test_init: Initial guess(es) at LV mean.
        :type h_mu_test_init: np.ndarray
        :param h_s_test_init: Initial guess(es) at LV variance
        :type h_s_test_init: np.ndarray
            If both mean and var are None, use init_strategy to guess.
        :param init_strategy: How to choose an initial guess for the LVs if none
            is provided
        :type init_strategy: String.  Options:
            * "prior"
            * "nearest_neighbor"
        :param opt: The optimizer to be used
        :type opt: gpflow.Optimizer
        :param maxiter: Maximum iterations for inference optimization
        :param disp: disp arg for optimization
        :param replace_objective: Whether we want to replace the objective back
            to training when we're done.
        :return:
        """

        # Checks:
        n_st = np.prod(self.n_subgrid[1:])
        assert y_test.shape[1] == n_st, \
            "y_test does not have correct number of spatiotemporal points"

        opt = ScipyOptimizer() or opt

        # Freeze all training variables
        self._freeze_training_params()

        # Get initial guesses
        if h_mu_test_init is None and h_s_test_init is None:
            d_0 = self.X0.shape[1]
            # TODO dispatch
            if init_strategy == "prior":
                h_mu_test_init = np.zeros((y_test.shape[0], d_0))
            elif init_strategy == "nearest_neighbor":
                y_train = self.Y.value.reshape((-1, n_st))
                i_min, _ = nearest_neighbor(y_train, y_test)
                if self.is_built_coherence(self.enquire_graph()) is Build.NO:
                    self.build()
                h_mu, _ = self.latent_variables()
                h_mu_test_init = h_mu[i_min].reshape((1, -1))
            else:
                raise Exception("Initialization strategy {} not recognized".
                                format(init_strategy))
            h_s_test_init = 1.0 * np.ones((y_test.shape[0], d_0))

        # Input test data:
        self._n_0_test = y_test.shape[0]
        self.y_test.assign(y_test.reshape((-1, self.output_dim)))
        self.h_mu_test.assign(h_mu_test_init)
        self.h_s_test.assign(h_s_test_init)

        # Ensure that everything is pre-computed
        self._precompute_infer()

        # Build the test-time graph
        if not self.infer_mode == 1:
            self.infer_mode = 1  # calls clear()
            # Note: if you want to train, you need to set test_mode to False
            # and then build() again!
        if self.is_built_coherence(self.enquire_graph()) is Build.NO:
            self.build()
        print("Inferring...")
        tic = time()
        if maxiter is None or maxiter > 0:  # Only skip if maxiter=0
            kwargs = {}
            if maxiter is not None:
                kwargs.update({"maxiter": maxiter})
            opt.minimize(self, **kwargs)
        toc = time() - tic
        h_mu_test, h_s_test = (self.h_mu_test.value, self.h_s_test.value)
        print("Inference done in {} sec.  d_mu={}, ds={}".format(
            toc,
            np.linalg.norm(h_mu_test - h_mu_test_init),
            np.linalg.norm(h_s_test - h_s_test_init),
            format_spec="%6f"))

        # Unfreeze training variables
        warn("Need to unfreeze training parameters if you want to train again!")

        return h_mu_test, h_s_test

    def infer_pcb(self, y_test, h_mu_test_init=None, h_s_test_init=None,
                  init_strategy="prior", test_noise=False, opt=None,
                  maxiter=None):
        """
        Infer the latent variable of a test observation via optimization using
        the "partially-collapsed bound" approach

        TODO: spatial inducing points aren't training spatials

        :param y_test: The observation to be inferred
        :type y_test: np.ndarray of size (n_xi* x n_st) ("Many dimensions")
            (Reshape to "many data, few dimensions" happens here.)
        :param h_mu_test_init: Initial guess(es) at LV mean.
        :type h_mu_test_init: np.ndarray
        :param h_s_test_init: Initial guess(es) at LV variance
        :type h_s_test_init: np.ndarray
            If both mean and var are None, use init_strategy to guess.
        :param init_strategy: How to choose an initial guess for the LVs if none
            is provided
        :type init_strategy: String.  Options:
            * "prior"
        :param test_noise: if True, this observation has its own noise parameter
            (Useful for inference with noisy observations that are different
            from the training data)
        :param opt: The optimizer to be used
        :type opt: gpflow.Optimizer
        :param maxiter: Maximum iterations for inference optimization
        :return:
        """

        # Initialize noise to training value
        self.sigma2_infer = self.likelihood.variance.value
        self.sigma2_infer.trainable = test_noise

        # Checks:
        n_st = np.prod([getattr(self, name).value.shape[0]
                        for name in self.input_st_infer_names])
        n_channels = self.output_dim
        assert y_test.shape[1] == n_st * n_channels, \
            "y_test does not have correct number of spatial points"

        opt = ScipyOptimizer() or opt

        # Freeze all training variables
        self._freeze_training_params()

        # Get initial guess
        # Can't use nearest neighbor because we aren't necessarily observing
        # anywhere in common!
        self._init_infer_lv_posterior(y_test.shape[0])

        # Input test data:
        self._n_0_test = y_test.shape[0]
        # MULTICHANNEL
        self.y_test.assign(y_test.reshape((-1, self.output_dim)))

        # Ensure that everything is pre-computed
        self._precompute_infer_pcb()

        # Build the test-time graph
        if not self.infer_mode == 2:
            self.infer_mode = 2  # calls clear()
            # Note: if you want to train, you need to set test_mode to False
            # and then build() again!
        if self.is_built_coherence(self.enquire_graph()) is Build.NO:
            self.build()
        print("Inferring...")
        tic = time()
        if maxiter is None or maxiter > 0:
            kwargs = {}
            if maxiter is not None:
                kwargs.update({"maxiter": maxiter})
            opt.minimize(self, **kwargs)
        toc = time() - tic
        print("Inference done in {} sec.".format(toc))
        if test_noise:
            print("sigma2*={}\n(train={})".format(
                self.sigma2_infer.value, self.likelihood.variance.value))

        # Unfreeze training variables
        warn("Need to unfreeze training parameters if you want to train again!")

        return self.inferred_latent_variables()

    @autoflow()
    @params_as_tensors
    def inferred_latent_variables(self):
        return self.h_mu_test, self.h_s_test

    @autoflow()
    @params_as_tensors
    def latent_variables(self, full_cov=False):
        """
        Get the actual latent variables, not their reparameterized parameters
        :return:
        """
        return self._latent_variables(full_cov)

    @autoflow((settings.float_type, [None, None]),
              (settings.float_type, [None, None]))
    def predict_y(self, mu_0_test, s_0_test=None):
        """
        Predictive mean & variance when the inputs are DISTRIBUTIONS
        with mean mu_0_test and variance s_0_test.
        """
        return self._predict_y(
            probability_distributions.DiagonalGaussian(mu_0_test, s_0_test))

    @autoflow((settings.float_type, [None, None]),
              (settings.float_type, [None, None, None]))
    def predict_y_full_cov_in(self, mu_0_test, s_0_test=None):
        """
        Predictive mean & variance when the inputs are DISTRIBUTIONS
        with mean mu_0_test and variance s_0_test.
        """
        return self._predict_y(
            probability_distributions.Gaussian(mu_0_test, s_0_test))

    @autoflow((settings.float_type, [None, None]),
              (settings.float_type, [None, None]))
    @params_as_tensors
    def predict_y_sigma2_infer(self, mu_0_test, s_0_test=None):
        """
        Predictive mean & variance using likelihood from inference.
        """
        assert isinstance(self.likelihood, gpflow.likelihoods.Gaussian), \
            "Need Gaussian likelihood"

        #X->F
        f_mean, f_var = self._build_predict(
            probability_distributions.DiagonalGaussian(mu_0_test, s_0_test))

        # F -> Y
        return f_mean, f_var + self.sigma2_infer

    @autoflow((settings.float_type, [None, None]),
              (settings.float_type, [None, None, None]))
    @params_as_tensors
    def predict_y_sigma2_infer_full_cov_in(self, mu_0_test, s_0_test=None):
        """
        Predictive mean & variance using likelihood from inference.
        """
        assert isinstance(self.likelihood, gpflow.likelihoods.Gaussian), \
            "Need Gaussian likelihood"

        # X->F
        f_mean, f_var = self._build_predict(
            probability_distributions.Gaussian(mu_0_test, s_0_test))

        # F -> Y
        return f_mean, f_var + self.sigma2_infer

    def predict_y_full_cov(self, mu_0_test):
        """
        For a single LV test input, at the same st points as the training
        data.

        :param mu_0_test: LV input
        :return:
        """

        # Precompute terms that don't have to do with test time:
        if not self._precomputed_predict_full_cov:
            self._kss_st, self._g_st, self._e_xi, self._d_tilde_sqrt_mtx = \
                self._precompute_predict_full_cov()
            self._precomputed_predict_full_cov = True

        ksu_xi, kss_xi = self._compute_test_time(mu_0_test)
        g_xi = ksu_xi @ self._e_xi
        kss = kss_xi * self._kss_st
        # Pre-multiply dtilde_sqrt against Ksu^xi (exploit vectorization):
        dksu = self._d_tilde_sqrt_mtx * g_xi.T

        x_st_test = [getattr(self, name) for name in self.input_st_test_names]
        n_st_test = np.prod([x.shape[0]for x in x_st_test])
        y_cov = np.zeros((n_st_test, n_st_test))
        for ksu_xi_i, dksu_i in zip(ksu_xi.flatten(), dksu):
            h = self._g_st * np.atleast_2d(dksu_i)
            y_cov -= h @ h.T
        y_cov += kss + self.likelihood.variance.value * np.eye(n_st_test)
        y_mean, _ = self.predict_y(mu_0_test, np.zeros(mu_0_test.shape))

        if DEBUG_PREDICT:
            warn("Debug prints!")
            np.savetxt("debug/g_st.dat", self._g_st)
            np.savetxt("debug/e_xi.dat", self._e_xi)
            np.savetxt("debug/kss_st.dat", self._kss_st)
            np.savetxt("debug/d_tilde_mtx.dat", self._d_tilde_sqrt_mtx)
            np.savetxt("debug/y_cov.dat", y_cov)
        return y_mean, y_cov

    # Protected...

    @params_as_tensors
    def _latent_variables(self, full_cov=False):
        """
        Means & variances
        """
        assert not full_cov, "full_cov=True not supported!"
        return self.X0, self.h_s

    def _freeze_training_params(self):
        """
        Bit of a kludge to turn off everything that was trainable
        TODO keep track of what got frozen
        """
        warn("Hard-freeze training params")
        self.likelihood.variance.trainable = False
        for kern in self.kern_list:
            kern.trainable = False
        self.X0.trainable = False
        self.h_s.trainable = False
        self.feature.trainable = False

    @params_as_tensors
    def _get_infer_lv_posterior(self):
        """
        Variational posterior on latent variables to be inferred at test time.
        :return:
        """
        return probability_distributions.DiagonalGaussian(
            self.h_mu_test, self.h_s_test)

    @params_as_tensors
    def _get_infer_lv_prior(self, **kwargs):
        """
        Prior on latent variables to be inferred at test time
        :return:
        """
        for key in kwargs.keys():
            warn("Unused kwarg: {}".format(key))
        # TODO More than one infer case at a time
        n_infer = 1
        return probability_distributions.DiagonalGaussian(
            tf.zeros((n_infer, self.d_xi), dtype=float_type),
            tf.ones((n_infer, self.d_xi), dtype=float_type))

    def _init_infer_lv_posterior(self, n, h_mu_test_init=None,
                                 h_s_test_init=None):
        """

        :return:
        """
        if h_mu_test_init is None and h_s_test_init is None:
            d_0 = self.X0.shape[1]
            h_mu_test_init = np.zeros((n, d_0))
            h_s_test_init = 1.0 * np.ones((n, d_0))

        # Assign
        self.h_mu_test.assign(h_mu_test_init)
        self.h_s_test.assign(h_s_test_init)

    def _precompute_train(self):
        """
        Pre-compute some terms used in training
        """
        self._tr_yyt = np.sum(self.Y.value ** 2)

    def _precompute_infer(self):
        """
        Runs Precomputations for inference and stores the results as np arrays.
        TODO Automatically detect when this needs to be re-evaluated
        """

        # Always have to pre-compute Tr(YY') for the test data:
        self._tr_yyt_test = np.sum(self.y_test.value ** 2)

        # "Safe mode": always pre-compute for now.
        if self.precomputed_infer:
            warn("infer detected precomputed training data; skip (be careful!)")
        else:
            if self.is_built_coherence() == Build.NO:
                self.build()
                self.initialize()
            self._l_kuu_0, self._l_kff_st, \
            self._psi0_train, \
            self._li_psi1t_0, \
            self._c_0_train, \
            self._lambda_c_st, self._q_c_st, \
            self._kl_train \
                = self._precompute_infer_tf()
            self._precomputed_infer = True

    @autoflow()
    @params_as_tensors
    def _precompute_infer_tf(self):
        """
        TensorFlow pre-compute instructions for terms that are constant at
        test time for inference.

        psi2
        psi1
        KL of training
        Chol of Kuu
        """
        n_i = self.n_subgrid
        n_st = np.prod(n_i[1:])
        x_st = [getattr(self, name) for name in self.input_names[1:]]
        h_mu, h_s = self._latent_variables()
        qx = probability_distributions.DiagonalGaussian(h_mu, h_s)
        kern_0 = self.kern_list[0]
        kern_st = self.kern_list[1:]
        psi0 = n_st * tf.reduce_sum(expectation(qx, kern_0))
        psi1_0 = expectation(qx, (kern_0, self.feature))
        psi2_0 = tf.reduce_sum(expectation(qx, (kern_0, self.feature),
                                           (kern_0, self.feature)), axis=0)
        l_kuu_0 = jit_cholesky(self.feature.Kuu(self.kern_list[0],
                                               jitter=self.jitter))
        kff_st = KroneckerProduct(
            [k.K(x_st_i) + self._st_jitter * tf.eye(n_ij, dtype=float_type)
             for n_ij, x_st_i, k in zip(n_i[1:], x_st, kern_st)])
        l_kff_st = kff_st.cholesky()

        tmp = tf.matrix_triangular_solve(l_kuu_0, psi2_0, lower=True)
        c_0 = tf.matrix_triangular_solve(l_kuu_0, tf.transpose(tmp), lower=True)
        c = KroneckerProduct([c_0] + (l_kff_st.transpose() @ l_kff_st).x)
        lambda_c_kp, q_c = c.self_adjoint_eig()
        lambda_c_st = lambda_c_kp.x[1:]
        q_c_st = q_c.x[1:]

        li_psi1t_0 = tf.matrix_triangular_solve(l_kuu_0, tf.transpose(psi1_0),
                                                lower=True)
        qx = probability_distributions.DiagonalGaussian(h_mu, h_s)
        px = self.x_prior
        kl_train = kl_divergence(qx, px)
        return l_kuu_0, l_kff_st.x, \
               psi0, \
               li_psi1t_0, \
               c_0, \
               lambda_c_st, q_c_st, \
               kl_train

    def _precompute_infer_pcb(self):
        """
        Runs precomputations for inference and stores the results as np arrays.
        TODO Automatically detect when this needs to be re-evaluated
        """

        # Always have to pre-compute Tr(YY') for the test data:
        self._tr_yyt_test = np.sum(self.y_test.value ** 2)

        # "Safe mode": always pre-compute for now.
        if self.precomputed_infer_pcb:
            warn("infer detected precomputed training data; skip (be careful!)")
        else:
            if self.is_built_coherence() == Build.NO:
                self.build()
                self.initialize()
            self._train_bound = self.compute_log_likelihood() + \
                                self.compute_log_prior()
            self._k_psi_inv_psi1t_y, \
            self._d_diag_inv, \
            self._lt_inv_q_0, \
            self._tr_c_infer_st, \
            self._l_kuu_0, \
            self._qlplq_st_diag \
                = self._precompute_infer_pcb_tf()
            self._precomputed_infer_pcb = True

    @autoflow()
    @params_as_tensors
    def _precompute_infer_pcb_tf(self):
        """
        TensorFlow pre-compute instructions for terms that are constant at
        test time for inference.

        """
        n_i = self.n_subgrid
        n = np.prod(n_i)
        m = np.prod(self.m_subgrid)
        d_y = self.output_dim
        x_st = [getattr(self, name) for name in self.input_names[1:]]
        x_st_infer = [getattr(self, name) for name in self.input_st_infer_names]
        h_mu, h_s = self._latent_variables()
        qx = probability_distributions.DiagonalGaussian(h_mu, h_s)
        kern_0 = self.kern_list[0]
        kern_st = self.kern_list[1:]
        sigma2 = self.likelihood.variance

        psi1_0 = expectation(qx, (kern_0, self.feature))
        psi2_0 = tf.reduce_sum(expectation(qx, (kern_0, self.feature),
                                           (kern_0, self.feature)), axis=0)
        l_kuu_0 = jit_cholesky(self.feature.Kuu(self.kern_list[0],
                                                jitter=self.jitter))
        kff_st = KroneckerProduct(
            [k.K(x_st_i) + self._st_jitter * tf.eye(n_ij, dtype=float_type)
             for n_ij, x_st_i, k in zip(n_i[1:], x_st, kern_st)])
        l_kff_st = kff_st.cholesky()

        l_kuu = KroneckerProduct([l_kuu_0] + l_kff_st.x)
        # IPSPATIAL
        psi1 = KroneckerProduct([psi1_0] + kff_st.x)

        # C's and D's
        tmp = tf.matrix_triangular_solve(l_kuu_0, psi2_0, lower=True)
        c_0 = tf.matrix_triangular_solve(l_kuu_0, tf.transpose(tmp), lower=True)
        # IPSPATIAL
        c_st = l_kff_st.transpose() @ l_kff_st
        c = KroneckerProduct([c_0] + c_st.x)
        lambda_c_kp, q_c = c.self_adjoint_eig()
        d_diag_inv = 1.0 / (lambda_c_kp.eval() + sigma2)
        # Infer case:
        # IPSPATIAL
        kus_st_infer = KroneckerProduct([kern_i.K(x_st_i, x_infer_st_i)
                                         for kern_i, x_st_i, x_infer_st_i
                                         in zip(kern_st, x_st, x_st_infer)])
        tmp_infer = l_kff_st.matrix_triangular_solve(kus_st_infer, lower=True)
        c_st_infer = tmp_infer @ tmp_infer.transpose()

        lt_inv_q = l_kuu.transpose().matrix_triangular_solve(q_c, lower=False)
        qt_l_inv = lt_inv_q.transpose()
        qt_l_inv_psi1t = qt_l_inv @ psi1.transpose()
        mm_shape = [(m_ij, m_ij) for m_ij in self.m_subgrid]
        mn_shape = [(m_ij, n_ij)
                    for m_ij, n_ij in zip(self.m_subgrid, self.n_subgrid)]
        lt_inv_q.shape_hint = mm_shape
        qt_l_inv.shape_hint = mm_shape
        qt_l_inv_psi1t.shape_hint = mn_shape
        qt_l_inv_psi1t_y = qt_l_inv_psi1t.matmul(self.Y, shape_hint=(n, d_y))
        k_psi_inv_psi1t_y = lt_inv_q.matmul(d_diag_inv * qt_l_inv_psi1t_y,
                                      shape_hint=(m, d_y))
        trace_c_test_st = np.prod([tf.trace(c_st_infer_i)
                                   for c_st_infer_i in c_st_infer.x])
        # "qlplq_st" is spatiotemporal submatrices of
        # Q_C^T * L^-1 * Psi2^(*) * L^-T * Q_C
        # Change this once we stop using training spatials as IP spatials
        psi2_test_st = kus_st_infer @ kus_st_infer.transpose()
        qt_l_inv_st = KroneckerProduct(qt_l_inv.x[1:])
        qlplq_st = qt_l_inv_st @ psi2_test_st @ qt_l_inv_st.transpose()

        return k_psi_inv_psi1t_y, \
               d_diag_inv, \
               lt_inv_q.x[0], \
               trace_c_test_st, \
               l_kuu.x[0], \
               qlplq_st.diag_part(col=True).eval()

    @autoflow()
    @params_as_tensors
    def _precompute_predict_full_cov(self):
        """
        Pre-compute matrices used for full-cov predictions

        :return:
        """
        n_i = self.n_subgrid
        n_st = np.prod(n_i[1:])
        m_xi = self.m_subgrid[0]
        # m = m_xi * n_st
        x_st = [getattr(self, name) for name in self.input_names[1:]]
        x_st_test = [getattr(self, name) for name in self.input_st_test_names]
        h_mu, h_s = self._latent_variables()
        qx = probability_distributions.DiagonalGaussian(h_mu, h_s)
        sigma2 = self.likelihood.variance

        kern_0 = self.kern_list[0]
        kern_st = self.kern_list[1:]
        # psi0 = n_st * tf.reduce_sum(expectation(qx, kern_0))
        # psi1_0 = expectation(qx, (kern_0, self.feature))
        psi2_0 = tf.reduce_sum(expectation(qx, (kern_0, self.feature),
                                           (kern_0, self.feature)), axis=0)
        l_kuu_0 = jit_cholesky(self.feature.Kuu(self.kern_list[0],
                                                jitter=self.jitter))
        kff_st = KroneckerProduct(
            [k.K(x_st_i) + self._st_jitter * tf.eye(n_ij, dtype=float_type)
             for n_ij, x_st_i, k in zip(n_i[1:], x_st, kern_st)])
        l_kff_st = kff_st.cholesky()

        tmp = tf.matrix_triangular_solve(l_kuu_0, psi2_0, lower=True)
        c_0 = tf.matrix_triangular_solve(l_kuu_0, tf.transpose(tmp), lower=True)
        c = KroneckerProduct([c_0] + (l_kff_st.transpose() @ l_kff_st).x)
        lambda_c_kp, q_c = c.self_adjoint_eig()
        q_c_st = KroneckerProduct(q_c.x[1:])
        lambda_c = lambda_c_kp.eval()
        d_diag = lambda_c + sigma2  # Diagonal elements (2D column vec)
        d_tilde = 1.0 - sigma2 / d_diag
        d_tilde_sqrt = tf.sqrt(d_tilde)

        # L^-T is UPPER triangular!
        e_xi = tf.matrix_triangular_solve(tf.transpose(l_kuu_0), q_c.x[0],
                                          lower=False)
        e_st = l_kff_st.transpose().matrix_triangular_solve(q_c_st, lower=False)
        ksu_st = KroneckerProduct([k.K(xti, xi) for k, xti, xi
                                   in zip(kern_st, x_st_test, x_st)])
        g_st = ksu_st @ e_st
        # g_st = l_kff_st @ q_c_st  # So long as Ksu=Kff for st
        kss_st = tf.identity(
            KroneckerProduct(
                [k.K(x_st_i) +
                 self._st_jitter * tf.eye(ni, dtype=float_type)
                 for x_st_i, ni, k
                 in zip(x_st_test, self.n_test_subgrid, kern_st)]).eval(),
            name="Kss_st")

        return kss_st, g_st.eval(), e_xi, \
               tf.reshape(d_tilde_sqrt, (m_xi, n_st))

    @params_as_tensors
    def _predict_y(self, qx):
        """
        X->F->Y

        :param qx: Either DiagonalGaussian or Gaussian density over LV input
        :return:
        """
        f_mean, f_var = self._build_predict(qx)
        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    @autoflow((settings.float_type, [None, None]))
    @params_as_tensors
    def _compute_test_time(self, h_mu_test):
        """
        Things that depend on the test stochastic input
        :return: Ksu^xi, Kss^xi
        """
        return tf.transpose(self.feature.Kuf(self.kern_list[0], h_mu_test)), \
               self.kern_list[0].K(h_mu_test)

    @params_as_tensors
    def _build_likelihood(self, qx=None, px=None, qx_diag=None):
        """
        Variational lower bound on the marginal log likelihood
        For definitions of the variables, consult the technical report.

        A lot of the computations here look similar to what you'd do for a 
        normal Bayesian GP-LVM.  If you're used to GPflow, here's some notes on
        variables that might look similar:
        GPflow    | My convention
        -------------------------
        sigma2    | beta^(-1)
        AAT       | C
        B         | beta * A
        log_det_B | log|A| + m log(beta)

        :param qx: The variational posterior over the latent variables
            in subgrid 0.
            If None, uses self.h_mu and self.h_s to make a DiagonalGaussian
        :param px: The prior over the LVs in subgrid 0. If none, use
            self.x_prior
        :return:
        """
        n_i = self.n_subgrid
        n_st = np.prod(n_i[1:])
        n = self.n  # All training data
        m_0 = len(self.feature)
        m = n_st * m_0  # Include inducing points for st inputs
        x_st = [getattr(self, name) for name in self.input_names[1:]]
        h_mu, h_s = self._latent_variables()
        qx = qx or probability_distributions.DiagonalGaussian(h_mu, h_s)
        if qx_diag is None:
            qx_diag = qx if \
                isinstance(qx, probability_distributions.DiagonalGaussian) \
                else qx.diag()
        px = px or self.x_prior
        kern_list = [getattr(self, name) for name in self.kern_names]
        kern_0 = kern_list[0]
        kern_st = kern_list[1:]

        sigma2 = self.likelihood.variance

        # Kernel matrices for spatiotemporal points:
        kff_st = KroneckerProduct(
            [k.K(x_st_i) + self._st_jitter * tf.eye(n_ij, dtype=float_type)
             for n_ij, x_st_i, k in zip(n_i[1:], x_st, kern_st)])
        l_kff_st = kff_st.cholesky()

        # Inducing points
        if self.tie_inducing_points:
            self.feature.Z = tf.reshape(qx.mu, tf.shape(self.feature.Z),
                                        name="H_u")
        kuu_0 = tf.identity(self.feature.Kuu(kern_0, jitter=self.jitter),
                            name="Kuu_0")
        if DEBUG:
            kuu_0 = debug_check(kuu_0, tf.is_nan)
            kuu_0 = debug_check(kuu_0, tf.is_inf)
            kuu_0 = tf.identity(kuu_0, name="Kuu_0")  # Keep name
        l_kuu_0 = jit_cholesky(kuu_0, name="L_Kuu_0")
        l_kuu = KroneckerProduct([l_kuu_0] + l_kff_st.x)

        # Statistics
        # (Only depend on the diagonal of q(x))
        psi0 = n_st * tf.reduce_sum(expectation(qx_diag, kern_0))
        psi1_0 = expectation(qx_diag, (kern_0, self.feature))
        psi2_0 = tf.reduce_sum(expectation(qx_diag, (kern_0, self.feature),
                                           (kern_0, self.feature)), axis=0,
                               name="psi2_0")

        if DEBUG:
            # If all inducing points are too far away from latent variables,
            # then Psi2 will be zero to machine precision, and ensuing
            # computations will have NaNs and zeros.
            # Check here:
            ip_min = tf.reduce_min(self.feature.Z)
            ip_max = tf.reduce_max(self.feature.Z)
            mu_min = tf.reduce_min(h_mu)
            mu_max = tf.reduce_max(h_mu)
            s_min = tf.reduce_min(h_s)
            s_max = tf.reduce_max(h_s)
            psi1_0_max = tf.reduce_max(tf.abs(psi1_0))
            psi2_0_max = tf.reduce_max(tf.abs(psi2_0))

        # IPSPATIAL
        psi1 = KroneckerProduct([psi1_0] + kff_st.x)

        # Compute intermediate matrices
        # c = inv(L) * psi2 * inv(L')
        # st parts are just the Kff's!
        tmp = tf.matrix_triangular_solve(l_kuu_0, psi2_0, lower=True)
        c_0 = tf.matrix_triangular_solve(l_kuu_0, tf.transpose(tmp), lower=True,
                                         name="c_0")
        # IPSPATIAL
        c = KroneckerProduct([c_0] + (l_kff_st.transpose() @ l_kff_st).x)
        lambda_c_kp, q_c = c.self_adjoint_eig()
        lambda_c = lambda_c_kp.eval()
        d_diag = lambda_c + sigma2  # Diagonal elements (2D column vec)
        logdet_a = tf.reduce_sum(tf.log(d_diag))  # a = q_c * d * q_c'
        trace_c = tf.reduce_sum(lambda_c)

        # Tr(Phi * inv(k_psi) * Phi') = Tr(D^(-1)BB')
        qlp1 = q_c.transpose() @ (l_kuu.matrix_triangular_solve(
            psi1.transpose(), lower=True))
        mn_size = [(m_i, n_i)
                   for m_i, n_i in zip(self.m_subgrid, self.n_subgrid)]
        qlp1.shape_hint = mn_size
        b = qlp1.matmul(self.Y, shape_hint=(self.n, self.output_dim))
        diag_bbt = tf.reduce_sum(b ** 2, 1)
        trace_pkp = tf.reduce_sum(diag_bbt / tf.reshape(d_diag, [-1]))

        # Bring it together
        log_2pi = tf.cast(tf.log(2.0 * np.pi), float_type)
        elbo_1 = 0.5 * self.output_dim * (-(n - m) * tf.log(sigma2)
                                          - n * log_2pi - logdet_a)

        elbo_2 = - 0.5 / sigma2 * (
            self._tr_yyt - trace_pkp + self.output_dim * (psi0 - trace_c)
        )
        if self._train_kl:
            ll = elbo_1 + elbo_2 - kl_divergence(qx, px)
        else:
            warn("Training without KL on latent variables.  " +
                 "Be sure you know what you're doing!")
            ll = elbo_1 + elbo_2

        return ll

    @params_as_tensors
    def _build_predict(self, qx_star):
        """
        Compute predictive posterior density over latent outputs given a LV

        TODO more than 1 point at a time
        TODO mean function

        :param mu_test: Mean of point to predict at.
        :param s_test: covariance of test point distribution

        :returns: mean & var/cov of the predictive distribution over latent
            outputs
        """
        n_i = self.n_subgrid
        m_i = self.m_subgrid
        n = self.n  # All training data
        m = np.prod(m_i)
        d_out = self.output_dim

        x = [getattr(self, name) for name in self.input_names]
        x_st = x[1:]
        x_st_test = [getattr(self, name) for name in self.input_st_test_names]
        n_star_i = [1] + [x_st_test_i.get_shape().as_list()[0]
                          for x_st_test_i in x_st_test]
        h_mu, h_s = self._latent_variables()
        kern_list = [getattr(self, name) for name in self.kern_names]
        kern_0 = kern_list[0]
        kern_st = kern_list[1:]
        # Var post on learned LVs
        qx = probability_distributions.DiagonalGaussian(h_mu, h_s)
        sigma2 = self.likelihood.variance

        # Tie IPs:
        if self.tie_inducing_points:
            self.feature.Z = tf.reshape(qx.mu, tf.shape(self.feature.Z),
                                        name="H_u")

        # Kernel matrices
        kuu_0 = self.feature.Kuu(kern_0, jitter=self.jitter)
        kff_st = [k.K(x_st_i) + self._st_jitter * tf.eye(n_ij, dtype=float_type)
                  for n_ij, x_st_i, k in zip(n_i[1:], x_st, kern_st)]
        # Choleskies
        l_kuu_0 = jit_cholesky(kuu_0)
        l_kff_st = KroneckerProduct([jit_cholesky(kff_st_i)
                                     for kff_st_i in kff_st])
        l_kuu = KroneckerProduct([l_kuu_0] + l_kff_st.x)

        # Statistics
        psi1_0 = expectation(qx, (kern_0, self.feature))
        psi2_0 = tf.reduce_sum(expectation(qx, (kern_0, self.feature),
                                           (kern_0, self.feature)), axis=0)

        # Compute intermediate matrices
        tmp = tf.matrix_triangular_solve(l_kuu_0, psi2_0, lower=True)
        c_0 = tf.matrix_triangular_solve(l_kuu_0, tf.transpose(tmp), lower=True)
        c = KroneckerProduct([c_0] + (l_kff_st.transpose() @ l_kff_st).x)
        lambda_c_kp, q_c = c.self_adjoint_eig()
        lambda_c = lambda_c_kp.eval()
        d_diag = lambda_c + sigma2  # Diagonal elements (2D column vec)
        d_tilde = 1.0 - sigma2 / d_diag

        # Test point...
        kus_0 = self.feature.Kuf(kern_0, qx_star.mu)
        if qx_star.cov is None:
            # Kernel expectation is just a point since input has no variance
            psi1st_0 = kus_0
        else:
            # psi1st = "Psi1 star transpose" (test inputs)
            psi1st_0 = tf.transpose(
                expectation(qx_star, (kern_0, self.feature)))
        kus = KroneckerProduct([kus_0] +
                               [kern_i.K(x_st_i, x_st_test_i)
                                for kern_i, x_st_i, x_st_test_i
                                in zip(kern_list[1:], x_st, x_st_test)])
        # L^(-1) @ K_{u*}
        li_kus = l_kuu.matrix_triangular_solve(kus, lower=True)

        # klq = K_{su} * L^(-T) * Q_C
        # TODO more than one point at a time
        klq = li_kus.transpose() @ q_c
        # n* x m
        ns_m_shape = [(n_star_ij, m_ij)
                          for n_star_ij, m_ij in zip(n_star_i, m_i)]
        klq.shape_hint = ns_m_shape

        # pslq = Psi1^* @ L^(-t) @ Q_C
        # li_psi1st = L^(-1) @ Psi1^(*,T)
        # li_psi1st_0 = tf.matrix_triangular_solve(l_kuu_0, psi1st_0, lower=True)
        li_psi1st = l_kuu.matrix_triangular_solve(
            KroneckerProduct([psi1st_0] + kus.x[1:]), lower=True)
        pslq = li_psi1st.transpose() @ q_c
        pslq.shape_hint = ns_m_shape

        # L^(-1) @ Psi1^T (no "*" on Psi1 here!)
        li_psi1t_0 = tf.matrix_triangular_solve(l_kuu_0, tf.transpose(psi1_0),
                                                lower=True)
        # This will change once there are st inducing points:
        li_psi1t = KroneckerProduct([li_psi1t_0] + l_kff_st.transpose().x)
        qlp1 = q_c.transpose() @ li_psi1t
        mn_size = [(m_ij, n_ij) for m_ij, n_ij in zip(m_i, n_i)]
        qlp1.shape_hint = mn_size
        b = qlp1.matmul(self.Y, shape_hint=(n, d_out))
        db = b / d_diag
        mean = pslq.matmul(db, shape_hint=(m, d_out))

        kss_diag = KroneckerProduct(
            [tf.reshape(kern_0.Kdiag(qx_star.mu), [-1, 1])] +
            [tf.reshape(kern_i.Kdiag(x_st_test_i), [-1, 1])
             for kern_i, x_st_test_i in zip(kern_list[1:], x_st_test)]).\
            eval()
        klq2 = klq * klq
        klq2.shape_hint = klq.shape_i
        term2 = klq2.matmul(d_tilde, shape_hint=(m, 1))
        var = tf.tile(kss_diag - term2, (1, d_out))

        return mean, var

    @params_as_tensors
    def _build_infer_likelihood(self):
        """
        Given a new observation, infer its latent variable posterior

        i.e. Y->X (inverse problem)

        :return:
        """
        n_i_train = self.n_subgrid
        n_i = n_i_train.copy()
        n_i[0] += self._n_0_test
        m_i = self.m_subgrid
        d_out = self.output_dim
        n_st = np.prod(n_i_train[1:])
        n_train = self.n  # All training data
        n_test = self._n_0_test * n_st
        n_tot = n_train + n_test
        m = np.prod(m_i)
        h_mu_test = self.h_mu_test
        h_s_test = self.h_s_test
        kern_list = [getattr(self, name) for name in self.kern_names]
        kern_0 = kern_list[0]
        qx_test = self._get_infer_lv_posterior()
        sigma2 = self.likelihood.variance

        # Basic kernel matrices:
        l_kuu_0 = tf.constant(self._l_kuu_0, dtype=float_type)
        l_kff_st = KroneckerProduct([tf.constant(l_kff_st_i, dtype=float_type)
                                     for l_kff_st_i in self._l_kff_st])

        # Statistics
        psi0_train = tf.constant(self._psi0_train, dtype=float_type)

        psi0_test = n_st * tf.reduce_sum(expectation(qx_test, kern_0))
        psi1_0_test = expectation(qx_test, (kern_0, self.feature))
        psi2_0_test = tf.reduce_sum(expectation(qx_test, (kern_0, self.feature),
                                                (kern_0, self.feature)), axis=0)
        psi0 = psi0_train + psi0_test

        # Intermediate matrices...
        c_0_train = tf.constant(self._c_0_train, dtype=float_type)
        tmp = tf.matrix_triangular_solve(l_kuu_0, psi2_0_test, lower=True)
        c_0_test = tf.matrix_triangular_solve(l_kuu_0, tf.transpose(tmp),
                                              lower=True)
        c_0 = c_0_train + c_0_test
        lambda_c_0, q_c_0 = jit_self_adjoint_eig(c_0)
        lambda_c_st = [tf.constant(lc_st_i, dtype=float_type)
                       for lc_st_i in self._lambda_c_st]
        q_c_st = [tf.constant(q_c_st_i, dtype=float_type)
                  for q_c_st_i in self._q_c_st]
        lambda_c = KroneckerProduct([tf.reshape(lambda_c_0, (-1, 1))]
                                    + lambda_c_st).eval()
        d_diag = lambda_c + sigma2  # Diagonal elements (2D column vec)
        logdet_a = tf.reduce_sum(tf.log(d_diag))  # a = q_c * d * q_c'
        trace_c = tf.reduce_sum(lambda_c)

        # The infamous "Tr(PKP)"
        # Approach is to concatenate li_psi1t_0's (train & test),
        # then continue as normal.
        # I'm not sure that this is the best way to do it, but it works for now.
        q_c = KroneckerProduct([q_c_0] + q_c_st)
        li_psi1t_0_train = tf.constant(self._li_psi1t_0, dtype=float_type)
        li_psi1t_0_test = tf.matrix_triangular_solve(l_kuu_0,
                                                     tf.transpose(psi1_0_test),
                                                     lower=True)
        li_psi1t_0 = tf.concat((li_psi1t_0_train, li_psi1t_0_test), 1)
        li_psi1t = KroneckerProduct([li_psi1t_0] + l_kff_st.transpose().x)
        qlp1 = q_c.transpose() @ li_psi1t
        mn_size = [(m_ij, n_ij) for m_ij, n_ij in zip(m_i, n_i)]
        qlp1.shape_hint = mn_size
        y_aug = tf.concat((self.Y, self.y_test), 0)
        b = qlp1.matmul(y_aug, shape_hint=(n_tot, d_out))
        diag_bbt = tf.reduce_sum(b ** 2, 1)
        trace_pkp = tf.reduce_sum(diag_bbt / tf.reshape(d_diag, [-1]))

        # Final ELBO assembly
        log_2pi = tf.cast(tf.log(2.0 * np.pi), float_type)
        elbo_1 = 0.5 * self.output_dim * (-(n_tot - m) * tf.log(sigma2)
                                          - n_tot * log_2pi - logdet_a)

        elbo_2 = - 0.5 / sigma2 * (
                self._tr_yyt + self._tr_yyt_test - trace_pkp
                + self.output_dim * (psi0 - trace_c)
        )

        # Combine with KL's
        qx = probability_distributions.DiagonalGaussian(h_mu_test, h_s_test)
        px = probability_distributions.DiagonalGaussian(
            tf.zeros((1, self.num_latent), dtype=float_type),
            tf.ones((1, self.num_latent), dtype=float_type))
        return elbo_1 + elbo_2 - kl_divergence(qx, px) \
               - tf.constant(self._kl_train, dtype=float_type)

    @params_as_tensors
    def _build_infer_pcb_likelihood(self):
        """
        Given a new observation with non-training spatials, infer its latent
        variable posterior using the partially-collapsed bound approach.

        i.e. Y->X (inverse problem)

        :return:
        """
        x_st = [getattr(self, name) for name in self.input_names[1:]]
        x_st_infer = [getattr(self, name) for name in self.input_st_infer_names]
        n_i_infer = [1] + [x_i.shape[0] for x_i in x_st_infer]
        m_i = self.m_subgrid
        d_y = self.output_dim
        n_st_infer = np.prod(n_i_infer[1:])
        # n_train = self.n  # All training data
        n_infer = np.prod(n_i_infer)
        # m = np.prod(m_i)
        kern_list = [getattr(self, name) for name in self.kern_names]
        kern_0 = kern_list[0]
        qx_test = self._get_infer_lv_posterior()
        sigma2 = self.likelihood.variance
        sigma2_infer = self.sigma2_infer

        # Basic kernel matrices:
        l_kuu_0 = tf.constant(self._l_kuu_0, dtype=float_type)

        # Statistics
        psi0_infer = tf.cast(n_st_infer, float_type) * \
                     tf.reduce_sum(expectation(qx_test, kern_0))
        # IPSPATIAL
        kus_st_infer_matrices = [kern_i.K(x_st_i, x_st_infer_i)
                                 for kern_i, x_st_i, x_st_infer_i
                                 in zip(kern_list[1:], x_st, x_st_infer)]
        psi1_0_infer = expectation(qx_test, (kern_0, self.feature))
        psi1t_infer = KroneckerProduct(
            [tf.transpose(psi1_0_infer)] +
            kus_st_infer_matrices
            )
        psi1t_infer.shape_hint = [(m_ij, n_ij_infer)
                                  for m_ij, n_ij_infer in zip(m_i, n_i_infer)]
        psi2_0_infer = tf.reduce_sum(
            expectation(qx_test, (kern_0, self.feature),
                        (kern_0, self.feature)), axis=0, name="Psi2_0_infer")

        if DEBUG_INFER:
            psi2_0_infer = debug_check(psi2_0_infer, tf.is_nan)

        lambda_psi2_0_infer, q_psi2_0_infer = tf.linalg.eigh(psi2_0_infer)
        # Note: lambda is 1D, but tf broadcasts as if it were a row vector.
        # This is exactly what we want.
        # matrix square root:
        # Hint: because psi2 is <K_uf @ K_fu>, then psi2_0 is at most rank q?...
        # Also, eigh returns eigenvalues in ascending order--> use last q cols.
        # For now, just set the previous ones to zeroes.
        first_col = m_i[0] - self.num_latent
        # bump up eigenvalues to at least 0 just to be sure we don't get NaNs.
        l_psi2_0_infer = q_psi2_0_infer * \
                         tf.sqrt(tf.maximum(lambda_psi2_0_infer, 0.0))
        # shape hint needed?
        l_psi2 = KroneckerProduct([l_psi2_0_infer] +
                                  kus_st_infer_matrices)
        l_psi2t = l_psi2.transpose()
        l_psi2t.shape_hint = [l_psi2t.shape_i[0]] + \
                            [(n_ij_infer, m_i)
                             for n_ij_infer, m_i
                             in zip(n_i_infer[1:], self.m_subgrid[1:])]

        # c_infer...
        tmp = tf.matrix_triangular_solve(l_kuu_0, psi2_0_infer, lower=True)
        c_0_infer = tf.matrix_triangular_solve(l_kuu_0, tf.transpose(tmp),
                                              lower=True)
        tr_c_infer = tf.trace(c_0_infer) * self._tr_c_infer_st

        train_bound = tf.constant(self._train_bound, dtype=float_type)
        tr_yy_term = -0.5 / sigma2_infer * \
                     tf.constant(self._tr_yyt_test, dtype=float_type)
        # Trace term involving mean of q(U) times itself
        k_psi_inv_psi1t_y = tf.constant(self._k_psi_inv_psi1t_y,
                                        dtype=float_type)
        t1 = l_psi2t.matmul(k_psi_inv_psi1t_y)
        tr_uu_term = -0.5 / sigma2_infer * tf.reduce_sum(t1 ** 2)

        # Trace term involving the cov of q(U)
        qlplq_0_diag = tf.reshape(tf.diag_part(
            tf.constant(self._lt_inv_q_0.T, dtype=float_type) @
            psi2_0_infer @
            tf.constant(self._lt_inv_q_0, dtype=float_type)),
            (m_i[0], 1))
        qlplq_diag = KroneckerProduct([qlplq_0_diag] +
                                      [tf.constant(self._qlplq_st_diag,
                                                   dtype=float_type)]).eval()
        tr_psi2_k_psi = tf.reduce_sum(qlplq_diag *
                                      tf.constant(self._d_diag_inv,
                                                  dtype=float_type))
        tr_ucov_term = -0.5 * sigma2 * d_y / sigma2_infer * tr_psi2_k_psi
        # Trace term involving the "cross" between u mean & training Y:
        psi1st_y_infer = psi1t_infer.matmul(self.y_test,
                                            shape_hint=(n_infer, d_y))
        tr_yu_term = 1.0 / sigma2_infer * \
            tf.reduce_sum(psi1st_y_infer * tf.constant(self._k_psi_inv_psi1t_y,
                                                       dtype=float_type))
        psi0_trc_term = -0.5 * d_y / sigma2_infer * (psi0_infer - tr_c_infer)

        # Final ELBO assembly
        log_2pi = tf.cast(tf.log(2.0 * np.pi), float_type)

        # Combine with KL's
        px_test = self._get_infer_lv_prior(for_kl=True)
        return train_bound \
            - 0.5 * tf.cast(n_infer, float_type) * d_y \
            * (tf.log(sigma2_infer) + log_2pi) \
            + tr_yy_term \
            + tr_uu_term \
            + tr_ucov_term \
            + tr_yu_term \
            + psi0_trc_term \
            - kl_divergence(qx_test, px_test)

    def _build(self):
        """
        Overrides Model._build() so that we can optionally build for inference
        instead of training by setting self.test_mode

        :return:
        """
        super(Model, self)._build()
        if self.infer_mode == 0:
            likelihood = self._build_likelihood()
        elif self.infer_mode == 1:
            likelihood = self._build_infer_likelihood()
        elif self.infer_mode == 2:
            likelihood = self._build_infer_pcb_likelihood()
        else:
            raise ValueError("Unrecognized infer_mode")

        prior = self.prior_tensor
        objective = self._build_objective(likelihood, prior)
        self._likelihood_tensor = likelihood
        self._objective = objective

    # Useful for debugging:

    @autoflow()
    @params_as_tensors
    def kuu(self):
        return self.feature.Kuu(self.kern_list[0], jitter=self.jitter)

    @autoflow()
    @params_as_tensors
    def kff(self):
        x = [getattr(self, name) for name in self.input_names]
        return [k.K(x_st_i) + self._st_jitter * tf.eye(n_ij, dtype=float_type)
                for n_ij, x_st_i, k
                in zip(self.n_subgrid[1:], x[1:], self.kern_list[1:])]


class SgplvmFullCovInfer(Sgplvm):
    """
    Use full covariance for inference datum's LV variational posterior
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.h_s_test
        # Cholesky of the posterior covariance for infer-time:
        self.lh_s_test = Parameter(np.ones((1, self.d_xi, self.d_xi)),
                                   transform=transforms.LowerTriangular(
                                       self.d_xi))

    @autoflow()
    @params_as_tensors
    def inferred_latent_variables(self):
        h_s_test = tf.matmul(self.lh_s_test, self.lh_s_test, transpose_b=True)
        return self.h_mu_test, h_s_test

    @params_as_tensors
    def _get_infer_lv_posterior(self):
        """
        Variational posterior on latent variables to be inferred at test time.
        :return:
        """
        h_s_test = tf.matmul(self.lh_s_test, self.lh_s_test, transpose_b=True)
        return probability_distributions.Gaussian(
            self.h_mu_test, h_s_test, chol_cov=self.lh_s_test)

    @params_as_tensors
    def _get_infer_lv_prior(self, for_kl=False, **kwargs):
        """
        Prior on latent variables to be inferred at test time
        :param cov_2d: Only provide a single tf.eye(d_xi) [d_xi x d_xi] instead
            of the full 3D stack [n_xi x d_xi x d_xi].  This "hack" is useful in
            ensuring compatibility with the KL divergence code.
        divergence
        :return:
        """
        for key in kwargs.keys():
            warn("Unused kwarg: {}".format(key))
        # TODO More than one infer case at a time
        n_infer = 1
        mu = tf.zeros((n_infer, self.d_xi), dtype=float_type)
        cov = tf.eye(self.d_xi, batch_shape=[n_infer], dtype=float_type) if \
            not for_kl else tf.eye(self.d_xi, dtype=float_type)
        return probability_distributions.Gaussian(mu, cov)

    def _init_infer_lv_posterior(self, n, h_mu_test_init=None,
                                 h_s_test_init=None):
        """
        Provide an initial guess for the LV posterior before inference at test
        time.
        """
        if h_mu_test_init is None and h_s_test_init is None:
            d_0 = self.X0.shape[1]
            h_mu_test_init = np.zeros((n, d_0))
            h_s_test_init = np.tile(np.eye(d_0), (n, 1, 1))

        # Assign
        self.h_mu_test.assign(h_mu_test_init)
        self.lh_s_test.assign(h_s_test_init)
