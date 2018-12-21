# Base class for structured GP models
# Steven Atkinson
# satkinso@nd.edu
# January 30, 2018

from __future__ import absolute_import

from ..kernels import turn_off_variances
from ..util import as_data_holder

from gpflow.params import DataHolder
# from gpflow.kernels import make_kernel_names
import gpflow.kernels
from gpflow import mean_functions
from gpflow.models import GPModel
from warnings import warn
import numpy as np


def _make_kernel_names(kern_list):
    """
    Take a list of kernels and return a list of strings, giving each kernel a
    unique name.

    Each name is made from the lower-case version of the kernel's class name.

    Duplicate kernels are given trailing numbers.
    """
    names = []
    counting_dict = {}
    for k in kern_list:
        inner_name = k.__class__.__name__.lower()

        # check for duplicates: start numbering if needed
        if inner_name in counting_dict:
            if counting_dict[inner_name] == 1:
                names[names.index(inner_name)] = inner_name + '_1'
            counting_dict[inner_name] += 1
            name = inner_name + '_' + str(counting_dict[inner_name])
        else:
            counting_dict[inner_name] = 1
            name = inner_name
        names.append(name)
    return names


class SgpModel(GPModel):
    """
    A base class for "Kronecker" Gaussian process models where:
    * the inputs have a grid structure
    * the kernel function exhibits separability.

    No mean functions for now.
    """

    def __init__(self, x, y, kern_list, likelihood, mean_function,
                 num_latent=None, name=None):
        """
        :param x: Inputs
        :type x: list of DataHolder arrays
        :param y: Outputs
        :type y: (same as GPModel)
        :param kern: the terms in the separable kernel, matching the inputs
        :type kern: list of kernels
        :param likelihood:
        :param mean_function: has to be Zero for now.
        :param num_latent: the number of outputs
        :param name: name of the model
        """
        # Skip the inputs and kernel on the super init; we'll do those here.
        super().__init__(None, y, None, likelihood, mean_function,
                                       num_latent, name)

        x = [as_data_holder(x_i) for x_i in x]
        y = as_data_holder(y)
        self.X, self.Y = x, y
        self.num_subgrids = len(x)
        # Number of data points within each subgrid
        self.n_subgrid = [xi.shape[0] for xi in x]
        self.n = np.prod(self.n_subgrid)
        self.d_in_subgrid = [xi.shape[1] for xi in x]
        self.d_in = np.sum(self.d_in_subgrid)

        # Make sure that at most one kernel has an optimizable variance:
        turn_off_variances(kern_list)
        self.kern_list = kern_list

        # Alternate storage of kernels and inputs so that they are found during
        # params_as_tensors.
        # TODO could use GPflow ParamList now..
        kern_names = _make_kernel_names(kern_list)
        input_names = ["X" + str(i) for i in range(len(x))]
        # Omit first subspace (not spatiotemporal!)
        input_st_test_names = [name + "_test" for name in input_names[1:]]
        self.input_names = input_names
        self.input_st_test_names = input_st_test_names
        self.kern_names = kern_names
        [setattr(self, name, k) for name, k in zip(kern_names, self.kern_list)]
        [setattr(self, name, x_i) for name, x_i in zip(input_names, self.X)]

    def compute_objective(self):
        return -(self.compute_log_likelihood() + self.compute_log_prior())
