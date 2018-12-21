# Utilities for the package
# Steven Atkinson
# satkinso@nd.edu
# February 13, 2018

import gpflow
import tensorflow as tf
import numpy as np

tensor_type = gpflow.settings.float_type  # tf.float64


def as_variable(x):
    """
    Convert the input to a tf.Variable

    :param x:
    :return: (tf.Variable)
    """
    return tf.Variable(x, dtype=tensor_type)


def as_data_holder(input):
    return input if isinstance(input, gpflow.DataHolder) \
        else gpflow.DataHolder(input)


def pca(y, q):
    """
    PCA dimensionality reduction of y (n x d) to x (n x q)

    "dims > data" case is based on implementation in SheffieldML/GPmat,
    https://github.com/SheffieldML/GPmat

    :param y: data
    :param q: latent dimensions
    :return:
    """
    n, d = y.shape
    assert(min(n, d) >= q), "Cannot have more latent dimensions than features"

    if n >= d:
        evals, evecs = np.linalg.eigh(np.cov(y.T))
        # Get q biggest evals and their associated evecs.
        i = np.argsort(evals)[::-1]
        w = evecs[:, i]
        w = w[:, :q]
        x = (y - y.mean(0)).dot(w)
        x = x / np.std(x, 0)  # Ensure unit variance
    else:
        y_mean = np.mean(y, 0).reshape(1, -1)
        y_center = y - y_mean
        inner_y = y_center @ y_center.T
        evals, evecs = np.linalg.eigh(inner_y)
        i = np.argsort(evals)[::-1]
        w = evecs[:, i]
        w = w[:, :q]
        evals[evals < 0] = 0
        x = w * np.sqrt(n)
        # v = v / sqrt(size(Y, 1))
        # sigma2 = (trace(innerY) - sum(v)) / (size(Y, 2) - dims)
        # W = X'*Ycentre
    return x


def nearest_neighbor(x, y):
    """
    Which row in x is closest to y

    :param x:
    :type x: np.ndarray (2D)
    :param y:
    :type y: np.ndarray (2D)

    :return: (int, np.ndarray): The index of the nearest neighbor and that
        neighbor.
    """
    i_min = np.argmin(np.sum((y - x) ** 2, 1))
    return i_min, x[i_min].reshape((1, -1))


def get_param_dict(obj):
    """
    Create a dictionary containing all of the parameters in obj
    (not Parameterized)

    :param obj:
    :type obj: gpflow.Parameter
    :return:
    """

    d = {}
    if isinstance(obj, gpflow.params.Parameter):
        d.update({obj.full_name: obj.value})
    elif isinstance(obj, gpflow.params.Parameterized):
        for param in obj.params:
            d.update(get_param_dict(param))
    return d


def np_least_squares(x, y):
    """
    Given x, y, find a, b for y=xa+b via least-squares.

    (Just a simple wrapper that I can make sense of quickly.)

    :param x: inputs (n x d_x)
    :param y: outputs (n x d_y)
    :return: a (d_x x d_y), b(1D, length d_y)
    """

    phi = np.vstack((x.T, np.ones((1, x.shape[0])))).T
    res = np.linalg.lstsq(phi, y)[0]
    a, b = res[:-1], res[-1]
    return a, b


def debug_check(x, check):
    """
    Checks for bad behavior.
    :param x: Tensor to check
    :param check: tf op (e.g. tf.is_nan)
    :return:
    """

    x = tf.cond(
        tf.reduce_any(check(x)),
        lambda: tf.Print(x, [x], "{} found issue with {}!".
                         format(check.__name__, x.name)),
        lambda: tf.identity(x))
    return x
