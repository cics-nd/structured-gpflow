# Dealing with structured data
# Steven Atkinson
# satkinso@nd.edu
# February 26, 2018

import numpy as np


def expand_grid(x):
    """
    Take a list of matrices and grid them out
    Quick and dirty implementation; could be done more elegantly...

    :param x: the arrays to be "gridded out"
    :type x: list of np.ndarrays
    :return: np.ndarray
    """
    assert all([isinstance(x_i, np.ndarray) for x_i in x])

    n_sub = len(x)
    n_list = [x_i.shape[0] for x_i in x]
    n = int(np.prod(n_list))
    m = int(np.sum([x_i.shape[1] for x_i in x]))

    def increment(v, n):
        for i in range(len(v) - 1, -1, -1):
            v[i] += 1
            if v[i] < n[i]:
                break
            v[i] = 0
        return v

    v = np.zeros(n_sub, dtype=int)
    y = np.zeros((n, m))
    for i in range(n):
        y[i, :] = np.concatenate([x[j][v[j], :] for j in range(n_sub)])
        v = increment(v, n_list)
    return y
