# Kernel functions
# Steven Atkinson
# satkinso@nd.edu
# February 21, 2018

import gpflow.kernels


def turn_off_variances(kern_list, from_outside=True):
    """
    Ensure that we don't have redundant variance parameters to optimize.
    THIS DOESN'T WORK VERY WELL!  WATCH OUT IF YOUR KERNELS ARE CRAZY!
    :param kern_list: list of kernels
    :param top: whether this is the top layer of the kernel (called from outside)
    :return: (done by reference)
    """

    if from_outside:
        kern_list = kern_list[1:]
    for kern in kern_list:
        if isinstance(kern, gpflow.kernels.Combination):
            turn_off_variances(kern.kern_list, False)
        elif hasattr(kern, "variance"):
            kern.variance.set_trainable(False)
