import numpy as np


def minmax_scale(x):
    """Scale data so it is in the range (0, 1)"""
    data_min = np.min(x)
    data_max = np.max(x)
    data_range = data_max - data_min

    # don't divide by zero
    if data_range == 0.:
        data_range = 1.

    x -= data_min
    x /= data_range
