import numpy as np


def running_mean(x, windwow_size):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    output = (cumsum[windwow_size:] - cumsum[:-windwow_size]) / float(windwow_size)
    n_padding = len(x) - len(output)
    return n_padding * [None] + output.tolist()
