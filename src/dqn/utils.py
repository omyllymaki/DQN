import cProfile
import inspect
import logging
import os
import pstats
from functools import wraps
from time import time

import numpy as np

logger = logging.getLogger(__name__)


def running_mean(x, windwow_size):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    output = (cumsum[windwow_size:] - cumsum[:-windwow_size]) / float(windwow_size)
    n_padding = len(x) - len(output)
    return n_padding * [None] + output.tolist()


def timer(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        t_ms = 1000 * (te - ts)

        calling_module = inspect.getmodule(inspect.currentframe().f_back)
        calling_module_name = calling_module.__name__ if calling_module else "unknown_module"
        calling_class_name = args[0].__class__.__name__ if args and hasattr(args[0], '__class__') else "unknown_class"
        f_name = f.__name__

        print(f"func {f_name} (module {calling_module_name}, class {calling_class_name}) took {t_ms:0.1f} ms")
        return result

    return wrap


def profile_decorator(filename):
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()

            result = func(*args, **kwargs)

            profiler.disable()
            profiler.dump_stats(filename)
            stats = pstats.Stats(filename)
            stats.sort_stats("cumulative")
            stats.print_stats()
            stats.dump_stats(f"{filename}.stats")
            os.system(f"snakeviz {filename}.stats")

            return result

        return wrapper

    return decorator
