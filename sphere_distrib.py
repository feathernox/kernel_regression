import numpy as onp
import jax.numpy as np
from jax import jit, random
from scipy.special import factorial2
from functools import partial


def generate_sample(key, n, d):
    X = random.normal(key, shape=(n, d))
    X = X / np.linalg.norm(X, axis=-1)[:, np.newaxis] * np.sqrt(d)
    return X


def dsphere_1c_expectation(d, m):
    # for [- sqrt(d), sqrt(d)]
    if m % 2 == 1:
        return 0
    k = m // 2
    res = factorial2(2 * k - 1) / onp.prod(1 + 2 * onp.arange(0, k) / d)
    return res


def dsphere_2c_expectation(d, k, m):
    if k % 2 == 1 or m % 2 == 1:
        return 0
    k = k // 2
    m = m // 2
    res = factorial2(2 * k - 1) * factorial2(2 * m - 1) / onp.prod(1 + 2 * onp.arange(0, k + m) / d)
    return res

