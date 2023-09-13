import math
import numpy as onp
import jax.numpy as np
from scipy.special import factorial2

import sphere_distrib


def check_validity(coef):
    assert len(coef) > 0 and (len(coef) == 1 or coef[-1] != 0), "Wrong polynomial format"


def classic_poly_norm_expectation(classic_coef):
    expectation = classic_coef[0]
    for idx in range(2, len(classic_coef), 2):
        expectation += classic_coef[idx] * factorial2(idx - 1)
    return expectation


def classic_poly_square(classic_coef):
    # TODO: optimize?
    classic_coef = onp.array(classic_coef)
    n_coef_old = classic_coef.shape[-1]
    n_coef_new = (n_coef_old - 1) * 2 + 1
    square_coef = onp.zeros(list(classic_coef.shape[:-1]) + [n_coef_new])
    for i in range(n_coef_old):
        square_coef[..., 2 * i] += classic_coef[..., i] ** 2
        for j in range(i + 1, n_coef_old):
            square_coef[..., i + j] += 2 * classic_coef[..., i] * classic_coef[..., j]
    return square_coef


def poly_square(coef):
    n_coef = coef.shape[-1]
    new_coef = (coef[..., np.newaxis] * coef[..., np.newaxis, :])[..., ::-1]
    new_coef = np.stack([np.trace(new_coef, offset=i, axis1=-2, axis2=-1)
                         for i in range(n_coef - 1, -n_coef, -1)], axis=-1)
    return new_coef


def classic_poly_dsphere_1c_expectation(classic_coef, d):
    expectation = 0
    for i in range(0, classic_coef.shape[-1], 2):
        expectation += classic_coef[..., i] * sphere_distrib.dsphere_1c_expectation(d, i)
    return expectation


def classic_poly_dsphere_2c_expectation(classic_coef, d):
    expectation = 0
    for k in range(0, classic_coef.shape[-2], 2):
        for m in range(0, classic_coef.shape[-1], 2):
            expectation += classic_coef[..., k, m] * sphere_distrib.dsphere_2c_expectation(d, k, m)
    return expectation


def classic_poly_mul(classic_coef_a, classic_coef_b):
    # first can be array, second must be 1d poly
    n_coef_a = classic_coef_a.shape[-1]
    n_coef_b = classic_coef_b.shape[-1]
    zero_pad = [(0, 0)] * (classic_coef_a.ndim - 1)
    res_coef = np.stack([np.pad(classic_coef_a * classic_coef_b[..., i:i + 1],
                                pad_width=zero_pad + [(i, n_coef_b - i - 1)], mode='constant', constant_values=0)
                         for i in range(n_coef_b)]).sum(axis=0)
    return res_coef


def classic_poly_mul_jit(classic_coef_a, classic_coef_b):
    # first can be array, second must be 1d poly
    n_coef_a = classic_coef_a.shape[-1]
    n_coef_b = classic_coef_b.shape[-1]
    n_coef = n_coef_a + n_coef_b - 1
    zero_pad = [(0, 0)] * (classic_coef_a.ndim - 1)
    res_coef = np.stack([np.pad(classic_coef_a * classic_coef_b[..., i:i + 1],
                                pad_width=zero_pad + [(i, n_coef_b - i - 1)], mode='constant', constant_values=0)
                         for i in range(n_coef_b)]).sum(axis=0)
    return res_coef



def classic_poly_1d_rescale(classic_coef, scale):
    # in case we have p(x) and want p'(x) = p(scale * x)
    n_coef = len(classic_coef)
    return (scale[..., np.newaxis] ** np.arange(n_coef)) * classic_coef


def classic_poly_2d_normalize(classic_coef, dimension):
    # in case we have p(x) and want p'(x * sqrt(d)) = p(x)all
    n_coef_a = classic_coef.shape[-2]
    n_coef_b = classic_coef.shape[-1]
    return classic_coef * (1 / np.sqrt(dimension) ** np.arange(n_coef_a))[:, np.newaxis] * \
           (1 / np.sqrt(dimension) ** np.arange(n_coef_b))


def polynomial_1d_expansion(classic_coef, alpha, beta):
    # i.e. we have polynomial p(x) and want to expand p(alpha_i x + beta_i y) (can be indexed by ij, ijk, etc.)
    # axis up to -3: index of sample based on alpha/beta shapes (p is 1d array); -2: degree of x; -1: degree of y
    n_coef = classic_coef.shape[-1]

    new_coefs = np.array([[0 if i + j >= n_coef else classic_coef[i + j] * math.comb(i + j, j)
                           for j in range(n_coef)] for i in range(n_coef)])

    polynomial_2d = ((alpha[..., np.newaxis] ** np.arange(n_coef))[..., np.newaxis] *
                             (beta[..., np.newaxis] ** np.arange(n_coef))[..., np.newaxis, :])  * new_coefs
    return polynomial_2d


if __name__ == "__main__":
    from jax import random, jit
    import timeit

    key = random.PRNGKey(555)
    poly_a = random.normal(key, (3,))
    poly_b = random.normal(key, (3,))
    print(poly_a, poly_b)

    # key, subkey_a, subkey_b = random.split(key, 3)
    # poly_a = random.normal(subkey_a, (1,))
    # poly_b = random.normal(subkey_b, (6,))
    # print(poly_square(poly_b))
    # print(classic_poly_square(poly_b))
    # polynomial_1d_expansion(poly_b, poly_a, poly_a)
    # poly_a = random.normal(key, (5, 6, 7))
    # print(classic_poly_square(poly_a) - poly_square(poly_a))


    # n = 1000
    # print(timeit.timeit(lambda: classic_poly_mul(poly_a, poly_b), number=n) / n)
    #
    # classic_poly_mul_jitted = jit(classic_poly_mul_jit)
    # n = 1000
    # print(timeit.timeit(lambda: classic_poly_mul_jitted(poly_a, poly_b), number=n) / n)

