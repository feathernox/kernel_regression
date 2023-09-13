import jax.numpy as np
from utils import match_shapes


def compute_delta_K(n, d, K):
    return n / d ** K


def _compute_bias_variance(rho, delta_K):
    rho, delta_K = match_shapes(rho, delta_K)
    r = np.sqrt((1 + rho - rho * delta_K) ** 2 + 4 * rho * delta_K)
    Bh = (1 - delta_K) / 2 + (1 + delta_K + rho * (1 - delta_K) ** 2) / (2 * r)
    Vh = (1 + rho + delta_K * rho) / (2 * r) - 0.5
    return {'Bh': Bh, 'Vh': Vh}


def compute_test_error_parameterized(snr, rho, delta_K):
    res = _compute_bias_variance(rho, delta_K)
    Etest = snr * res['Bh'] + res['Vh']
    return {**res, 'Etest': Etest}


def compute_test_error(g_fn, activation, K, sigma_z, sigma_W2, lreg, delta_K):
    rho = activation.rho(K, lreg=lreg, sigma_W2=sigma_W2)
    res = _compute_bias_variance(rho, delta_K)
    _, Bh = match_shapes(sigma_z, res['Bh'])
    sigma_z, Vh = match_shapes(sigma_z, res['Vh'])
    Bh_K = g_fn.get_hermite_coef_sqr(K) * Bh
    Bh_gK = g_fn.get_hermite_coef_sqr_remainder(K + 1) * (Vh + 1)
    Vh = sigma_z ** 2 * Vh
    Etest = Bh_K + Bh_gK + Vh
    return {'Bh_K': Bh_K, 'Bh_gK': Bh_gK, 'Vh': Vh, 'Etest': Etest}
