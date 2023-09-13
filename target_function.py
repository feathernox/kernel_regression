import numpy as onp
import jax.numpy as np

import hermite
import polynomial
from utils import read_fraction_list, prod_list
from scipy.special import factorial


class TargetFunction():
    def __init__(self):
        return

    def __call__(self, points):
        raise NotImplementedError()

    def SNR(self, K, sigma_z=0):
        raise NotImplementedError()

    def expectation(self):
        raise NotImplementedError()

    def expectation_sqr(self):
        raise NotImplementedError()

    def get_hermite_coef_sqr(self, K):
        raise NotImplementedError()

    def get_hermite_coef_sqr_remainder(self, K):
        raise NotImplementedError()


class PolynomialFunction(TargetFunction):
    """
    NOTE: Hermite coefficients "self.hermite_coef" are unscaled here!!
    I.e. sum^n_{i=0} hermite_coef[i] He[i]  is the original polynomial
    if He[i] is standard probabilist's Hermite polynomial,
    i.e integrating He^2[i] w.r.t. to standard Gaussian measure gives i!, not 1.
    """
    def __init__(self, coef, poly_type='classic'):
        super(PolynomialFunction, self).__init__()
        polynomial.check_validity(coef)
        coef = onp.array(coef, dtype='float')
        self.deg = len(coef) - 1  # degree of polynomial
        if poly_type == 'classic':
            self.classic_coef = np.array(coef)
            self.hermite_coef = np.array(hermite.classic2hermite(coef))
        elif poly_type == 'hermite':
            self.hermite_coef = np.array(coef)
            self.classic_coef = np.array(hermite.hermite2classic(coef))
        else:
            raise ValueError('Unknown polynomial type')

    def __call__(self, points):
        return np.polyval(self.classic_coef[::-1], points)

    def expectation(self):
        return self.hermite_coef[0]

    def expectation_sqr(self):
        return (self.hermite_coef ** 2 * prod_list(0, self.deg + 1)).sum()

    def get_hermite_coef_sqr(self, K, scaled=True):
        # scaled means normalized
        # unscaled means probabilist's
        if K > self.deg:
            return 0.

        if scaled:
            return self.hermite_coef[K] ** 2 * factorial(K)
        else:
            return self.hermite_coef[K] ** 2

    def get_hermite_coef_sqr_remainder(self, K, scaled=True):
        # including K!
        # unscaled means divided by (K - 1)!
        if K > self.deg:
            return 0.

        rem = self.hermite_coef[K:] ** 2 * prod_list(K, self.deg + 1)

        if scaled:
            rem *= factorial(max(0, K - 1))

        return rem.sum()

    def SNR(self, K, sigma_z=0):
        if K > self.deg:
            snr_val = np.where(sigma_z == 0., np.nan, 0.)
            return snr_val

        snr_val = self.get_hermite_coef_sqr(K, scaled=False) / \
                  (self.get_hermite_coef_sqr_remainder(K + 1, scaled=False) + sigma_z ** 2 / factorial(K))
        return snr_val


def get_target_function(kind='poly', **kwargs):
    if kind == 'poly':
        kwargs['coef'] = read_fraction_list(kwargs['coef'])
        return PolynomialFunction(**kwargs)
    else:
        raise NotImplementedError()
