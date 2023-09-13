import numpy as onp
import jax.numpy as np
from neural_tangents import stax

import hermite
import polynomial
from utils import read_fraction_list, prod_list, match_shapes
from scipy.special import factorial


class ActivationFunction:
    def __init__(self):
        pass

    def __call__(self, points):
        raise NotImplementedError()

    def get_layer(self):
        raise NotImplementedError

    def get_hermite_coef_sqr(self, K,  include_RF=True, sigma_W2=0.):
        raise NotImplementedError()

    def get_hermite_coef_sqr_remainder(self, K, include_RF=True, sigma_W2=0.):
        raise NotImplementedError()

    def get_lambda_crit(self, K):
        raise NotImplementedError()

    def rho(self, K, lreg=0., sigma_W2=0.):
        raise NotImplementedError()

    def rho_K1(self, K):
        raise NotImplementedError()

    def rho_K2(self, K):
        raise NotImplementedError()


class PolynomialActivation(ActivationFunction):
    """
    NOTE: Hermite coefficients "self.hermite_coef" are unscaled here!!
    I.e. sum^n_{i=0} hermite_coef[i] He[i]  is the original polynomial
    if He[i] is standard probabilist's Hermite polynomial,
    i.e integrating He^2[i] w.r.t. to standard Gaussian measure gives i!, not 1.
    """
    def __init__(self, coef, poly_type='classic'):
        super(PolynomialActivation, self).__init__()
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
        self.classic_deriv_coef = self.classic_coef[1:] * np.arange(1, self.deg + 1)
        self._init_kernel_coef()

    def __call__(self, points):
        return np.polyval(self.classic_coef[::-1], points)

    def apply_deriv(self, points):
        return np.polyval(self.classic_deriv_coef[::-1], points)

    def get_layer(self):
        return stax.Polynomial(self.classic_coef)

    def _init_kernel_coef(self):
        self.kernel_coef = self.hermite_coef ** 2 * factorial(onp.arange(0, self.deg + 1))
        self.kernel_deriv_coef = onp.zeros(self.deg + 1)
        self.kernel_deriv_coef[1:] = onp.array(self.kernel_coef[1:]) * np.arange(1, self.deg + 1)

    def ntk_coef(self, sigma_W2=0.):
        return self.kernel_coef + sigma_W2 ** 2 * self.kernel_deriv_coef

    def kernel(self, X, Y=None, sigma_W2=0.):
        # mostly for sanity check purpose
        if Y is None:
            Y = X
        d = X.shape[1]
        scalar_prod = (X @ Y.T / d)
        result = np.polyval(self.ntk_coef(sigma_W2)[::-1], scalar_prod)
        return result

    def get_hermite_coef_sqr(self, K, include_RF=True, sigma_W2=0., scaled=True):
        # scaled means normalized
        # unscaled means probabilist's
        if K > self.deg:
            return np.zeros(np.shape(sigma_W2))

        coef = (include_RF + K * sigma_W2 ** 2) * self.hermite_coef[K] ** 2
        if scaled:
            return coef * factorial(K)
        else:
            return coef

    def get_hermite_coef_sqr_remainder(self, K, include_RF=True, sigma_W2=0., scaled=True):
        # including K!
        # unscaled means divided by (K - 1)!
        if K > self.deg:
            return np.zeros(np.shape(sigma_W2))

        rem = (include_RF + (np.array(sigma_W2)[..., np.newaxis] ** 2) * np.arange(K, self.deg + 1)) * \
              self.hermite_coef[K:] ** 2 * prod_list(K, self.deg + 1)

        if scaled:
            rem *= factorial(max(0, K - 1))

        return rem.sum(axis=-1)

    def get_lambda_crit(self, K):
        s_k = self.get_hermite_coef_sqr_remainder(K + 1, include_RF=False, sigma_W2=1.)
        s_1 = self.get_hermite_coef_sqr_remainder(K + 1, sigma_W2=0.)
        lreg = s_k / K - s_1
        return lreg

    def rho(self, K, sigma_W2=0., lreg=0.):
        # first axes: sigma_W2
        # last axes: lreg

        sigma_W2, lreg = match_shapes(sigma_W2, lreg)

        if K > self.deg:
            val = np.where(lreg == 0, np.nan, 0.) * sigma_W2
        else:
            val = self.get_hermite_coef_sqr(K, sigma_W2=sigma_W2, scaled=False) / \
                  (self.get_hermite_coef_sqr_remainder(K + 1, sigma_W2=sigma_W2, scaled=False) +
                   lreg / factorial(K))

        return val

    def rho_K1(self, K):
        if K > self.deg:
            return np.nan
        a_k = self.get_hermite_coef_sqr(K, include_RF=False, sigma_W2=1, scaled=False)
        s_k = self.get_hermite_coef_sqr_remainder(K + 1, include_RF=False, sigma_W2=1., scaled=False)
        return a_k / s_k

    def rho_K2(self, K):
        if K > self.deg:
            return np.nan
        a_k = self.get_hermite_coef_sqr(K, scaled=False)
        s_k = self.get_hermite_coef_sqr_remainder(K + 1, scaled=False)
        return a_k / s_k


def get_activation_function(kind='poly', **kwargs):
    if kind == 'poly':
        kwargs['coef'] = read_fraction_list(kwargs['coef'])
        return PolynomialActivation(**kwargs)
    else:
        raise NotImplementedError()


