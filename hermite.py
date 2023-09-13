import numpy as onp
import jax.numpy as np

# Probabilist's non-normalized Hermite polynomials
# I use standard numpy here because I operate with arrays of different lengths which I typically sum
# No implementation of Gegenbauer's polynomials --
# seems not to be easily computable for big d which is the case we are interested in
class HermitePolynomials:
    def __init__(self):
        self.hermite_polynomials = [
            onp.array([1.]),
            onp.array([0., 1.])
        ]

    def __getitem__(self, item):
        self._update_poly(item)
        return self.hermite_polynomials[item]

    def _update_poly(self, degree):
        if degree < len(self.hermite_polynomials):
            return
        self._update_poly(degree - 1)

        new_poly = onp.zeros(degree + 1)
        new_poly[1:] += onp.array(self.hermite_polynomials[degree - 1])
        new_poly[:-2] += - (degree - 1) * onp.array(self.hermite_polynomials[degree - 2])
        self.hermite_polynomials.append(new_poly)


HERMITE_COMPUTED = HermitePolynomials()


# todo: convert to jax / make multidimensional?
def hermite2classic(hermite_coef):
    n_coef = len(hermite_coef)
    classic_coef = onp.zeros(n_coef)
    for idx_poly in range(n_coef):
        classic_coef[:idx_poly + 1] += hermite_coef[idx_poly] * HERMITE_COMPUTED[idx_poly]
    return classic_coef


def classic2hermite(classic_coef):
    n_coef = len(classic_coef)
    classic_coef = onp.array(classic_coef)
    hermite_coef = onp.zeros(n_coef)

    for idx_poly in range(n_coef - 1, -1, -1):
        hermite_coef[idx_poly] = classic_coef[idx_poly]
        classic_coef[:idx_poly + 1] -= hermite_coef[idx_poly] * HERMITE_COMPUTED[idx_poly]

    return hermite_coef
