import jax.numpy as np
from polynomial import *


def compute_krr_mse_poly_infinite(X_train, g_fn_coef, xi_weights, activation_coef, learned_weights):
    dimension = X_train.shape[1]
    total_error = classic_poly_dsphere_1c_expectation(
        poly_square(g_fn_coef), dimension)  # first component, g^2

    weight_xi = np.clip(X_train @ xi_weights.T / dimension, -1, 1)[:, 0]  # second component, g * K
    weight_xi_orth = np.sqrt(1 - weight_xi ** 2)
    poly_kernel_gK = polynomial_1d_expansion(
        activation_coef, weight_xi, weight_xi_orth
    ).transpose((1, 2, 0)).dot(learned_weights[:, 0])
    poly_kernel_gK = classic_poly_2d_normalize(poly_kernel_gK, dimension)
    total_error -= 2 * classic_poly_dsphere_2c_expectation(
        classic_poly_mul(poly_kernel_gK.T, g_fn_coef).T, dimension)

    weight_xi = np.clip(X_train @ X_train.T / dimension, -1, 1)
    weight_xi_orth = np.sqrt(1 - weight_xi ** 2)
    poly_kernel_KK = polynomial_1d_expansion(
        activation_coef, weight_xi, weight_xi_orth
    ).transpose((2, 3, 0, 1)).dot(learned_weights[:, 0]).dot(learned_weights[:, 0])

    total_error += classic_poly_dsphere_2c_expectation(
        classic_poly_2d_normalize(classic_poly_mul(poly_kernel_KK.T, activation_coef).T, dimension),
        dimension)
    return total_error


def compute_krr_mse_poly_finite(X_train, params, g_fn, xi_weights, activation, learned_weights):
    dimension = X_train.shape[1]

    th_error_g_g = classic_poly_dsphere_1c_expectation(
        poly_square(g_fn.classic_coef), dimension)

    W1, W2 = params[0][0], params[2][0][:, 0]
    N_width = W1.shape[1]

    weight_W1 = np.linalg.norm(W1, axis=0) / np.sqrt(dimension)
    W1_normalized = W1 / weight_W1
    weight_W1_X = (X_train @ W1_normalized / dimension).T

    F_train = activation(X_train @ W1 / np.sqrt(dimension))
    F_deriv_train = activation.apply_deriv(X_train @ W1 / np.sqrt(dimension))
    F_train_weighted = F_train.T @ learned_weights[:, 0]

    weight_W1_xi = np.clip(W1_normalized.T @ xi_weights.T / dimension, -1, 1)[:, 0]
    weight_W1_xi_orth = np.sqrt(1 - weight_W1_xi ** 2)

    poly_g_K2 = polynomial_1d_expansion(
        activation.classic_coef, weight_W1_xi * weight_W1, weight_W1_xi_orth * weight_W1
    )
    poly_g_K2 = (F_train_weighted[:, np.newaxis, np.newaxis] * poly_g_K2).mean(axis=0)
    th_error_g_K2 = classic_poly_dsphere_2c_expectation(
        classic_poly_mul(poly_g_K2.T, g_fn.classic_coef).T, dimension)

    weight_X_xi = np.clip(X_train @ xi_weights.T / dimension, -1, 1)[:, 0]
    weight_X_xi_orth = np.sqrt(1 - weight_X_xi ** 2)
    weight_coord_x1 = F_deriv_train.T @ (learned_weights[:, 0] * weight_X_xi)
    weight_coord_x2 = ((weight_W1_X - weight_W1_xi[:, np.newaxis] * weight_X_xi[np.newaxis, :]) *
                       F_deriv_train.T @ learned_weights[:, 0]) / weight_W1_xi_orth
    poly_g_K1 = (W2[:, np.newaxis, np.newaxis] ** 2) * polynomial_1d_expansion(
        activation.classic_deriv_coef, weight_W1_xi * weight_W1, weight_W1_xi_orth * weight_W1
    )
    poly_g_K1_x1 = (poly_g_K1 * weight_coord_x1[:, np.newaxis, np.newaxis]).mean(axis=0)
    poly_g_K1_x2 = (poly_g_K1 * weight_coord_x2[:, np.newaxis, np.newaxis]).mean(axis=0)
    poly_g_K1 = onp.zeros(onp.array(poly_g_K1_x1.shape) + 1)
    poly_g_K1[1:, :-1] += poly_g_K1_x1
    poly_g_K1[:-1, 1:] += poly_g_K1_x2

    th_error_g_K1 = classic_poly_dsphere_2c_expectation(
        classic_poly_mul(poly_g_K1.T, g_fn.classic_coef).T, dimension) / np.sqrt(dimension)

    weight_W1_W1 = np.clip(W1_normalized.T @ W1_normalized / dimension, -1, 1)
    weight_W1_W1_orth = np.sqrt(1 - weight_W1_W1 ** 2)
    sigma_W1_expansion = classic_poly_1d_rescale(activation.classic_coef, weight_W1)

    poly_K2_K2 = polynomial_1d_expansion(activation.classic_coef, weight_W1_W1 * weight_W1[np.newaxis, :],
                                         weight_W1_W1_orth * weight_W1[np.newaxis, :])
    poly_K2_K2 = poly_K2_K2.transpose((0, 2, 3, 1)) @ F_train_weighted / N_width
    poly_K2_K2 = classic_poly_mul(poly_K2_K2.transpose((0, 2, 1)),
                                  sigma_W1_expansion[:, np.newaxis, :])
    poly_K2_K2 = poly_K2_K2.transpose((2, 1, 0)) @ F_train_weighted / N_width
    th_error_K2_K2 = classic_poly_dsphere_2c_expectation(poly_K2_K2, dimension)

    F_deriv_weighted_2coord = weight_W1_X * learned_weights[:, 0] @ F_deriv_train / np.sqrt(dimension)
    F_deriv_weighted_1coord = (weight_W1_X * F_deriv_train.T) @ learned_weights[:, 0] / np.sqrt(dimension)
    sigma_deriv_W1_W1_expansion = polynomial_1d_expansion(activation.classic_deriv_coef,
                                                          weight_W1_W1 * weight_W1[np.newaxis, :],
                                                          weight_W1_W1_orth * weight_W1[np.newaxis, :])

    def reduce_K2_K1(poly):
        poly = (sigma_deriv_W1_W1_expansion *
                poly[..., np.newaxis, np.newaxis]).transpose((0, 2, 3, 1)) @ W2 ** 2 / N_width
        poly = classic_poly_mul(poly.transpose((0, 2, 1)), sigma_W1_expansion[:, np.newaxis, :])
        poly = poly.transpose((2, 1, 0)) @ F_train_weighted / N_width
        return poly

    poly_K2_K1_x1 = F_deriv_weighted_2coord
    poly_K2_K1_x2 = (
                F_deriv_weighted_1coord[np.newaxis, :] - weight_W1_W1 * F_deriv_weighted_2coord)  # / weight_W1_W1_orth
    poly_K2_K1_x2 = np.where(np.abs(weight_W1_W1) == 1, 0, poly_K2_K1_x2 / weight_W1_W1_orth)
    poly_K2_K1_x1 = reduce_K2_K1(poly_K2_K1_x1)
    poly_K2_K1_x2 = reduce_K2_K1(poly_K2_K1_x2)
    poly_K2_K1 = onp.zeros(onp.array(poly_K2_K1_x1.shape) + 1)
    poly_K2_K1[1:, :-1] += poly_K2_K1_x1
    poly_K2_K1[:-1, 1:] += poly_K2_K1_x2
    th_error_K2_K1 = classic_poly_dsphere_2c_expectation(poly_K2_K1, dimension)

    sigma_deriv_W1_expansion = classic_poly_1d_rescale(activation.classic_deriv_coef, weight_W1)

    poly_K1_K1_x1_x1 = F_deriv_weighted_1coord[:, np.newaxis] * F_deriv_weighted_2coord
    poly_K1_K1_x1_x2 = (F_deriv_weighted_1coord[:, np.newaxis] * F_deriv_weighted_1coord[np.newaxis, :] +
                        F_deriv_weighted_2coord * F_deriv_weighted_2coord.T -
                        2 * weight_W1_W1 * F_deriv_weighted_1coord[:,
                                           np.newaxis] * F_deriv_weighted_2coord) / weight_W1_W1_orth
    poly_K1_K1_x2_x2 = (F_deriv_weighted_2coord.T * F_deriv_weighted_1coord[np.newaxis, :]
                        - (F_deriv_weighted_1coord[:, np.newaxis] * F_deriv_weighted_1coord[np.newaxis, :] +
                           F_deriv_weighted_2coord.T * F_deriv_weighted_2coord) * weight_W1_W1 +
                        F_deriv_weighted_1coord[:, np.newaxis] * F_deriv_weighted_2coord * weight_W1_W1 ** 2) / \
                       (1 - weight_W1_W1 ** 2)
    poly_K1_K1_x3_x3 = ((F_deriv_train.T * learned_weights[:, 0]) @ X_train / dimension @ X_train.T @
                        (F_deriv_train.T * learned_weights[:, 0]).T / dimension)
    poly_K1_K1_x3_x3_diag = poly_K1_K1_x3_x3 - F_deriv_weighted_1coord[np.newaxis, :] ** 2
    poly_K1_K1_x3_x3_out_diag = poly_K1_K1_x3_x3 + (
            weight_W1_W1 * (F_deriv_weighted_1coord[:, np.newaxis] * F_deriv_weighted_1coord[np.newaxis, :] +
                            F_deriv_weighted_2coord * F_deriv_weighted_2coord.T) -
            (F_deriv_weighted_1coord[:, np.newaxis] * F_deriv_weighted_2coord +
             F_deriv_weighted_2coord.T * F_deriv_weighted_1coord[np.newaxis, :])
    ) / (1 - weight_W1_W1 ** 2)

    poly_K1_K1_x1_x2 = np.where(np.abs(weight_W1_W1) == 1, 0, poly_K1_K1_x1_x2)
    poly_K1_K1_x2_x2 = np.where(np.abs(weight_W1_W1) == 1, 0, poly_K1_K1_x2_x2)
    poly_K1_K1_x3_x3 = np.where(np.abs(weight_W1_W1) == 1, poly_K1_K1_x3_x3_diag,
                                poly_K1_K1_x3_x3_out_diag)

    def reduce_K1_K1(poly):
        poly = (sigma_deriv_W1_W1_expansion *
                poly[..., np.newaxis, np.newaxis]).transpose((0, 2, 3, 1)) @ W2 ** 2 / N_width
        poly = classic_poly_mul(poly.transpose((0, 2, 1)), sigma_deriv_W1_expansion[:, np.newaxis, :])
        poly = poly.transpose((2, 1, 0)) @ W2 ** 2 / N_width
        return poly

    poly_K1_K1_x1_x1 = reduce_K1_K1(poly_K1_K1_x1_x1)
    poly_K1_K1_x1_x2 = reduce_K1_K1(poly_K1_K1_x1_x2)
    poly_K1_K1_x2_x2 = reduce_K1_K1(poly_K1_K1_x2_x2)
    poly_K1_K1_x3_x3 = reduce_K1_K1(poly_K1_K1_x3_x3)

    poly_K1_K1 = onp.zeros(onp.array(poly_K1_K1_x1_x1.shape) + 2)
    poly_K1_K1[2:, :-2] += poly_K1_K1_x1_x1
    poly_K1_K1[1:-1, 1:-1] += poly_K1_K1_x1_x2
    poly_K1_K1[:-2, 2:] += poly_K1_K1_x2_x2
    poly_K1_K1[:-2, :-2] += (dimension / (dimension + np.arange(poly_K1_K1_x3_x3.shape[0])[:, np.newaxis]
                                          + np.arange(poly_K1_K1_x3_x3.shape[1])[np.newaxis, :])) * poly_K1_K1_x3_x3
    th_error_K1_K1 = classic_poly_dsphere_2c_expectation(poly_K1_K1, dimension)
    total_error = th_error_g_g - 2 * (th_error_g_K1 + th_error_g_K2) + \
                  (th_error_K1_K1 + 2 * th_error_K2_K1 + th_error_K2_K2)

    # return {'test_error': total_error.item(), 'g_g': th_error_g_g.item(),
    #         'g_K1': th_error_g_K1.item(), 'g_K2': th_error_g_K2.item(),
    #         'K1_K1': th_error_K1_K1.item(), 'K1_K2': th_error_K2_K1.item(), 'K2_K2': th_error_K2_K2.item()}
    return total_error.item()
