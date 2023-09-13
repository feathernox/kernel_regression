import hydra
from omegaconf import OmegaConf

import numpy as onp
import jax.numpy as np
from jax import random, jit

import neural_tangents as nt
from neural_tangents import stax

import copy

import regression_error
import sphere_distrib
from target_function import get_target_function
from activation import get_activation_function

from scipy.stats import bootstrap
import time


@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config: OmegaConf):
    key = random.PRNGKey(config.seed)
    np_rng_key = onp.random.default_rng(seed=config.seed)

    n = config.hyperparams.n
    d = config.hyperparams.d

    g_fn = get_target_function(**config.target_function)
    sigma_z = config.hyperparams.sigma_z

    sigma_W2 = config.hyperparams.sigma_W2
    lreg = config.hyperparams.lmbd
    activation = get_activation_function(**config.activation)

    n_exp = config.n_experiments
    use_test_sample = config.use_test_sample
    compute_infinite_width = config.compute_infinite_width
    if compute_infinite_width:
        N_width = 1
    else:
        N_width = config.hyperparams.N_width

    init_fn, apply_fn, kernel_ntk_inf_fn = stax.serial(
        stax.Dense(N_width, W_std=np.sqrt(d), b_std=None, parameterization='standard'),
        activation.get_layer(),
        stax.Dense(1, W_std=sigma_W2 * np.sqrt(N_width), b_std=None, parameterization='standard')
    )

    if compute_infinite_width:
        kernel_ntk_inf_fn = jit(kernel_ntk_inf_fn, static_argnames='get')
        error_ifinite = jit(regression_error.compute_krr_mse_poly_infinite)
    else:
        kwargs = dict(
            f=apply_fn,
            trace_axes=(),
            vmap_axes=0
        )

        empirical_ntk = jit(nt.empirical_ntk_fn(
            **kwargs, implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES))

    test_errors = []

    for i_exp in range(n_exp):
        key, X_train_key, xi_key, z_train_key = random.split(key, 4)
        # todo: introduce storing datasets in case X_train is too large
        X_train = sphere_distrib.generate_sample(X_train_key, n, d)
        xi_weights = sphere_distrib.generate_sample(xi_key, 1, d)
        z_train = sigma_z * random.normal(z_train_key, (n, 1))
        y_train = g_fn(X_train @ xi_weights.T / np.sqrt(d)) + z_train

        if use_test_sample:
            key, X_test_key = random.split(key, 2)
            n_test = config.n_test
            X_test = sphere_distrib.generate_sample(X_test_key, n_test, d)
            y_test = g_fn(X_test @ xi_weights.T / np.sqrt(d))

        if compute_infinite_width:
            K_ntk_inf_train = kernel_ntk_inf_fn(X_train / np.sqrt(d), X_train / np.sqrt(d), 'ntk') / N_width

            predictor = nt.predict.gp_inference(K_ntk_inf_train, y_train, diag_reg=lreg, diag_reg_absolute_scale=True)

            # hack to obtain weights:
            learned_weights = predictor('ntk', np.eye(n))


            # TODO: organize checks that it's polynomial
            total_error = error_ifinite(X_train, g_fn.classic_coef, xi_weights,
                                        activation.ntk_coef(sigma_W2), learned_weights)
            stat_dict = {'mse_precise': total_error}

            if use_test_sample:
                K_ntk_inf_test_train = kernel_ntk_inf_fn(X_test / np.sqrt(d), X_train / np.sqrt(d), 'ntk').reshape(
                    (n_test, n)) / N_width
                y_pred = predictor('ntk', K_ntk_inf_test_train)
                errs = (y_test - y_pred) ** 2
                ci = bootstrap((errs,), onp.mean, random_state=np_rng_key).confidence_interval
                test_error = errs.mean()

                stat_dict['mse_sample'] = {'mean': test_error, 'ci': (ci.low[0], ci.high[0])}

            test_errors.append(stat_dict)
        else:
            key, net_key = random.split(key)
            _, params = init_fn(net_key, (-1, d))

            K_ntk_emp_train = empirical_ntk(X_train / np.sqrt(d), X_train / np.sqrt(d), params)[:, :, 0, 0] / N_width
            predictor = nt.predict.gp_inference(K_ntk_emp_train, y_train, diag_reg=lreg, diag_reg_absolute_scale=True)
            # hack to obtain weights:
            learned_weights = predictor('ntk', np.eye(n))
            # TODO: adapt the finite error predictor
            total_error = regression_error.compute_krr_mse_poly_finite(X_train, params, g_fn,
                                                                       xi_weights, activation, learned_weights)

            if use_test_sample:
                K_ntk_emp_test_train = empirical_ntk(X_test / np.sqrt(d), X_train / np.sqrt(d), params)[:, :, 0,
                                       0] / N_width
                y_pred = predictor('ntk', K_ntk_emp_test_train)
                params_K2 = copy.deepcopy(params)
                params_K2[2] = (params_K2[2][0] * 0, None)
                K_ntk_2_emp_test_train = empirical_ntk(X_test / np.sqrt(d), X_train / np.sqrt(d), params_K2)[:, :,
                                         0, 0] / N_width
                K_ntk_1_emp_test_train = K_ntk_emp_test_train - K_ntk_2_emp_test_train
                y_pred_K2 = predictor('ntk', K_ntk_2_emp_test_train)
                y_pred_K1 = predictor('ntk', K_ntk_1_emp_test_train)
                print(((y_test - y_pred) ** 2).mean())
    print(test_errors)


if __name__ == "__main__":
    main()
