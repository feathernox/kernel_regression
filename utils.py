import jax.numpy as np
import numpy as onp


def read_fraction_list(arr):
    return list(map(lambda x: eval(x) if isinstance(x, str) else x, arr))


def prod_list(start, end):
    if start == end:
        return onp.array([])
    pl = [start if start > 0 else 1]
    for i in range(start + 1, end):
        pl.append(pl[-1] * i)
    return onp.array(pl)


def match_shapes(A, B):
    # meshgrid-like function
    # i.e. we have A of shape (a_1, ..., a_m) and B of shape (b_1, ..., b_n);
    # then, the result will be (m+n)-dimensional array A_ext (a_1, ..., a_m, 1, ..., 1)
    # and (m+n)-dimensional array B_ext (1, ..., 1, b_1, ..., b_n)
    A_arr = np.array(A)
    B_arr = np.array(B)
    A_ext = np.expand_dims(A_arr, A_arr.ndim + np.arange(B_arr.ndim))
    B_ext = np.expand_dims(B_arr, np.arange(A_arr.ndim))
    new_shape = tuple([*A_arr.shape, *B_arr.shape])
    A_ext = np.broadcast_to(A_ext, new_shape)
    B_ext = np.broadcast_to(B_ext, new_shape)
    return A_ext, B_ext
