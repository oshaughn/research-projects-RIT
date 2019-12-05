from __future__ import division
import cupy
import os

#ILE_base = os.environ["ILE_CODE_PATH"]  # default: store code inside main repo. Maintainer controls.
_cuda_code = None


def Q_inner_product_cupy(Q, A, start_indices, window_size):
    num_time_points, num_lms = Q.shape
    num_extrinsic_samples, _ = A.shape

    assert not cupy.isfortran(Q)
    assert not cupy.isfortran(A)

    out = cupy.empty(
        (num_extrinsic_samples, window_size),
        dtype=cupy.complex128,
        order="C",
    )

    global _cuda_code
    if _cuda_code is None:
        # it's assumed that cuda_Q_inner_product.cu is placed in the same folder as this code
        path = os.path.join(os.path.dirname(__file__), 'cuda_Q_inner_product.cu')
        # alternative to deal with packaging in another directory
        if not (os.path.isfile(path)):
            path = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'cuda_Q_inner_product.cu')
        with open(path, 'r') as f:
            _cuda_code = f.read()
            Q_prod_fn = cupy.RawKernel(_cuda_code, "Q_inner")
    else:
        Q_prod_fn = cupy.RawKernel(_cuda_code, "Q_inner")

    float_prec = 16
    num_threads_x = 4
    num_threads_y = 1024 // 4
    block_size = num_threads_x, num_threads_y, 0
    grid_size = (
        (num_extrinsic_samples+num_threads_x-1)//num_threads_x,
        0,
        0,
    )
    args = (
        Q, A, start_indices, window_size,
        num_time_points, num_extrinsic_samples, num_lms,
        out,
    )
    Q_prod_fn(
        grid_size, block_size, args,
        shared_mem=cupy.int32(num_threads_x*num_lms*float_prec),
    )

    return out
