from __future__ import division
import cupy

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

    mod = cupy.cuda.function.Module()
    mod.load_file("cuda_Q_inner_product.cubin")
    Q_prod_fn = mod.get_function("Q_inner")

    float_prec = 16
    num_threads = 128
    grid_size = num_threads, 0, 0
    block_size = (num_extrinsic_samples+num_threads-1)//num_threads, 0, 0
    args = (
        Q, A, start_indices, window_size,
        num_time_points, num_extrinsic_samples, num_lms,
        out,
    )
    Q_prod_fn(
        grid_size, block_size, args,
        shared_mem=cupy.int32(num_threads*num_lms*float_prec),
    )

    return out
