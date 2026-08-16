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


def Q_inner_product_cubic_cupy(Q, A, start_indices, fractional_offsets, window_size):
    """Cubic-interpolated Q inner product for fractional detector-time offsets.

    ``start_indices`` are the integer floor indices of the first requested time
    sample and ``fractional_offsets`` are the corresponding fractional parts in
    [0, 1).  The CUDA kernel uses a four-point cubic Lagrange stencil along the
    time axis and zero extension outside the precomputed Q buffer.
    """
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
        path = os.path.join(os.path.dirname(__file__), 'cuda_Q_inner_product.cu')
        if not (os.path.isfile(path)):
            path = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'cuda_Q_inner_product.cu')
        with open(path, 'r') as f:
            _cuda_code = f.read()
            Q_prod_fn = cupy.RawKernel(_cuda_code, "Q_inner_cubic")
    else:
        Q_prod_fn = cupy.RawKernel(_cuda_code, "Q_inner_cubic")

    float_prec = 16
    # The cubic stencil uses more registers than the nearest-neighbor kernel, so
    # use a conservative default block shape to keep launches portable across
    # older GPUs.  Tune upward on newer cards with RIFT_Q_CUBIC_THREADS_X/Y.
    num_threads_x = int(os.environ.get("RIFT_Q_CUBIC_THREADS_X", "4"))
    num_threads_y = int(os.environ.get("RIFT_Q_CUBIC_THREADS_Y", "128"))
    block_size = num_threads_x, num_threads_y, 0
    grid_size = (
        (num_extrinsic_samples+num_threads_x-1)//num_threads_x,
        0,
        0,
    )
    args = (
        Q, A, start_indices, fractional_offsets, window_size,
        num_time_points, num_extrinsic_samples, num_lms,
        out,
    )
    Q_prod_fn(
        grid_size, block_size, args,
        shared_mem=cupy.int32(0),
    )

    return out


def Q_inner_product_sinc_cupy(Q, A, start_indices, fractional_offsets, window_size,
                              halfwidth=None):
    """Band-limited (Lanczos windowed-sinc) Q inner product for fractional detector-time offsets.

    Same contract as ``Q_inner_product_cubic_cupy``: ``start_indices`` are the integer floor
    indices of the first requested time sample, ``fractional_offsets`` the corresponding
    fractional parts in [0, 1).  The stencil is 2*halfwidth taps wide (default
    ``factored_likelihood.SINC_HALFWIDTH_DEFAULT``) with zero extension outside the precomputed
    Q buffer.

    The tap weights come from the same ``factored_likelihood._sinc_lanczos_weight_matrix`` the
    CPU window builder uses, evaluated with the cupy backend so they are built ON THE DEVICE:
    deriving them a second time in CUDA would put two independent definitions of the stencil in
    the tree, and pulling the offsets back to the host to use the numpy path would move tens of
    MB per detector per call at production n_extrinsic.  The weight work is O(n_ex * 2a) against
    the kernel's O(n_ex * window * n_lms * 2a), so it is negligible either way.

    Which stencil to use depends on the oversampling factor fNyq/fmax -- see
    ``_sinc_Q_window_numpy`` for the measured crossover.  This one is the accurate choice near
    Nyquist, which is where production runs sit.
    """
    # Deferred import: factored_likelihood imports this module, so a top-level import would be
    # circular.  By call time factored_likelihood is always fully imported (it is the caller).
    from .factored_likelihood import _sinc_lanczos_weight_matrix, SINC_HALFWIDTH_DEFAULT

    if halfwidth is None:
        halfwidth = SINC_HALFWIDTH_DEFAULT

    num_time_points, num_lms = Q.shape
    num_extrinsic_samples, _ = A.shape

    assert not cupy.isfortran(Q)
    assert not cupy.isfortran(A)

    _offsets, tap_weights = _sinc_lanczos_weight_matrix(
        cupy.asarray(fractional_offsets), halfwidth, xpy=cupy)
    # Derived from halfwidth, NOT read back off _offsets: indexing a cupy array to get a Python
    # int forces a device sync, and this runs once per detector per likelihood call.  The two
    # must agree, so assert it rather than trusting the comment -- cheap, host-side only.
    n_taps = 2 * halfwidth
    tap_first = -halfwidth + 1
    assert _offsets.shape == (n_taps,), \
        "weight-matrix stencil width %r disagrees with 2*halfwidth=%d" % (_offsets.shape, n_taps)
    tap_weights_d = cupy.ascontiguousarray(tap_weights.astype(cupy.float64))

    out = cupy.empty(
        (num_extrinsic_samples, window_size),
        dtype=cupy.complex128,
        order="C",
    )

    global _cuda_code
    if _cuda_code is None:
        path = os.path.join(os.path.dirname(__file__), 'cuda_Q_inner_product.cu')
        if not (os.path.isfile(path)):
            path = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'cuda_Q_inner_product.cu')
        with open(path, 'r') as f:
            _cuda_code = f.read()
            Q_prod_fn = cupy.RawKernel(_cuda_code, "Q_inner_sinc")
    else:
        Q_prod_fn = cupy.RawKernel(_cuda_code, "Q_inner_sinc")

    # 2a taps against the cubic's 4, so this kernel is heavier still; keep the same conservative
    # default block shape and the same env-tunable override.
    num_threads_x = int(os.environ.get("RIFT_Q_SINC_THREADS_X", "4"))
    num_threads_y = int(os.environ.get("RIFT_Q_SINC_THREADS_Y", "128"))
    block_size = num_threads_x, num_threads_y, 0
    grid_size = (
        (num_extrinsic_samples+num_threads_x-1)//num_threads_x,
        0,
        0,
    )
    args = (
        Q, A, start_indices, tap_weights_d, n_taps, tap_first, window_size,
        num_time_points, num_extrinsic_samples, num_lms,
        out,
    )
    Q_prod_fn(
        grid_size, block_size, args,
        # one double per tap per threadIdx.x, staged so the innermost loop reads shared not global
        shared_mem=cupy.int32(num_threads_x*n_taps*8),
    )

    return out
