import os

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--require-gpu", action="store_true", default=False,
        help="fail instead of skip unless CuPy can execute on a CUDA device",
    )


@pytest.fixture(params=("numpy", "cupy"))
def backend(request):
    if request.param == "numpy":
        return np
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy reports no CUDA devices")
        _ = (cupy.arange(4) + 1).sum()
        cupy.cuda.Stream.null.synchronize()
        return cupy
    except Exception as exc:
        if request.config.getoption("--require-gpu") or os.environ.get("RIFT_REQUIRE_GPU_PRECOMPUTE") == "1":
            pytest.fail("real CUDA device required: %r" % (exc,))
        pytest.skip("no usable CUDA device: %r" % (exc,))


def to_host(x):
    try:
        import cupy
        if isinstance(x, cupy.ndarray):
            return cupy.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)
