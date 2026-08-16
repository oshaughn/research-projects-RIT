"""Shared helper for the GPU consistency tests: skip HONESTLY when there is no GPU.

These files are dual-use -- runnable as plain scripts on a GPU node, and collectable by pytest.
Their original no-GPU path printed a line and `return`ed, which is right for the script mode but
WRONG under pytest: a test that returns without asserting is reported as PASSED.  A CI run on a
machine with no GPU would then show green for the GPU parity checks while having verified
nothing, which is precisely the false-green that hid the zero-collection bug in
test_q_window_interp.py.

skip_without_gpu() reports a real pytest skip when running under pytest, and falls back to the
printed message when the file is run as a script.
"""
from __future__ import print_function

import sys


def skip_without_gpu(have_gpu, why, label="GPU"):
    """Return True if the caller should bail out because no GPU is available.

    Under pytest this raises Skipped instead of returning, so the test is recorded as SKIPPED
    rather than PASSED.  Run as a script, it prints and returns True.
    """
    if have_gpu:
        return False
    msg = "cupy/GPU unavailable (%s)" % (why,)
    if "pytest" in sys.modules:                      # collected by pytest -> real skip
        import pytest
        pytest.skip(msg)
    print("(%s) SKIPPED: %s" % (label, msg))         # run as a script -> just say so
    return True
