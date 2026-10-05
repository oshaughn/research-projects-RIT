"""Decide whether this pytest process can run a cupy kernel, without crashing on the way.

NOT a pytest target -- a helper two GPU-parity test files import.  It is named without a
``test_`` prefix so pytest does not collect it and so the ``test_*.py`` scope globs in
.travis/test-q-window-stencil.sh do not demand a registration marker for it.

THE DEFECT IT EXISTS FOR, and WHICH pytest has it.  From pytest 9.1,
``pytest.importorskip('cupy')`` skips on ModuleNotFoundError and RE-RAISES every other
ImportError.  The IGWN CVMFS environment ships the cupy PACKAGE on every CIT host, so cupy
always exists in the ModuleNotFoundError sense; on a host with no CUDA driver the import gets
as far as ``libcuda.so.1: cannot open shared object file``, which is a plain ImportError.
importorskip therefore turns a GPU-less host into a test FAILURE.

The version matters, and the first draft of this file got it wrong by naming the environment
instead.  Measured on ldas-grid 2026-09-18, same interpreter
(/cvmfs/software.igwn.org/conda/envs/igwn/bin/python, 3.11, cupy 12.0.0):

    pytest 9.1.1   from ~/.local/lib/python3.11/site-packages   1 failed
    pytest 8.3.5   CVMFS's own, reached with PYTHONNOUSERSITE=1  1 skipped

8.3.5 defaults ``exc_type=ImportError``, so it still skips, and warns that it will stop:
``PytestDeprecationWarning: Module 'cupy' was found, but when imported by pytest it raised``.
The user site-packages on the NFS home shadows CVMFS, so 9.1.1 is what an ordinary
interactive run on these hosts gets, and the two callers did fail there.  A container, a
condor execute node, or anyone with PYTHONNOUSERSITE set sees 8.3.5 and the old guard skipping
-- so "it reproduces on CIT" is not the claim; "it reproduces under pytest >= 9.1" is, and
that becomes universal when CVMFS bumps pytest.  GitHub's runners have no cupy installed at
all, get the clean ModuleNotFoundError, and skip either way, so CI cannot see this at all.

THE SECOND FAILURE, which looks identical from a distance.  ``import cupy`` SUCCEEDS on the
CIT GPU head nodes while the visible device is one this cupy cannot compile for.  Measured on
ldas-pcdev2 2026-09-18: with CUDA_VISIBLE_DEVICES=1 the import succeeds and the first kernel
answers ``nvrtc: error: invalid value for --gpu-architecture`` -- that slot is a Blackwell
cc 12.0 and this is cupy 12.0.0.  So the probe RUNS a kernel rather than trusting the import.

It probes AT DISPATCH, per slot, because neither the slot map nor its ordering is stable.  On
that same host nvidia-smi lists the RTX 3080 at 0, the Blackwell at 1 and the A100 at 2, while
CUDA's own ordering puts the A100 at 0 and the 3080 at 2.  A remembered index, or an index
read off nvidia-smi, picks a different card than the one it names.

Approach and verdict grammar are taken from ``_GPU_PROBE``/``gpu_slot``/``_no_gpu`` in
test_e2e_analytic_pipeline.py, which settled this first and is deliberately left alone: that
gate runs ILE CHILDREN and must hold no CUDA context in the pytest process, while the two
callers here use cupy IN THIS PROCESS and re-index nothing.  The two copies answer the same
question for different consumers; if you change the verdict grammar, change both.
"""

import os
import subprocess
import sys

import pytest

# Every verdict is ONE line beginning with a known word, and every message is flattened,
# because cupy's ImportError is a multi-line banner: reading "the last line of the probe's
# output" turns it into the skip reason "no usable GPU (If you installed CuPy via whee)".
_GPU_PROBE = r"""
import numpy as np
def _flat(e):
    return ("%s: %s" % (type(e).__name__, e)).replace("\n", " ")[:150]
try:
    import cupy
except Exception as e:
    print("VERDICT NOCUPY %s" % _flat(e)); raise SystemExit(0)
bad = []
for d in range(cupy.cuda.runtime.getDeviceCount()):
    try:
        with cupy.cuda.Device(d):
            cupy.asnumpy(cupy.cos(cupy.asarray(np.zeros(2), dtype=float)))
    except Exception as e:
        bad.append("%d:%s" % (d, type(e).__name__)); continue
    print("VERDICT SLOT %d" % d); raise SystemExit(0)
print("VERDICT NOSLOT %s" % (",".join(bad) or "no devices at all"))
"""

# Keyed by CUDA_VISIBLE_DEVICES, because that is the only input the verdict depends on and a
# test that pins it would otherwise be answered from a stale cache.  Costs one interpreter
# startup plus one cupy import per distinct value, which is why it is cached at all.
#
# Only a REAL verdict is cached.  The two failure strings below -- the probe would not run, and
# the probe ran but said nothing -- describe a transient outage, and caching one would make the
# rest of the session answer from an outage that has since cleared.  A cached SLOT is safe
# because cupy_or_skip re-checks it in this process every call.
_VERDICT_CACHE = {}
_REAL_VERDICTS = ("SLOT ", "NOCUPY ", "NOSLOT ")


def _flat(e):
    return ("%s: %s" % (type(e).__name__, e)).replace("\n", " ")[:150]


def probe_verdict():
    """``SLOT <d>`` / ``NOCUPY <why>`` / ``NOSLOT <why>``, measured in a SUBPROCESS.

    A subprocess because the question cannot be asked in-process without paying the cost of
    the wrong answer: ``import cupy`` on a driverless host raises out of wherever it is
    called, and probing device 0 on a Blackwell slot leaves the failure in this process.

    ``d`` indexes the CURRENTLY VISIBLE device list, and the child inherits this process's
    CUDA_VISIBLE_DEVICES, so ``d`` means the same thing to both.  The e2e gate re-indexes its
    answer because it hands an explicit CUDA_VISIBLE_DEVICES to a grandchild; callers here
    use the slot in this process and must not.

    THE COST OF PROBING AT ALL, which .travis/test-integrate.sh:165 already records for the e2e
    gate: break the probe and the lane skips with a reason naming the GPU, which a reason-
    matching gate scores as fine.  Measured 2026-09-18 on ldas-pcdev2 with a working A100 --
    an unimportable module inside _GPU_PROBE silently removes both callers and the peak-local
    gate stays green.  What closes it is RIFT_CI_REQUIRE_GPU=1, under which no_gpu FAILS; the
    gpu_integration job in .gitlab-ci.yml sets it (with CUDA_VISIBLE_DEVICES=0), so CI is
    covered and an interactive hand-run on a GPU node without it is not.
    """
    key = os.environ.get("CUDA_VISIBLE_DEVICES")
    if key in _VERDICT_CACHE:
        return _VERDICT_CACHE[key]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"          # the CIT head nodes cap THREADS, not processes
    env["MPLBACKEND"] = "Agg"
    try:
        proc = subprocess.run([sys.executable, "-c", _GPU_PROBE], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    except (subprocess.TimeoutExpired, OSError) as e:
        # Not re-raised, which is the whole subject of this module: a CVMFS stall or a wedged
        # driver is an environment outage, and a non-GPU run must not turn red for one.  It
        # still goes through no_gpu(), so RIFT_CI_REQUIRE_GPU=1 keeps failing on it.
        return "the GPU probe could not be run: %s" % _flat(e)
    verdicts = [l for l in proc.stdout.decode().splitlines() if l.startswith("VERDICT ")]
    line = verdicts[-1][len("VERDICT "):] if verdicts else (
        "probe produced no verdict (rc=%d): %s" % (proc.returncode,
                                                   proc.stdout.decode()[-300:]))
    if line.startswith(_REAL_VERDICTS):
        _VERDICT_CACHE[key] = line
    return line


def no_gpu(reason):
    """Skip, or FAIL when the environment promised a device.

    RIFT_CI_REQUIRE_GPU=1 is the GPU runner saying it HAS a device; there, a skipped device
    lane is a green report for a lane that never ran.

    DO NOT DELETE THIS AS A DUPLICATE OF THE SHELL.  Re-checked 2026-09-19, because the first
    version of this paragraph went stale within a day and said three wrong things:

      * all four skip-scoring blocks in .travis/test-integrate.sh now carry an "under
        RIFT_CI_REQUIRE_GPU any skip is fatal" branch.  Two did when this was written.
      * .travis/test-q-window-stencil.sh has no RIFT_CI_REQUIRE_GPU branch and no reason
        matching either.  It scores by junit COUNTS -- EXPECTED_PASSED=75, MAX_SKIPS=3 -- so a
        new skip there drops `passed` below the floor and fails on the count, not the reason.
      * this helper has three callers, not two.

    What the shell still cannot do is survive `pytest <file>` run by hand on the GPU node,
    which is what someone does to reproduce a CI failure, and that run would report green with
    the device lane skipped.  That is what this function is for.
    """
    if os.environ.get("RIFT_CI_REQUIRE_GPU", "0") == "1":
        pytest.fail("RIFT_CI_REQUIRE_GPU=1 promised a usable device and there is none: %s.  "
                    "On this runner a skipped device lane is a failure, not a pass." % reason)
    pytest.skip("%s  A skip is NOT a pass: pin CUDA_VISIBLE_DEVICES to a slot the installed "
                "cupy supports and rerun." % reason)


def _rift_backend_complaints(cupy):
    """Why RIFT itself is not on the device, as a list of strings, empty when it is.

    RIFT's ``try: import cupy`` probes run AT IMPORT and on the DEFAULT visible device, so a
    later ``Device(d).use()`` cannot move them.  SphericalHarmonics_gpu runs
    ``junk_to_check_installed = cupy.array(5)``; on an unusable default slot that raises, it
    sets ``cupy_here = False`` and keys ``_coeffs`` by numpy alone, and a caller that then
    passes ``xpy=cupy`` into the likelihood dies on ``KeyError: <module 'cupy'>``.
    """
    import numpy
    complaints = []
    from RIFT.likelihood import factored_likelihood
    if factored_likelihood.xpy_default is numpy:
        complaints.append("factored_likelihood.xpy_default is numpy")
    from RIFT.likelihood import SphericalHarmonics_gpu
    if not getattr(SphericalHarmonics_gpu, "cupy_here", False):
        complaints.append("SphericalHarmonics_gpu.cupy_here is False")
    return complaints


def cupy_or_skip(require_rift_backend=False):
    """The imported cupy module, with a slot that can build a kernel already selected.

    Skips (or fails under RIFT_CI_REQUIRE_GPU=1) instead of raising, on both of the failures
    above.  Every skip reason names cupy/GPU/CUDA, which is what the reason-matching guards in
    .travis/test-integrate.sh accept when RIFT_CI_REQUIRE_GPU is unset.
    (.travis/test-q-window-stencil.sh does not match reasons; see no_gpu.)

    ``require_rift_backend`` additionally demands that RIFT's own import-time probes took the
    device.  OPT-IN, because the two answers differ and a blanket check costs a lane that
    works.  Measured on ldas-pcdev2 2026-09-18 with CUDA_VISIBLE_DEVICES="1,0" and "1,2",
    where slot 1 is a Blackwell cc 12.0 this cupy cannot compile for and the second entry is
    usable: the probe returns SLOT 1 and .use() does put arrays on it, so
    test_peak_local_runs_on_the_gpu_backend_and_matches_numpy PASSES -- it takes xpy as an
    argument and never touches SphericalHarmonics_gpu.  test_gpu_offset_is_per_row_too goes
    through the likelihood and FAILS on KeyError.  Pass True from callers of the second kind.
    """
    line = probe_verdict()
    if not line.startswith("SLOT "):
        no_gpu("no usable GPU for this test -- %s." % line)
    d = int(line.split()[1])
    try:
        import cupy
        # .use(), not CUDA_VISIBLE_DEVICES: the caller's module-level `from RIFT... import`
        # may already have imported cupy through a `try: import cupy` in RIFT, after which
        # rewriting the environment decides nothing.  This sets the CURRENT device, which is
        # what the callers' bare cupy.asarray()/kernel calls use.
        cupy.cuda.Device(d).use()
        cupy.asnumpy(cupy.zeros(1) + 1)
    except Exception as e:
        # REACHABLE, and not only through a probe/parent disagreement: the CIT pcdevs are
        # shared, so a slot the child used can be busy by the time this process asks
        # (cudaErrorDevicesUnavailable).  Deleting the .use() above also lands here, which is
        # how it was shown to be load-bearing rather than decorative.
        no_gpu("cupy slot %d passed the subprocess probe but is unusable here -- %s."
               % (d, _flat(e)))
    if require_rift_backend:
        complaints = _rift_backend_complaints(cupy)
        if complaints:
            no_gpu("cupy can build a kernel on CUDA slot %d, but RIFT fell back to numpy when "
                   "it was imported (%s), and Device(%d).use() cannot undo that.  Pin "
                   "CUDA_VISIBLE_DEVICES so the FIRST visible slot is the usable one."
                   % (d, "; ".join(complaints), d))
    return cupy
