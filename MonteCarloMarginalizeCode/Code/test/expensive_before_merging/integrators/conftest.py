"""Make the pytest entry point of this suite agree with run_shape_recovery.sh.

The shell driver exports four things before it runs anything; pytest exported none of them, so
`RIFT_RUN_EXPENSIVE=1 pytest test_shape_recovery.py` -- which the suite documents as an equivalent
way in -- was a materially different experiment from the gate it claims to be.  Two of those
exports are environment (here), one decides which RIFT is imported at all, and that one is NOT set
here on purpose:

  CUDA_VISIBLE_DEVICES=""   set here.  The suite documents itself as CPU-only and deterministic,
                            and the CPU path is also the configuration being exercised on purpose
                            (cupy installed, no device -- the worker layout that has repeatedly
                            bitten production).  Left unset, a pytest run on a GPU box measures a
                            different code path than the gate does.

  OMP/MKL/OPENBLAS threads  set here, to the shell driver's default of 4, and only if the caller
                            has not chosen.  Best effort: this binds only if no BLAS has been
                            loaded yet, which under pytest means before the first numpy import.

  PYTHONPATH                deliberately NOT set.  Prepending the checkout would make the numbers
                            right and leave the operator's invocation wrong, so the next thing
                            they run by hand measures the installed RIFT again.  Instead
                            test_shape_recovery.py hard-fails with the export to run.  See
                            shape_recovery.assert_rift_under_test() and FOLLOWUPS item 6.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
for _var in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ["OMP_NUM_THREADS"])
