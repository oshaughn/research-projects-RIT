"""
test_selfterm_reduction.py -- unit test for the fused-calmarg self-term FIX
reduction (analyses/calmarg_selfterm_bias/NOTE.md).

Unlike backtest_calmarg.py (which checks the cal REDUCTION with a single, shared,
cal-independent rho_sq), this exercises the PER-REALIZATION self-term path: each
calibration realization c carries its OWN cross terms U_c, V_c (=> its own
rho_sq_c = <C_c h|C_c h>), supplied via ctUArrayDict_cal/ctVArrayDict_cal.

Ground truth (brute force): run the UNCHANGED n_cal==1 likelihood on realization
block c using that realization's OWN (U_c, V_c) as the cross terms, then combine
logsumexp_c(lnL_c) - log(n_cal).  This is exactly what "per-realization rho_sq_c"
must reproduce.  We then check that:
  * loop  reduction (cal_method='loop',  n_cal>1, +cal cross terms) == brute force
  * fused reduction (cal_method='fused', n_cal>1, +cal cross terms) == brute force
on CPU (numpy fused) and, if available, GPU (CUDA fused).  Distance-marginalization
and phase-marginalization variants are covered too.

Run:
  export PYTHONPATH=<checkout>/MonteCarloMarginalizeCode/Code
  python3 -m RIFT.calmarg.test_selfterm_reduction --backend cpu
  python3 -m RIFT.calmarg.test_selfterm_reduction --backend gpu
"""
from __future__ import print_function
import argparse
import numpy as np
import lal
from scipy.special import logsumexp

import RIFT.likelihood.factored_likelihood as fl
import RIFT.calmarg.backtest_calmarg as bt


def _make_cal_crossterms(case, rng):
    """Per-realization Hermitian U_c and symmetric V_c (n_cal, n_lms, n_lms), one set
    per detector.  Positive-definite U_c so rho_sq_c>0 (needed by the distmarg
    transforms); arbitrary but self-consistent, exactly like backtest's psd_UV path."""
    U_cal = {}
    V_cal = {}
    n_lms = case["n_lms"]; n_cal = case["n_cal"]
    for det in case["dets"]:
        Uc = np.zeros((n_cal, n_lms, n_lms), dtype=complex)
        Vc = np.zeros((n_cal, n_lms, n_lms), dtype=complex)
        for c in range(n_cal):
            # positive-definite Hermitian U_c = M M^H + n_lms I  (well-conditioned)
            M = rng.standard_normal((n_lms, n_lms)) + 1j*rng.standard_normal((n_lms, n_lms))
            Uc[c] = M @ M.conj().T + n_lms*np.eye(n_lms)
            Vc[c] = 0.0   # V=0 keeps rho_sq_c manifestly real/positive (as in physical <h|h>)
        U_cal[det] = Uc
        V_cal[det] = Vc
    return U_cal, V_cal


def _to_backend_dicts(d, xpy):
    return {k: xpy.asarray(v) for k, v in d.items()}


def _brute_force(case, xpy, U_cal, V_cal, phase_marginalization, loglikelihood):
    """Per-realization n_cal==1 reference using each realization's OWN (U_c, V_c)."""
    P = bt._build_P(case, xpy)
    tvals = xpy.asarray(case["tvals"])
    n_cal = case["n_cal"]
    lnL_blocks = np.zeros((n_cal, case["npts_extrinsic"]))
    for c in range(n_cal):
        lookupNKDict, rholmsArrayDict, _ctU, _ctV, epochDict = bt._dicts(
            case, xpy, bt._block_rholms(case, c))
        # use realization c's OWN cross terms as the (single) U,V
        ctU_c = {det: xpy.asarray(U_cal[det][c]) for det in case["dets"]}
        ctV_c = {det: xpy.asarray(V_cal[det][c]) for det in case["dets"]}
        out = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            tvals, P, lookupNKDict, rholmsArrayDict, ctU_c, ctV_c, epochDict,
            Lmax=2, xpy=xpy, n_cal=1, loglikelihood=loglikelihood,
            phase_marginalization=phase_marginalization)
        lnL_blocks[c] = bt._to_host(out)
    return logsumexp(lnL_blocks, axis=0) - np.log(n_cal)


def _cal_method(case, xpy, U_cal, V_cal, method, phase_marginalization, loglikelihood, cal_distmarg=None):
    P = bt._build_P(case, xpy)
    lookupNKDict, rholmsArrayDict, ctU, ctV, epochDict = bt._dicts(case, xpy, case["rholms"])
    ctU_cal = _to_backend_dicts(U_cal, xpy)
    ctV_cal = _to_backend_dicts(V_cal, xpy)
    out = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        xpy.asarray(case["tvals"]), P, lookupNKDict, rholmsArrayDict, ctU, ctV, epochDict,
        Lmax=2, xpy=xpy, n_cal=case["n_cal"], cal_method=method,
        loglikelihood=loglikelihood, phase_marginalization=phase_marginalization,
        cal_distmarg=cal_distmarg,
        ctUArrayDict_cal=ctU_cal, ctVArrayDict_cal=ctV_cal)
    return bt._to_host(out)


def run(backend="cpu", n_cal=12, npts_extrinsic=48, seed=20240611):
    xpy = bt._backend(backend)
    rng = np.random.default_rng(seed)
    ok_all = True

    # ---- (A) default helper, no phase marg ----
    case = bt.make_synthetic_case(n_cal=n_cal, npts_extrinsic=npts_extrinsic, seed=seed)
    U_cal, V_cal = _make_cal_crossterms(case, rng)
    ref = _brute_force(case, xpy, U_cal, V_cal, False, fl._factored_lnL_helper)
    loop = _cal_method(case, xpy, U_cal, V_cal, 'loop', False, fl._factored_lnL_helper)
    fused = _cal_method(case, xpy, U_cal, V_cal, 'fused', False, fl._factored_lnL_helper)
    for name, v, tol in [("loop", loop, 1e-9), ("fused", fused, 1e-9)]:
        err = float(np.max(np.abs(v - ref)))
        flag = "OK" if err < tol else "**DIFF**"
        if err >= tol: ok_all = False
        print("  [default   ] %-6s vs brute: %.3e  %s" % (name, err, flag))

    # ---- (B) phase marginalization ----
    case = bt.make_synthetic_case(n_cal=n_cal, npts_extrinsic=npts_extrinsic, seed=seed+1)
    U_cal, V_cal = _make_cal_crossterms(case, rng)
    ref = _brute_force(case, xpy, U_cal, V_cal, True, fl._factored_lnL_helper)
    loop = _cal_method(case, xpy, U_cal, V_cal, 'loop', True, fl._factored_lnL_helper)
    fused = _cal_method(case, xpy, U_cal, V_cal, 'fused', True, fl._factored_lnL_helper)
    for name, v, tol in [("loop", loop, 1e-9), ("fused", fused, 1e-9)]:
        err = float(np.max(np.abs(v - ref)))
        flag = "OK" if err < tol else "**DIFF**"
        if err >= tol: ok_all = False
        print("  [phase-marg] %-6s vs brute: %.3e  %s" % (name, err, flag))

    # ---- (C) distance marginalization ----
    case = bt.make_synthetic_case(n_cal=n_cal, npts_extrinsic=npts_extrinsic, seed=seed+2, psd_UV=True)
    case["dist"] = np.full(case["npts_extrinsic"], fl.distMpcRef) * (lal.PC_SI*1e6)
    U_cal, V_cal = _make_cal_crossterms(case, rng)
    params = bt.make_distmarg_table(xpy)
    dm_loglike = bt.make_distmarg_loglikelihood(params, xpy)
    ref = _brute_force(case, xpy, U_cal, V_cal, False, dm_loglike)
    loop = _cal_method(case, xpy, U_cal, V_cal, 'loop', False, dm_loglike)
    fused = _cal_method(case, xpy, U_cal, V_cal, 'fused', False, dm_loglike, cal_distmarg=params)
    for name, v, tol in [("loop", loop, 1e-6), ("fused", fused, 1e-6)]:
        err = float(np.max(np.abs(v - ref)))
        flag = "OK" if err < tol else "**DIFF**"
        if err >= tol: ok_all = False
        print("  [distmarg  ] %-6s vs brute: %.3e  %s" % (name, err, flag))

    print("# RESULT:", "PASS" if ok_all else "MISMATCH")
    return ok_all


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--n-cal", type=int, default=12)
    p.add_argument("--npts-extrinsic", type=int, default=48)
    p.add_argument("--seed", type=int, default=20240611)
    a = p.parse_args()
    print("# self-term reduction test  backend=%s  n_cal=%d" % (a.backend, a.n_cal))
    ok = run(a.backend, a.n_cal, a.npts_extrinsic, a.seed)
    raise SystemExit(0 if ok else 1)
