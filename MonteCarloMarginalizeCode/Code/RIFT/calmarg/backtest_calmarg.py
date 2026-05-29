"""
backtest_calmarg.py  --  backtest harness for in-loop calibration marginalization

PURPOSE
-------
Compare different implementations of the calibration-marginalized factored
likelihood against each other and against a brute-force reference, on controlled
inputs that exercise the per-realization block structure
(rholmsArrayDict[det] holding n_cal contiguous length-N_window blocks, selected by
ifirst -> ifirst + c*N_window).

This is the rig Option C (a fused CUDA kernel) is developed against: register the
new implementation in METHODS and the harness reports lnL agreement (vs the
brute-force reference and vs Option B) and timing, on both CPU and GPU backends.

It is deliberately self-contained (synthetic inputs, no frames/PSDs/cache needed),
so it runs anywhere RIFT + lal import.  See run_physics_backtest() below for the
heavier real-data comparison vs bilby's calibration_reweighting.py.

METHODS (the registry being backtested)
----------------------------------------
  reference : brute force -- run the unchanged n_cal==1 likelihood on each
              realization block separately, combine logsumexp_c(lnL_c) - log(n_cal).
              This is the ground truth the others must reproduce.
  in_loop_B : DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(..., n_cal=n_cal)
              -- Option B (cal loop reusing the existing Q kernel, streaming LSE).
  in_loop_C : Option C fused kernel -- STUB, raises NotImplementedError. Wire the
              new implementation here when it exists; the harness will validate it.

USAGE
-----
  python -m RIFT.calmarg.backtest_calmarg --backend cpu --n-cal 20
  python -m RIFT.calmarg.backtest_calmarg --backend gpu --n-cal 100 --npts-extrinsic 4096 --repeat 5
  python -m RIFT.calmarg.backtest_calmarg --backend gpu --methods reference,in_loop_B,in_loop_C
"""
from __future__ import print_function

import argparse
import time

import numpy as np
import lal
from scipy.special import logsumexp

import RIFT.likelihood.factored_likelihood as fl


# ---------------------------------------------------------------------------
# Synthetic case construction
# ---------------------------------------------------------------------------
def make_synthetic_case(n_cal=20, npts_extrinsic=64, N_window=256, npts=16,
                        deltaT=1.0/4096, det="H1", seed=1234):
    """Build a controlled set of likelihood inputs with embedded cal-block
    structure.  The n_cal realization blocks are independent random rholm draws
    (their physical relationship is irrelevant for backtesting that the *reduction*
    over blocks is computed correctly -- the loglikelihood callback is applied
    identically across methods, so method agreement holds regardless).

    N_window must exceed the sky time-delay spread (+-0.021 s) plus npts so the
    per-sample window stays inside each block.

    Returns a dict ('case') of plain numpy arrays / scalars; convert to a backend
    with to_backend().
    """
    rng = np.random.default_rng(seed)
    n_lms = 2
    npts_full = N_window * n_cal

    case = dict(
        det=det, n_cal=n_cal, n_lms=n_lms, N_window=N_window, npts=npts,
        deltaT=deltaT, npts_extrinsic=npts_extrinsic,
        lookupNK=np.array([[2, 2], [2, -2]], dtype=int),
        rholms=(rng.standard_normal((n_lms, npts_full))
                + 1j*rng.standard_normal((n_lms, npts_full))),
        tref=1000000000.0,
    )
    U = rng.standard_normal((n_lms, n_lms)) + 1j*rng.standard_normal((n_lms, n_lms))
    V = rng.standard_normal((n_lms, n_lms)) + 1j*rng.standard_normal((n_lms, n_lms))
    case["U"] = U + U.conj().T
    case["V"] = V + V.conj().T
    # epoch placed so the integration window sits near the middle of each block
    case["epoch"] = case["tref"] - 0.03
    case["tvals"] = np.linspace(-(npts//2)*deltaT, (npts//2)*deltaT, npts)

    # extrinsic parameter arrays
    case["phi"] = rng.uniform(0, 2*np.pi, npts_extrinsic)
    case["theta"] = rng.uniform(0.2, np.pi-0.2, npts_extrinsic)
    case["psi"] = rng.uniform(0, np.pi, npts_extrinsic)
    case["incl"] = rng.uniform(0.2, np.pi-0.2, npts_extrinsic)
    case["phiref"] = rng.uniform(0, 2*np.pi, npts_extrinsic)
    case["dist"] = np.full(npts_extrinsic, 500.0) * (lal.PC_SI*1e6)  # 500 Mpc
    return case


class _PVec(object):
    """Minimal stand-in for the vectorized ChooseWaveformParams object that the
    likelihood reads (phi, theta, psi, incl, phiref, dist arrays; tref, deltaT
    scalars)."""
    pass


def _build_P(case, xpy):
    P = _PVec()
    for name in ("phi", "theta", "psi", "incl", "phiref", "dist"):
        setattr(P, name, xpy.asarray(case[name]))
    P.tref = case["tref"]
    P.deltaT = case["deltaT"]
    return P


def _backend(name):
    if name == "cpu":
        return np
    if name == "gpu":
        import cupy as cp
        return cp
    raise ValueError("backend must be 'cpu' or 'gpu', got %r" % name)


def _to_host(x):
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except ImportError:
        pass
    return np.asarray(x)


def _block_rholms(case, c, xpy):
    """rholms restricted to realization block c (as a single-block array)."""
    N = case["N_window"]
    return xpy.asarray(case["rholms"][:, c*N:(c+1)*N])


# ---------------------------------------------------------------------------
# Method implementations (the registry being backtested)
# ---------------------------------------------------------------------------
def method_reference(case, xpy, phase_marginalization=False, loglikelihood=None):
    """Brute force: per-block n_cal==1 evaluation, combined by hand."""
    if loglikelihood is None:
        loglikelihood = fl._factored_lnL_helper
    P = _build_P(case, xpy)
    det = case["det"]
    tvals = xpy.asarray(case["tvals"])
    lookupNKDict = {det: case["lookupNK"]}
    ctU = {det: xpy.asarray(case["U"])}
    ctV = {det: xpy.asarray(case["V"])}
    epochDict = {det: case["epoch"]}
    n_cal = case["n_cal"]
    lnL_blocks = np.zeros((n_cal, case["npts_extrinsic"]))
    for c in range(n_cal):
        out = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            tvals, P, lookupNKDict, {det: _block_rholms(case, c, xpy)}, ctU, ctV,
            epochDict, Lmax=2, xpy=xpy, n_cal=1,
            loglikelihood=loglikelihood, phase_marginalization=phase_marginalization)
        lnL_blocks[c] = _to_host(out)
    return logsumexp(lnL_blocks, axis=0) - np.log(n_cal)


def method_in_loop_B(case, xpy, phase_marginalization=False, loglikelihood=None):
    """Option B: single call with n_cal>1."""
    if loglikelihood is None:
        loglikelihood = fl._factored_lnL_helper
    P = _build_P(case, xpy)
    det = case["det"]
    out = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        xpy.asarray(case["tvals"]), P, {det: case["lookupNK"]},
        {det: xpy.asarray(case["rholms"])}, {det: xpy.asarray(case["U"])},
        {det: xpy.asarray(case["V"])}, {det: case["epoch"]},
        Lmax=2, xpy=xpy, n_cal=case["n_cal"],
        loglikelihood=loglikelihood, phase_marginalization=phase_marginalization)
    return _to_host(out)


def method_in_loop_C(case, xpy, phase_marginalization=False, loglikelihood=None):
    """Option C: fused CUDA kernel (Q + loglikelihood + cal log-sum-exp on-board).

    STUB. When the fused kernel exists, call it here and return lnL of shape
    (npts_extrinsic,) on the host.  The harness will then validate it against
    'reference' and 'in_loop_B' automatically.
    """
    raise NotImplementedError(
        "Option C (fused kernel) is not implemented yet -- wire it into "
        "method_in_loop_C in RIFT/calmarg/backtest_calmarg.py")


METHODS = {
    "reference": method_reference,
    "in_loop_B": method_in_loop_B,
    "in_loop_C": method_in_loop_C,
}


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------
def _sync(xpy):
    if xpy is not np:
        xpy.cuda.Stream.null.synchronize()


def run_backtest(methods, backend="cpu", repeat=3, phase_marginalization=False,
                 **case_kwargs):
    """Evaluate each method, time it, and report agreement vs 'reference' (if run)
    and vs 'in_loop_B'."""
    xpy = _backend(backend)
    case = make_synthetic_case(**case_kwargs)
    print("# calmarg backtest  backend=%s  n_cal=%d  npts_extrinsic=%d  N_window=%d  npts=%d  phase_marg=%s"
          % (backend, case["n_cal"], case["npts_extrinsic"], case["N_window"],
             case["npts"], phase_marginalization))

    results = {}
    timings = {}
    for name in methods:
        fn = METHODS[name]
        try:
            out = fn(case, xpy, phase_marginalization=phase_marginalization)  # warm-up / compile
            _sync(xpy)
            best = float("inf")
            for _ in range(repeat):
                t0 = time.perf_counter()
                out = fn(case, xpy, phase_marginalization=phase_marginalization)
                _sync(xpy)
                best = min(best, time.perf_counter() - t0)
            results[name] = np.asarray(out)
            timings[name] = best
            print("  %-12s ok    best %8.2f ms" % (name, best*1e3))
        except NotImplementedError as e:
            print("  %-12s SKIP  (%s)" % (name, e))
        except Exception as e:
            print("  %-12s FAIL  %s: %s" % (name, type(e).__name__, e))

    # agreement
    baseline = "reference" if "reference" in results else (
        "in_loop_B" if "in_loop_B" in results else None)
    if baseline:
        print("# max |lnL - %s| :" % baseline)
        ok = True
        for name, vals in results.items():
            if name == baseline:
                continue
            err = float(np.max(np.abs(vals - results[baseline])))
            flag = "OK" if err < 1e-9 else "**DIFF**"
            if err >= 1e-9:
                ok = False
            print("    %-12s %.3e   %s" % (name, err, flag))
        print("# RESULT:", "PASS" if ok else "MISMATCH")
        return ok
    return True


# ---------------------------------------------------------------------------
# Physics backtest vs bilby calibration_reweighting.py  (scaffold -- needs data)
# ---------------------------------------------------------------------------
def run_physics_backtest(precompute_or_config=None, cal_envelope_dir=None,
                         bilby_data_dump=None, **kwargs):
    """Compare in-loop calibration marginalization to the bilby postprocessor on a
    REAL event.  This needs frames/PSDs/cache (or a saved ILE precompute) plus the
    bilby data_dump used by calibration_reweighting.py, so it does NOT run in the
    self-contained harness above.

    Intended flow (TODO, to run on the stable host):
      1. Build data_dict / psd_dict (real or injected) the same way ILE does, OR
         load a saved precompute.
      2. cal = RIFT.calmarg.generate_realizations.create_realizations(env, ...) for
         each detector from cal_envelope_dir.
      3. PrecomputeLikelihoodTerms(..., calibration_realizations=cal) -> cal-extended
         rholms; pack with PackLikelihoodDataStructuresAsArrays.
      4. Evaluate the in-loop calmarg likelihood over the SAME extrinsic samples the
         bilby reweighter used (read its posterior + weights), compare per-sample
         lnL and the integrated log-evidence shift.
      5. Compare to bilby calibration_likelihood from calibration_reweighting.py.
         Expect agreement to first order in cal amplitude; the apply-to-data
         (RIFT) vs apply-to-template (bilby) convention differs at second order --
         quantify and record that difference here.
    """
    raise NotImplementedError(
        "Physics backtest needs real data/precompute + a bilby data_dump; "
        "see docstring for the intended flow. Run on the stable host post-update.")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--methods", default="reference,in_loop_B,in_loop_C",
                   help="comma-separated subset of: %s" % ",".join(METHODS))
    p.add_argument("--n-cal", type=int, default=20)
    p.add_argument("--npts-extrinsic", type=int, default=64)
    p.add_argument("--N-window", type=int, default=256)
    p.add_argument("--npts", type=int, default=16)
    p.add_argument("--repeat", type=int, default=3, help="timing repetitions (best-of)")
    p.add_argument("--phase-marginalization", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit("unknown methods: %s (known: %s)"
                         % (unknown, list(METHODS)))
    ok = run_backtest(
        methods, backend=args.backend, repeat=args.repeat,
        phase_marginalization=args.phase_marginalization,
        n_cal=args.n_cal, npts_extrinsic=args.npts_extrinsic,
        N_window=args.N_window, npts=args.npts, seed=args.seed)
    raise SystemExit(0 if ok else 1)
