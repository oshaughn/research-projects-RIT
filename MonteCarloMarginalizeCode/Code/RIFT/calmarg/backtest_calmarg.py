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
                        deltaT=1.0/4096, dets=("H1", "L1"), seed=1234,
                        psd_UV=False):
    """Build a controlled set of likelihood inputs with embedded cal-block
    structure.  The n_cal realization blocks are independent random rholm draws
    (their physical relationship is irrelevant for backtesting that the *reduction*
    over blocks is computed correctly -- the loglikelihood callback is applied
    identically across methods, so method agreement holds regardless).

    Multiple detectors exercise the kernel's detector loop and the function's
    per-detector stacking; each detector gets its own random rholms/U/V, and the
    likelihood derives a distinct per-detector ifirst from the (real) detector
    location, so the stacked-ifirst path is genuinely tested.

    N_window must exceed the sky time-delay spread (+-0.021 s) plus npts so the
    per-sample window stays inside each block.

    Returns a dict ('case') of plain numpy arrays / scalars (rholms/U/V are dicts
    keyed by detector); convert to a backend in the method functions.
    """
    rng = np.random.default_rng(seed)
    n_lms = 2
    npts_full = N_window * n_cal
    dets = tuple(dets)

    case = dict(
        dets=dets, n_cal=n_cal, n_lms=n_lms, N_window=N_window, npts=npts,
        deltaT=deltaT, npts_extrinsic=npts_extrinsic,
        lookupNK=np.array([[2, 2], [2, -2]], dtype=int),
        tref=1000000000.0,
    )
    case["rholms"] = {}
    case["U"] = {}
    case["V"] = {}
    for det in dets:
        case["rholms"][det] = (rng.standard_normal((n_lms, npts_full))
                               + 1j*rng.standard_normal((n_lms, npts_full)))
        if psd_UV:
            # positive-definite U, V=0  -> rho_sq>0, required by the distmarg
            # transforms (asinh(rho_sq/bref), x0=kappa/rho_sq); mirrors physical <h|h>
            case["U"][det] = np.eye(n_lms, dtype=complex)
            case["V"][det] = np.zeros((n_lms, n_lms), dtype=complex)
        else:
            U = rng.standard_normal((n_lms, n_lms)) + 1j*rng.standard_normal((n_lms, n_lms))
            V = rng.standard_normal((n_lms, n_lms)) + 1j*rng.standard_normal((n_lms, n_lms))
            case["U"][det] = U + U.conj().T
            case["V"][det] = V + V.conj().T
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


def _dicts(case, xpy, rholms):
    """Build the per-detector dicts the likelihood expects from a rholms map."""
    dets = case["dets"]
    lookupNKDict = {d: case["lookupNK"] for d in dets}
    rholmsArrayDict = {d: xpy.asarray(rholms[d]) for d in dets}
    ctU = {d: xpy.asarray(case["U"][d]) for d in dets}
    ctV = {d: xpy.asarray(case["V"][d]) for d in dets}
    epochDict = {d: case["epoch"] for d in dets}
    return lookupNKDict, rholmsArrayDict, ctU, ctV, epochDict


def _block_rholms(case, c):
    """Per-detector rholms restricted to realization block c."""
    N = case["N_window"]
    return {d: case["rholms"][d][:, c*N:(c+1)*N] for d in case["dets"]}


# ---------------------------------------------------------------------------
# Distance-marginalization table + loglikelihood (mirror of the ILE driver, so the
# fused distmarg kernel can be validated against reference/Option B using the SAME
# table and transforms)
# ---------------------------------------------------------------------------
def _bilinear(s0, ds, t0, dt, fgrid, xpy):
    """Mirror of EvenBivariateLinearInterpolator in the ILE driver."""
    dx_inv, dy_inv = 1.0/ds, 1.0/dt

    def call(x, y):
        i_mid = dx_inv * (x - s0)
        j_mid = dy_inv * (y - t0)
        i_lo = xpy.floor(i_mid).astype(int); i_hi = xpy.ceil(i_mid).astype(int)
        j_lo = xpy.floor(j_mid).astype(int); j_hi = xpy.ceil(j_mid).astype(int)
        p = i_mid - i_lo; q = j_mid - j_lo
        p_ = 1 - p; q_ = 1 - q
        f = p_*q_ * fgrid[i_lo, j_lo]
        f += p*q_ * fgrid[i_hi, j_lo]
        f += p_*q * fgrid[i_lo, j_hi]
        f += p*q * fgrid[i_hi, j_hi]
        return f
    return call


def make_distmarg_table(xpy, ns=64, nt=48, xmin=-1.0e4, xmax=1.0e4,
                        sqrt_bmax=1.0, bref=1.0, tmax=10.0, seed=7):
    """Build a synthetic-but-self-consistent distance-marginalization table.

    s_array spans x0_to_s(xmin)..x0_to_s(xmax), so any x0 in (xmin,xmax) maps to an
    in-bounds s; wide (xmin,xmax) keeps realized x0=kappa/rho_sq in range.  lnI_array
    is an arbitrary smooth surface -- physical values are irrelevant for backtesting
    that the kernel reproduces the same transform the Python closure applies.
    """
    def x0_to_s(x0):
        return (np.arcsinh(sqrt_bmax*(x0 - xmin))
                - np.arcsinh(sqrt_bmax*(xmax - x0)))
    smin = float(x0_to_s(xmin))
    smax = float(x0_to_s(xmax))
    s_array = np.linspace(smin, smax, ns)
    t_array = np.linspace(0.0, tmax, nt)
    SS, TT = np.meshgrid(s_array, t_array, indexing='ij')
    lnI_array = -0.3*SS**2 + np.cos(TT) - 0.05*TT   # smooth, arbitrary

    return dict(
        lnI_array=xpy.asarray(lnI_array),
        s0=float(s_array[0]), ds=float(s_array[1]-s_array[0]),
        smin=float(s_array[0]), smax=float(s_array[-1]),
        t0=float(t_array[0]), dt=float(t_array[1]-t_array[0]),
        tmax=float(t_array[-1]),
        xmin=float(xmin), xmax=float(xmax),
        sqrt_bmax=float(sqrt_bmax), bref=float(bref),
    )


def make_distmarg_loglikelihood(params, xpy):
    """Python distmarg loglikelihood closure (mirror of the ILE driver), consuming
    the same table the fused kernel uses."""
    xmin, xmax = params["xmin"], params["xmax"]
    sqrt_bmax, bref = params["sqrt_bmax"], params["bref"]
    smin, smax, tmax = params["smin"], params["smax"], params["tmax"]
    intp = _bilinear(params["s0"], params["ds"], params["t0"], params["dt"],
                     params["lnI_array"], xpy)

    def loglikelihood(kappa_sq, rho_sq):
        x0 = kappa_sq / rho_sq
        s = (xpy.arcsinh(sqrt_bmax*(x0 - xmin))
             - xpy.arcsinh(sqrt_bmax*(xmax - x0)))
        t = xpy.arcsinh(rho_sq / bref)
        lnI = xpy.full_like(x0, -xpy.inf)
        in_bounds = (s > smin) & (s < smax) & (t < tmax)
        lnI[in_bounds] = intp(s[in_bounds], t[in_bounds])
        x0c = xpy.clip(x0, xmin, xmax)
        return rho_sq * x0c * (x0 - 0.5*x0c) + lnI
    return loglikelihood


# ---------------------------------------------------------------------------
# Method implementations (the registry being backtested)
# ---------------------------------------------------------------------------
def method_reference(case, xpy, phase_marginalization=False, loglikelihood=None):
    """Brute force: per-block n_cal==1 evaluation, combined by hand."""
    if loglikelihood is None:
        loglikelihood = fl._factored_lnL_helper
    P = _build_P(case, xpy)
    tvals = xpy.asarray(case["tvals"])
    n_cal = case["n_cal"]
    lnL_blocks = np.zeros((n_cal, case["npts_extrinsic"]))
    for c in range(n_cal):
        lookupNKDict, rholmsArrayDict, ctU, ctV, epochDict = _dicts(
            case, xpy, _block_rholms(case, c))
        out = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            tvals, P, lookupNKDict, rholmsArrayDict, ctU, ctV,
            epochDict, Lmax=2, xpy=xpy, n_cal=1,
            loglikelihood=loglikelihood, phase_marginalization=phase_marginalization)
        lnL_blocks[c] = _to_host(out)
    return logsumexp(lnL_blocks, axis=0) - np.log(n_cal)


def method_in_loop_B(case, xpy, phase_marginalization=False, loglikelihood=None):
    """Option B: single call with n_cal>1 (cal_method='loop')."""
    if loglikelihood is None:
        loglikelihood = fl._factored_lnL_helper
    P = _build_P(case, xpy)
    lookupNKDict, rholmsArrayDict, ctU, ctV, epochDict = _dicts(
        case, xpy, case["rholms"])
    out = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        xpy.asarray(case["tvals"]), P, lookupNKDict, rholmsArrayDict, ctU, ctV,
        epochDict, Lmax=2, xpy=xpy, n_cal=case["n_cal"], cal_method='loop',
        loglikelihood=loglikelihood, phase_marginalization=phase_marginalization)
    return _to_host(out)


def method_in_loop_C(case, xpy, phase_marginalization=False, loglikelihood=None):
    """Option C: fused CUDA kernel (Q + default helper + cal log-sum-exp on-board).

    GPU-only and (for now) default helper / no phase marginalization; raises
    NotImplementedError otherwise, so the harness SKIPs it on CPU.
    """
    if loglikelihood is None:
        loglikelihood = fl._factored_lnL_helper
    P = _build_P(case, xpy)
    lookupNKDict, rholmsArrayDict, ctU, ctV, epochDict = _dicts(
        case, xpy, case["rholms"])
    out = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        xpy.asarray(case["tvals"]), P, lookupNKDict, rholmsArrayDict, ctU, ctV,
        epochDict, Lmax=2, xpy=xpy, n_cal=case["n_cal"], cal_method='fused',
        cal_distmarg=case.get("cal_distmarg"),
        loglikelihood=loglikelihood, phase_marginalization=phase_marginalization)
    return _to_host(out)


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
                 loglikelihood_mode="default", **case_kwargs):
    """Evaluate each method, time it, and report agreement vs 'reference' (if run)
    and vs 'in_loop_B'.

    loglikelihood_mode:
      'default'  -- the distance-unmarginalized helper.
      'distmarg' -- the distance-marginalization loglikelihood (uses positive-definite
                    U so rho_sq>0, builds a self-consistent table; reference/Option B
                    use the Python closure, Option C uses the fused distmarg kernel).
    """
    xpy = _backend(backend)
    loglikelihood = None
    if loglikelihood_mode == "distmarg":
        case_kwargs["psd_UV"] = True
        case = make_synthetic_case(**case_kwargs)
        case["dist"] = np.full(case["npts_extrinsic"], fl.distMpcRef) * (lal.PC_SI*1e6)
        params = make_distmarg_table(xpy)
        case["cal_distmarg"] = params           # consumed by the fused distmarg kernel
        loglikelihood = make_distmarg_loglikelihood(params, xpy)
    else:
        case = make_synthetic_case(**case_kwargs)
    # distmarg's asinh/bilinear differ at ULP level between numpy and the kernel, so
    # the fused-vs-loop agreement is float-level rather than bit-level.
    tol = 1e-9 if loglikelihood_mode == "default" else 1e-6
    print("# calmarg backtest  backend=%s  dets=%s  n_cal=%d  npts_extrinsic=%d  N_window=%d  npts=%d  phase_marg=%s  loglike=%s"
          % (backend, ",".join(case["dets"]), case["n_cal"], case["npts_extrinsic"],
             case["N_window"], case["npts"], phase_marginalization, loglikelihood_mode))

    results = {}
    timings = {}
    for name in methods:
        fn = METHODS[name]
        try:
            out = fn(case, xpy, phase_marginalization=phase_marginalization,
                     loglikelihood=loglikelihood)  # warm-up / compile
            _sync(xpy)
            best = float("inf")
            for _ in range(repeat):
                t0 = time.perf_counter()
                out = fn(case, xpy, phase_marginalization=phase_marginalization,
                         loglikelihood=loglikelihood)
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
        print("# max |lnL - %s|  (tol %.0e):" % (baseline, tol))
        ok = True
        for name, vals in results.items():
            if name == baseline:
                continue
            err = float(np.max(np.abs(vals - results[baseline])))
            flag = "OK" if err < tol else "**DIFF**"
            if err >= tol:
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
    p.add_argument("--dets", default="H1,L1", help="comma-separated detector prefixes")
    p.add_argument("--npts-extrinsic", type=int, default=64)
    p.add_argument("--N-window", type=int, default=256)
    p.add_argument("--npts", type=int, default=16)
    p.add_argument("--repeat", type=int, default=3, help="timing repetitions (best-of)")
    p.add_argument("--loglikelihood", default="default", choices=["default", "distmarg"],
                   help="default helper, or distance-marginalization loglikelihood")
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
    dets = tuple(d.strip() for d in args.dets.split(",") if d.strip())
    ok = run_backtest(
        methods, backend=args.backend, repeat=args.repeat,
        phase_marginalization=args.phase_marginalization,
        loglikelihood_mode=args.loglikelihood,
        n_cal=args.n_cal, npts_extrinsic=args.npts_extrinsic,
        N_window=args.N_window, npts=args.npts, seed=args.seed, dets=dets)
    raise SystemExit(0 if ok else 1)
