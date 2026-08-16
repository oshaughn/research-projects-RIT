"""
test_noloop_gpu_stencils : GPU-vs-CPU parity for the BASELINE NoLoop likelihood, over all
three Q_lm sub-sample time stencils and BOTH of its GPU dispatch sites.

factored_likelihood.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop is the baseline
consumer of the Q-window machinery, and it reaches the device through
factored_likelihood._q_inner_product_gpu at two structurally different places:

  (1) the n_cal == 1 path (no calibration marginalization), which calls the kernel once
      per detector on the full Q buffer;
  (2) the n_cal > 1 calibration-marginalization 'loop' path, which caches
      (Q, FY_conj, ifirst, N_window_block, frac_first) per detector in `cal_cache` and
      then calls the kernel once per (realization, detector) on a *block slice*
      Q_det[c*N_window_block:(c+1)*N_window_block] with the within-block offset
      `ifirst_within`.

Site (2) is not covered by the kernel-level test (test_q_window_interp_gpu.py) nor by the
rotation/freqresponse tests, and it is the site where a stencil can be wired into the
plain path and forgotten in the calibration path: the block slicing changes the buffer
length seen by the kernel, so the zero-extension guard is exercised differently.

Both sites are run here with xpy=numpy and with xpy=cupy on the SAME packed data (the
Q banks, U/V cross terms and the extrinsic parameter vector are moved to device exactly
as bin/integrate_likelihood_extrinsic_batchmode does under --gpu), for every stencil in
factored_likelihood.TIME_INTERP_CHOICES, and asserted to agree.

TOLERANCE (chosen a priori, not fitted to the observed numbers): the two backends
evaluate the same real sum in a different order -- the CPU builds the
(n_extrinsic, npts, n_lm) Q window and contracts it with einsum, the device kernel fuses
the lm contraction -- so only floating-point reassociation should separate them.  For a
reduction of this length that is ~sqrt(N)*eps*|lnL| ~ 1e-14*|lnL|.  We require
    max|lnL_gpu - lnL_cpu| < 1e-8 + 1e-11 * max|lnL_cpu|
i.e. ~1000x the reassociation floor, which is still many orders of magnitude tighter
than any genuine stencil/dispatch error (a wrong or missing stencil moves lnL by O(1)
nats or more, as the 'sinc' vs 'cubic' separation printed by
test_calmarg_stencil_gating demonstrates).

SKIPPED (not failed) if cupy / a GPU is unavailable, following test_slowrot_gpu.py.

Run on a GPU node (an sm_75 card -- the installed cupy 10.6/CUDA 11.2 cannot compile
for sm_120):
    CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 \
      PYTHONPATH=<worktree>/MonteCarloMarginalizeCode/Code \
      ~/RIFT_develUWM/bin/python RIFT/likelihood/test_noloop_gpu_stencils.py
"""
from __future__ import print_function, division

import numpy as np
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl

# Same pre-existing environment workaround used by test_slowrot_noloop.py /
# test_slowrot_gpu.py: when numba's @vectorize decoration is unavailable at import time,
# factored_likelihood falls back to a SCALAR lalylm which its own array call sites
# (ComputeYlmsArrayVector) cannot use.  Rebinding it here affects only this process.
if not getattr(fl, "numba_on", True):
    fl.lalylm = np.vectorize(lal.SpinWeightedSphericalHarmonic, otypes=[complex])

from RIFT.likelihood._gpu_test_support import skip_without_gpu

try:
    import cupy
    _ = cupy.array(1.0) + 1.0    # force a real device op
    HAVE_GPU = True
    _WHY = None
except Exception as e:                                   # pragma: no cover - env dependent
    HAVE_GPU = False
    _WHY = str(e)


fSample = 4096.0
fmin = 30.0
fmax = 1700.0
event_time = 1e9
t_window = 0.1
Lmax = 2
deltaT = 1. / fSample
deltaF = 1. / 4.

N_CAL = 4          # calibration realizations for the 'loop' path
N_EXTRINSIC = 64
T_HALFWIDTH = 0.03  # lnL(t) window half width

# Injected distance 2 Gpc (SNR ~ 12), NOT the 200 Mpc used by the rotation tests.  Those
# tests compare lnL(t) arrays; this one compares the TIME-INTEGRATED lnL, whose reduction
# is  lnL = lnLmax + log simps(exp(lnL_t - lnLmax))  with lnLmax the GLOBAL max over all
# extrinsic samples.  At SNR ~ 120 the spread of lnL across random sky positions is
# ~1e4 nats, so exp() underflows to 0 for the poorly-placed samples and the CPU result is
# -inf for them (a real property of the reduction, reproduced on both backends -- not a
# bug in the stencils, but it makes a difference comparison vacuous).  A realistic SNR
# keeps the whole extrinsic vector in range.
Psig = lsu.ChooseWaveformParams(
    fmin=fmin, radec=True, incl=0.3, phiref=0.0, theta=0.2, phi=1.0, psi=0.4,
    m1=30 * lal.MSUN_SI, m2=25 * lal.MSUN_SI, detector='H1',
    dist=2000e6 * lal.PC_SI, deltaT=deltaT, tref=event_time, deltaF=deltaF)

data_dict = {}
for _det in ("H1", "L1", "V1"):
    _P = Psig.manual_copy()
    _P.detector = _det
    data_dict[_det] = lsu.non_herm_hoff(_P)
psd_dict = {det: lalsim.SimNoisePSDaLIGOZeroDetHighPower for det in data_dict}


def _P_vec(K=N_EXTRINSIC, seed=1234):
    """Vector of extrinsic samples, exactly the shape the ILE hands the NoLoop path."""
    rng = np.random.RandomState(seed)
    Pv = Psig.manual_copy()
    Pv.phi = rng.uniform(0, 2 * np.pi, K)
    Pv.theta = np.arcsin(rng.uniform(-1, 1, K))
    Pv.psi = rng.uniform(0, np.pi, K)
    Pv.incl = np.arccos(rng.uniform(-1, 1, K))
    Pv.phiref = rng.uniform(0, 2 * np.pi, K)
    Pv.dist = rng.uniform(1500, 4000, K) * 1e6 * lsu.lsu_PC
    Pv.tref = float(event_time)
    Pv.deltaT = deltaT
    return Pv


def _P_vec_to_gpu(Pv):
    """Cast the sampled extrinsic arrays to device arrays, as the driver does
    (integrate_likelihood_extrinsic_batchmode: ``P.phi = xpy_default.asarray(...)``)."""
    Pg = Pv.manual_copy()
    for attr in ("phi", "theta", "psi", "incl", "phiref", "dist"):
        Pg.__dict__[attr] = cupy.asarray(np.asarray(getattr(Pv, attr), dtype=np.float64))
    Pg.tref = float(Pv.tref)
    Pg.deltaT = float(Pv.deltaT)
    return Pg


def _pack(rholms, crossTerms, crossTermsV):
    """Array-pack the precompute output for the NoLoop path (one entry per detector).

    NOTE: pass None for the interpolant dict -- PackLikelihoodDataStructuresAsArrays has a
    pre-existing py2-ism (`rholm_intpArray = range(nKeys)`) that raises TypeError whenever
    that argument is truthy.  The NoLoop array path does not use the interpolants.
    """
    lookupNKDict, rholmArrayDict, ctUArrayDict, ctVArrayDict, epochDict = {}, {}, {}, {}, {}
    for det in rholms:
        pairKeys = list(rholms[det].keys())
        lookupNK, _lkn, _conj, ctU, ctV, rholmArray, _intp, epoch = \
            fl.PackLikelihoodDataStructuresAsArrays(
                pairKeys, None, rholms[det], crossTerms[det], crossTermsV[det])
        lookupNKDict[det] = lookupNK
        rholmArrayDict[det] = rholmArray
        ctUArrayDict[det] = ctU
        ctVArrayDict[det] = ctV
        epochDict[det] = epoch
    return lookupNKDict, rholmArrayDict, ctUArrayDict, ctVArrayDict, epochDict


def _banks_to_gpu(rholmArrayDict, ctUArrayDict, ctVArrayDict):
    return (
        {d: cupy.asarray(rholmArrayDict[d]) for d in rholmArrayDict},
        {d: cupy.asarray(ctUArrayDict[d]) for d in ctUArrayDict},
        {d: cupy.asarray(ctVArrayDict[d]) for d in ctVArrayDict},
    )


def _calibration_realizations(data, n_cal, seed=7):
    """Smooth, physically-shaped complex calibration draws C_c(f), shape (n_freq, n_cal).

    ComputeModeIPTimeSeries iterates ``calibration_realizations.T``, applies each draw to
    the DATA, and concatenates the resulting per-realization rho_lm(t) blocks -- which is
    exactly the n_cal-contiguous-block layout the NoLoop 'loop' path assumes.  A few
    percent in amplitude and a few tens of mrad in phase is the realistic O4 scale; the
    point here is only that the blocks genuinely DIFFER, so the per-realization kernel
    calls cannot be accidentally satisfied by a single block.
    """
    n = data.data.length
    f = float(data.f0) + np.arange(n) * float(data.deltaF)
    rng = np.random.RandomState(seed)
    out = np.empty((n, n_cal), dtype=np.complex128)
    for c in range(n_cal):
        a0, a1, p0, p1 = rng.uniform(-1, 1, 4)
        dA = 0.03 * (a0 * np.sin(2 * np.pi * f / 512.) + a1 * np.cos(2 * np.pi * f / 1024.))
        dphi = 0.03 * (p0 * np.cos(2 * np.pi * f / 700.) + p1 * np.sin(2 * np.pi * f / 300.))
        out[:, c] = (1.0 + dA) * np.exp(1j * dphi)
    return out


_CACHE = {}


def _setup():
    """Precompute + pack, once: the plain (n_cal=1) banks and the n_cal=N_CAL banks."""
    if _CACHE:
        return _CACHE
    _, ct, ctV, rho, _snr, _rest = fl.PrecomputeLikelihoodTerms(
        event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
        analyticPSD_Q=True, verbose=False, quiet=True, ignore_threshold=None,
        skip_interpolation=True)
    _CACHE['plain'] = _pack(rho, ct, ctV)

    cal = {det: _calibration_realizations(data_dict[det], N_CAL, seed=11 + i)
           for i, det in enumerate(sorted(data_dict))}
    _, ct_c, ctV_c, rho_c, _snr_c, _rest_c = fl.PrecomputeLikelihoodTerms(
        event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
        analyticPSD_Q=True, verbose=False, quiet=True, ignore_threshold=None,
        skip_interpolation=True, calibration_realizations=cal)
    _CACHE['cal'] = _pack(rho_c, ct_c, ctV_c)
    return _CACHE


def _tolerance(lnL_cpu):
    return 1e-8 + 1e-11 * float(np.max(np.abs(np.asarray(lnL_cpu))))


def _run_pair(banks, n_cal, interp, Pv, tvals):
    """Return (lnL_cpu, lnL_gpu) for one (bank, n_cal, stencil) combination."""
    lookupNKDict, rholmArrayDict, ctUArrayDict, ctVArrayDict, epochDict = banks
    lnL_cpu = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, Pv, lookupNKDict, rholmArrayDict, ctUArrayDict, ctVArrayDict, epochDict,
        Lmax=Lmax, xpy=np, n_cal=n_cal, cal_method='loop', time_interp=interp)

    rG, uG, vG = _banks_to_gpu(rholmArrayDict, ctUArrayDict, ctVArrayDict)
    lnL_gpu = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        cupy.asarray(tvals), _P_vec_to_gpu(Pv), lookupNKDict, rG, uG, vG, epochDict,
        Lmax=Lmax, xpy=cupy, n_cal=n_cal, cal_method='loop', time_interp=interp)
    return np.asarray(lnL_cpu), cupy.asnumpy(lnL_gpu)


def test_noloop_gpu_matches_cpu_all_stencils():
    """Both GPU dispatch sites of the baseline NoLoop, all three stencils."""
    if not HAVE_GPU:
        if skip_without_gpu(HAVE_GPU, _WHY): return
    cache = _setup()
    Pv = _P_vec()
    tvals = np.arange(int(2 * T_HALFWIDTH / deltaT)) * deltaT - T_HALFWIDTH

    results = {}
    failures = []
    for label, key, n_cal in (("n_cal=1      ", 'plain', 1),
                              ("n_cal=%d loop" % N_CAL, 'cal', N_CAL)):
        for interp in fl.TIME_INTERP_CHOICES:
            lnL_cpu, lnL_gpu = _run_pair(cache[key], n_cal, interp, Pv, tvals)
            assert lnL_cpu.shape == lnL_gpu.shape, \
                "shape mismatch %s %s: %s vs %s" % (label, interp, lnL_cpu.shape, lnL_gpu.shape)
            assert np.all(np.isfinite(lnL_cpu)), "non-finite CPU lnL (%s, %s)" % (label, interp)
            assert np.all(np.isfinite(lnL_gpu)), "non-finite GPU lnL (%s, %s)" % (label, interp)
            d = float(np.max(np.abs(lnL_cpu - lnL_gpu)))
            tol = _tolerance(lnL_cpu)
            results[(label, interp)] = (d, tol, float(np.max(np.abs(lnL_cpu))))
            print("(GPU) NoLoop %s  interp=%-7s : max|GPU-CPU| = %.3e   (tol %.3e, "
                  "max|lnL| = %.4g)" % (label, interp, d, tol, np.max(np.abs(lnL_cpu))))
            if not (d < tol):
                failures.append("%s / %s: max|GPU-CPU| = %.6e >= tol %.6e" % (label, interp, d, tol))
    assert not failures, "GPU disagrees with CPU:\n  " + "\n  ".join(failures)
    return results


def test_stencils_are_distinguishable_on_gpu():
    """Guard against a silent dispatch collapse.

    The parity test above would still pass if _q_inner_product_gpu quietly returned the
    'nearest' result for every stencil (both backends would just be wrong together --
    except they would not, since the CPU dispatch is separate; but a shared upstream
    collapse, e.g. frac_first left as None, would).  So also require that the three
    stencils give DIFFERENT GPU lnL, at both dispatch sites.
    """
    if not HAVE_GPU:
        if skip_without_gpu(HAVE_GPU, _WHY): return
    cache = _setup()
    Pv = _P_vec()
    tvals = np.arange(int(2 * T_HALFWIDTH / deltaT)) * deltaT - T_HALFWIDTH
    for label, key, n_cal in (("n_cal=1      ", 'plain', 1),
                              ("n_cal=%d loop" % N_CAL, 'cal', N_CAL)):
        lnL = {}
        for interp in fl.TIME_INTERP_CHOICES:
            _, lnL[interp] = _run_pair(cache[key], n_cal, interp, Pv, tvals)
        for a, b in (('nearest', 'cubic'), ('nearest', 'sinc'), ('cubic', 'sinc')):
            sep = float(np.max(np.abs(lnL[a] - lnL[b])))
            print("(GPU) NoLoop %s  stencil separation %-7s vs %-7s : max|diff| = %.3e"
                  % (label, a, b, sep))
            assert sep > 0.0, \
                "GPU stencils %s and %s are bit-identical (%s) -- dispatch collapsed" % (a, b, label)


def test_both_gpu_dispatch_sites_are_reached():
    """Structural proof that the parity test above really covered BOTH device call sites.

    Counting the calls into factored_likelihood._q_inner_product_gpu distinguishes them
    unambiguously: the n_cal==1 path calls it once per detector, the calibration 'loop'
    path once per (realization, detector).  Without this, a refactor that routed the loop
    path back through the CPU builder would leave the parity numbers above looking fine
    while silently testing nothing on the device.
    """
    if not HAVE_GPU:
        if skip_without_gpu(HAVE_GPU, _WHY): return
    cache = _setup()
    Pv = _P_vec()
    tvals = np.arange(int(2 * T_HALFWIDTH / deltaT)) * deltaT - T_HALFWIDTH
    n_det = len(data_dict)
    orig = fl._q_inner_product_gpu
    for label, key, n_cal, expect in (("n_cal=1", 'plain', 1, n_det),
                                      ("n_cal=%d loop" % N_CAL, 'cal', N_CAL, n_det * N_CAL)):
        for interp in fl.TIME_INTERP_CHOICES:
            counter = {'n': 0, 'lens': set()}

            def _counting(Q, A, si, fo, npts, ti, _o=orig, _c=counter):
                _c['n'] += 1
                _c['lens'].add(int(Q.shape[0]))
                return _o(Q, A, si, fo, npts, ti)

            fl._q_inner_product_gpu = _counting
            try:
                lookupNKDict, rholmArrayDict, ctUArrayDict, ctVArrayDict, epochDict = cache[key]
                rG, uG, vG = _banks_to_gpu(rholmArrayDict, ctUArrayDict, ctVArrayDict)
                fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
                    cupy.asarray(tvals), _P_vec_to_gpu(Pv), lookupNKDict, rG, uG, vG,
                    epochDict, Lmax=Lmax, xpy=cupy, n_cal=n_cal, cal_method='loop',
                    time_interp=interp)
            finally:
                fl._q_inner_product_gpu = orig
            print("(GPU) NoLoop %-13s interp=%-7s : _q_inner_product_gpu calls = %d "
                  "(expected %d), device Q buffer lengths = %s"
                  % (label, interp, counter['n'], expect, sorted(counter['lens'])))
            assert counter['n'] == expect, \
                "%s / %s reached the GPU dispatch %d times, expected %d" \
                % (label, interp, counter['n'], expect)


if __name__ == "__main__":
    test_noloop_gpu_matches_cpu_all_stencils()
    test_stencils_are_distinguishable_on_gpu()
    test_both_gpu_dispatch_sites_are_reached()
    print("NOLOOP GPU STENCIL CHECK DONE" if HAVE_GPU else "NOLOOP GPU STENCIL CHECK SKIPPED (no GPU)")
