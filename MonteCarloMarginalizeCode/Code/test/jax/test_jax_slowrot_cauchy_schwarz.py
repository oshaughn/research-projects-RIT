"""test_jax_slowrot_cauchy_schwarz : the JAX rotation likelihood must be a real <d|h> - (1/2)<h|h>.

The JAX twin of ``RIFT/likelihood/test_slowrot_cauchy_schwarz.py``, which guards the numpy/cupy
NoLoop.  Read that file first -- the physics, the reason the arrival offset must be nonzero, and
the (A)/(B)/(C) ladder are documented there and are not repeated in full here.

WHY THIS FILE EXISTS SEPARATELY FROM test_jax_slowrot.py.
``test_jax_slowrot.py`` gate (a) checks that the JAX path AGREES with the NoLoop.  That is
necessary but NOT sufficient, and the difference is not academic: a likelihood that drops the
arrival-time post-phase from BOTH terms is perfectly self-consistent, satisfies Cauchy-Schwarz,
and was measured ~95 nats from the correct value.  Agreement pins the two implementations to each
other; only a bound and an independently constructed model pin the VALUE.

Three checks, in order (the later ones are worthless without the earlier ones):

  (A) TEETH.  With the modulation switched off (f_sidereal=0) against the SAME rotating data the
      deficit must be LARGE, or this configuration does not exercise rotation at all and (B),(C)
      would pass on an untested code path.
  (B) THE BOUND.  No sampled lnL(t) may exceed (1/2)<d|d>.  The data IS the exact model at the
      p_max under test (see data_for), so at the true arrival sample lnL sits ON the bound:
      maximum sensitivity, no slack.  Measured deficit at the peak: 0.0 nats (p_max=0) and
      5.1e-04 out of 3.2e+05 (p_max=1).
  (C) THE MECHANISM.  lnL(t) must equal a directly constructed <d|h> - (1/2)<h|h> for the model
      the likelihood implies, built explicitly in the time domain and contracted with the same
      band-limited, noise-weighted inner product.  (B) can only detect a violation; (C) pins the
      value.

  (D) is a bonus cross-check: the JAX lnL(t) against the numpy NoLoop lnL(t) on the same bank.

(C)'s tolerance is ABSOLUTE (1e-6 nats) OR RELATIVE to 0.5<h|h> (1e-6), whichever passes, and the
relative arm is not slack bought to make p_max=1 go green.  At p_max=1 with INFL=1350 the delay
Taylor series is deliberately far past its radius of convergence (the p=1 band is ~5x the p=0
one), so the explicit time-domain reference -- which reconstructs the model from a circularly
rolled, FD-differentiated series -- is itself only conditioned to ~4e-07 of the model norm.
That residual is a property of THE REFERENCE, not of the likelihood, and the test proves it every
run: it prints the numpy NoLoop's disagreement with the SAME reference alongside the JAX one, and
they are identical to the digit (1.360e-01 nats both).  What pins the JAX path to the reference
implementation at that scale is (D), at 1.3e-09 nats out of 3.2e+05.  The mutation numbers below
show the relative arm still catches a dropped post-phase by 3000x.

The whole ladder runs at p_max=0 (Path A) AND p_max=1 (Path B).  Path B is a distinct code path
for this port, not a wider bank: several ``p`` then share a sidereal harmonic ``n``, so the
post-phase buckets ``m = n_a' - n_a`` collect (a,a') pairs from DIFFERENT p (4-20 pairs per bucket
at p_max=1 vs 1-5 at p_max=0) and the V-term reflection ``(p,n)->(p,-n)`` has to resolve within p.
p_max=2 is NOT run by default: it is a 15-band bank whose 225 U/V cross terms dominate the
precompute, and it adds no new branch -- the same duplicate-m scatter-add and within-p reflection
p_max=1 already exercises.  Pass it explicitly to run_ladder() if you want it.

THE ARRIVAL OFFSET MUST BE NONZERO.  The post-phase is exp(i n Omega (t - tref)); at t = tref it
is the identity and a broken implementation passes every check.  The data is therefore placed at
the detector's true geometric arrival time (+10.2 ms for H1 here, 42 samples).

MUTATION TEST (measured; 0.5<d|d> = 50960.387223 at p_max=0, 324843.955893 at p_max=1).
  * Drop the post-phase from BOTH terms (the pre-#131 code).  Self-consistent, so (B) does NOT
    fire -- it lands 0.057 nats (p_max=0) / 1.805 nats (p_max=1) UNDER the bound.  (C) catches
    it at 95.31 nats (p_max=0) and 965.67 nats = 2.97e-03 of 0.5<h|h> (p_max=1), i.e. 3000-7000x
    the gate; (D) at 3.6e+03 nats (p_max=1).  This is exactly why (C) and (D) exist and why
    NoLoop agreement alone is not enough -- though test_jax_slowrot.py gate (a) does also fire,
    at max|rel| 1.33e-05 (p_max=0) and 4.86e-05 (p_max=1).
  * Drop it from the model norm only (the asymmetric form).  (B) fires: 10.57 nats OVER the
    bound at p_max=0, 1122.48 nats OVER at p_max=1.
Neither check subsumes the other; keep both.

Run: JAX_PLATFORMS=cpu PYTHONPATH=<tree>/MonteCarloMarginalizeCode/Code \\
     python test/jax/test_jax_slowrot_cauchy_schwarz.py
"""
from __future__ import print_function, division
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.factored_likelihood_with_rotation as flwr
import RIFT.likelihood.slowrot_response as srr

from RIFT.likelihood.jax_ile.banded import build_rotation_data
# _accumulate_unit is the (private) kernel that produces the per-time-bin kappa and rho^2.
# The public entry points marginalize over t, which would smear exactly the arrival-time
# dependence this file is about; every sampled lnL_t below is a genuine lnL for ONE arrival
# time, which is what makes (B) tolerance-free.
from RIFT.likelihood.jax_ile.core import _accumulate_unit

fmin = 30.; fmax = 1700.; event_time = 1e9; t_window = 0.1; Lmax = 2
deltaT = 1. / 4096.; seglen = 4.; deltaF = 1. / seglen
fNyq = 1. / 2. / deltaT; N = int(round(seglen / deltaT))
det = 'H1'
HARM = (-2, -1, 0, 1, 2)
# Omega * T_segment equal to a 90-minute (5400 s) signal at the true sidereal rate.  The
# 5-harmonic antenna expansion is EXACT at any Omega, so inflating it costs no accuracy.
INFL = 5400. / seglen
OMEGA = flwr.OMEGA_EARTH * INFL
FSID = OMEGA / (2.0 * np.pi)
RA, DEC, PSI, INCL, PHIREF = 1.0, 0.2, 0.5, 0.7, 0.9
DLOUD = fl.distMpcRef * 1e6 * lsu.lsu_PC / 30.      # loud, so lnL sits near the bound

TOL_BOUND = 1e-6           # nats above (1/2)<d|d> that we call a violation
TOL_DIRECT_ABS = 1e-6      # nats of disagreement with the explicit model
TOL_DIRECT_REL = 1e-6      # ... or, for an ill-conditioned model, of 0.5<h|h> (see run_ladder)
TOL_NOLOOP = 1e-8          # nats of disagreement with the numpy NoLoop lnL(t)
MIN_STATIC_DEFICIT = 1.0   # (A): rotation must be worth at least this much here
NPTS_SCAN = 164            # +-20 ms
SCAN_HALF = 10             # (C) samples either side of the arrival sample

TVALS = -0.02 + np.arange(NPTS_SCAN) * deltaT


def _ifft_arr(hf):
    n = hf.data.length; dt = 1. / (n * hf.deltaF)
    ts = lal.CreateCOMPLEX16TimeSeries("h", hf.epoch, 0., dt, lal.DimensionlessUnit, n)
    lal.COMPLEX16FreqTimeFFT(ts, hf, lal.CreateReverseCOMPLEX16FFTPlan(n, 0))
    return np.array(ts.data.data)


def _to_fd(arr, epoch, dt, n):
    ts = lal.CreateCOMPLEX16TimeSeries("h", epoch, 0., dt, lal.DimensionlessUnit, n)
    ts.data.data[:] = arr[:n]
    hf = lal.CreateCOMPLEX16FrequencySeries("hf", epoch, 0., 1. / dt / n, lsu.lsu_HertzUnit, n)
    lal.COMPLEX16TimeFreqFFT(hf, ts, lal.CreateForwardCOMPLEX16FFTPlan(n, 0))
    return hf


Psig = lsu.ChooseWaveformParams(
    fmin=fmin, radec=True, incl=INCL, phiref=PHIREF, theta=DEC, phi=RA, psi=PSI,
    m1=30 * lal.MSUN_SI, m2=25 * lal.MSUN_SI, detector=det, dist=200e6 * lal.PC_SI,
    deltaT=deltaT, tref=event_time, deltaF=deltaF)

lald = lalsim.DetectorPrefixToLALDetector(det)
DELAY = float(lal.TimeDelayFromEarthCenter(np.asarray(lald.location), RA, DEC,
                                           lal.LIGOTimeGPS(event_time)))
K_ARR = int(round(DELAY / deltaT))       # arrival sample offset from tref
assert K_ARR > 0, ("this test needs the signal placed at a POSITIVE arrival offset (see the "
                   "module docstring): the post-phase is the identity at zero offset, and a "
                   "negative one wraps the inspiral onset.  Geometric delay here is %g s." % DELAY)

# ---------------------------------------------------------------- data: the exact Path-A model,
# placed at the detector's geometric arrival time.
Pm = Psig.manual_copy(); Pm.dist = DLOUD
hlms_d, _ = fl.internal_hlm_generator(Pm, Lmax, verbose=False, quiet=True)
lm0 = list(hlms_d.keys())[0]
epoch_intr = float(hlms_d[lm0].epoch)
u_grid = epoch_intr + np.arange(N) * deltaT          # data-grid intrinsic time = t' - tref
hY_data = np.zeros(N, dtype=complex)
for lm in hlms_d:
    hY_data += _ifft_arr(hlms_d[lm]) * lal.SpinWeightedSphericalHarmonic(INCL, -PHIREF, -2,
                                                                        lm[0], lm[1])
g_ev = float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(event_time))) - RA
Atil = {n: v * np.exp(1j * n * g_ev)
        for n, v in srr.antenna_harmonics(lald.response, DEC, PSI).items()}
F_of_u = sum(Atil[n] * np.exp(1j * n * OMEGA * u_grid) for n in Atil)
DATA_PATH_A = _to_fd(np.real(F_of_u * np.roll(hY_data, K_ARR)),
                     lal.LIGOTimeGPS(epoch_intr + event_time), deltaT, N)
psd_dict = {det: lalsim.SimNoisePSDaLIGOZeroDetHighPower}
IPc = lsu.ComplexIP(fmin, fmax, fNyq, deltaF, psd_dict[det], True, False, 0.)
INV_DIST = fl.distMpcRef / (DLOUD / (lsu.lsu_PC * 1e6))
print("INFL=%.1f (Omega*T_seg=%.3f rad)  arrival offset %+d samples (%+.2f ms)"
      % (INFL, OMEGA * seglen, K_ARR, 1e3 * K_ARR * deltaT))


def _Pv():
    Pv = Psig.manual_copy()
    for key, v in [('phi', RA), ('theta', DEC), ('incl', INCL), ('phiref', PHIREF),
                   ('psi', PSI), ('dist', DLOUD)]:
        setattr(Pv, key, np.ones(1) * v)
    Pv.tref = event_time; Pv.deltaT = deltaT
    return Pv


def rotation_lnL_t(f_sidereal, p_max=0):
    """(jax lnL(t), numpy NoLoop lnL(t), arrival sample offsets, a_list) on one shared bank."""
    P = Psig.manual_copy()
    data_dict = data_for(p_max)[1]
    bank = flwr.PrecomputeLikelihoodTermsWithRotation(
        event_time, t_window, P, data_dict, psd_dict, Lmax, fmax, harmonics=HARM,
        p_max=p_max, f_sidereal=f_sidereal, analyticPSD_Q=True, verbose=False, quiet=True,
        skip_interpolation=True)
    meta = bank[4]
    assert len(meta['a_list']) == (p_max + 1) * len(HARM), "unexpected a_list size"
    lk, rho_b, U_b, V_b, epd = flwr.pack_rotation_arrays(meta, bank[3], bank[1], bank[2])
    Pv = _Pv()

    lnL_ref = flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(
        TVALS, Pv, meta, lk, rho_b, U_b, V_b, epd, Lmax=Lmax, array_output=True)[0]

    jdata = build_rotation_data(meta, lk, rho_b, U_b, V_b, epd, deltaT, TVALS)
    kappa, rho_sq = _accumulate_unit(
        jdata, Pv.phi, Pv.theta, Pv.psi, Pv.incl, Pv.phiref, "nearest", False)
    lnL_jax = np.asarray(kappa.real * INV_DIST - 0.5 * rho_sq * INV_DIST ** 2)[0]

    # Reproduce the shared indexing so we know which arrival sample each output is.
    off = float(Pv.tref - float(epd[det]))
    ifirst = int(np.round((off + DELAY + TVALS[0]) / deltaT))
    kvals = ifirst + np.arange(NPTS_SCAN) - int(round(off / deltaT))
    return lnL_jax, np.asarray(lnL_ref), kvals, list(meta['a_list'])


# ---------------------------------------------------------------- the explicit model for (C)
# The model the likelihood implies, built explicitly on the data grid:
#
#   h(u) = invDist * Re[ sum_a C~_a(t) chi_a(u - t) ],   chi_a(u) = e^{i n_a Omega u} hY^(p_a)(u)
#
# With the arrival at sample k (t = k*deltaT) the post-phase cancels the shift inside the
# modulation, C~_{(p,n)} e^{i n Omega (u - k dt)} = C_{(p,n)} e^{i n Omega u}, so
#
#   h(u) = invDist * Re[ sum_p G_p(u) * roll(hY^(p), k) ],  G_p(u) = sum_n C_{(p,n)} e^{i n Omega u}
#
# and at p_max=0 this is exactly the F(u)*roll(hY,k) of the numpy twin (G_0 == F).
#
# G_p reuses flwr.rotation_coefficients and the FD derivative weight rather than re-deriving
# them: what (C) is pinning is the arrival-time post-phase and the band contraction, not the
# response algebra (test_jax_slowrot_coeffs, 2e-16) or the FD derivative (test_slowrot_fd_ops).
Pref = Psig.manual_copy()
Pref.dist = fl.distMpcRef * 1e6 * lsu.lsu_PC
Pref.deltaF = deltaF
hlms_r, _ = fl.internal_hlm_generator(Pref, Lmax, verbose=False, quiet=True)
Ylm_r = fl.ComputeYlms(Lmax, INCL, -PHIREF, selected_modes=list(hlms_r.keys()))
hY_ref = np.zeros(N, dtype=complex)
for lm in hlms_r:
    hY_ref += Ylm_r[lm] * _ifft_arr(hlms_r[lm])
data_epoch = lal.LIGOTimeGPS(epoch_intr + event_time)
_hY_ref_fd = _to_fd(hY_ref, data_epoch, deltaT, N)
_FVALS = flwr.evaluate_fvals_from_length(N, _hY_ref_fd.deltaF)


def _hY_deriv(p):
    """p-th time derivative of hY_ref on the data grid (FD weight, RIFT fvals packing)."""
    if p == 0:
        return hY_ref
    hfp = lal.CreateCOMPLEX16FrequencySeries(
        "hfp", _hY_ref_fd.epoch, 0., _hY_ref_fd.deltaF, lsu.lsu_HertzUnit, N)
    hfp.data.data[:] = _hY_ref_fd.data.data * flwr.time_derivative_weight(_FVALS, p)
    return _ifft_arr(hfp)


def _explicit_model_fd(k, p_max, a_list):
    """FD of h(u) above, for arrival sample k, at fiducial distance scaled by INV_DIST.

    ``a_list`` is the bank's band list and the sum is RESTRICTED to it, which matters from
    p_max=1 on: the delay convolution gives rotation_coefficients keys (p, n+m) with
    m in {-1,0,1}, so it emits harmonics OUTSIDE the requested set (n=+-3 for HARM=+-2), and
    the bank has no band for them.  Both evaluators silently drop them (the NoLoop's Cg()
    indexes C by a_list; pack_coefficients does the same), so the truncated sum IS the model
    the likelihood implies.  Summing the full coefficient dict here instead disagrees by
    2.2e+05 nats at p_max=1 in this configuration -- the dropped bands are the same order as
    the ones kept, because at INFL=1350 the first-order delay term dominates.
    """
    C = flwr.rotation_coefficients(det, RA, DEC, PSI, event_time, p_max)   # {(p,n): C_a}
    keep = set((int(p), int(n)) for (p, n) in a_list)
    h_td = np.zeros(N, dtype=complex)
    for p in range(p_max + 1):
        G_p = np.zeros(N, dtype=complex)
        for (pa, na), c in C.items():
            if pa == p and (pa, na) in keep:
                G_p = G_p + c * np.exp(1j * na * OMEGA * u_grid)
        h_td = h_td + G_p * np.roll(_hY_deriv(p), k)
    return _to_fd(np.real(h_td) * INV_DIST, data_epoch, deltaT, N)


_DATA_CACHE = {}


def data_for(p_max):
    """(data, data_dict, 0.5<d|d>, a_list) with the data EQUAL to the exact model at this p_max.

    That is what makes (B) maximally tight: with the data equal to the model the likelihood can
    represent, lnL at the true arrival sample sits exactly ON (1/2)<d|d>, leaving no slack for an
    inconsistency to hide in.  A p_max=0 dataset used against a p_max=1 bank would instead leave
    the p>=1 bands fitting nothing, and (B) would pass with 1e5 nats of margin.

    p_max=0 uses the INDEPENDENT construction above (srr.antenna_harmonics -> F(u) -> Re[F*roll]),
    which shares nothing with rotation_coefficients; the assert below pins the two together at
    p_max=0 so the p>=1 datasets inherit that provenance.
    """
    if p_max not in _DATA_CACHE:
        a_list = flwr._elementary_index_set(HARM, p_max)
        if p_max == 0:
            d = DATA_PATH_A
            chk = _explicit_model_fd(K_ARR, 0, a_list)
            dd = np.max(np.abs(chk.data.data - d.data.data))
            ref = np.max(np.abs(d.data.data))
            assert dd <= 1e-12 * ref, (
                "the explicit model and the independent antenna_harmonics data construction "
                "disagree at p_max=0 by %g (rel %g) -- (C)'s reference is not the Path-A model"
                % (dd, dd / ref))
        else:
            d = _explicit_model_fd(K_ARR, p_max, a_list)
        _DATA_CACHE[p_max] = (d, {det: d}, 0.5 * IPc.ip(d, d).real, a_list)
    return _DATA_CACHE[p_max]


def run_ladder(p_max=0, verbose=True):
    """The (A)-(D) ladder at one p_max.  Returns a dict of the measured numbers."""
    tag = "Path %s, p_max=%d" % ("A" if p_max == 0 else "B", p_max)
    data, _dd, HALF_DD, _al = data_for(p_max)
    if verbose:
        print("\n=== JAX SLOWROT CAUCHY-SCHWARZ (%s, A=%d bands, 0.5<d|d>=%.6f) ==="
              % (tag, (p_max + 1) * len(HARM), HALF_DD))

    # ------------------------------------------------------------ (A) teeth
    lnL_static, _, _, _ = rotation_lnL_t(0.0, p_max=p_max)
    static_deficit = HALF_DD - float(np.max(lnL_static))
    print("(A) rotation OFF vs rotating data: deficit = %.4f nats" % static_deficit)
    assert static_deficit > MIN_STATIC_DEFICIT, (
        "this configuration does not exercise rotation (static deficit %g <= %g), so the bound "
        "and direct-model checks below would be vacuous" % (static_deficit, MIN_STATIC_DEFICIT))

    # ------------------------------------------------------------ (B) the bound
    lnL_rot, lnL_noloop, kvals, a_list = rotation_lnL_t(FSID, p_max=p_max)
    overshoot = float(np.max(lnL_rot)) - HALF_DD
    jpeak = int(np.argmax(lnL_rot))
    print("(B) rotation ON : max lnL = %.6f at k=%+d   deficit = %+.6e"
          % (np.max(lnL_rot), kvals[jpeak], HALF_DD - np.max(lnL_rot)))
    assert kvals[jpeak] == K_ARR, (
        "lnL peaks at arrival sample %d, not the %d the data was built at -- the test is no "
        "longer sitting on the bound and (B) has lost its teeth" % (kvals[jpeak], K_ARR))
    assert overshoot <= TOL_BOUND, (
        "Cauchy-Schwarz VIOLATED: max JAX lnL exceeds 0.5<d|d> by %g nats.  lnL = <d|h> - "
        "(1/2)<h|h> cannot exceed (1/2)<d|d> for any h, so term1 and term2 are being evaluated "
        "for different templates -- see rotation_post_phase() and "
        "core._accumulate_unit_banded." % overshoot)

    # ------------------------------------------------------------ (C) the mechanism
    # (C) scans only NON-NEGATIVE arrival offsets: a circular shift to earlier times wraps real
    # signal across the segment boundary, where the FFT correlation the precompute uses and an
    # explicit time-domain roll legitimately disagree.  See the numpy twin's docstring.
    worst = 0.0; worst_ref = 0.0; n_cmp = 0; scale = 0.0
    for j in range(max(0, jpeak - SCAN_HALF), min(NPTS_SCAN, jpeak + SCAN_HALF + 1)):
        k = int(kvals[j])
        if k < 0:
            continue
        hf = _explicit_model_fd(k, p_max, a_list)
        hh = IPc.ip(hf, hf).real
        lnL_direct = IPc.ip(hf, data).real - 0.5 * hh
        worst = max(worst, abs(lnL_direct - lnL_rot[j]))
        worst_ref = max(worst_ref, abs(lnL_direct - lnL_noloop[j]))
        scale = max(scale, 0.5 * hh); n_cmp += 1
    print("(C) vs explicit time-domain model over %d samples about the peak: max|d lnL| = %.3e"
          "  (rel to 0.5<h|h>=%.3e: %.2e; numpy NoLoop vs the same reference: %.3e)"
          % (n_cmp, worst, scale, worst / scale, worst_ref))

    # ------------------------------------------------------------ (D) vs the numpy NoLoop
    d_noloop = float(np.max(np.abs(lnL_rot - lnL_noloop)))
    print("(D) vs numpy NoLoop lnL(t) over the whole %d-sample scan: max|d lnL| = %.3e"
          % (NPTS_SCAN, d_noloop))

    assert n_cmp >= SCAN_HALF, "too few comparable samples (%d) for (C) to mean anything" % n_cmp
    assert worst < TOL_DIRECT_ABS or worst / scale < TOL_DIRECT_REL, (
        "JAX rotation likelihood disagrees with the explicit <d|h> - (1/2)<h|h> for the model "
        "it implies by %g nats (%.2e of 0.5<h|h>) at p_max=%d" % (worst, worst / scale, p_max))
    assert d_noloop < TOL_NOLOOP, "JAX vs NoLoop lnL(t) disagree by %g nats" % d_noloop

    return dict(p_max=p_max, static_deficit=static_deficit, max_lnL=float(np.max(lnL_rot)),
                overshoot=overshoot, direct=worst, noloop=d_noloop)


# pytest collects these; running the file as a script executes the same thing (see __main__).
def test_cauchy_schwarz_path_a():
    run_ladder(p_max=0)


def test_cauchy_schwarz_path_b():
    run_ladder(p_max=1)


if __name__ == "__main__":
    run_ladder(p_max=0)
    run_ladder(p_max=1)
    print("\nALL JAX SLOWROT CAUCHY-SCHWARZ CHECKS PASSED")
