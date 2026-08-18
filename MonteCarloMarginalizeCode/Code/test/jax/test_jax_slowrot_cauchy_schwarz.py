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
  (B) THE BOUND.  No sampled lnL(t) may exceed (1/2)<d|d>.  The data IS the exact Path-A model,
      so at the true arrival sample lnL sits ON the bound: maximum sensitivity, no slack.
  (C) THE MECHANISM.  lnL(t) must equal a directly constructed <d|h> - (1/2)<h|h> for the model
      the likelihood implies, built explicitly in the time domain and contracted with the same
      band-limited, noise-weighted inner product.  (B) can only detect a violation; (C) pins the
      value.

  (D) is a bonus cross-check: the JAX lnL(t) against the numpy NoLoop lnL(t) on the same bank.

THE ARRIVAL OFFSET MUST BE NONZERO.  The post-phase is exp(i n Omega (t - tref)); at t = tref it
is the identity and a broken implementation passes every check.  The data is therefore placed at
the detector's true geometric arrival time (+10.2 ms for H1 here, 42 samples).

MUTATION TEST (measured, this configuration; 0.5<d|d> = 50960.387223).
  * Drop the post-phase from BOTH terms (the pre-#131 code): self-consistent, so (B) does NOT
    fire -- max lnL 50960.330459, 0.057 nats UNDER the bound -- and (C) catches it at 95.31
    nats.  This is exactly why (C) exists and why NoLoop agreement alone is not enough:
    test_jax_slowrot.py gate (a) also fires here, at max|rel| = 1.33e-05.
  * Drop it from the model norm only (the asymmetric form): (B) fires -- max lnL 50970.953046,
    10.57 nats OVER the bound.
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
TOL_DIRECT = 1e-6          # nats of disagreement with the explicit model
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
data = _to_fd(np.real(F_of_u * np.roll(hY_data, K_ARR)),
              lal.LIGOTimeGPS(epoch_intr + event_time), deltaT, N)
data_dict = {det: data}
psd_dict = {det: lalsim.SimNoisePSDaLIGOZeroDetHighPower}
IPc = lsu.ComplexIP(fmin, fmax, fNyq, data.deltaF, psd_dict[det], True, False, 0.)
HALF_DD = 0.5 * IPc.ip(data, data).real
INV_DIST = fl.distMpcRef / (DLOUD / (lsu.lsu_PC * 1e6))
print("INFL=%.1f (Omega*T_seg=%.3f rad)  arrival offset %+d samples (%+.2f ms)  0.5<d|d>=%.6f"
      % (INFL, OMEGA * seglen, K_ARR, 1e3 * K_ARR * deltaT, HALF_DD))


def _Pv():
    Pv = Psig.manual_copy()
    for key, v in [('phi', RA), ('theta', DEC), ('incl', INCL), ('phiref', PHIREF),
                   ('psi', PSI), ('dist', DLOUD)]:
        setattr(Pv, key, np.ones(1) * v)
    Pv.tref = event_time; Pv.deltaT = deltaT
    return Pv


def rotation_lnL_t(f_sidereal):
    """(jax lnL(t), numpy NoLoop lnL(t), arrival sample offsets) on one shared bank."""
    P = Psig.manual_copy()
    bank = flwr.PrecomputeLikelihoodTermsWithRotation(
        event_time, t_window, P, data_dict, psd_dict, Lmax, fmax, harmonics=HARM, p_max=0,
        f_sidereal=f_sidereal, analyticPSD_Q=True, verbose=False, quiet=True,
        skip_interpolation=True)
    meta = bank[4]
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
    return lnL_jax, np.asarray(lnL_ref), kvals


# ---------------------------------------------------------------- (A) teeth
lnL_static, _, _ = rotation_lnL_t(0.0)
static_deficit = HALF_DD - float(np.max(lnL_static))
print("(A) rotation OFF vs rotating data: deficit = %.4f nats" % static_deficit)
assert static_deficit > MIN_STATIC_DEFICIT, (
    "this configuration does not exercise rotation (static deficit %g <= %g), so the bound and "
    "direct-model checks below would be vacuous" % (static_deficit, MIN_STATIC_DEFICIT))

# ---------------------------------------------------------------- (B) the bound
lnL_rot, lnL_noloop, kvals = rotation_lnL_t(FSID)
overshoot = float(np.max(lnL_rot)) - HALF_DD
jpeak = int(np.argmax(lnL_rot))
print("(B) rotation ON : max lnL = %.6f at k=%+d   deficit = %+.6e"
      % (np.max(lnL_rot), kvals[jpeak], HALF_DD - np.max(lnL_rot)))
assert kvals[jpeak] == K_ARR, (
    "lnL peaks at arrival sample %d, not the %d the data was built at -- the test is no longer "
    "sitting on the bound and (B) has lost its teeth" % (kvals[jpeak], K_ARR))
assert overshoot <= TOL_BOUND, (
    "Cauchy-Schwarz VIOLATED: max JAX lnL exceeds 0.5<d|d> by %g nats.  lnL = <d|h> - (1/2)<h|h> "
    "cannot exceed (1/2)<d|d> for any h, so term1 and term2 are being evaluated for different "
    "templates -- see rotation_post_phase() and core._accumulate_unit_banded." % overshoot)

# ---------------------------------------------------------------- (C) the mechanism
# The model the likelihood implies, built explicitly:
#   h(t') = invDist * Re[ F(t'-tref) * hY(t' - t_arr) ],   F from the SAME A_tilde harmonics.
Pref = Psig.manual_copy()
Pref.dist = fl.distMpcRef * 1e6 * lsu.lsu_PC
Pref.deltaF = data.deltaF
hlms_r, _ = fl.internal_hlm_generator(Pref, Lmax, verbose=False, quiet=True)
Ylm_r = fl.ComputeYlms(Lmax, INCL, -PHIREF, selected_modes=list(hlms_r.keys()))
hY_ref = np.zeros(N, dtype=complex)
for lm in hlms_r:
    hY_ref += Ylm_r[lm] * _ifft_arr(hlms_r[lm])
data_epoch = lal.LIGOTimeGPS(epoch_intr + event_time)

# (C) scans only NON-NEGATIVE arrival offsets: a circular shift to earlier times wraps real
# signal across the segment boundary, where the FFT correlation the precompute uses and an
# explicit time-domain roll legitimately disagree.  See the numpy twin's docstring.
worst = 0.0; n_cmp = 0
for j in range(max(0, jpeak - SCAN_HALF), min(NPTS_SCAN, jpeak + SCAN_HALF + 1)):
    k = int(kvals[j])
    if k < 0:
        continue
    hf = _to_fd(np.real(F_of_u * np.roll(hY_ref, k)) * INV_DIST, data_epoch, deltaT, N)
    lnL_direct = IPc.ip(hf, data).real - 0.5 * IPc.ip(hf, hf).real
    worst = max(worst, abs(lnL_direct - lnL_rot[j])); n_cmp += 1
print("(C) vs explicit time-domain model over %d samples about the peak: max|d lnL| = %.3e"
      % (n_cmp, worst))
assert n_cmp >= SCAN_HALF, "too few comparable samples (%d) for (C) to mean anything" % n_cmp
assert worst < TOL_DIRECT, (
    "JAX rotation likelihood disagrees with the explicit <d|h> - (1/2)<h|h> for the model it "
    "implies by %g nats" % worst)

# ---------------------------------------------------------------- (D) vs the numpy NoLoop
d_noloop = float(np.max(np.abs(lnL_rot - lnL_noloop)))
print("(D) vs numpy NoLoop lnL(t) over the whole %d-sample scan: max|d lnL| = %.3e"
      % (NPTS_SCAN, d_noloop))
assert d_noloop < TOL_NOLOOP, "JAX vs NoLoop lnL(t) disagree by %g nats" % d_noloop

print("ALL JAX SLOWROT CAUCHY-SCHWARZ CHECKS PASSED")
