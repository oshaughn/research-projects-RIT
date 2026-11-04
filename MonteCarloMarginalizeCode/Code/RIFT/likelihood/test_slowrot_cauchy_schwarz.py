"""test_slowrot_cauchy_schwarz : the rotation likelihood must be a real <d|h> - (1/2)<h|h>.

For ANY single template h, lnL = <d|h> - (1/2)<h|h> <= (1/2)<d|d>.  That is Cauchy-Schwarz, not
an approximation, so it holds whatever the model error is -- a truncated delay expansion, a wrong
sky position, the wrong waveform family.  The only way a "likelihood" can exceed it is by
evaluating its two terms for DIFFERENT h.

That is exactly the failure this file guards.  The precompute builds its elementary templates
chi_a(u) = e^{i n Omega u} h^{(p)}(u) on the template's INTRINSIC time u, while the physical
response modulation e^{i n Omega (t'-tref)} lives on absolute time.  Placing the template at
arrival time t makes the two differ by exp(i n Omega (t - tref)) -- the post-phase carried by
rotation_post_phase().  Drop it from the model norm, or apply it to only one of the two terms,
and lnL overshoots the bound by O(n Omega (t - tref)) * <d|d>: ~1e-4 of <d|d> at the physical
90-minute-BNS rate.  That is invisible next to any ordinary convergence check and fatal to the
one statement about a likelihood that cannot be argued with.

THE ARRIVAL OFFSET MUST BE NONZERO, AND THAT IS THE WHOLE POINT.
The post-phase is exp(i n Omega (t - tref)); at t = tref it is the identity and the defect is
invisible.  So the data here places the signal at the detector's true geometric arrival time
(+10.2 ms for H1 at this sky position, 42 samples), which is where a real analysis evaluates it.
A version of this test with the signal at t = tref passes on the BROKEN code.

Three checks, in order, because the later ones are worthless without the earlier ones:

  (A) TEETH.  With the modulation switched off (f_sidereal=0) against the SAME rotating data, the
      deficit must be LARGE.  If it is not, this configuration does not exercise rotation and
      (B),(C) would pass on an untested code path.
  (B) THE BOUND.  No sampled lnL(t) may exceed (1/2)<d|d>.  No interpolation is involved, so no
      estimator tolerance is needed: every sampled value is a genuine lnL for its arrival time.
      The data is the exact Path-A model, so at the true arrival sample lnL sits ON the bound and
      the check is maximally tight -- there is no slack for an inconsistency to hide in.
  (C) THE MECHANISM.  lnL(t) must equal a directly constructed <d|h> - (1/2)<h|h> for the model
      the likelihood implies, built explicitly in the time domain and contracted with the same
      band-limited, noise-weighted inner product.  (B) can only detect a violation; (C) pins the
      value from an independent construction.

(C) scans only NON-NEGATIVE arrival offsets.  RIFT's mode arrays start with the tapered onset of
the inspiral at index 0 and park the merger near the end, so a circular shift to earlier times
wraps real signal across the segment boundary, where the FFT correlation the precompute uses and
an explicit time-domain roll legitimately disagree (by exp(i n Omega * seglen) on the wrapped
samples).  That is a property of the finite segment, not of the likelihood; shifting later wraps
only the decayed ringdown and is clean to machine precision.

Run: source ~/RIFT_develUWM/bin/activate;
     PYTHONPATH=<this tree>/MonteCarloMarginalizeCode/Code python <this file>
"""
from __future__ import print_function, division
import numpy as np
import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.factored_likelihood_with_rotation as flwr
import RIFT.likelihood.slowrot_response as srr

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
MIN_STATIC_DEFICIT = 1.0   # (A): rotation must be worth at least this much here
NPTS_SCAN = 164            # +-20 ms
SCAN_HALF = 10             # (C) samples either side of the arrival sample


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
print("INFL=%.1f (Omega*T_seg=%.3f rad)  arrival offset %+d samples (%+.2f ms)  0.5<d|d>=%.6f"
      % (INFL, OMEGA * seglen, K_ARR, 1e3 * K_ARR * deltaT, HALF_DD))


def rotation_lnL_t(f_sidereal):
    """lnL(t) from the maintained rotation NoLoop, plus the arrival sample offsets it used."""
    P = Psig.manual_copy()
    bank = flwr.PrecomputeLikelihoodTermsWithRotation(
        event_time, t_window, P, data_dict, psd_dict, Lmax, fmax, harmonics=HARM, p_max=0,
        f_sidereal=f_sidereal, analyticPSD_Q=True, verbose=False, quiet=True,
        skip_interpolation=True)
    meta = bank[4]
    lk, rho_b, U_b, V_b, epd = flwr.pack_rotation_arrays(meta, bank[3], bank[1], bank[2])
    Pv = Psig.manual_copy()
    for key, v in [('phi', RA), ('theta', DEC), ('incl', INCL), ('phiref', PHIREF),
                   ('psi', PSI), ('dist', DLOUD)]:
        setattr(Pv, key, np.ones(1) * v)
    Pv.tref = event_time; Pv.deltaT = deltaT
    tvals = -0.02 + np.arange(NPTS_SCAN) * deltaT
    lnL_t = flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(
        tvals, Pv, meta, lk, rho_b, U_b, V_b, epd, Lmax=Lmax, array_output=True)[0]
    # Reproduce the NoLoop's own indexing so we know which arrival sample each output is.
    off = float(Pv.tref - float(epd[det]))
    ifirst = int(np.round((off + DELAY + tvals[0]) / deltaT))
    kvals = ifirst + np.arange(NPTS_SCAN) - int(round(off / deltaT))
    return np.asarray(lnL_t), kvals


# ---------------------------------------------------------------- (A) teeth
lnL_static, _ = rotation_lnL_t(0.0)
static_deficit = HALF_DD - float(np.max(lnL_static))
print("(A) rotation OFF vs rotating data: deficit = %.4f nats" % static_deficit)
assert static_deficit > MIN_STATIC_DEFICIT, (
    "this configuration does not exercise rotation (static deficit %g <= %g), so the bound and "
    "direct-model checks below would be vacuous" % (static_deficit, MIN_STATIC_DEFICIT))

# ---------------------------------------------------------------- (B) the bound
lnL_rot, kvals = rotation_lnL_t(FSID)
overshoot = float(np.max(lnL_rot)) - HALF_DD
jpeak = int(np.argmax(lnL_rot))
print("(B) rotation ON : max lnL = %.6f at k=%+d   deficit = %+.6e"
      % (np.max(lnL_rot), kvals[jpeak], HALF_DD - np.max(lnL_rot)))
assert kvals[jpeak] == K_ARR, (
    "lnL peaks at arrival sample %d, not the %d the data was built at -- the test is no longer "
    "sitting on the bound and (B) has lost its teeth" % (kvals[jpeak], K_ARR))
assert overshoot <= TOL_BOUND, (
    "Cauchy-Schwarz VIOLATED: max lnL exceeds 0.5<d|d> by %g nats.  lnL = <d|h> - (1/2)<h|h> "
    "cannot exceed (1/2)<d|d> for any h, so term1 and term2 are being evaluated for different "
    "templates -- see rotation_post_phase()." % overshoot)

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
invDist = fl.distMpcRef / (DLOUD / (lsu.lsu_PC * 1e6))
data_epoch = lal.LIGOTimeGPS(epoch_intr + event_time)

worst = 0.0; n_cmp = 0
for j in range(max(0, jpeak - SCAN_HALF), min(NPTS_SCAN, jpeak + SCAN_HALF + 1)):
    k = int(kvals[j])
    if k < 0:                      # see the docstring: negative shifts wrap the inspiral onset
        continue
    hf = _to_fd(np.real(F_of_u * np.roll(hY_ref, k)) * invDist, data_epoch, deltaT, N)
    lnL_direct = IPc.ip(hf, data).real - 0.5 * IPc.ip(hf, hf).real
    worst = max(worst, abs(lnL_direct - lnL_rot[j])); n_cmp += 1
print("(C) vs explicit time-domain model over %d samples about the peak: max|d lnL| = %.3e"
      % (n_cmp, worst))
assert n_cmp >= SCAN_HALF, "too few comparable samples (%d) for (C) to mean anything" % n_cmp
assert worst < TOL_DIRECT, (
    "rotation NoLoop disagrees with the explicit <d|h> - (1/2)<h|h> for the model it implies "
    "by %g nats" % worst)

print("ALL SLOWROT CAUCHY-SCHWARZ CHECKS PASSED")
