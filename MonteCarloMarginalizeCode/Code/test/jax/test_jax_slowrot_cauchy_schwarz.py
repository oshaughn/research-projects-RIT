"""test_jax_slowrot_cauchy_schwarz : the JAX rotation likelihood must be a real <d|h> - (1/2)<h|h>.

The JAX twin of ``RIFT/likelihood/test_slowrot_cauchy_schwarz.py``, which guards the numpy/cupy
NoLoop.  Read that file first -- the physics and the reason the arrival offset must be nonzero are
documented there and are not repeated here.

WHY THIS FILE EXISTS SEPARATELY FROM test_jax_slowrot.py.  That file's gate (a) checks the JAX
path AGREES with the NoLoop.  Necessary, not sufficient: a likelihood that drops the arrival-time
post-phase from BOTH terms is self-consistent, satisfies Cauchy-Schwarz, and is badly wrong.
Agreement pins the two implementations to each other; only a bound and an independently
constructed model pin the VALUE.

Four checks, in order (the later ones are worthless without the earlier ones):

  (A) TEETH.  With the modulation switched off (f_sidereal=0) against the SAME rotating data the
      deficit must be LARGE, or this configuration does not exercise rotation and (B),(C) would
      pass on an untested code path.  (A) compares the evaluator against ITSELF at f_sidereal=0,
      so it guards the CONFIGURATION, not the post-phase -- a defect common to both arms cancels.
  (B) THE BOUND.  No sampled lnL(t) may exceed (1/2)<d|d>.  The data IS the exact model at the
      p_max under test (see data_for), so at the true arrival sample lnL sits ON the bound.
      NOTE THE SIGN: (B) PRINTS a deficit, 0.5<d|d> - max lnL, and ASSERTS on the overshoot,
      max lnL - 0.5<d|d>.  They are negatives of each other; a violation is a POSITIVE overshoot.
  (C) THE MECHANISM.  lnL(t) must equal a directly constructed <d|h> - (1/2)<h|h> for the model
      the likelihood implies, built explicitly in the time domain.  (B) can only detect a
      violation; (C) pins the value.  Its reference shifts the MODULATED template circularly and
      repairs the phase with rotation_post_phase, because that is what the bank does; modulating
      on the unrolled grid instead disagrees on the samples that wrap the segment boundary.
  (D) a cross-check of the JAX lnL(t) against the numpy NoLoop on the same bank.

Both rungs run: p_max=0 (Path A) and p_max=1 (Path B).  Path B is a distinct code path, not a
wider bank -- several ``p`` share a sidereal harmonic ``n``, so the post-phase buckets
``m = n_a' - n_a`` collect (a,a') pairs from DIFFERENT p, and the V-term reflection
``(p,n)->(p,-n)`` has to resolve within p.  p_max=2 is not run by default: it is a 27-band bank
whose 729 U/V cross terms dominate the precompute and it adds no new branch.  config_for()
RAISES for it rather than guessing a rate: give it a CONFIG entry, with the measurement
justifying whatever tolerance it needs, before calling run_ladder(p_max=2).

THE ARRIVAL OFFSET MUST BE NONZERO.  The post-phase is exp(i n Omega (t - tref)); at t = tref it
is the identity and a broken implementation passes every check.  The data is therefore placed at
the detector's true geometric arrival time.

Path B runs at a higher rotation rate than Path A, and that is (A)'s requirement alone: the static
deficit grows with Omega, and at Path A's rate it falls below MIN_STATIC_DEFICIT.

DO NOT "fix" a failure here by widening TOL_BOUND, TOL_DIRECT_* or MIN_STATIC_DEFICIT.  Every
gate has orders of margin over what it catches; a failure is a defect, not a tolerance being
tight.

TWO THINGS THIS LADDER DELIBERATELY DOES NOT CLAIM.  It does not claim the p-expansion CONVERGES
here -- it does not, and that is fine, because the data is built as the exact model at the p_max
under test, so what is validated is that the evaluator computes lnL for the model the bank
implies.  And it does not claim the bank's CIRCULARLY shifted model matches a physically modulated
one; they differ on the wrapped samples, which is a property of FFT-correlation banks that a
Path-B production run inherits, and no assert here covers it.

PORTABILITY -- READ BEFORE FILING A FINDING ON A DIGIT PRINTED BY THIS FILE.  Numbers here are
bit-stable within a host but NOT across CPU families: cells built from a near-total cancellation
of large numbers keep only a couple of significant figures, and those figures differ between
Intel and AMD.  Do NOT "fix" a cell because your host differs, and do not derive an argument from
a digit that is not stable.  THERE IS NO SHORTCUT FOR CLASSIFYING A NEW CELL -- measure it on both
families.  A digit-count rule of the form 16 - log10(operand/result) was tried and REFUTED: it
over-predicts stability and cannot separate cells that split from cells that do not.

The spread is harmless ONLY BECAUSE no gate here is a tolerance on one of these numbers --
every assert compares against a TOL_* constant, not against a recorded digit.  Pinning any
host-split cell as an expected value would make the spread live and this suite host-dependent.
If you must pin one, use a tolerance that survives both families, or state the host.

DO NOT ATTACH A MECHANISM TO A MEASURED TABLE WITHOUT CHECKING IT AT MORE THAN ONE ROW.  Two
explanations for the split were adopted on partial evidence and later withdrawn; the disconfirming
row was already in the table both times.

Evidence, sweeps, mutation tables and measured impact: PRs #117 and #163, and
RIFT_roboto_paper analyses/slowrot_bound_violation/ + analyses/slowrot_nyquist_bin/NOTE.md.

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

fmin = 30.; event_time = 1e9; t_window = 0.1; Lmax = 2
deltaT = 1. / 4096.; seglen = 4.; deltaF = 1. / seglen
fNyq = 1. / 2. / deltaT; N = int(round(seglen / deltaT))
det = 'H1'
HARM = (-2, -1, 0, 1, 2)
psd_dict = {det: lalsim.SimNoisePSDaLIGOZeroDetHighPower}


def _harm_for(p_max):
    """The harmonic set the PRECOMPUTE will actually carry for this p_max.

    rotation_coefficients emits keys (p, n+m) with |m| <= 1, so the coefficient index widens
    by one per derivative order, and PrecomputeLikelihoodTermsWithRotation widens a too-narrow
    `harmonics` to |n| <= 2 + p_max rather than silently dropping bands (#142/#143).  Derive
    the set from that same helper: assuming HARM here instead would put the data, the bank and
    the explicit reference model on THREE different harmonic sets at p_max >= 1.
    """
    return flwr.widen_harmonics_for_p_max(HARM, p_max)[0]
RA, DEC, PSI, INCL, PHIREF = 1.0, 0.2, 0.5, 0.7, 0.9
DLOUD = fl.distMpcRef * 1e6 * lsu.lsu_PC / 30.      # loud, so lnL sits near the bound

# ---------------------------------------------------------------- per-rung configuration
# The two knobs the rung's conditioning turns on: the rotation rate (through INFL, the factor
# by which the sidereal rate is inflated so that Omega*T_segment matches a long signal) and the
# upper end of the band.  They are PER p_max because the p >= 1 rungs need a different balance
# from Path A -- see CONFIG and config_for below.
INFL_DEFAULT = 5400. / seglen      # Omega * T_segment as for a 90-minute signal
FMAX_DEFAULT = 1700.


class Config(object):
    """One rung's (INFL, fmax), plus everything derived from them.

    Everything that does NOT depend on these two knobs -- the waveform modes, hY_data, hY_ref
    and its FD derivatives -- stays at module level and is shared across configurations, so a
    sweep over (INFL, fmax) does not regenerate waveforms.
    """

    def __init__(self, infl=INFL_DEFAULT, fmax=FMAX_DEFAULT):
        self.infl = float(infl)
        self.fmax = float(fmax)
        # The 5-harmonic ANTENNA expansion is exact at any Omega, so inflating Omega costs no
        # accuracy at p_max=0.  The DELAY expansion is a Taylor series and does not share that
        # property: see _delay_expansion_ratio.
        self.omega = flwr.OMEGA_EARTH * self.infl
        self.fsid = self.omega / (2.0 * np.pi)
        self.ipc = lsu.ComplexIP(fmin, self.fmax, fNyq, deltaF, psd_dict[det], True, False, 0.)
        self._data_cache = {}

    def __repr__(self):
        return "Config(INFL=%.1f, fmax=%.0f, Omega*T_seg=%.3f rad)" % (
            self.infl, self.fmax, self.omega * seglen)


# The configuration each rung runs at.  An unlisted p_max >= 1 RAISES in config_for() below;
# the bare-Config() fallback is reachable only for p_max < 1 and not in CONFIG -- i.e.
# negative, or a non-integer below 1 -- since 0 is a key here.  Neither occurs in practice.
#
# Path B runs FASTER than Path A, at Omega*T_segment for a 6-hour signal rather than a
# 90-minute one, and that is (A)'s requirement, not (B)'s or (C)'s.  With the model
# non-truncated (#142/#143) the static approximation is good to 0.39 nats at the 90-minute
# rate -- below MIN_STATIC_DEFICIT, i.e. the rung would not be exercising rotation.  The
# deficit grows FASTER THAN LINEARLY but slower than Omega^2 over this range (measured:
# 0.0046 / 0.107 / 0.389 / 1.296 / 3.923 nats at INFL = 135 / 675 / 1350 / 2700 / 5400 --
# that is 10.1x for the last 4x, i.e. ~Omega^1.66, where Omega^2 would predict 16x; this
# comment said "like Omega^2" against that same list).  So 4x the rate buys 10x the teeth.
# Nothing else
# pays for it: (B) and (C) are at machine precision across that whole range once the two
# defects issue #159 turned up are fixed (see PRs #117 and #163).
CONFIG = {
    0: Config(),
    1: Config(infl=21600. / seglen),
}


def config_for(p_max):
    """Rotation rate for this rung.  REFUSES an unlisted p_max >= 1 rather than guessing.

    The old fallback handed any unlisted p_max the Path-A default, the rate this file argues is
    too slow for p >= 1, so run_ladder(p_max=2) silently ran at a rate its own asserts reject.

    p_max=2 is unsupported because (D), JAX against the numpy NoLoop, exceeds TOL_NOLOOP at
    every configuration tried, and TOL_NOLOOP is absolute-only.  Supporting the rung means
    giving (D) the `abs OR rel` shape (C) already has -- a change to what the test ASSERTS,
    not a tolerance bump, and not a loosening of TOL_NOLOOP.  Add a CONFIG entry only together
    with that change and the measurements justifying it.

    DO NOT WRITE A MECHANISM FOR (D)'s SIZE HERE.  Three attempts were made and all three were
    refuted by measuring a second configuration.  Raising the rate does move (D); it does not
    move it far enough.

    Measurements: PR #163, and RIFT_roboto_paper analyses/slowrot_bound_violation/.
    """
    if p_max in CONFIG:
        return CONFIG[p_max]
    if p_max >= 1:
        raise ValueError(
            "no CONFIG entry for p_max=%r: this ladder's rate is chosen per rung, and the "
            "old fallback silently used the Path-A rate (INFL=1350), which p >= 1 asserts "
            "reject.  See this function's docstring for the p_max=2 measurements." % (p_max,))
    return Config()

TOL_BOUND = 1e-6           # nats above (1/2)<d|d> that we call a violation
TOL_DIRECT_ABS = 1e-6      # nats of disagreement with the explicit model
TOL_DIRECT_REL = 1e-6      # ... or, as a backstop, of 0.5<h|h>.  A BACKSTOP, not slack
                           # bought to make p_max=1 pass: both rungs clear the ABSOLUTE arm.
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
INV_DIST = fl.distMpcRef / (DLOUD / (lsu.lsu_PC * 1e6))


def _path_a_data(cfg):
    """The exact Path-A model F(u) * roll(hY, K_ARR) at this configuration's Omega."""
    F_of_u = sum(Atil[n] * np.exp(1j * n * cfg.omega * u_grid) for n in Atil)
    return _to_fd(np.real(F_of_u * np.roll(hY_data, K_ARR)),
                  lal.LIGOTimeGPS(epoch_intr + event_time), deltaT, N)


def delay_expansion_ratio(cfg):
    """max |2 pi f delta_tau| over the band: the p-expansion's convergence parameter.

    The p >= 1 bands are the Taylor series of h(t - delta_tau(t)) in the delay DRIFT
    delta_tau(t) = tau(t) - tau(tref), so the p-th band is smaller than the p-1'th by roughly
    this factor.  Above 1 the series diverges at the top of the band, and every construction
    that rebuilds the model from it -- including (C)'s explicit reference -- inherits that.

    It is a max over the whole u_grid evaluated at fmax, so it is an UPPER BOUND at the band
    edge.  What it licenses is refusing p >= 3, where the reconstruction blows up.  IT IS NOT
    THE REASON THE RUNG STOPS AT p_max = 1 -- at the shipped rate p = 2 is the most
    perturbative order of all, so this metric says nothing against it; p_max = 2 is
    unsupported for reasons that are config_for's business, not this metric's.

    Its TREND across rates is the informative part; a single value is a band-edge bound and
    says nothing on its own about which p you can afford.

    Measured norms per p_max: PR #163, and RIFT_roboto_paper analyses/slowrot_bound_violation/.
    """
    Bd = srr.delay_harmonics(lald.location, DEC)
    Btil = {m: Bd[m] * np.exp(1j * m * g_ev) for m in Bd}
    D = dict(Btil)
    D[0] = D[0] - np.real(sum(Btil.values()))
    dtau = sum(D[m] * np.exp(1j * m * cfg.omega * u_grid) for m in D)
    return 2.0 * np.pi * cfg.fmax * float(np.max(np.abs(np.real(dtau))))


def _Pv():
    Pv = Psig.manual_copy()
    for key, v in [('phi', RA), ('theta', DEC), ('incl', INCL), ('phiref', PHIREF),
                   ('psi', PSI), ('dist', DLOUD)]:
        setattr(Pv, key, np.ones(1) * v)
    Pv.tref = event_time; Pv.deltaT = deltaT
    return Pv


def rotation_lnL_t(f_sidereal, p_max, cfg):
    """(jax lnL(t), numpy NoLoop lnL(t), arrival sample offsets, a_list) on one shared bank."""
    P = Psig.manual_copy()
    data_dict = data_for(p_max, cfg)[1]
    bank = flwr.PrecomputeLikelihoodTermsWithRotation(
        event_time, t_window, P, data_dict, psd_dict, Lmax, cfg.fmax, harmonics=HARM,
        p_max=p_max, f_sidereal=f_sidereal, analyticPSD_Q=True, verbose=False, quiet=True,
        skip_interpolation=True)
    meta = bank[4]
    _harm = _harm_for(p_max)
    assert len(meta['a_list']) == (p_max + 1) * len(_harm), (
        "unexpected a_list size: %d bands for p_max=%d over %d harmonics"
        % (len(meta['a_list']), p_max, len(_harm)))
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
# with C~_a = C_a e^{i n_a Omega k dt} the arrival-time post-phase at arrival sample k
# (rotation_post_phase).
#
# THE SHIFT IS APPLIED TO THE MODULATED TEMPLATE, and that is not interchangeable with the
# obvious-looking alternative.  Analytically the post-phase cancels the shift inside the
# modulation -- C~_{(p,n)} e^{i n Omega (u - k dt)} = C_{(p,n)} e^{i n Omega u} -- so one is
# tempted to modulate on the UNROLLED grid and write
#     h(u) = invDist Re[ sum_p G_p(u) roll(hY^(p), k) ],  G_p(u) = sum_n C_{(p,n)} e^{i n Omega u}.
# But the shift here is CIRCULAR, and e^{i n Omega u} is not periodic on the segment, so the
# two forms differ by e^{i n Omega T_seg} on exactly the k samples that wrap the boundary.
# At p_max=0 that costs nothing -- hY^(0) is machine zero over the last K_ARR samples
# (1.2e-16 of its peak) -- but hY^(1) is NOT: the FD derivative leaves 5.9e-04 of its peak
# there, and the wrapped mismatch then shows up as ~1e-02 nats of disagreement with the
# bank, which computes the shift by FFT correlation and is circular in exactly this sense.
# See issue #159.  The post-phase is still applied EXPLICITLY below, so (C) keeps its teeth
# against a dropped rotation_post_phase (mutation numbers: PRs #117 and #163).
#
# At p_max=0 the sum reduces to F(u)*roll(hY,k), the numpy twin's construction (G_0 == F),
# and data_for() asserts that equality at 1e-12.
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


def _explicit_model_fd(k, p_max, a_list, cfg):
    """FD of h(u) above, for arrival sample k, at fiducial distance scaled by INV_DIST.

    ``a_list`` is the bank's band list and the sum is RESTRICTED to it.  Since #142/#143 the
    precompute WIDENS a too-narrow harmonic set to |n| <= 2 + p_max, so for a bank built that
    way the restriction is a no-op and nothing is dropped -- keep it anyway, because it is what
    makes this reference track the bank rather than assume it, and a bank built with
    widen_harmonics=False genuinely is a truncated model that this sum must match.

    Historical note, because the number is instructive: before #142 the bank had no band for
    the |n| = 3 coefficients at p_max=1, both evaluators silently dropped them, and summing the
    full coefficient dict here instead of restricting to a_list disagreed by 2.2e+05 nats at
    this configuration -- the dropped bands were the same order as the ones kept, because at
    INFL=1350, the rate this rung then ran at, the first-order delay term dominates.
    """
    C = flwr.rotation_coefficients(det, RA, DEC, PSI, event_time, p_max)   # {(p,n): C_a}
    keep = set((int(p), int(n)) for (p, n) in a_list)
    h_td = np.zeros(N, dtype=complex)
    for p in range(p_max + 1):
        hp = _hY_deriv(p)
        for (pa, na), c in C.items():
            if pa != p or (pa, na) not in keep:
                continue
            chi_a = np.exp(1j * na * cfg.omega * u_grid) * hp          # chi_a(u)
            post = np.exp(1j * na * cfg.omega * k * deltaT)            # rotation_post_phase
            h_td = h_td + c * post * np.roll(chi_a, k)                 # C~_a chi_a(u - k dt)
    return _to_fd(np.real(h_td) * INV_DIST, data_epoch, deltaT, N)


def data_for(p_max, cfg):
    """(data, data_dict, 0.5<d|d>, a_list) with the data EQUAL to the exact model at this p_max.

    That is what makes (B) maximally tight: with the data equal to the model the likelihood can
    represent, lnL at the true arrival sample sits exactly ON (1/2)<d|d>, leaving no slack for an
    inconsistency to hide in.  A p_max=0 dataset used against a p_max=1 bank would instead leave
    the p>=1 bands fitting nothing, and (B) would pass with 1e5 nats of margin.

    p_max=0 uses the INDEPENDENT construction above (srr.antenna_harmonics -> F(u) -> Re[F*roll]),
    which shares nothing with rotation_coefficients; the assert below pins the two together at
    p_max=0 so the p>=1 datasets inherit that provenance.
    """
    if p_max not in cfg._data_cache:
        a_list = flwr._elementary_index_set(_harm_for(p_max), p_max)
        if p_max == 0:
            d = _path_a_data(cfg)
            chk = _explicit_model_fd(K_ARR, 0, a_list, cfg)
            dd = np.max(np.abs(chk.data.data - d.data.data))
            ref = np.max(np.abs(d.data.data))
            assert dd <= 1e-12 * ref, (
                "the explicit model and the independent antenna_harmonics data construction "
                "disagree at p_max=0 by %g (rel %g) -- (C)'s reference is not the Path-A model"
                % (dd, dd / ref))
        else:
            d = _explicit_model_fd(K_ARR, p_max, a_list, cfg)
        cfg._data_cache[p_max] = (d, {det: d}, 0.5 * cfg.ipc.ip(d, d).real, a_list)
    return cfg._data_cache[p_max]


def run_ladder(p_max=0, cfg=None, verbose=True):
    """The (A)-(D) ladder at one p_max.  Returns a dict of the measured numbers."""
    if cfg is None:
        cfg = config_for(p_max)
    tag = "Path %s, p_max=%d" % ("A" if p_max == 0 else "B", p_max)
    data, _dd, HALF_DD, _al = data_for(p_max, cfg)
    if verbose:
        print("\n=== JAX SLOWROT CAUCHY-SCHWARZ (%s, A=%d bands, 0.5<d|d>=%.6f) ==="
              % (tag, len(_al), HALF_DD))
        print("    %s  arrival offset %+d samples (%+.2f ms)  max|2 pi f dtau| = %.3f"
              % (cfg, K_ARR, 1e3 * K_ARR * deltaT, delay_expansion_ratio(cfg)))

    # ------------------------------------------------------------ (A) teeth
    lnL_static, _, _, _ = rotation_lnL_t(0.0, p_max, cfg)
    static_deficit = HALF_DD - float(np.max(lnL_static))
    print("(A) rotation OFF vs rotating data: deficit = %.4f nats" % static_deficit)
    assert static_deficit > MIN_STATIC_DEFICIT, (
        "this configuration does not exercise rotation (static deficit %g <= %g), so the "
        "bound and direct-model checks below would be vacuous"
        % (static_deficit, MIN_STATIC_DEFICIT))

    # ------------------------------------------------------------ (B) the bound
    lnL_rot, lnL_noloop, kvals, a_list = rotation_lnL_t(cfg.fsid, p_max, cfg)
    overshoot = float(np.max(lnL_rot)) - HALF_DD
    jpeak = int(np.argmax(lnL_rot))
    print("(B) rotation ON : max lnL = %.6f at k=%+d   deficit = %+.6e"
          % (np.max(lnL_rot), kvals[jpeak], HALF_DD - np.max(lnL_rot)))
    assert kvals[jpeak] == K_ARR, (
        "lnL peaks at arrival sample %d, not the %d the data was built at -- the test is no "
        "longer sitting on the bound and (B) has lost its teeth" % (kvals[jpeak], K_ARR))
    assert overshoot <= TOL_BOUND, (
        "Cauchy-Schwarz VIOLATED: max JAX lnL exceeds 0.5<d|d> by %g nats.  lnL = <d|h> - "
        "(1/2)<h|h> cannot exceed (1/2)<d|d> for any h, so term1 and term2 are being "
        "evaluated for different templates -- see rotation_post_phase() and "
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
        hf = _explicit_model_fd(k, p_max, a_list, cfg)
        hh = cfg.ipc.ip(hf, hf).real
        lnL_direct = cfg.ipc.ip(hf, data).real - 0.5 * hh
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

    return dict(p_max=p_max, infl=cfg.infl, fmax=cfg.fmax, half_dd=HALF_DD,
                static_deficit=static_deficit, max_lnL=float(np.max(lnL_rot)),
                overshoot=overshoot, direct=worst, direct_rel=worst / scale,
                noloop=d_noloop, expansion_ratio=delay_expansion_ratio(cfg))


# pytest collects these; running the file as a script executes the same thing (see __main__).
def test_cauchy_schwarz_path_a():
    run_ladder(p_max=0)


def test_cauchy_schwarz_path_b():
    run_ladder(p_max=1)


if __name__ == "__main__":
    run_ladder(p_max=0)
    run_ladder(p_max=1)
    print("\nALL JAX SLOWROT CAUCHY-SCHWARZ CHECKS PASSED")
