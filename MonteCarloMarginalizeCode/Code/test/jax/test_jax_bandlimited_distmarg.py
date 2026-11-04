"""Band-limited time quadrature on the DISTANCE-marginalized JAX likelihood.

`fused_log_likelihood_distmarg` used to reduce over distance on the data time
grid and hand the reduced field to the terminal selector, which refuses
`bandlimited`: interpolating an already-reduced nonlinear field can converge to
the wrong function.  It now refines the complex primitive kappa(t) first and
applies the SAME distance quadrature at every refined node.

WHAT THE REFERENCE IS, AND WHY IT IS NOT THE CODE UNDER TEST.  The fine-grid
reference below reconstructs the primitive with a PLAIN PERIODIC zero-padded FFT
on a guarded window, reduces over distance with a numpy log-sum-exp, and
integrates with a numpy trapezoid.  Its extension is periodic where the shipped
one is an even reflection, and its guard taper is a half-cosine on the offset
grid (k+1)/(g+1) where the shipped one is on k/g, so agreement is not two
spellings of one routine.  It is checked for convergence in BOTH the guard and
the refinement factor, because a reference that has not converged is not one.

The taper is not optional and is not decoration.  An UNTAPERED periodic
reconstruction leaves a step at the periodic seam whose Gibbs ringing decays
only like 1/guard: doubling the guard halves the error instead of removing it,
and the sequence never converges to the tolerance this file asserts.  That is
what the reference's own convergence test would have caught, and it is why the
reference tapers.  The DESIGN record carries the untapered ladder.

A sample-rate ladder cannot serve as that reference here and the numbers say
why: the integrand's width is sigma_t ~ 1/(2 pi rho sigma_f), so Simpson at 8x
the native rate is still coarse at production amplitude.  What the ladder DOES
show, and is asserted below, is that native Simpson moves monotonically toward
the band-limited value as the rate rises -- which is the claim the option makes.

Numbers and method: RIFT/likelihood/jax_ile/DESIGN_jax_bandlimited_distmarg.md.
"""
import inspect
import pathlib
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

lal = pytest.importorskip("lal")
lalsim = pytest.importorskip("lalsimulation")

import RIFT.lalsimutils as lalsimutils                              # noqa: E402
from RIFT.likelihood.jax_ile import core, wrapper                   # noqa: E402
from RIFT.likelihood.jax_ile.wrapper import (                       # noqa: E402
    JAXDistanceMarginalizedLikelihood, build_data_from_precompute,
    bandlimited_storage_requirement)

MSUN, PC = lal.MSUN_SI, lal.PC_SI
_CODE = pathlib.Path(__file__).parents[2]

EPOCH = 1126259462.0
DETECTORS = ("H1", "L1")
IWH = 0.075                       # marginalization half-window, seconds
FMIN, FREF, FMAX = 40.0, 40.0, 300.0
DELTAF = 0.25
SRATE = 1024.0
ANGLES = (1.2, -0.4, 0.7, 0.9, 2.1)
D_MIN, D_MAX, N_GRID = 50.0, 4000.0, 512
# The injected angles are NOT where this likelihood peaks, and every test here
# evaluates at a fixed point, so none needs them to be.  The precompute below
# receives the same P as the injection, and for IMRPhenomD the template route
# (hlmoft -> SimInspiralTDModesFromPolarizations) bakes that P's phiref and psi
# into the modes; ILE then applies both again through Y_lm and F.  Production
# zeroes P.phiref/P.psi before the precompute (batchmode driver, and the JAX
# driver's make_template).  Measured on the 900 Mpc case: lnL 84.0 at the
# injection against a 2-D (psi, phiref) maximum of 194.9; the conventional
# factored likelihood gives the same numbers, so this is the waveform interface,
# not the quadrature.

# Injected distances.  Named by the amplitude they produce, and the fixture
# asserts the amplitude rather than trusting the label.
DIST_QUIET, DIST_LOUD = 390.0, 48.5

# Agreement between the shipped quadrature and the independently reconstructed
# fine-grid reference, in nats.  Measured; see the DESIGN record.
TOL_REFERENCE = 1e-3


def _params(dist_mpc, srate):
    P = lalsimutils.ChooseWaveformParams()
    P.m1, P.m2 = 35.0 * MSUN, 30.0 * MSUN
    P.s1z, P.s2z = 0.1, -0.2
    P.fmin, P.fref = FMIN, FREF
    P.deltaT = 1.0 / srate
    P.deltaF = DELTAF
    P.dist = dist_mpc * 1e6 * PC
    P.fmax = 0.0
    P.approx = lalsim.IMRPhenomD
    P.radec = True
    P.tref = EPOCH
    P.phi, P.theta, P.psi, P.incl, P.phiref = ANGLES
    return P


def _build(dist_mpc, srate, storage_half):
    P = _params(dist_mpc, srate)
    data_dict, psd_dict = {}, {}
    for det in DETECTORS:
        Pdet = P.copy()
        Pdet.detector = det
        data_dict[det] = lalsimutils.non_herm_hoff(Pdet)
        psd_dict[det] = lalsim.SimNoisePSDaLIGOZeroDetHighPower
    data, extras = build_data_from_precompute(
        P.copy(), data_dict, psd_dict, EPOCH, storage_half, IWH, 2, FMAX,
        analyticPSD_Q=True, verbose=False)
    extras["network_snr"] = _network_snr(P, data_dict)
    return data, extras


def _network_snr(P, data_dict):
    """Optimal network SNR of the zero-noise data, sqrt(sum_det <d|d>), on the
    band the likelihood integrates.  Not ``extras["guess_snr"]``: that is the
    precompute's deliberately deflated estimate, sqrt(sum max|Q_lm|^2 / U_lm)
    divided by 2.3, and sits 2.305x below this on every fixture here."""
    IP = lalsimutils.ComplexIP(
        fLow=FMIN, fNyq=0.5 / P.deltaT, deltaF=P.deltaF, fMax=FMAX,
        psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower, analyticPSD_Q=True)
    return float(np.sqrt(sum(IP.norm(d) ** 2 for d in data_dict.values())))


#: Guard and refinement factor at which the numpy reference is converged; both
#: are certified below by halving them.  See the DESIGN record for the ladders.
REF_FACTOR = 512


def _reference_guard(srate=SRATE):
    """Guard for the numpy reference gathers: one doubling past the certified one."""
    return 2 * bandlimited_storage_requirement(1.0 / srate, IWH)[2]


def _storage_half(guard, srate=SRATE):
    return IWH + guard / srate + 0.05 + 16.0 / srate


def _angles():
    return [np.atleast_1d(np.asarray(v, dtype=float)) for v in ANGLES]


def _like(data, quadrature):
    return JAXDistanceMarginalizedLikelihood(
        data, D_MIN, D_MAX, n_grid=N_GRID, time_quadrature=quadrature)


def _value(like):
    return float(np.asarray(like.log_likelihood(*_angles()))[0])


# --------------------------------------------------------------------------
# Independent numpy reference
# --------------------------------------------------------------------------
def _periodic_fft_upsample(x, factor):
    """Plain periodic zero-padded FFT interpolation, Nyquist bin split evenly.

    Deliberately NOT the shipped even extension: the reference must not be able
    to inherit a boundary-handling mistake from the code it certifies.  Its own
    periodic seam is pushed far outside the integrated window by the guard.
    """
    n = x.shape[-1]
    X = np.fft.fft(x)
    half, n_out = n // 2, n * factor
    if n % 2 == 0:
        nyq = X[half:half + 1] * 0.5
        Y = np.concatenate([X[:half], nyq,
                            np.zeros(n_out - n - 1, dtype=complex), nyq,
                            X[half + 1:]])
    else:
        Y = np.concatenate([X[:half + 1], np.zeros(n_out - n, dtype=complex),
                            X[half + 1:]])
    return np.fft.ifft(Y) * factor


def _numpy_distance_reduction(K, R, x_grid, log_w, block=16):
    """log sum_g exp(K x_g - R x_g^2 / 2 + log w_g), blocked, in numpy."""
    m = np.full(K.shape, -np.inf)
    s = np.zeros(K.shape)
    for start in range(0, len(x_grid), block):
        xs = x_grid[start:start + block]
        e = (K[:, None] * xs[None, :]
             - 0.5 * R[:, None] * np.square(xs)[None, :]
             + log_w[start:start + block][None, :])
        m_b = np.max(e, axis=-1)
        s_b = np.sum(np.exp(e - m_b[:, None]), axis=-1)
        m_new = np.maximum(m, m_b)
        s = s * np.exp(m - m_new) + s_b * np.exp(m_b - m_new)
        m = m_new
    return m + np.log(s)


def _numpy_log_trapezoid(lnL, dx):
    peak = np.max(lnL)
    y = np.exp(lnL - peak)
    return peak + np.log(dx * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1]))


def _taper_guard(kappa, guard):
    """Half-cosine ramp over the guard samples, on the OFFSET grid (k+1)/(g+1).

    The shipped taper ramps on k/g.  Both drive the seam to zero; neither is the
    other, so a mistake in the shipped window shape cannot be inherited here.
    """
    if not guard:
        return kappa
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(1, guard + 1) / (guard + 1)))
    w = np.concatenate([ramp, np.ones(kappa.shape[-1] - 2 * guard), ramp[::-1]])
    return kappa * w


def _fine_grid_reference(data, x_grid, log_w, guard, factor, angles=None):
    angles = _angles() if angles is None else angles
    kappa, rho = core._accumulate_unit(
        data, *angles, core.JAX_INTERP_DEFAULT, False, guard=guard)
    kappa = np.asarray(kappa)[0]
    rho = np.asarray(rho)[0]
    npts = int(data.npts)
    fine = _periodic_fft_upsample(_taper_guard(kappa, guard), factor)
    keep = slice(guard * factor, guard * factor + (npts - 1) * factor + 1)
    K = fine[keep].real
    R = np.full(K.shape, rho[guard])
    lnL = _numpy_distance_reduction(K, R, x_grid, log_w)
    return _numpy_log_trapezoid(lnL, float(data.deltaT) / factor)


def _reduce_then_refine(data, x_grid, log_w, factor):
    """The WRONG order, built explicitly: reduce on the data grid, then refine.

    This is what `_time_marginalize_terminal` refuses, and what the code would
    do if the primitive gather were dropped.  Nothing asserts it is close to
    anything; it exists so the ordering assertions below are load-bearing.
    """
    kappa, rho = core._accumulate_unit(
        data, *_angles(), core.JAX_INTERP_DEFAULT, False, guard=0)
    K = np.asarray(kappa)[0].real
    R = np.asarray(rho)[0]
    lnL_t = _numpy_distance_reduction(K, R, x_grid, log_w)
    fine = _periodic_fft_upsample(lnL_t.astype(complex), factor).real
    fine = fine[:(len(lnL_t) - 1) * factor + 1]
    return _numpy_log_trapezoid(fine, float(data.deltaT) / factor)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cases():
    guard = _reference_guard()
    out = {}
    for tag, dist in (("quiet", DIST_QUIET), ("loud", DIST_LOUD)):
        data, extras = _build(dist, SRATE, _storage_half(guard))
        like_s = _like(data, "simpson")
        out[tag] = dict(
            data=data, snr=float(extras["network_snr"]),
            x_grid=np.asarray(like_s.x_grid),
            log_w=np.asarray(like_s.log_w_grid),
            simpson=_value(like_s),
            bandlimited=_value(_like(data, "bandlimited")),
            guard=guard)
    return out


def test_fixture_amplitudes_are_the_two_regimes_claimed(cases):
    """Guard on the fixture.  Both assertions below are amplitude claims, and a
    distance that quietly stopped producing the intended SNR would make the
    'differs at high amplitude' test pass or fail for the wrong reason.
    The bounds are on the network optimal SNR of the data (measured 45.9 and
    369.2), not on ``guess_snr``, which is 2.305x lower by construction."""
    assert 40.0 < cases["quiet"]["snr"] < 55.0, cases["quiet"]["snr"]
    assert cases["loud"]["snr"] > 300.0, cases["loud"]["snr"]


# --------------------------------------------------------------------------
# (a) agreement with a converged, independently built reference
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tag", ["quiet", "loud"])
def test_bandlimited_distmarg_matches_the_independent_fine_grid_reference(
        cases, tag):
    case = cases[tag]
    ref = _fine_grid_reference(case["data"], case["x_grid"], case["log_w"],
                               case["guard"], REF_FACTOR)
    assert abs(case["bandlimited"] - ref) < TOL_REFERENCE, (
        "%s: bandlimited %.6f vs independent reference %.6f (%+.3e nats)"
        % (tag, case["bandlimited"], ref, case["bandlimited"] - ref))


@pytest.mark.parametrize("tag", ["quiet", "loud"])
def test_the_reference_is_converged_in_both_of_its_own_knobs(cases, tag):
    """An unconverged reference certifies nothing.  Halving the factor and
    halving the guard must each leave it alone at the tolerance the agreement
    test uses."""
    case = cases[tag]
    args = (case["data"], case["x_grid"], case["log_w"])
    full = _fine_grid_reference(*args, guard=case["guard"], factor=REF_FACTOR)
    coarse_factor = _fine_grid_reference(*args, guard=case["guard"],
                                         factor=REF_FACTOR // 2)
    narrow_guard = _fine_grid_reference(*args, guard=case["guard"] // 2,
                                        factor=REF_FACTOR)
    assert abs(full - coarse_factor) < TOL_REFERENCE, (
        "%s: reference not converged in the factor (%+.3e nats)"
        % (tag, full - coarse_factor))
    assert abs(full - narrow_guard) < TOL_REFERENCE, (
        "%s: reference not converged in the guard (%+.3e nats)"
        % (tag, full - narrow_guard))


# --------------------------------------------------------------------------
# (b) the knob is live
# --------------------------------------------------------------------------
def test_bandlimited_differs_from_native_simpson_at_high_amplitude(cases):
    """If this stops biting, the option is inert and everything above is
    measuring the Simpson value twice."""
    loud = cases["loud"]
    delta = loud["bandlimited"] - loud["simpson"]
    assert abs(delta) > 100 * TOL_REFERENCE, (
        "bandlimited and native Simpson agree to %+.3e nats at rho %.1f; the "
        "option is not changing the answer" % (delta, loud["snr"]))


def test_native_simpson_moves_toward_the_bandlimited_value_as_the_rate_rises(
        cases):
    """The physical claim, on regenerated data: Simpson's error is a resolution
    error, so refining the DATA rate must close the gap the option closes at the
    native rate.  Not a converged reference -- 8x is still coarse against
    sigma_t ~ 1/(2 pi rho sigma_f) -- which is why the assertion is on the
    ordering of the residuals and not on their size."""
    loud = cases["loud"]
    target = loud["bandlimited"]
    gaps = [abs(loud["simpson"] - target)]
    guard = bandlimited_storage_requirement(1.0 / SRATE, IWH)[2]
    for mult in (4, 8):
        data, _ = _build(DIST_LOUD, SRATE * mult, _storage_half(guard))
        gaps.append(abs(_value(_like(data, "simpson")) - target))
    assert gaps[1] < gaps[0], "4x Simpson did not improve on native: %r" % (gaps,)
    assert gaps[2] < gaps[1], "8x Simpson did not improve on 4x: %r" % (gaps,)


def test_the_reduction_order_is_what_carries_the_agreement(cases):
    """Reduce-then-refine is a different number, not a rounding difference.

    This is the assertion that was mutation-tested: swapping the shipped order
    makes the agreement test above fail by orders of magnitude, and this records
    by how much rather than leaving it to the reviewer to imagine."""
    loud = cases["loud"]
    wrong = _reduce_then_refine(loud["data"], loud["x_grid"], loud["log_w"],
                                REF_FACTOR)
    assert abs(loud["bandlimited"] - wrong) > 100 * TOL_REFERENCE, (
        "refining the already-reduced field lands within %.3e nats of the "
        "shipped answer, so the ordering assertions prove nothing"
        % abs(loud["bandlimited"] - wrong))


# --------------------------------------------------------------------------
# (c) the refusal set still refuses, and the enabled one no longer does
# --------------------------------------------------------------------------
def test_the_refusal_set_still_refuses_and_names_the_endpoint(cases):
    data = cases["quiet"]["data"]
    refusing = [
        (wrapper.JAXDistPhiMargLikelihood, dict(nphi=4)),
        (wrapper.JAXDistPsiMargLikelihood, dict(npsi=4)),
        (wrapper.JAXDistPhiPsiMargLikelihood, dict(nphi=4, npsi=4,
                                                   angle_marg="grid")),
    ]
    for cls, kwargs in refusing:
        with pytest.raises(ValueError, match="primitive time fields"):
            cls(data, D_MIN, D_MAX, n_grid=32,
                time_quadrature="bandlimited", **kwargs)
        # ... and the same construction is fine on the default quadrature, so
        # the refusal is about the COMBINATION and not about the wrapper.
        cls(data, D_MIN, D_MAX, n_grid=32, **kwargs)


def test_only_the_angle_wrappers_call_the_nonlinear_refusal(cases):
    """The distance wrapper must not merely stop raising -- it must stop calling
    the validator, or a later edit re-enabling it would be invisible here."""
    src = inspect.getsource(wrapper.JAXDistanceMarginalizedLikelihood.__init__)
    assert "_validate_nonlinear_time_quadrature" not in src
    for cls in (wrapper.JAXDistPhiMargLikelihood,
                wrapper.JAXDistPsiMargLikelihood,
                wrapper.JAXDistPhiPsiMargLikelihood):
        assert "_validate_nonlinear_time_quadrature" in inspect.getsource(
            cls.__init__), cls.__name__


def test_bandlimited_distmarg_refuses_what_it_cannot_honour(cases):
    """Two fail-closed doors.  `return_lnLt` has no meaning once the reduction
    moves to the refined grid, and rotation data's norm depends on arrival time,
    which the refinement holds fixed."""
    import types
    data = cases["quiet"]["data"]
    like = _like(data, "simpson")
    with pytest.raises(ValueError, match="return_lnLt"):
        core.fused_log_likelihood_distmarg(
            data, *_angles(), like.x_grid, like.log_w_grid,
            time_quadrature="bandlimited", return_lnLt=True)
    with pytest.raises(ValueError, match="arrival time"):
        core.fused_log_likelihood_distmarg(
            types.SimpleNamespace(feature="rotation"), *_angles(),
            like.x_grid, like.log_w_grid, time_quadrature="bandlimited")


def test_the_distance_reduction_has_one_definition_for_both_grids():
    """Coarse and refined paths must reach the same quadrature helper.  A second
    copy is how a future adaptive/GH distance branch would land on one grid and
    not the other -- silently, since both would still return a number."""
    src = inspect.getsource(core.fused_log_likelihood_distmarg)
    assert "_logsumexp_grid_blocked(" in src and "_logsumexp_grid_scanned(" in src
    assert "reduce_fn=_reduce" in src


def test_the_guard_pair_has_one_definition(cases):
    """Gathered support and integrated guard must be the same number.  Three
    sites need it; a re-typed copy is a wrong likelihood, not an error."""
    for fn in (core.fused_log_likelihood, core.fused_log_likelihood_distmarg,
               wrapper.bandlimited_storage_requirement):
        assert "bandlimited_time_guard(" in inspect.getsource(fn), fn.__name__
    npts = int(cases["quiet"]["data"].npts)
    g0, gcert = core.bandlimited_time_guard(npts)
    assert gcert == 2 * g0 and g0 >= core.default_time_guard(npts)
    like = _like(cases["quiet"]["data"], "bandlimited")
    assert (like.time_guard_initial, like.time_guard_certified) == (g0, gcert)


# --------------------------------------------------------------------------
# (d) the endpoint certificate on a floored field
# --------------------------------------------------------------------------
# The distance-marginalized field has a floor: at every node the distance sum
# is at least the far-distance prior mass, so a row's peak-to-endpoint contrast
# is bounded by its own peak height and the fixed-distance kernel's 15-nat
# endpoint gap rejects every low-contrast row -- converged or not.  Blind
# full-sky draws, which every prior-seeded driver mode evaluates by the
# thousand, are mostly low-contrast rows.  Measured on 256 such rows at 20 ms
# half-window: 35% rejected by the gap alone, all of them agreeing with the
# independent reference to 1e-4 nat.  The distance path therefore runs with
# that certificate off and the guard-agreement and doubling certificates on.
# Numbers: the DESIGN record, "The endpoint certificate".
N_BLIND = 128
#: The blind census runs at this distance, not on the ``quiet`` fixture: the
#: floor argument needs low-contrast rows, and at 390 Mpc the gap rejects none
#: of 256 (measured); at 900 Mpc it rejects 14 of 256 at this half-window.
DIST_BLIND = 900.0


def _blind_draws(n, seed):
    """The driver's ``sample_prior`` for the five angles, re-typed on purpose:
    the point is a full-sky, isotropic-orientation draw, not a driver import."""
    rng = np.random.default_rng(seed)
    return [rng.uniform(0.0, 2 * np.pi, n), np.arcsin(rng.uniform(-1.0, 1.0, n)),
            rng.uniform(0.0, np.pi, n), np.arccos(rng.uniform(-1.0, 1.0, n)),
            rng.uniform(0.0, 2 * np.pi, n)]


def _primitive_with_gap(like, angles, endpoint_log_gap):
    """The shipped refinement on the distance field with a CHOSEN endpoint gap.

    Rebuilds ``fused_log_likelihood_distmarg``'s reduction so the certificate
    can be switched without touching the module; the wrapper's own value is
    asserted equal to the ``None`` setting below, so this helper cannot drift
    from the code silently."""
    import jax.numpy as jnp
    data = like.data
    guard = like.time_guard_certified
    kappa, rho = core._accumulate_unit(
        data, *[jnp.asarray(v) for v in angles], like.interp, False, guard=guard)
    a = jnp.asarray(like.x_grid)
    b = -0.5 * jnp.square(a)
    log_w = jnp.asarray(like.log_w_grid)

    def reduce_fn(k, r):
        kk = k.real
        n = int(np.prod(kk.shape))
        block = min(max(1, core._BANDLIMITED_GRID_ELEMENTS // max(n, 1)),
                    int(a.shape[0]))
        return core._logsumexp_grid_scanned(
            kk.reshape(n), r.reshape(n), a, b, log_w, block).reshape(kk.shape)

    return np.asarray(core._time_marginalize_reflected_primitive(
        kappa, rho, data.deltaT, False, guard=guard, reduce_fn=reduce_fn,
        endpoint_log_gap=endpoint_log_gap))


@pytest.fixture(scope="module")
def blind():
    data, _ = _build(DIST_BLIND, SRATE, _storage_half(_reference_guard()))
    like = _like(data, "bandlimited")
    angles = _blind_draws(N_BLIND, seed=0)
    return dict(
        data=data, like=like, angles=angles,
        shipped=np.asarray(like.log_likelihood(*angles)),
        gap_off=_primitive_with_gap(like, angles, None),
        gap_on=_primitive_with_gap(like, angles, core._TIME_ENDPOINT_LOG_GAP_MIN))


def test_blind_draws_are_certified_without_the_endpoint_gap(blind):
    """Every blind row gets a number from the shipped wrapper, and that number
    is the ``endpoint_log_gap=None`` refinement and nothing else."""
    assert np.all(np.isfinite(blind["shipped"])), (
        "%d of %d blind rows uncertified with the endpoint gap off"
        % (int(np.sum(~np.isfinite(blind["shipped"]))), N_BLIND))
    # The wrapper is jitted and the direct call is not; XLA fusion moves the
    # last bits, so this is a tolerance and not an equality.
    np.testing.assert_allclose(blind["shipped"], blind["gap_off"], rtol=0, atol=1e-8)


def test_the_endpoint_gap_was_rejecting_converged_rows(blind):
    """The certificate is live on this field (it rejects a material fraction),
    and what it rejects agrees with the independent reference.  Without the
    first assertion the second would be vacuous; without the second the first
    would only show the gate is loud."""
    rejected = np.where(np.isnan(blind["gap_on"]) & np.isfinite(blind["gap_off"]))[0]
    assert len(rejected) >= 4, (
        "the 15-nat gap rejected only %d of %d blind rows; the floor argument "
        "is not exercised by this draw" % (len(rejected), N_BLIND))
    guard = _reference_guard()
    like = blind["like"]
    worst = 0.0
    for i in rejected[:8]:
        angles = [np.atleast_1d(v[i]) for v in blind["angles"]]
        ref = _fine_grid_reference(blind["data"], np.asarray(like.x_grid),
                                   np.asarray(like.log_w_grid), guard,
                                   REF_FACTOR, angles=angles)
        worst = max(worst, abs(blind["gap_off"][i] - ref))
        assert abs(blind["gap_off"][i] - ref) < TOL_REFERENCE, (
            "rejected row %d: refinement %.6f vs reference %.6f"
            % (i, blind["gap_off"][i], ref))


def test_no_production_caller_applies_the_endpoint_gap():
    """Both fields run without the endpoint gap.  The fixed-distance kernel
    kept it when this file was written; the follow-up of 2026-09-08 measured
    the same signature there (test_jax_bandlimited_6d_blind.py; DESIGN record,
    "The fixed-distance kernel").  The kernel default is ``None``, the
    threshold constant stays for the tests that pin what the gap rejected, and
    no production caller passes one."""
    sig = inspect.signature(core._time_marginalize_reflected_primitive)
    assert sig.parameters["endpoint_log_gap"].default is None
    assert core._TIME_ENDPOINT_LOG_GAP_MIN == 15.0
    assert "endpoint_log_gap" not in inspect.getsource(core.fused_log_likelihood)
    assert "endpoint_log_gap=None" in inspect.getsource(core.fused_log_likelihood_distmarg)


# --------------------------------------------------------------------------
# (e) the driver
# --------------------------------------------------------------------------
def _load_driver():
    import importlib.machinery
    import importlib.util
    path = _CODE / "bin" / "integrate_likelihood_extrinsic_jax"
    loader = importlib.machinery.SourceFileLoader("_jax_bl_distmarg_driver", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_driver_eval_lnL_still_fails_closed_and_names_the_rows():
    """The hard stop on an uncertified row is kept (no coarse likelihood is
    substituted); what changed is that the message counts the rows and prints
    their parameters, so a failed run says where it failed."""
    import types
    drv = _load_driver()
    like = types.SimpleNamespace(
        time_quadrature="bandlimited",
        log_likelihood=lambda *cols: np.array([1.0, np.nan, 3.0]))
    opts = types.SimpleNamespace(n_chunk=8000)
    theta = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]] * 3)
    theta[1, 0] = 2.5
    with pytest.raises(RuntimeError) as err:
        drv.eval_lnL(like, theta, opts, with_distance=False)
    msg = str(err.value)
    assert "1 of 3 rows" in msg and "ra=2.5000" in msg
    assert "no coarse likelihood is substituted" in msg


@pytest.mark.parametrize("mode", ["prior-mc", "map"])
def test_driver_prior_seeded_modes_run_distance_marginalized_bandlimited(
        tmp_path, mode):
    """The modes that evaluate blind prior draws through the driver's own
    ``eval_lnL`` -- which stops the run on one uncertified row.  Before the
    endpoint change, every seed of this command failed inside the first chunk
    (measured: 2-7 uncertified rows per 16 draws, seeds 0-29).  ``map`` is
    the one whose pilot is hard-coded to 4000 blind draws.  No flowMC
    dependency, so this is the executable coverage the CI ``jax-ile-check``
    job actually runs.

    ``laplace-is`` is deliberately not here.  On this injection it gets
    through every evaluation and then refuses its own evidence (adapted lnZ
    below the prior pilot's Markov floor, neff 6 at n_max 4000) while the
    Simpson control at the same budget passes with neff 16: the resolved peak
    is narrower than its single moment-matched Gaussian covers.  That is the
    estimator's documented failure, not a certificate, and a test of it would
    be a test of luck.

    The half-window is 50 ms, not the 20 ms of the flowMC test above: a
    wrong-sky draw shifts a detector's arrival by up to 2 R_earth / c, about
    43 ms, and a row whose arrival peak sits at the window edge is one the
    trapezoid and guard certificates legitimately cannot converge on."""
    import os
    out = tmp_path / "ile"
    env = dict(os.environ, PYTHONPATH=str(_CODE), OMP_NUM_THREADS="1",
               JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    proc = subprocess.run(
        [sys.executable, str(_CODE / "bin" / "integrate_likelihood_extrinsic_jax"),
         "--inj-mode", "--mass1", "35", "--mass2", "30",
         "--inj-deltaF", "0.25", "--inj-detectors", "H1,L1",
         "--inj-distance", "900",
         "--fmin-template", "40", "--reference-freq", "40", "--fmax", "300",
         "--l-max", "2", "--approximant", "IMRPhenomD", "--srate", "1024",
         "--data-integration-window-half", "0.05",
         "--internal-data-storage-window-half", "0.08",
         "--d-min", "50", "--d-max", "4000", "--distance-grid-points", "32",
         "--distance-marginalization", "--mode", mode,
         "--time-marginalization-quadrature", "bandlimited",
         "--n-max", "400", "--seed", "3", "--output-file", str(out)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=3600)
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, log[-3000:]
    assert "failed a certificate" not in log, log[-3000:]
    assert "time-marginalization quadrature: bandlimited" in log, log[-3000:]
    row = np.atleast_2d(np.loadtxt(str(out) + "_0_.dat"))
    assert row.shape[1] == 13 and np.isfinite(row[0, 9]), row
    if mode == "prior-mc":
        assert np.isfinite(row[0, 12]) and row[0, 11] == 400, row


# --------------------------------------------------------------------------
# (f) the flowMC driver path
# --------------------------------------------------------------------------
def test_driver_runs_flowmc_distance_marginalized_bandlimited(tmp_path):
    """End to end through the shipped executable, small budget.  A library test
    cannot see the storage-window widening, the option plumbing, or the fact
    that the run publishes a row rather than a NaN.

    flowMC is an optional dependency the CI ``jax-ile-check`` job does not
    install, so ``pytest.importorskip`` below is a real skip there, not a
    pass; this is still full executable coverage on any host that has flowMC
    (e.g. local development).  A non-flowMC ``--mode`` (laplace-is, prior-mc,
    map, nuts) was tried here first and dropped: all four reach the SAME
    ``JAXDistanceMarginalizedLikelihood(..., time_quadrature="bandlimited")``
    construction (driver ``analyze_one``, the ``elif
    opts.distance_marginalization:`` branch) via a blind full-sky prior draw,
    and MEASURED against this exact injection that draw's reflected-FFT
    convergence check hard-fails (no coarse likelihood substituted, by
    design) on roughly a quarter of blind draws regardless of seed -- so a
    small ``--n-max`` does not make the combination reliable, only rarer to
    catch in one CI run.  See the flagged follow-up on this fragility.

    The budget is set by memory, not by wall clock.  flowMC unrolls its
    per-step proposal, so the compiled graph -- already eleven refinement
    branches wide -- is multiplied by ``--n-local-steps``; at twenty steps this
    same run needs 23 GB, and at two it needs 4.5 GB.  ``--n-prior-pilot`` is
    what actually brackets the peak here, so the steps are what gets cut.
    ``--internal-data-storage-window-half`` is set BELOW the band-limited
    requirement on purpose, so the auto-widening has something to do and the
    assertion on it is not vacuous."""
    pytest.importorskip("flowMC")
    import os
    out = tmp_path / "ile"
    env = dict(os.environ, PYTHONPATH=str(_CODE), OMP_NUM_THREADS="1",
               JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    proc = subprocess.run(
        [sys.executable, str(_CODE / "bin" / "integrate_likelihood_extrinsic_jax"),
         "--inj-mode", "--mass1", "35", "--mass2", "30",
         "--inj-deltaF", "0.25", "--inj-detectors", "H1,L1",
         "--inj-distance", "900",
         "--fmin-template", "40", "--reference-freq", "40", "--fmax", "300",
         "--l-max", "2", "--approximant", "IMRPhenomD", "--srate", "1024",
         "--data-integration-window-half", "0.02",
         "--internal-data-storage-window-half", "0.08",
         "--d-min", "50", "--d-max", "4000", "--distance-grid-points", "32",
         "--distance-marginalization", "--mode", "flowmc",
         "--time-marginalization-quadrature", "bandlimited",
         "--n-training-loops", "1", "--n-production-loops", "1",
         "--n-epochs", "2", "--n-local-steps", "2", "--n-global-steps", "2",
         "--n-prior-pilot", "2000", "--output-file", str(out)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=3600)
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, log[-3000:]
    assert "is not valid for" not in log, log[-3000:]
    assert "widening rholm storage for bandlimited time support" in log, (
        "the storage window was not widened for the distance-marginalized "
        "bandlimited run: %s" % log[-3000:])
    assert "time-marginalization quadrature: bandlimited" in log, log[-3000:]
    row = np.atleast_2d(np.loadtxt(str(out) + "_0_.dat"))
    assert row.shape[1] == 13 and np.isfinite(row[0, 9]), row
