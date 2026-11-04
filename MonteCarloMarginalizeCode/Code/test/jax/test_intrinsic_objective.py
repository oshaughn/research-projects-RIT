"""IntrinsicMarginalObjective: one compile across intrinsic points, same values as the stock wrapper."""
import numpy as np
import pytest
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lalsimutils
from RIFT.likelihood.jax_ile import build_data_from_precompute
from RIFT.likelihood.jax_ile.wrapper import JAXDistPhiPsiMargLikelihood
from RIFT.likelihood.jax_ile.intrinsic_objective import IntrinsicMarginalObjective

MSUN, PC = lal.MSUN_SI, lal.PC_SI
FID = 1126259462.0


def _P(m1, m2, s1x=0.0):
    P = lalsimutils.ChooseWaveformParams()
    P.m1, P.m2 = m1 * MSUN, m2 * MSUN
    P.s1x, P.s1z, P.s2z = s1x, 0.1, -0.2
    P.fmin = P.fref = 30.0
    P.deltaT, P.deltaF = 1.0 / 2048, 0.25
    # Not IMRPhenomPv2: RIFT builds its modes in hlmoft_IMRPv2_dict, whose (2,-2) mode
    # from the inclination-pi evaluation is non-finite in ~1 of 10 fresh processes.
    P.approx = lalsim.IMRPhenomXPHM
    P.radec = True
    P.tref = FID
    return P


def _data():
    P = _P(35, 30, 0.3)
    P.phi, P.theta, P.psi, P.incl, P.phiref = 1.2, -0.4, 0.7, 0.9, 2.1
    P.dist = 600e6 * PC
    dd, pp = {}, {}
    for det in ("H1", "L1", "V1"):
        Pd = P.copy(); Pd.detector = det
        dd[det] = lalsimutils.non_herm_hoff(Pd)
        pp[det] = lalsim.SimNoisePSDaLIGOZeroDetHighPower
    return dd, pp


def test_objective_matches_wrapper_and_compiles_once():
    dd, pp = _data()
    obj = IntrinsicMarginalObjective(
        dd, pp, _P(35, 30), FID, 1.0, 5000.0, l_max=2, fmax=1000.0, n_sky=64,
        nphi=8, npsi=4, n_dist=64, waveform_kwargs=dict(ignore_threshold=None),
        analyticPSD_Q=True)
    rng = np.random.default_rng(3)
    ra, dec, incl = rng.uniform(0, 2 * np.pi, 64), rng.uniform(-1, 1, 64), rng.uniform(0, np.pi, 64)
    for P in (_P(35, 30, 0.3), _P(36, 29, 0.0)):
        mine, _ = obj.lnL_sky(P, ra, dec, incl)
        data, _ = build_data_from_precompute(
            obj._prepare(P), dd, pp, FID, 0.15, 0.075, 2, 1000.0, analyticPSD_Q=True,
            ignore_threshold=None)
        ref = JAXDistPhiPsiMargLikelihood(data, 1.0, 5000.0, nphi=8, npsi=4, n_grid=64,
                                          angle_marg="grid")
        ref_v = np.asarray(ref.log_likelihood(ra, dec, incl))
        assert np.allclose(mine, ref_v, rtol=0, atol=1e-8), np.max(np.abs(mine - ref_v))
    assert obj.n_compiles == 1


def test_device_precompute_matches_cpu_numpy_backend():
    """Device path (numpy backend of gpu_precompute) reproduces the CPU precompute."""
    dd, pp = _data()
    kw = dict(l_max=2, fmax=1000.0, n_sky=64, nphi=8, npsi=4, n_dist=64,
              waveform_kwargs=dict(ignore_threshold=None), analyticPSD_Q=True, seed=5)
    cpu = IntrinsicMarginalObjective(dd, pp, _P(35, 30), FID, 1.0, 5000.0, **kw)
    dev = IntrinsicMarginalObjective(dd, pp, _P(35, 30), FID, 1.0, 5000.0,
                                     device_precompute="numpy", **kw)
    rng = np.random.default_rng(4)
    ra, dec, incl = rng.uniform(0, 2 * np.pi, 64), rng.uniform(-1, 1, 64), rng.uniform(0, np.pi, 64)
    dev.lnL_sky(_P(35, 30, 0.3), ra, dec, incl)          # CPU template + device self-check
    for P in (_P(36, 29, 0.1), _P(34, 31, 0.5)):
        a, d_dev = dev.lnL_sky(P, ra, dec, incl)
        b, _ = cpu.lnL_sky(P, ra, dec, incl)
        assert np.allclose(a, b, rtol=0, atol=1e-7), np.max(np.abs(a - b))
    assert dev.n_compiles == 1


def _numeric_rb_objective(bad_component=None, full_nonfinite=False):
    """Separate the RB/full dispatch contract from waveform compilation."""
    obj = object.__new__(IntrinsicMarginalObjective)
    obj.mode_backend = "ripple"
    obj.batch_size = 2
    obj._template = SimpleNamespace(detectors={"H1": dict(epoch=0.)}, detector_names=["H1"])
    obj._key = ("mock-shape",)
    obj.n_calls = obj.n_nonfinite = 0
    obj.t_waveform = obj.t_kernel = 0.
    obj.reduce = "profile"
    obj.sky = SimpleNamespace(ra=jnp.zeros(1), dec=jnp.zeros(1), incl=jnp.zeros(1))
    obj._kernel = jax.jit(lambda a, ra, dec, incl: jnp.full_like(ra, a["H1"][0][0,0]))

    def full(m1, m2, c1, c2):
        q = jnp.ones((1,1)) * (jnp.nan if full_nonfinite else m1)
        return {"H1": (q, jnp.ones((1,1)), jnp.zeros((1,1)))}

    def approximate(m1, m2, c1, c2, state):
        values = [-m1[:,None,None], jnp.ones((len(m1),1,1)), jnp.zeros((len(m1),1,1))]
        if bad_component is not None:
            values[bad_component] = jnp.full_like(values[bad_component], jnp.inf)
        return {"H1": tuple(values)}

    obj._ripple = dict(rb=object(), rb_state={}, fn=full,
                       modes=SimpleNamespace(lms=[(2,2)]))
    obj._vfn_rb = lambda: approximate
    return obj


@pytest.mark.parametrize("method", ["scalar", "batch"])
@pytest.mark.parametrize("bad_component", [None,0,1,2])
def test_nonfinite_rb_arrays_use_full_grid_only_when_needed(method, bad_component):
    obj = _numeric_rb_objective(bad_component)
    p = _P(35,30,.3)
    if method == "scalar":
        value = np.asarray(obj._ripple_precompute(p).detectors["H1"]["Q"])[0,0]
    else:
        value = np.asarray(obj.batch([p]))[0]
    # The finite approximation has a distinct sign, so this also proves that
    # valid RB arrays keep their fast path rather than always using full grid.
    np.testing.assert_allclose(value, -35. if bad_component is None else 35.)


def test_nonfinite_full_grid_retains_existing_zero_likelihood_behavior():
    obj = _numeric_rb_objective(bad_component=1, full_nonfinite=True)
    assert np.isneginf(np.asarray(obj.batch([_P(35,30,.3)]))[0])
    assert obj.n_nonfinite == 1


@pytest.mark.parametrize("fmin, expect_fallback", [(63., True), (20., False)])
def test_automatic_mesh_initial_fallback_matches_full_grid(monkeypatch, fmin, expect_fallback):
    from RIFT.likelihood.jax_ile import ripple_modes

    class ConstantModes:
        # Isolate the automatic mesh and real Q/U/V/likelihood path from a
        # costly waveform compile. Intrinsic amplitude still varies per row.
        def __init__(self, freqs, fmin, fmax, fref, **kwargs):
            self.freqs = jnp.asarray(freqs)
            self.lms = [(2,2)]
            self._fn = self._modes

        def _modes(self, m1, m2, c1, c2):
            return jnp.ones((1, len(self.freqs)), dtype=complex) * m1

        def modes_at(self, freqs, m1, m2, c1, c2):
            values = jnp.ones((1, len(freqs)), dtype=complex) * m1
            return values, values

    monkeypatch.setattr(ripple_modes, "RippleXPHMModes", ConstantModes)
    p = _P(35.,30.,.3)
    p.fmin = p.fref = fmin
    p.deltaF, p.deltaT = 1., 1/128.
    data = lal.CreateCOMPLEX16FrequencySeries(
        "H1", lal.LIGOTimeGPS(FID-.5), -64., 1., lal.DimensionlessUnit, 128)
    data.data.data[:] = 1.
    kwargs = dict(l_max=2, fmax=64., n_sky=4, nphi=4, npsi=4, n_dist=8,
                  analyticPSD_Q=True, mode_backend="ripple", batch_size=2,
                  storage_window_half=.125, integration_window_half=.0625, seed=7)
    inputs = ({"H1":data}, {"H1":lambda _:1.}, p, FID, 100., 2000.)
    obj = IntrinsicMarginalObjective(*inputs, rb_eps=.3, **kwargs)
    reference = IntrinsicMarginalObjective(*inputs, rb_eps=None, **kwargs)
    got, _ = obj.precompute(p)
    want, _ = reference.precompute(p)
    assert bool(obj._ripple.get("rb_fallback")) is expect_fallback
    assert (obj._ripple["rb_state"] is None) is expect_fallback
    if expect_fallback:
        assert "two distinct edges" in obj._ripple["rb_fallback"]
    else:
        assert len(obj._ripple["rb"].f_edges) > 2
    for key in ("Q", "U", "V"):
        np.testing.assert_allclose(got.detectors["H1"][key], want.detectors["H1"][key],
                                   rtol=1e-12, atol=1e-9)
    assert float(np.real(np.asarray(want.detectors["H1"]["U"])[0,0])) > 0.
    p2 = p.copy()
    p2.m1 = 36.*MSUN
    scalar = np.array([obj(p),obj(p2)])
    expected = np.array([reference(p),reference(p2)])
    assert np.isfinite(scalar).all()
    np.testing.assert_allclose(scalar, expected, rtol=0., atol=1e-8)
    np.testing.assert_allclose(obj.batch([p,p2]), expected, rtol=0., atol=1e-8)
    assert bool(obj._ripple.get("rb_fallback")) is expect_fallback
