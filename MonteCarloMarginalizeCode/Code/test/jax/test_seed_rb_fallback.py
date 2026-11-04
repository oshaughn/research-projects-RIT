"""The real ripple program retains modes absent in an RB fiducial."""
import numpy as np
import pytest
import lal
import lalsimulation as lalsim
import jax.numpy as jnp

from RIFT import lalsimutils
from RIFT.likelihood.jax_ile.intrinsic_objective import IntrinsicMarginalObjective


def test_ripple_scalar_batch_fallback_and_reactivation(tmp_path, monkeypatch):
    pytest.importorskip("ripplegw")
    pytest.importorskip("flatbuffers")
    fid = 1126259462.
    p = lalsimutils.ChooseWaveformParams()
    p.m1, p.m2 = 30*lal.MSUN_SI, 20*lal.MSUN_SI
    p.s1x, p.s1z, p.s2z = .3, .2, -.1
    p.fmin = p.fref = 20.
    p.deltaT, p.deltaF = 1/512., 1.
    p.approx = lalsim.IMRPhenomXPHM
    p.radec = True
    p.tref = fid
    rng = np.random.default_rng(1)
    d = lal.CreateCOMPLEX16FrequencySeries(
        "H1", lal.LIGOTimeGPS(fid-.5), -256., 1., lal.DimensionlessUnit, 512)
    d.data.data[:] = (rng.normal(size=512)+1j*rng.normal(size=512))*1e-23
    obj = IntrinsicMarginalObjective(
        {"H1": d}, {"H1": lalsim.SimNoisePSDaLIGOZeroDetHighPower}, p,
        fid, 100., 2000., n_sky=8, nphi=4, npsi=4, n_dist=8,
        l_max=2, fmax=200., analyticPSD_Q=True, mode_backend="ripple",
        rb_eps=.3, batch_size=2, export_dir=str(tmp_path))
    # Establish the RB program and its export cache using a supported fiducial.
    assert np.isfinite(obj(p))
    assert len(list(tmp_path.glob("jaxexport-*.bin"))) == 2
    full_modes = obj._full_modes
    actual = np.asarray(full_modes(p))
    absent = int(np.argmax(np.max(np.abs(actual), axis=1)))
    assert np.any(actual[absent] != 0)

    # Model a structurally absent fiducial mode, keeping the candidate program
    # unmodified: at the candidate this same physical ripple mode is nonzero.
    def missing_mode_fiducial(point):
        return jnp.asarray(full_modes(point)).at[absent].set(0.)

    monkeypatch.setattr(obj, "_full_modes", missing_mode_fiducial)
    obj.rb_refresh(p)
    pre, _ = obj.precompute(p)
    args = (jnp.asarray(p.m1/lal.MSUN_SI), jnp.asarray(p.m2/lal.MSUN_SI),
            jnp.asarray([p.s1x,p.s1y,p.s1z]), jnp.asarray([p.s2x,p.s2y,p.s2z]))
    q, u, v = obj._ripple["fn"](*args)["H1"]
    for name, expected in (("Q",q),("U",u),("V",v)):
        np.testing.assert_allclose(pre.detectors["H1"][name], expected, rtol=1e-12, atol=0)
    assert float(np.real(np.asarray(u)[absent, absent])) > 0
    p2 = p.copy()
    p2.s1x = .4
    scalar = np.array([obj(p), obj(p2)])
    batch = np.asarray(obj.batch([p,p2]))
    assert np.isfinite(scalar).all()
    np.testing.assert_allclose(batch, scalar, rtol=0, atol=1e-9)

    # Returning to representable modes must reuse the RB program, not the full
    # batch callable (the signatures differ because RB has a state argument).
    monkeypatch.setattr(obj, "_full_modes", full_modes)
    obj.rb_refresh(p)
    np.testing.assert_allclose(obj.batch([p,p]), [obj(p),obj(p)], rtol=0, atol=1e-9)
    assert np.isfinite(obj(p))
