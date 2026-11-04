"""A scalar angular-kernel rebuild also replaces its batched callable."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("lal")
jnp = pytest.importorskip("jax.numpy")

from RIFT.likelihood.jax_ile import anglemarg
from RIFT.likelihood.jax_ile.intrinsic_objective import IntrinsicMarginalObjective


def test_exact_angle_amplitude_refresh_updates_batch_kernel(monkeypatch):
    # Isolate cache invalidation from waveform generation: the angular stand-in
    # returns the amplitude sizing captured by the real kernel builder. Exercise
    # the actual build, refresh and batched-dispatch methods with unchanged shape.
    def captured_amplitude(data, ra, dec, incl, *args, amp_sizing=None, **kwargs):
        return jnp.ones_like(ra) * amp_sizing

    monkeypatch.setattr(anglemarg, "fused_log_likelihood_distphipsimarg_exact", captured_amplitude)
    obj = object.__new__(IntrinsicMarginalObjective)
    obj.mode_backend = "ripple"
    obj._key = ("same-shape",)
    obj._amp = 1.
    obj.angle_scheme = "exact"
    obj._template = SimpleNamespace(detectors={"H1": {}}, detector_names=["H1"])
    obj.x_grid = obj.log_w_grid = obj.phi_grid = obj.psi_grid = jnp.array([0.])
    obj.interp = "nearest"
    obj.n_compiles = obj.n_calls = obj.n_nonfinite = 0
    obj.t_waveform = obj.t_kernel = 0.
    obj.verbose = False
    obj.reduce = "profile"
    obj.sky = SimpleNamespace(ra=jnp.array([0.]), dec=jnp.array([0.]),
                             incl=jnp.array([0.]), log_w=np.array([0.]))
    arrays = {"H1": (jnp.zeros((1, 1)), jnp.zeros((1, 1)), jnp.zeros((1, 1)))}
    obj._ripple = {"fn": lambda *args: arrays}
    obj._build_kernel(obj._template)
    p = SimpleNamespace(m1=1., m2=1., s1x=0., s1y=0., s1z=0.,
                        s2x=0., s2y=0., s2z=0.)
    np.testing.assert_array_equal(obj.batch([p]), [1.])
    original = obj._kernel
    obj._amp_bound = lambda data: 10.
    obj._refresh_amp(obj._template)
    assert obj._kernel is not original
    assert obj.n_compiles == 2
    scalar = np.asarray(obj._kernel(arrays, obj.sky.ra, obj.sky.dec, obj.sky.incl))
    np.testing.assert_array_equal(scalar, [10.])
    np.testing.assert_array_equal(obj.batch([p]), scalar)
