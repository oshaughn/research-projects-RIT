"""ripple_modes: the JAX Q/U/V reduction matches gpu_precompute; ripple backend smoke test."""
import numpy as np
import pytest
import jax.numpy as jnp

from RIFT.likelihood import gpu_precompute as gp
from RIFT.likelihood.jax_ile.ripple_modes import conj_modes, reduce_qvu


def test_reduce_qvu_matches_compound_precompute():
    rng = np.random.default_rng(0)
    K, N = 3, 512
    h = rng.normal(size=(K, N)) + 1j * rng.normal(size=(K, N))
    d = rng.normal(size=N) + 1j * rng.normal(size=N)
    w = rng.uniform(0.1, 1.0, size=N)
    hc = np.conj(h[:, (N - np.arange(N)) % N])
    q0, u0, v0 = gp.compound_precompute_arrays(h[None], hc[None], d, w, 0.25, 1.0 / (N * 0.25),
                                               17, 64, backend=np)
    q, u, v = reduce_qvu(jnp.asarray(h), jnp.asarray(d), jnp.asarray(w), 0.25, 17, 64)
    assert np.allclose(np.asarray(q), q0[0].T, atol=1e-10)
    assert np.allclose(np.asarray(u), u0[0, 0], atol=1e-10)
    assert np.allclose(np.asarray(v), v0[0, 0], atol=1e-10)
    assert np.allclose(np.asarray(conj_modes(jnp.asarray(h))), hc)


def test_ripple_modes_smoke():
    pytest.importorskip("ripplegw")
    from RIFT.likelihood.jax_ile.ripple_modes import RippleXPHMModes
    df, N = 0.25, 8192
    f = -0.5 * N * df + df * np.arange(N)
    m = RippleXPHMModes(f, 20.0, 800.0, 20.0, l_max=2)
    h = np.asarray(m(30.0, 20.0, [0.4, 0.1, 0.2], [0.0, 0.1, -0.1]))
    assert h.shape == (5, N) and np.all(np.isfinite(h))
    band = (np.abs(f) >= 20) & (np.abs(f) <= 800)
    assert np.all(h[:, ~band] == 0) and np.max(np.abs(h[:, band])) > 0


def _bp_like():
    return 30.0, 20.0, [0.4, 0.1, 0.2], [0.0, 0.1, -0.1]


def test_ripple_modes_rebuild_native_polarizations():
    """Extracted modes contracted with -2Y_lm reproduce ripple's own h+ - i hx at some view."""
    pytest.importorskip("ripplegw")
    import lal
    import jax
    from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM import generate_xphm
    from RIFT.likelihood.jax_ile.ripple_modes import RippleXPHMModes
    df, N = 0.25, 8192
    f = -0.5 * N * df + df * np.arange(N)
    m1, m2, c1, c2 = _bp_like()
    M = RippleXPHMModes(f, 20.0, 800.0, 20.0, l_max=4)
    H = np.asarray(M(m1, m2, c1, c2))
    pos = (f >= 20) & (f <= 800)
    hp, hc = jax.jit(generate_xphm, static_argnums=(12,))(m1, m2, *c1, *c2, 1000.0, 1.3, 0.3,
                                                           jnp.asarray(f[pos]), 20.0)
    target = np.asarray(hp) - 1j * np.asarray(hc)
    best = 0.0
    for th in np.linspace(0, np.pi, 91):            # the J-frame view is unknown: search it
        for ph in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            R = sum(H[k, pos] * lal.SpinWeightedSphericalHarmonic(th, ph, -2, l, m)
                    for k, (l, m) in enumerate(M.lms))
            best = max(best, abs(np.vdot(R, target)) / np.sqrt(np.vdot(R, R).real * np.vdot(target, target).real))
    assert best > 0.9995, best


def test_ripple_matches_lalsim_raw_inner_product():
    """ripple XPHM h+ vs lalsim, raw normalized inner product (no maximization)."""
    pytest.importorskip("ripplegw")
    import lal, lalsimulation as lalsim, jax
    from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM import generate_xphm
    m1, m2, c1, c2 = _bp_like()
    df = 0.25
    f = np.arange(int(20 / df), int(800 / df)) * df
    hp, _ = jax.jit(generate_xphm, static_argnums=(12,))(m1, m2, *c1, *c2, 400.0, 0.7, 0.4, jnp.asarray(f), 20.0)
    HP, _ = lalsim.SimInspiralChooseFDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *c1, *c2, 400e6 * lal.PC_SI,
                                               0.7, 0.4, 0, 0, 0, df, 20.0, 800.0, 20.0, lal.CreateDict(),
                                               lalsim.IMRPhenomXPHM)
    a = HP.data.data[int(20 / df):int(20 / df) + len(f)]
    r = np.vdot(a, np.asarray(hp)) / np.vdot(a, a)
    assert abs(r - 1) < 1e-3, r
