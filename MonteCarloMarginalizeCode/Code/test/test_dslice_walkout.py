"""walk_out_slices: extend quantile-placed distance slices until the conditional is delta below its peak.

The mock likelihood is Gaussian in u = 1/d with a flat prior, so lp at any d is known exactly and the
walk-out's stopping point can be checked against it.
"""
import numpy as np
import pytest

from RIFT.misc.distance_slices import walk_out_slices

U0, SU = 1 / 1000.0, 0.15 / 1000.0          # peak at 1000 Mpc, width ~15% in u


def lnL(d):
    return 50.0 - 0.5 * ((1.0 / np.asarray(d, float) - U0) / SU) ** 2


def flat(d):
    return np.zeros_like(np.asarray(d, float))


class Fresh:
    def __init__(self, f=lnL):
        self.f, self.calls = f, []

    def __call__(self, d):
        self.calls.append(float(d[0]))
        return self.f(d), np.full(len(d), 0.1), np.full(len(d), 30.0), np.full(len(d), 1000)


def narrow_core(n=50):
    # quantile slices of a too-narrow main loop: they end ~1 nat below the peak
    u = U0 + SU * np.linspace(-1.4, 1.4, n)
    return 1.0 / u


def test_walk_reaches_target_on_both_sides():
    d = narrow_core()
    f = Fresh()
    dw, lw, sw, nw, tw = walk_out_slices(d, lnL(d), flat, f, 1.0, 1e4, delta_lnL=7.0)
    assert len(dw) == len(f.calls) > 0
    near, far = dw[dw < d.min()], dw[dw > d.max()]
    assert len(near) and len(far)
    peak = lnL(1 / U0)
    assert peak - lnL(near.min()) >= 7.0 and peak - lnL(far.max()) >= 7.0
    # every walk slice but the last on each side is still above the target
    for side in (np.sort(near)[::-1], np.sort(far)):
        assert np.all(peak - lnL(side[:-1]) < 7.0)
    assert sw.shape == nw.shape == tw.shape == dw.shape and tw.dtype.kind == "i"


def test_without_walk_the_core_edges_are_short_of_target():
    d = narrow_core()
    peak = lnL(1 / U0)
    assert peak - lnL(d.min()) < 1.5 and peak - lnL(d.max()) < 1.5


def test_steps_double_in_log_d():
    d = narrow_core()
    f = Fresh()
    dw = walk_out_slices(d, lnL(d), flat, f, 1.0, 1e4, delta_lnL=7.0)[0]
    far = np.r_[d.max(), np.sort(dw[dw > d.max()])]
    g = np.diff(np.log(far))
    assert len(g) >= 2 and np.allclose(g[1:] / g[:-1], 2.0)


def test_prior_counts_toward_the_drop():
    d = narrow_core()
    steep = lambda dd: 2.0 * np.log(np.asarray(dd, float))   # ~ d^2 prior raises the far side
    dw_flat = walk_out_slices(d, lnL(d), flat, Fresh(), 1.0, 1e4, delta_lnL=7.0)[0]
    dw_vol = walk_out_slices(d, lnL(d), steep, Fresh(), 1.0, 1e4, delta_lnL=7.0)[0]
    assert dw_vol.max() >= dw_flat.max()


def test_stops_at_range_limit():
    d = narrow_core()
    f = Fresh()
    dmax = d.max() * 1.05
    dw = walk_out_slices(d, lnL(d), flat, f, 1.0, dmax, delta_lnL=7.0)[0]
    assert dw.max() <= dmax and np.sum(dw > d.max()) == 1


def test_edge_beyond_range_limit_never_walks_inward():
    d = narrow_core()
    f = Fresh()
    dw = walk_out_slices(d, lnL(d), flat, f, 1.0, d.max() * 0.9, delta_lnL=7.0)[0]
    assert not np.any(dw > d.min()), dw     # far side adds nothing; nothing lands inside the core


@pytest.mark.parametrize("bad_value", [-np.inf, np.nan])
def test_stops_on_nonfinite_slice_and_respects_max_steps(bad_value):
    d = narrow_core()
    bad = Fresh(lambda dd: np.full(len(dd), bad_value))
    dw, lw = walk_out_slices(d, lnL(d), flat, bad, 1.0, 1e4, delta_lnL=7.0)[:2]
    assert len(dw) == 2 and not np.any(np.isfinite(lw))      # one per side, then stop
    capped = Fresh(lambda dd: np.full(len(dd), 50.0))           # flat: never reaches target
    dw = walk_out_slices(d, lnL(d), flat, capped, 1.0, 1e9, delta_lnL=7.0, max_steps=3)[0]
    assert len(dw) == 6


@pytest.mark.parametrize("delta, n", [(0.0, 50), (7.0, 1)])
def test_disabled_or_degenerate_input_adds_nothing(delta, n):
    d = narrow_core(n)
    f = Fresh()
    dw = walk_out_slices(d, lnL(d), flat, f, 1.0, 1e4, delta_lnL=delta)[0]
    assert len(dw) == 0 and not f.calls


def test_running_peak_raises_the_target_when_the_walk_crosses_the_peak():
    # the core sits entirely on the near side of the peak; the far walk crosses it
    u = U0 + SU * np.linspace(2.0, 4.0, 50)
    d = 1.0 / u
    why = []
    dw, lw = walk_out_slices(d, lnL(d), flat, Fresh(), 1.0, 1e4, delta_lnL=7.0, reasons=why)[:2]
    far = np.sort(dw[dw > d.max()])
    lp_far = lnL(far)
    run_peak = max(lnL(d).max(), lp_far.max())
    assert run_peak > lnL(d).max() + 1.0                      # the walk found a higher peak
    assert run_peak - lp_far[-1] >= 7.0
    assert np.all(run_peak - lp_far[:-1] < 7.0)               # stopped at the first slice past the new target
    assert why == ["target", "target"]


def test_stop_reasons():
    d = narrow_core()
    why = []
    walk_out_slices(d, lnL(d), flat, Fresh(), 1.0, d.max() * 1.05, delta_lnL=7.0, reasons=why)
    assert why == ["target", "range"]
    why = []
    walk_out_slices(d, lnL(d), flat, Fresh(lambda dd: np.full(len(dd), np.nan)), 1.0, 1e4, delta_lnL=7.0, reasons=why)
    assert why == ["nonfinite", "nonfinite"]
    why = []
    walk_out_slices(d, lnL(d), flat, Fresh(lambda dd: np.full(len(dd), 50.0)), 1.0, 1e9, delta_lnL=7.0,
                    max_steps=3, reasons=why)
    assert why == ["max_steps", "max_steps"]


def test_walk_out_fresh_slices_restores_the_overflow_offset(monkeypatch):
    # fresh_sample_slices integrates the offset-subtracted likelihood; the walk-out rows must be
    # put back on ILE's axis before they are compared with the existing slices or written.
    import RIFT.misc.distance_slices as ds
    offset = 123.0
    calls = []

    def fake_fresh(sampler, like, d_new, n_max, n_eff_target, n_chunk, return_lnL):
        calls.append(float(d_new[0]))
        return lnL(d_new) - offset, np.full(1, 0.1), np.full(1, 30.0), np.full(1, 1000)

    monkeypatch.setattr(ds, "fresh_sample_slices", fake_fresh)
    d = narrow_core()
    why = []
    dw, lw, sw, nw, tw = ds.walk_out_fresh_slices(None, None, d, lnL(d), flat, 1.0, 1e4,
                                                  lnL_offset=offset, delta_lnL=7.0, max_steps=8,
                                                  n_max=1, n_eff_target=1, n_chunk=1, reasons=why)
    ref = walk_out_slices(d, lnL(d), flat, Fresh(), 1.0, 1e4, delta_lnL=7.0)
    assert len(calls) == len(dw) > 0
    np.testing.assert_allclose(lw, lnL(dw))          # on the slices' axis, not offset below it
    np.testing.assert_allclose(dw, ref[0])           # same stopping points as the offset-free walk
    assert why == ["target", "target"]


def test_merge_walkout_appends_rows_with_integer_ntotal():
    from RIFT.misc.distance_slices import merge_walkout
    base = (np.array([1.0, 2.0]), np.array([5.0, 6.0]), np.array([0.1, 0.1]), np.array([30.0, 30.0]),
            np.array([100, 200]))
    added = (np.array([0.5, 3.0]), np.array([1.0, 2.0]), np.array([0.2, 0.2]), np.array([20.0, 20.0]),
             np.array([300.0, 400.0]))
    d, l, s, n, t = merge_walkout(*base, added)
    np.testing.assert_array_equal(d, [1.0, 2.0, 0.5, 3.0])
    np.testing.assert_array_equal(l, [5.0, 6.0, 1.0, 2.0])
    np.testing.assert_array_equal(t, [100, 200, 300, 400])
    assert t.dtype.kind == "i" and len(s) == len(n) == 4
    empty = tuple(np.array([]) for _ in range(5))
    d0 = merge_walkout(*base, empty)
    np.testing.assert_array_equal(d0[0], base[0]) and np.testing.assert_array_equal(d0[4], base[4])


def test_disabled_walk_reports_off():
    d = narrow_core()
    why = []
    walk_out_slices(d, lnL(d), flat, Fresh(), 1.0, 1e4, delta_lnL=7.0, max_steps=0, reasons=why)
    assert why == ["off", "off"]
