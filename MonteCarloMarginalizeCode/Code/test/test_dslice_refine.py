"""refine_slice_gaps: bisect slice gaps that a conditional crosses too fast to interpolate.

Mock: a conditional flat in d out to an edge, then falling steeply -- the shape at S240615dg MR-1 grid
point 6, where the walk-out's doubling step jumped from the flat part to 15 nats down.
"""
import numpy as np
import pytest

from RIFT.misc.distance_slices import refine_slice_gaps, refine_fresh_slices, walk_out_slices

EDGE, WIDTH = 1850.0, 60.0


def lnL(d):
    d = np.asarray(d, float)
    return 300.0 - 0.5 * np.maximum(d - EDGE, 0.0) ** 2 / WIDTH ** 2


def flat(d):
    return np.zeros_like(np.asarray(d, float))


class Fresh:
    def __init__(self, f=lnL):
        self.f, self.calls = f, []

    def __call__(self, d):
        self.calls.append(float(d[0]))
        return self.f(d), np.full(len(d), 0.1), np.full(len(d), 30.0), np.full(len(d), 1000)


def walked():
    core = np.linspace(1000.0, 1714.0, 25)
    w = walk_out_slices(core, lnL(core), flat, Fresh(), 1.0, 1e4, delta_lnL=5.0)
    return np.r_[core, w[0]], np.r_[lnL(core), w[1]]


def max_live_jump(d, y, delta=5.0):
    o = np.argsort(d)
    lp = np.asarray(y)[o]
    j = np.abs(np.diff(lp))
    live = np.maximum(lp[1:], lp[:-1]) > lp.max() - delta
    return j[live].max()


def test_walkout_alone_leaves_the_edge_unresolved():
    d, y = walked()
    assert max_live_jump(d, y) > 6.0


def test_refinement_resolves_the_edge():
    d, y = walked()
    why = []
    f = Fresh()
    added = refine_slice_gaps(d, y, flat, f, dlnL_max=1.0, delta_lnL=5.0, max_add=20, reasons=why)
    assert why == ["resolved"] and len(added[0]) == len(f.calls) > 0
    D, Y = np.r_[d, added[0]], np.r_[y, added[1]]
    assert max_live_jump(D, Y) <= 1.0
    assert np.any((added[0] > EDGE) & (added[0] < EDGE + 3 * WIDTH))        # slices landed on the cliff
    np.testing.assert_allclose(added[1], lnL(added[0]))
    assert added[4].dtype.kind == "i"


def test_cap_and_off():
    d, y = walked()
    why = []
    added = refine_slice_gaps(d, y, flat, Fresh(), dlnL_max=1.0, delta_lnL=5.0, max_add=2, reasons=why)
    assert len(added[0]) == 2 and why == ["max_add"]
    for kw in (dict(dlnL_max=0.0, max_add=5), dict(dlnL_max=1.0, max_add=0)):
        why = []
        f = Fresh()
        added = refine_slice_gaps(d, y, flat, f, delta_lnL=5.0, reasons=why, **kw)
        assert len(added[0]) == 0 and not f.calls and why == ["off"]


def test_dead_region_is_left_alone():
    # jumps far below the peak (beyond delta) are not refined
    d = np.array([1000.0, 1500.0, 1800.0, 2500.0, 4000.0])
    y = np.array([300.0, 300.0, 300.0, 280.0, 200.0])
    f = Fresh(lambda dd: np.full(len(dd), np.interp(dd[0], d, y)))
    added = refine_slice_gaps(d, y, flat, f, dlnL_max=1.0, delta_lnL=5.0, max_add=50)
    assert np.all((added[0] > 1800.0) & (added[0] < 2500.0))


@pytest.mark.parametrize("bad", [np.nan, -np.inf])
def test_nonfinite_slice_is_recorded_and_not_retried(bad):
    d, y = walked()
    f = Fresh(lambda dd: np.full(len(dd), bad))
    why = []
    added = refine_slice_gaps(d, y, flat, f, dlnL_max=1.0, delta_lnL=5.0, max_add=20, reasons=why)
    assert len(added[0]) == len(f.calls) >= 1 and not np.any(np.isfinite(added[1]))
    assert len(set(np.round(f.calls, 6))) == len(f.calls)      # no pair bisected twice
    assert why == ["resolved"]


def test_refine_fresh_slices_restores_the_offset(monkeypatch):
    import RIFT.misc.distance_slices as ds
    offset = 77.0

    def fake_fresh(sampler, like, d_new, n_max, n_eff_target, n_chunk, return_lnL):
        return lnL(d_new) - offset, np.full(1, 0.1), np.full(1, 30.0), np.full(1, 1000)

    monkeypatch.setattr(ds, "fresh_sample_slices", fake_fresh)
    d, y = walked()
    added = refine_fresh_slices(None, None, d, y, flat, offset, dlnL_max=1.0, delta_lnL=5.0, max_add=20,
                                n_max=1, n_eff_target=1, n_chunk=1)
    np.testing.assert_allclose(added[1], lnL(added[0]))
    ref = refine_slice_gaps(d, y, flat, Fresh(), dlnL_max=1.0, delta_lnL=5.0, max_add=20)
    np.testing.assert_allclose(added[0], ref[0])


def test_prior_enters_the_jump():
    # equal lnL at d and 2d: only the d^2 prior makes the pair jump by 2 ln 2 > 1 nat
    d, y = np.array([1000.0, 2000.0]), np.array([300.0, 300.0])
    sq = lambda dd: 2.0 * np.log(np.asarray(dd, float))
    const = Fresh(lambda dd: np.full(len(dd), 300.0))
    assert len(refine_slice_gaps(d, y, sq, const, dlnL_max=1.0, delta_lnL=5.0, max_add=1)[0]) == 1
    assert len(refine_slice_gaps(d, y, flat, Fresh(lambda dd: np.full(len(dd), 300.0)),
                                 dlnL_max=1.0, delta_lnL=5.0, max_add=1)[0]) == 0


def test_bisects_at_the_geometric_mean_of_the_largest_jump_first():
    d = np.array([1000.0, 1500.0, 3000.0])
    y = np.array([300.0, 298.0, 290.0])           # jumps 2 and 8 nats: the second pair goes first
    f = Fresh(lambda dd: np.full(len(dd), 295.0))
    added = refine_slice_gaps(d, y, flat, f, dlnL_max=1.0, delta_lnL=20.0, max_add=1)
    np.testing.assert_allclose(added[0], [np.sqrt(1500.0 * 3000.0)], rtol=1e-12)


def test_narrow_gap_is_skipped_and_reported():
    d = np.array([1000.0, 1000.0 * np.exp(0.005)])  # ln gap 0.005 < 2 * min_log_step (0.01)
    y = np.array([300.0, 296.0])
    why = []
    f = Fresh()
    added = refine_slice_gaps(d, y, flat, f, dlnL_max=1.0, delta_lnL=5.0, max_add=5, reasons=why)
    assert len(added[0]) == 0 and not f.calls and why == ["min_step"]
