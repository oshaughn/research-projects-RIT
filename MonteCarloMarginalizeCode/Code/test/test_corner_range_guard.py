"""RIFT.misc.corner_range: the predicate the CIP driver uses to decline a corner panel.

The predicate has to agree with corner itself, in both directions.  If it is too
permissive the driver still dies; if it is too strict it silently stops drawing
plots that were fine.  So the cases below that matter are run through the real
``corner.corner`` as well, and the two verdicts are compared -- a predicate checked
only against its own reasoning would agree with itself whatever corner does.

The degenerate input is the one measured on the CI grid of .travis/test-posterior.sh:
the m1-m2 panel, whose range is the input GRID's mass box while the samples drawn in
it are the POSTERIOR, which had moved off that box entirely.
"""

import numpy as np
import pytest

from RIFT.misc.corner_range import (
    overlay_or_warn, pad_degenerate_intervals, unplottable_reason)

# The m1-m2 box the CI grid defines (m1 uniform in [50,60], m2 in [0.8 m1, m1]),
# and the posterior that arm produces: a constant lnL, so the posterior is the
# prior and spans decades of mass.  Measured on CIT 2026-09-17, IGWN conda
# python 3.11 / numpy 1.26.4 / corner 2.3.0.
CI_RANGE = [[50.00987243652344, 59.99585723876953], [40.82694625854492, 59.306819915771484]]


def _ci_posterior(n=2500, seed=0, inside=0):
    """n samples spanning [19.9, 1447] x [3.3, 280], of which `inside` land in CI_RANGE."""
    rng = np.random.default_rng(seed)
    m1 = np.exp(rng.uniform(np.log(19.91), np.log(1447.6), n))
    m2 = np.exp(rng.uniform(np.log(3.34), np.log(279.8), n))
    # Push out anything that happens to fall in the box, then put back exactly `inside`.
    hit = ((m1 >= CI_RANGE[0][0]) & (m1 <= CI_RANGE[0][1])
           & (m2 >= CI_RANGE[1][0]) & (m2 <= CI_RANGE[1][1]))
    m1[hit] = 1000.
    assert inside <= n
    for j in range(inside):
        m1[j], m2[j] = 55., 50.
    return np.column_stack([m1, m2])


# The two ValueErrors corner raises about a range or an empty histogram.  Anything
# else is re-raised: a test that accepted any ValueError would pass on a corner bug
# and report agreement that was never tested.
_CORNER_RANGE_ERRORS = ("is not valid or the sample is empty", "no dynamic range")


def _corner_raises(sample, ranges, **kw):
    """True if corner.corner rejects this (sample, range) pair for a range reason."""
    corner = pytest.importorskip("corner")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        corner.corner(np.asarray(sample), range=[list(r) for r in ranges], **kw)
    except ValueError as exc:
        if not any(m in str(exc) for m in _CORNER_RANGE_ERRORS):
            raise
        return True
    finally:
        plt.close("all")
    return False


# ---------------------------------------------------------------- the CI failure

def test_ci_degenerate_panel_is_reported():
    reason = unplottable_reason(_ci_posterior(inside=0), CI_RANGE, labels=["m1", "m2"])
    assert reason is not None
    assert "m1" in reason


def test_ci_degenerate_panel_is_what_corner_rejects():
    sample = _ci_posterior(inside=0)
    assert _corner_raises(sample, CI_RANGE)
    assert unplottable_reason(sample, CI_RANGE, labels=["m1", "m2"]) is not None


def test_one_sample_in_the_box_is_enough_for_both():
    """The measured counts were 0,1,2,3,5 across runs: ONE must still plot."""
    sample = _ci_posterior(inside=1)
    assert not _corner_raises(sample, CI_RANGE)
    assert unplottable_reason(sample, CI_RANGE, labels=["m1", "m2"]) is None


def test_pairwise_not_whole_box():
    """A row outside in a THIRD coordinate still fills the panel it is inside.

    Checking membership of the full n-D box would decline this, and corner draws it.
    """
    sample = np.array([[0.5, 0.5, 99.], [0.5, 99., 0.5], [99., 0.5, 0.5]])
    ranges = [[0., 1.], [0., 1.], [0., 1.]]
    assert unplottable_reason(sample, ranges) is None
    assert not _corner_raises(sample, ranges)


def test_pair_empty_while_each_axis_is_occupied():
    """No single axis is empty; the (0,1) panel still is.  This is the CI geometry."""
    sample = np.array([[0.5, 99.], [99., 0.5]])
    ranges = [[0., 1.], [0., 1.]]
    reason = unplottable_reason(sample, ranges, labels=["a", "b"])
    assert reason is not None and "panel" in reason
    assert _corner_raises(sample, ranges)


# ---------------------------------------------------------------- range shapes

def test_zero_width_range_is_not_a_reason_to_decline():
    """MEASURED, not assumed: corner collapses every bin edge onto one value and
    np.histogram2d accepts non-decreasing edges, so a sample sitting ON that value is
    still counted.  Declining here would drop a panel corner draws."""
    sample = np.column_stack([np.full(64, 7.0), np.linspace(0., 1., 64)])
    ranges = [[7., 7.], [0., 1.]]
    assert not _corner_raises(sample, ranges)
    assert unplottable_reason(sample, ranges, labels=["m", "q"]) is None


def test_zero_width_range_off_the_data_is_reported():
    """The zero-width case that DOES fail is the ordinary empty one."""
    sample = np.column_stack([np.full(64, 9.0), np.linspace(0., 1., 64)])
    ranges = [[7., 7.], [0., 1.]]
    assert _corner_raises(sample, ranges)
    assert unplottable_reason(sample, ranges, labels=["m", "q"]) is not None


def test_padding_removes_the_singular_axis():
    ranges = [[7., 7.], [0., 1.]]
    padded_ranges, padded = pad_degenerate_intervals(ranges)
    assert padded == [0]
    assert padded_ranges[1] == [0., 1.]          # untouched
    assert padded_ranges[0][0] < 7. < padded_ranges[0][1]
    sample = np.column_stack([np.full(64, 7.0), np.linspace(0., 1., 64)])
    assert unplottable_reason(sample, padded_ranges, labels=["m", "q"]) is None
    assert not _corner_raises(sample, padded_ranges)


def test_padding_at_zero_uses_the_absolute_floor():
    padded_ranges, padded = pad_degenerate_intervals([[0., 0.], [-1., 1.]])
    assert padded == [0]
    assert padded_ranges[0][0] < 0. < padded_ranges[0][1]


def test_inverted_range_is_not_reported_because_corner_sorts_it():
    """corner bins with min()/max() of each interval, so inversion alone is legal.

    Reporting it would decline a panel corner draws.
    """
    sample = np.column_stack([np.linspace(0., 1., 64), np.linspace(0., 1., 64)])
    ranges = [[1., 0.], [0., 1.]]
    assert unplottable_reason(sample, ranges) is None
    assert not _corner_raises(sample, ranges)


def test_non_finite_range_is_reported():
    sample = np.column_stack([np.linspace(0., 1., 8), np.linspace(0., 1., 8)])
    assert "not finite" in unplottable_reason(sample, [[0., np.inf], [0., 1.]])
    assert "not finite" in unplottable_reason(sample, [[np.nan, 1.], [0., 1.]])


def test_empty_sample_is_reported():
    assert unplottable_reason(np.zeros((0, 2)), [[0., 1.], [0., 1.]]) is not None


def test_all_nan_column_is_reported():
    sample = np.column_stack([np.full(8, np.nan), np.linspace(0., 1., 8)])
    ranges = [[0., 1.], [0., 1.]]
    assert unplottable_reason(sample, ranges, labels=["a", "b"]) is not None
    assert _corner_raises(sample, ranges)


def test_range_endpoints_are_inclusive_like_the_histogram():
    """A sample sitting exactly on both edges is inside for np.histogram2d, so it must
    be inside here too."""
    sample = np.array([[0., 1.], [0., 1.]])
    ranges = [[0., 1.], [0., 1.]]
    assert unplottable_reason(sample, ranges) is None
    assert not _corner_raises(sample, ranges)


# ---------------------------------------------------------------- weights, shapes

def test_zero_weight_inside_is_reported():
    """corner rejects on the WEIGHTED sum, so weight-zero rows do not fill a panel."""
    sample = np.array([[0.5, 0.5], [99., 99.]])
    ranges = [[0., 1.], [0., 1.]]
    wt = np.array([0., 1.])
    assert unplottable_reason(sample, ranges, weights=wt) is not None
    assert _corner_raises(sample, ranges, weights=wt)
    assert unplottable_reason(sample, ranges) is None   # uniform weights: plottable


def test_shape_disagreement_raises_rather_than_declining():
    """A caller bug must stay loud: it is not an unplottable sample."""
    with pytest.raises(ValueError):
        unplottable_reason(np.zeros((4, 3)), [[0., 1.], [0., 1.]])
    with pytest.raises(ValueError):
        unplottable_reason(np.zeros(4), [[0., 1.]])
    with pytest.raises(ValueError):
        unplottable_reason(np.zeros((4, 2)), [[0., 1.], [0., 1.]], weights=np.ones(3))


# ---------------------------------------------------------------- driver wiring

def test_driver_declines_rather_than_catching():
    """The guard must not become a try/except.

    .travis/test-posterior.sh records that dormant try/except scaffolding around these
    blocks hid a real matplotlib failure for a release.  Corner 1 and Corner 3 are
    already `if True: ... else:` for that reason; assert the corner calls this change
    touches are not put back inside a handler.
    """
    import ast
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, os.pardir, "bin",
                        "util_ConstructIntrinsicPosterior_GenericCoordinates.py")
    src = open(path).read()
    assert "overlay_or_warn(" in src
    assert "unplottable_reason(dat_here" in src

    # Line number of the Corner 3 header, and of the guard, from the SOURCE; the
    # handlers are looked for in the AST, so the commented-out `# try:` left in this
    # block does not answer the question (a plain substring search says it does).
    lines = src.splitlines()
    start = next(i for i, ln in enumerate(lines, 1) if "Corner plot 3" in ln)
    guard = next(i for i, ln in enumerate(lines, 1) if "unplottable_reason(dat_here" in ln)
    assert guard > start
    handlers = [n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Try) and n.lineno >= start]
    assert not handlers, (
        "Corner 3 must not catch (handlers at lines {}): an unplottable sample is "
        "declined, a broken plotter is not".format([n.lineno for n in handlers]))


# ---------------------------------------------------------------- the overlay wrapper

def _driver_source():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    return open(os.path.join(here, os.pardir, "bin",
                             "util_ConstructIntrinsicPosterior_GenericCoordinates.py")).read()


def test_overlay_call_sites_do_not_collide_with_the_positional_names():
    """Every overlay_or_warn(...) in the driver, checked against the real signature.

    `labels` IS a corner keyword and was once this function's third positional, so the
    lalinference overlay -- the one call site that forwards labels= to corner, and the
    one no CI job runs -- raised TypeError before corner was reached.  A green pipeline
    said nothing.  So bind each call site's keywords against the signature instead of
    reading them.
    """
    import ast
    import inspect
    sig = inspect.signature(overlay_or_warn)
    positional = [n for n, prm in sig.parameters.items()
                  if prm.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional, "overlay_or_warn takes its coordinate labels positionally"

    calls = [n for n in ast.walk(ast.parse(_driver_source()))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "overlay_or_warn"]
    assert len(calls) >= 4, "expected the Corner 1 and Corner 3 overlays; found {}".format(len(calls))
    for call in calls:
        # A splat would make the bind below prove nothing about the real arguments.
        assert not any(isinstance(a, ast.Starred) for a in call.args), \
            "*args splat at line {}".format(call.lineno)
        assert all(k.arg is not None for k in call.keywords), \
            "**kwargs splat at line {}".format(call.lineno)
        # Bind the call the way Python will, with placeholders for the values.
        sig.bind(*[None] * len(call.args), **{k.arg: None for k in call.keywords})


def test_overlay_draws_when_plottable_and_declines_when_not():
    pytest.importorskip("corner")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        base = plt.figure()
        ok = np.column_stack([np.linspace(0., 1., 32), np.linspace(0., 1., 32)])
        out = overlay_or_warn(ok, [[0., 1.], [0., 1.]], ["a", "b"], "grid",
                              weights=np.ones(32) / 32., fig=base, plot_datapoints=True,
                              plot_density=False, plot_contours=False, quantiles=None)
        assert out is not None

        out = overlay_or_warn(ok + 99., [[0., 1.], [0., 1.]], ["a", "b"], "grid",
                              weights=np.ones(32) / 32., fig=base, plot_datapoints=True,
                              plot_density=False, plot_contours=False, quantiles=None)
        assert out is base, "a declined overlay returns the figure it was handed"
    finally:
        plt.close("all")


def test_overlay_forwards_the_labels_keyword_corner_takes():
    """The lalinference call site's shape: coordinate names positionally, corner's own
    `labels` (TeX) as a keyword.  These are different arguments and both must arrive."""
    pytest.importorskip("corner")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        dat = np.column_stack([np.linspace(0., 1., 32), np.linspace(0., 1., 32)])
        fig = overlay_or_warn(dat, [[0., 1.], [0., 1.]], ["m1", "m2"], "lalinference",
                              weights=np.ones(32) / 32., color="r", labels=["$m_1$", "$m_2$"],
                              quantiles=[0.05, 0.95], no_fill_contours=True,
                              plot_datapoints=False, plot_density=False,
                              fill_contours=False, levels=[0.9])
        assert fig is not None
    finally:
        plt.close("all")
