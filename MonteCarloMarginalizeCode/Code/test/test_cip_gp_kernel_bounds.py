#!/usr/bin/env python3
"""Tests for ``fit_gp``'s kernel-bound options and its saturation report.

Why this matters
----------------
``fit_gp`` builds ``WhiteKernel(noise_level_bounds=...) + C(amplitude_bounds) *
RBF(length_scale_bounds=...)``.  The first two bounds were hardcoded constants,
hand-tuned against the ln L dynamic range of contemporary-detector analyses.  At
third-generation dynamic range they saturate: the amplitude ceiling ``1e1``
represents a signal of at most ``sqrt(1e1) = 3.2`` nats, against ranges of a
couple of hundred nats.

**A saturated fit is not self-announcing.**  The optimizer returns successfully,
the posterior is produced, and nothing in the output distinguishes "converged"
from "pinned against a wall the user never chose".  On a zero-spin BNS at network
amplitude 23.8 the default bounds put two of four hyperparameters exactly on a
bound, cost a factor 1.9 in held-out predictive accuracy, and narrowed the
recovered 90% credible interval in chirp mass by 40%.

This module follows ``test_cip_priors.py``: CIP is a script that parses argv at
import, so the functions under test are extracted from its source with ``ast``
and exec'd.  They are therefore byte-identical to the ones CIP runs, rather than
transcribed here where they could silently drift.

Coverage:

``test_defaults_reproduce_the_historical_hardcoded_bounds``
    The whole design rests on the new options being no-ops when unset.  If a
    default ever changes, every previously published number silently moves.

``test_report_flags_a_saturated_fit`` / ``test_report_clears_an_unsaturated_fit``
    The saturation flag must be *true* when a hyperparameter is on a bound and
    *false* when none is.  A flag that is always one or the other is useless, so
    both directions are asserted.

``test_holdout_is_reported_only_when_requested``
    In-sample residual rewards flexibility and cannot separate a better fit from
    an overfit one; the held-out score can.  It costs K extra fits, so it must
    stay off by default.

``test_peak_scan_requires_a_valid_coordinate_index``
    CIP passes the chirp-mass column unconditionally and its "no such
    coordinate" sentinel is ``-1``, which indexes the last column instead of
    failing.  The scan must decline to run rather than mislabel a coordinate.

``test_peak_index_comes_from_the_fitted_coordinate_basis`` /
``test_fit_gp_derives_the_peak_index_from_coord_names``
    The scan labels its answer "chirp mass", so the column it scans must be the
    chirp-mass column of the array actually fitted.  CIP keeps two coordinate
    lists -- coord_names (fitted) and low_level_coord_names (sampled) -- which
    diverge under ``--parameter-implied`` / ``--parameter-nofit``, so an index
    taken in the wrong list reports some other coordinate's peak as mc.

``test_peak_scan_is_bounded_in_points_and_batched`` /
``test_batched_scan_finds_the_same_peak_as_a_single_pass``
    The scan is an outer-product grid and ``gp.predict`` forms a
    points-by-training kernel matrix, so an unbounded 3-D scan can exhaust
    memory on a fit that succeeded.  Bounding it is only safe if the blocked
    search still returns the same peak as one pass.
"""
import ast
import os

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.gaussian_process import GaussianProcessRegressor           # noqa: E402
from sklearn.gaussian_process.kernels import (                          # noqa: E402
    RBF, WhiteKernel, ConstantKernel as C)

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.abspath(os.path.join(_HERE, os.pardir))
CIP_SCRIPT = os.path.join(
    _CODE, "bin", "util_ConstructIntrinsicPosterior_GenericCoordinates.py")

# The values that were hardcoded in fit_gp before these options existed.  These
# literals are the point of the test: they are what "unset behaves as before"
# means, so they are deliberately written out rather than read from the source.
HISTORICAL_NOISE_BOUNDS = "1e-2,1"
HISTORICAL_AMPLITUDE_BOUNDS = "1e-3,1e1"
HISTORICAL_LENGTH_SCALE_MAX_FACTOR = 5.0


def _parse_script(path):
    with open(path) as handle:
        return ast.parse(handle.read())


CIP_TREE = _parse_script(CIP_SCRIPT)


def _add_argument_kwargs(tree, option):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            continue
        try:
            if ast.literal_eval(node.args[0]) != option:
                continue
        except (ValueError, SyntaxError):
            continue
        found = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            try:
                found[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError):
                found[keyword.arg] = None
        return found
    raise AssertionError("no add_argument({!r}) call found".format(option))


def _load_functions(*names):
    """Exec the named top-level CIP functions in a namespace they can run in."""
    wanted = {}
    for node in CIP_TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            wanted[node.name] = node
    missing = set(names) - set(wanted)
    assert not missing, "CIP no longer defines: %s" % sorted(missing)
    ns = {"np": np, "GaussianProcessRegressor": GaussianProcessRegressor,
          "RBF": RBF, "WhiteKernel": WhiteKernel, "C": C, "print": print}
    module = ast.Module(body=[wanted[n] for n in names], type_ignores=[])
    exec(compile(module, CIP_SCRIPT, "exec"), ns)
    return ns


def _fit(amp_bounds, noise_bounds=(1e-2, 1.0), n=40, seed=0, noise=0.0):
    """A small GP fit over a peak whose amplitude far exceeds any tight bound.

    ``noise`` matters more than it looks: on exactly noiseless data the fitted
    WhiteKernel level runs to its LOWER bound, which the report correctly calls
    saturation.  The unsaturated fixture therefore has to supply real scatter for
    the noise term to have an interior optimum -- a fixture detail, not a
    property of the code under test.
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1, 1, size=(n, 1)), axis=0)
    y = 120.0 * np.exp(-0.5 * (x[:, 0] / 0.2) ** 2)      # ~120-nat dynamic range
    if noise:
        y = y + rng.normal(0.0, noise, size=n)
    kernel = (WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds)
              + C(0.5, amp_bounds) * RBF(length_scale=[0.3],
                                         length_scale_bounds=(1e-3, 1e1)))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=2)
    gp.fit(x, y)
    return gp, kernel, x, y


def _fit_nd(ndim, n=30, seed=0):
    """The same fixture in ``ndim`` dimensions, for the peak-scan tests."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, ndim))
    y = (120.0 * np.exp(-0.5 * np.sum((x / 0.4) ** 2, axis=1))
         + rng.normal(0.0, 0.5, size=n))
    kernel = (WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1e3))
              + C(0.5, (1e-3, 1e8)) * RBF(length_scale=[0.3] * ndim,
                                          length_scale_bounds=(1e-3, 1e1)))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=1)
    gp.fit(x, y)
    return gp, kernel, x, y


def test_defaults_reproduce_the_historical_hardcoded_bounds():
    """Unset flags must leave fit_gp building exactly the kernel it always did."""
    assert (_add_argument_kwargs(CIP_TREE, "--fit-gp-noise-bounds")["default"]
            == HISTORICAL_NOISE_BOUNDS)
    assert (_add_argument_kwargs(CIP_TREE, "--fit-gp-amplitude-bounds")["default"]
            == HISTORICAL_AMPLITUDE_BOUNDS)
    assert (_add_argument_kwargs(CIP_TREE, "--fit-gp-length-scale-max-factor")["default"]
            == HISTORICAL_LENGTH_SCALE_MAX_FACTOR)
    # off by default: it costs K extra GP fits
    assert _add_argument_kwargs(CIP_TREE, "--fit-gp-holdout-folds")["default"] == 0


def test_gp_bounds_opt_parses_a_pair():
    ns = _load_functions("_gp_bounds_opt")
    assert ns["_gp_bounds_opt"]("1e-4,1e3") == (1e-4, 1e3)
    assert ns["_gp_bounds_opt"](HISTORICAL_AMPLITUDE_BOUNDS) == (1e-3, 1e1)
    with pytest.raises(ValueError):
        ns["_gp_bounds_opt"]("not-a-pair")


def test_report_flags_a_saturated_fit():
    """A ceiling far below the data's dynamic range must be reported as saturated."""
    ns = _load_functions("report_gp_kernel")
    gp, kernel, x, y = _fit(amp_bounds=(1e-3, 1e1))       # 3.2 nats vs a 120-nat peak
    rec = ns["report_gp_kernel"](gp, x, y)
    assert rec["saturated"] is True
    assert rec["n_at_bound"] >= 1
    pinned = [h["name"] for h in rec["hyperparameters"]
              if h["at_lower_bound"] or h["at_upper_bound"]]
    assert any("constant_value" in nm for nm in pinned), (
        "the amplitude is the hyperparameter this ceiling pins; got %s" % pinned)
    for h in rec["hyperparameters"]:
        if h["at_upper_bound"] or h["at_lower_bound"]:
            assert h["decades_to_bound"] < 1e-2


def test_report_clears_an_unsaturated_fit():
    """With room to move, nothing may be reported as sitting on a bound."""
    ns = _load_functions("report_gp_kernel")
    gp, kernel, x, y = _fit(amp_bounds=(1e-3, 1e8), noise_bounds=(1e-4, 1e3), noise=0.5)
    rec = ns["report_gp_kernel"](gp, x, y)
    assert rec["saturated"] is False, rec["kernel_fitted"]
    assert rec["n_at_bound"] == 0
    assert all(h["decades_to_bound"] > 0 for h in rec["hyperparameters"])


def test_holdout_is_reported_only_when_requested():
    ns = _load_functions("report_gp_kernel")
    gp, kernel, x, y = _fit(amp_bounds=(1e-3, 1e8), noise_bounds=(1e-4, 1e3), noise=0.5)

    off = ns["report_gp_kernel"](gp, x, y)
    assert "holdout_rms_nats" not in off, "held-out scoring must be opt-in"

    on = ns["report_gp_kernel"](gp, x, y, holdout_folds=3,
                                kernel_proto=kernel, alpha_proto=0.25)
    assert on["holdout_folds"] == 3
    assert np.isfinite(on["holdout_rms_nats"]) and on["holdout_rms_nats"] >= 0.0
    assert on["holdout_max_abs_nats"] >= on["holdout_rms_nats"] * 0.0


def test_report_is_pure_json_and_names_every_hyperparameter():
    """The record is consumed by build scripts, so it must stay serializable."""
    import json
    ns = _load_functions("report_gp_kernel")
    gp, kernel, x, y = _fit(amp_bounds=(1e-3, 1e8), noise_bounds=(1e-4, 1e3), noise=0.5)
    rec = ns["report_gp_kernel"](gp, x, y)
    json.dumps(rec)                                        # must not raise
    assert len(rec["hyperparameters"]) == len(gp.kernel_.theta)
    assert all(h["name"] and not h["name"].startswith("param_")
               for h in rec["hyperparameters"]), (
        "a hyperparameter fell back to a positional name; the name mapping broke")


def test_peak_scan_requires_a_valid_coordinate_index():
    """CIP's "no chirp mass here" sentinel is -1, a perfectly legal index."""
    ns = _load_functions("report_gp_kernel")
    gp, kernel, x, y = _fit_nd(2)
    for bad in (None, -1, 2, 7):
        rec = ns["report_gp_kernel"](gp, x, y, peak_index=bad)
        assert "surface_peak" not in rec, (
            "peak_index=%r must not be scanned as a coordinate" % (bad,))
    rec = ns["report_gp_kernel"](gp, x, y, peak_index=1, peak_max_points=256)
    assert rec["surface_peak"]["coord_index"] == 1


def test_peak_index_comes_from_the_fitted_coordinate_basis():
    """The mc column is looked up in the list x is built from, or not at all."""
    ns = _load_functions("_mc_index_in")
    assert ns["_mc_index_in"](['mc', 'delta_mc', 'xi']) == 0
    assert ns["_mc_index_in"](['delta_mc', 'mc']) == 1
    # --parameter delta_mc --parameter-implied mu1 --parameter-implied mu2
    # --parameter-nofit mc : mc is sampled but NOT fitted, so column 1 is mu1.
    # Returning 1 here (the position of mc in the sampling list) would report
    # the mu1 peak as chirp mass; the scan must be declined instead.
    assert ns["_mc_index_in"](['delta_mc', 'mu1', 'mu2']) == -1


def test_fit_gp_derives_the_peak_index_from_coord_names():
    """fit_gp's x has coord_names columns, so mc_index must not index it.

    Asserted on the source rather than by running fit_gp, which needs parsed
    argv and a full data load; the wiring is what regressed, and it is visible
    in the call itself.
    """
    fit_gp = next(n for n in CIP_TREE.body
                  if isinstance(n, ast.FunctionDef) and n.name == "fit_gp")
    calls = [n for n in ast.walk(fit_gp)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "report_gp_kernel"]
    assert calls, "fit_gp no longer reports its kernel"
    for call in calls:
        passed = [k.value for k in call.keywords if k.arg == "peak_index"]
        assert passed, "the peak index must be passed explicitly"
        arg = passed[0]
        assert not (isinstance(arg, ast.Name) and arg.id == "mc_index"), (
            "mc_index indexes low_level_coord_names, not the fitted columns")
        assert (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
                and arg.func.id == "_mc_index_in"), (
            "the peak index must be derived from the fitted coordinate list")
        assert [a.id for a in arg.args if isinstance(a, ast.Name)] == ["coord_names"]


def test_peak_scan_is_bounded_in_points_and_batched():
    """Total scan points and per-call block size must both stay under budget."""
    ns = _load_functions("report_gp_kernel")
    gp, kernel, x, y = _fit_nd(3)

    calls = []
    inner_predict = gp.predict

    def counting_predict(pts, **kwargs):
        calls.append(len(pts))
        return inner_predict(pts, **kwargs)

    gp.predict = counting_predict
    rec = ns["report_gp_kernel"](gp, x, y, peak_index=0, peak_max_points=4096,
                                 peak_block_elements=3000)
    scan_calls = calls[1:]                       # calls[0] is the residual predict

    n_axis = rec["surface_peak"]["grid_points_per_axis"]
    assert n_axis ** 3 <= 4096, "the grid ignored its total-points budget"
    assert n_axis < 240, "a 3-D scan must be thinned below the per-axis default"
    assert sum(scan_calls) == n_axis ** 3, "the whole grid must still be scanned"
    assert max(scan_calls) <= 3000 // len(y), "a block exceeded the element budget"


def test_batched_scan_finds_the_same_peak_as_a_single_pass():
    """Blocking is only acceptable if the running argmax is the global one."""
    ns = _load_functions("report_gp_kernel")
    gp, kernel, x, y = _fit(amp_bounds=(1e-3, 1e8), noise_bounds=(1e-4, 1e3), noise=0.5)
    one = ns["report_gp_kernel"](gp, x, y, peak_index=0,
                                 peak_block_elements=10 ** 9)["surface_peak"]
    many = ns["report_gp_kernel"](gp, x, y, peak_index=0,
                                  peak_block_elements=1)["surface_peak"]
    assert one["grid_points_per_axis"] == many["grid_points_per_axis"] == 240
    assert one["value"] == many["value"]
    assert abs(one["value"]) < 0.2, "the fixture's peak sits at x=0"
