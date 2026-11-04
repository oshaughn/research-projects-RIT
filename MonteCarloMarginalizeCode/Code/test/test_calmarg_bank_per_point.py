"""--calibration-bank-per-point: a fresh calibration bank for each intrinsic point.

1. The driver's own helper (lifted from the driver source): two points get different
   banks, the same (seed, point) reproduces its bank exactly, a different seed changes
   it, and the size resets to the starting --calibration-n-realizations.  Prior and
   export (node-retaining) branches both.
2. With the option off the helper is unreachable: its only call is guarded by the flag.
3. End to end: two intrinsic points in one job with the option on report two different
   banks, and each point still exports its own fair draws.
4. The redrawn bank follows --calibration-correction-type per detector, and its exported
   nodes are the nodes of the realizations in that bank.
5. End to end: with the option off, outputs are byte-identical to the driver with the
   option's code deleted; with it on, lnL changes (the likelihood sees the new bank).
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

DRIVER = Path(__file__).resolve().parents[1] / "bin" / "integrate_likelihood_extrinsic_batchmode"


@pytest.fixture(autouse=True)
def _restore_seed_state():
    """seed_everything sets process-global RNG state; leave it as we found it."""
    from RIFT.integrators import seeding
    import random
    saved = (seeding._seed_used, dict(seeding._stream_counters), np.random.get_state(), random.getstate())
    yield
    np.random.set_state(saved[2])
    random.setstate(saved[3])
    seeding._seed_used = saved[0]
    seeding._stream_counters.clear(); seeding._stream_counters.update(saved[1])


def _envelope(path):
    f = np.geomspace(10, 1024, 20)
    one, zero = np.ones_like(f), np.zeros_like(f)
    np.savetxt(path, np.column_stack([f, one, zero, one - .05, zero - .03, one + .05, zero + .03]))


def _lift(tmp_path, export, n_initial=4):
    """Exec the driver's _redraw_calibration_bank_for_point in a stand-in namespace."""
    for ifo in ("H1", "L1"):
        _envelope(tmp_path / ("%s.txt" % ifo))
    tree = ast.parse(DRIVER.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == "_redraw_calibration_bank_for_point"]
    assert len(funcs) == 1
    opts = SimpleNamespace(calibration_export_posterior=export, calibration_proposal_breadcrumb=None,
                           calibration_envelope_directory=str(tmp_path), fmin_template=20,
                           calibration_spline_count=4, calibration_n_realizations=3 * n_initial)
    ns = dict(np=np, opts=opts, P=SimpleNamespace(deltaF=1., deltaT=1 / 256.), fmax=120,
              cal_fmin_ifo={"H1": 20, "L1": 20},
              calibration_correction_types={"H1": "template", "L1": "template"},
              calibration_current_prior=None, calibration_realization_dict={},
              calibration_nodes=None, calibration_log_weights=None,
              _calibration_n_realizations_initial=n_initial)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(DRIVER), "exec"), ns)
    return ns


def _bank(ns, seed, point):
    from RIFT.integrators.seeding import seed_everything
    seed_everything(seed, verbose=False)
    ns["_redraw_calibration_bank_for_point"](point, {"H1": None, "L1": None})
    return ({k: v.copy() for k, v in ns["calibration_realization_dict"].items()},
            None if ns["calibration_nodes"] is None else ns["calibration_nodes"].copy())


@pytest.mark.parametrize("export", [False, True])
def test_banks_differ_between_points_and_reproduce(tmp_path, export):
    ns = _lift(tmp_path, export)
    a7, n7 = _bank(ns, 11, 7)
    assert ns["opts"].calibration_n_realizations == 4      # reset from the grown 12
    for ifo in ("H1", "L1"):
        assert a7[ifo].shape == (256, 4)
    a8, _ = _bank(ns, 11, 8)
    a7_again, n7_again = _bank(ns, 11, 7)
    a7_other_seed, _ = _bank(ns, 12, 7)
    for ifo in ("H1", "L1"):
        assert not np.allclose(a7[ifo], a8[ifo])
        np.testing.assert_array_equal(a7[ifo], a7_again[ifo])
        assert not np.allclose(a7[ifo], a7_other_seed[ifo])
        # realizations within one bank are distinct draws
        assert not np.allclose(a7[ifo][:, 0], a7[ifo][:, 1])
    if export:
        assert n7.shape == (4, 2 * 2 * 4)
        np.testing.assert_array_equal(n7, n7_again)
    else:
        assert n7 is None


def _rebuild(nodes, fmin_ifo=None):
    """The bank these exported nodes describe (H1 then L1, fmax 120, fmin 20 unless given)."""
    from RIFT.calmarg import generate_realizations as gr
    fmin_ifo = fmin_ifo or {"H1": 20, "L1": 20}
    out = {}
    for i, ifo in enumerate(("H1", "L1")):
        log_f = np.linspace(np.log10(fmin_ifo[ifo]), np.log10(120), 4)
        out[ifo] = gr.build_realizations_from_nodes(nodes[:, 8 * i:8 * i + 4].T, nodes[:, 8 * i + 4:8 * i + 8].T,
                                                    1., 1 / 256., fmin_ifo[ifo], 120, log_f)
    return out


def _skewed_envelope(path):
    """Median far from 1, so the data and template conventions give disjoint node priors:
    template amp 1.3 +- 0.02, phase 0.2 +- 0.01; data amp 1/1.3 +- 0.012, phase -0.2."""
    f = np.geomspace(10, 1024, 20)
    c = np.ones_like(f)
    np.savetxt(path, np.column_stack([f, 1.3 * c, .2 * c, 1.28 * c, .19 * c, 1.32 * c, .21 * c]))


@pytest.mark.parametrize("export", [False, True])
def test_redraw_follows_correction_types_and_exports_its_own_nodes(tmp_path, export):
    """H1 'template' is not H1's default ('data'), so a redraw that ignored
    calibration_correction_types would put H1's amplitude near 1/1.3, not 1.3.  L1 starts
    at 30 Hz (cal_fmin_ifo), so its 20-30 Hz rows stay exactly 1 while H1's do not.
    With export, the nodes must rebuild the bank's realizations exactly."""
    from RIFT.calmarg import generate_realizations as gr
    ns = _lift(tmp_path, export)
    for ifo in ("H1", "L1"):
        _skewed_envelope(tmp_path / ("%s.txt" % ifo))
    types = {"H1": "template", "L1": "data"}
    fmin_ifo = {"H1": 20, "L1": 30}
    ns["calibration_correction_types"] = types
    ns["cal_fmin_ifo"] = fmin_ifo
    bank, nodes = _bank(ns, 11, 7)
    for ifo, amp in (("H1", 1.3), ("L1", 1 / 1.3)):
        band = np.any(bank[ifo] != 1, axis=1)
        assert band.sum() > 50
        np.testing.assert_allclose(np.median(np.abs(bank[ifo][band])), amp, rtol=0.05)
    f_abs = np.abs(128 - np.arange(256))   # RIFT packing label, deltaF = 1 Hz, 256 bins
    low = (f_abs >= 21) & (f_abs <= 29)
    assert np.all(bank["L1"][low] == 1) and np.all(bank["H1"][low] != 1)
    if not export:
        return
    prior = gr.prior_cal_breadcrumb_dict(str(tmp_path), ["H1", "L1"], 20, 120, 4,
                                         fmin_ifo=fmin_ifo, correction_types=types)
    assert np.all(np.abs((nodes - prior["prior_mean"]) / prior["prior_sigma"]) < 6)
    rebuilt = _rebuild(nodes, fmin_ifo)
    for ifo in ("H1", "L1"):
        np.testing.assert_allclose(bank[ifo], rebuilt[ifo], rtol=0, atol=1e-12)


def test_breadcrumb_branch_redraws_from_the_proposal(tmp_path):
    """--calibration-proposal-breadcrumb path: realizations AND importance weights are
    replaced per point from the breadcrumb loaded at startup (_bc), and reproduce.  The
    weights use the LIVE prior (calibration_current_prior), not the one the breadcrumb
    stored, and the exported nodes rebuild the bank.  L1 starts at 30 Hz (cal_fmin_ifo)."""
    from RIFT.calmarg import generate_realizations as gr
    from RIFT.calmarg import adaptive
    ns = _lift(tmp_path, export=True)
    fmin_ifo = {"H1": 20, "L1": 30}
    ns["cal_fmin_ifo"] = fmin_ifo
    prior = gr.prior_cal_breadcrumb_dict(str(tmp_path), ["H1", "L1"], 20, 120, 4, fmin_ifo=fmin_ifo)
    # A proposal narrower than, and offset from, the prior: draws come from it and carry
    # nonzero importance weights, which a prior-branch draw could not imitate.  Its stored
    # prior is twice as wide as the live one, so weights built from it are wrong.
    proposal = dict(prior, proposal_mean=prior["prior_mean"] + 0.5 * prior["prior_sigma"],
                    proposal_cov=np.diag((0.3 * prior["prior_sigma"]) ** 2),
                    prior_sigma=2 * prior["prior_sigma"])
    ns["_bc"] = {"cal": proposal}
    ns["opts"].calibration_proposal_breadcrumb = "loaded-at-startup.npz"
    ns["calibration_current_prior"] = prior
    a7, n7 = _bank(ns, 11, 7)
    w7 = np.array(ns["calibration_log_weights"])
    a8, n8 = _bank(ns, 11, 8)
    w8 = np.array(ns["calibration_log_weights"])
    a7b, n7b = _bank(ns, 11, 7)
    assert w7.shape == (4,) and np.all(np.isfinite(w7))
    np.testing.assert_allclose(
        w7, adaptive.log_prior(n7, prior["prior_mean"], prior["prior_sigma"])
        - adaptive._mvn_logpdf(n7, proposal["proposal_mean"], proposal["proposal_cov"]), rtol=1e-10)
    rebuilt = _rebuild(n7, fmin_ifo)
    for ifo in ("H1", "L1"):
        np.testing.assert_allclose(a7[ifo], rebuilt[ifo], rtol=0, atol=1e-12)
    low = (np.abs(128 - np.arange(256)) >= 21) & (np.abs(128 - np.arange(256)) <= 29)
    assert np.all(a7["L1"][low] == 1) and np.all(a7["H1"][low] != 1)
    assert np.ptp(w7) > 0 and not np.allclose(w7, w8)
    # nodes sit near the proposal mean, within a few proposal sigmas
    z = (n7 - proposal["proposal_mean"]) / (0.3 * prior["prior_sigma"])
    assert np.all(np.abs(z) < 6), z
    for ifo in ("H1", "L1"):
        assert a7[ifo].shape == (256, 4)
        assert not np.allclose(a7[ifo], a8[ifo])
        np.testing.assert_array_equal(a7[ifo], a7b[ifo])
    np.testing.assert_array_equal(n7, n7b)
    assert not np.allclose(n7, n8)


def test_per_point_and_shared_banks_draw_the_same_prior(tmp_path):
    """Without export the shared bank comes from create_realizations (global RNG) and the
    per-point bank from draw_prior_realizations_with_nodes.  Same prior: compare the
    realization mean and spread in band over many draws."""
    from RIFT.calmarg import generate_realizations as gr
    f = np.geomspace(10, 1024, 20)
    np.savetxt(tmp_path / "H1.txt", np.column_stack(
        [f, 1 + .02 * np.log(f / 10), .01 * np.log(f / 10), .95 + 0 * f, -.03 + 0 * f, 1.06 + 0 * f, .04 + 0 * f]))
    n = 3000
    np.random.seed(5)
    # The driver passes the same per-IFO correction type to both; so does this test.
    shared = gr.create_realizations(str(tmp_path / "H1.txt"), 1., 1 / 256., 20, 120, 4, n,
                                    correction_type="data")
    per_point = gr.draw_prior_realizations_with_nodes(str(tmp_path), ["H1"], 1., 1 / 256., 20, 120, 4, n,
                                                      rng=np.random.default_rng(6),
                                                      correction_types={"H1": "data"})["realizations"]["H1"]
    band = np.ptp(np.abs(shared), axis=1) > 0   # in-band rows; both are exactly 1 outside
    assert band.sum() > 50
    for part in (np.abs, np.angle):
        a, b = part(shared[band]), part(per_point[band])
        se = np.sqrt(a.var(axis=1) / n + b.var(axis=1) / n) + 1e-12
        assert np.all(np.abs(a.mean(axis=1) - b.mean(axis=1)) < 5 * se)
        np.testing.assert_allclose(a.std(axis=1), b.std(axis=1), rtol=0.1, atol=1e-9)


def _option_readers(tree):
    """Every AST node that names the option's dest: opts.calibration_bank_per_point, a bare
    name, or a string such as getattr(opts, 'calibration_bank_per_point')."""
    name = "calibration_bank_per_point"
    return [n for n in ast.walk(tree)
            if (isinstance(n, ast.Attribute) and n.attr == name)
            or (isinstance(n, ast.Name) and n.id == name)
            or (isinstance(n, ast.Constant) and isinstance(n.value, str) and name in n.value)]


def test_the_redraw_runs_only_under_the_flag():
    """Option off => the shared startup bank, untouched: the helper's only call site is
    guarded by opts.calibration_bank_per_point."""
    tree = ast.parse(DRIVER.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_redraw_calibration_bank_for_point"]
    assert len(calls) == 1
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.If)
              and any(c is calls[0] for st in n.body for c in ast.walk(st))]
    assert len(guards) == 1, "redraw call must sit directly under one guard"
    assert ast.unparse(guards[0].test) == "calibration_marginalization and opts.calibration_bank_per_point"
    assert not guards[0].orelse, "option off must run nothing here"
    # ... and that guard is the driver's only reader of the option
    readers = _option_readers(tree)
    assert len(readers) == 1 and any(r is readers[0] for r in ast.walk(guards[0].test))
    # ... and outside its own block the feature appears nowhere else: no option-off code
    # (e.g. a size reset to _calibration_n_realizations_initial) runs for it
    _, outer = _feature_statements(tree)
    assert len(outer) == 1 and outer[0] is guards[0], [ast.unparse(st)[:80] for st in outer]
    # the point index is the GRID row, not the offset within this job
    assert ast.unparse(calls[0].args[0]) == "opts.event + indx_event"
    # ... and it runs BEFORE the bank is handed to the precompute, or the likelihood
    # would integrate over the old bank while the log and export report the new one.
    hand_off = [n for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign))
                and any("extra_kwargs['calibration_realizations']" == ast.unparse(t)
                        for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    assert len(hand_off) == 1, [n.lineno for n in hand_off]
    assert ast.unparse(hand_off[0].value) == "calibration_realization_dict"
    assert calls[0].lineno < hand_off[0].lineno
    opt = re.search(r'add_option\("--calibration-bank-per-point",action=\'store_true\',default=False',
                    DRIVER.read_text())
    assert opt, "--calibration-bank-per-point must default to off"


def test_two_points_get_different_banks_end_to_end(tmp_path):
    import test_calexport_multievent as tcm
    d = tcm._make_fixture(tmp_path)
    proc = tcm._run_driver(d, "--calibration-bank-per-point")
    assert proc.returncode == 0, proc.stdout[-4000:]
    lines = [ln for ln in proc.stdout.splitlines() if "[calmarg bank]" in ln]
    assert len(lines) == tcm.N_POINTS, proc.stdout[-4000:]
    assert "point 0:" in lines[0] and "point 1:" in lines[1], lines
    sums = [ln.rsplit("checksum", 1)[1] for ln in lines]
    assert sums[0] != sums[1], lines
    tcm._check_exports(d["outdir"], proc)
    # The same grid row analyzed alone (--event 1, one point) starts from the same bank.
    (tmp_path / "single").mkdir()
    d1 = tcm._make_fixture(tmp_path / "single")
    proc1 = tcm._run_driver(d1, "--calibration-bank-per-point", "--event", "1",
                            "--n-events-to-analyze", "1")
    assert proc1.returncode == 0, proc1.stdout[-4000:]
    single = [ln for ln in proc1.stdout.splitlines() if "[calmarg bank]" in ln]
    assert len(single) == 1 and "point 1:" in single[0], single
    assert single[0].rsplit("checksum", 1)[1] == sums[1], (single, lines)


_CHEAP = ("--n-max", "2000")
_GUARD = "calibration_marginalization and opts.calibration_bank_per_point"


def _names_the_feature(node):
    """Any identifier, attribute or string naming the option, or one of its helpers."""
    def hit(text):
        return "bank_per_point" in text or text in (
            "_calibration_n_realizations_initial", "_redraw_calibration_bank_for_point")
    return any((isinstance(n, ast.Attribute) and hit(n.attr)) or (isinstance(n, ast.Name) and hit(n.id))
               or (isinstance(n, ast.Constant) and isinstance(n.value, str) and hit(n.value))
               for n in ast.walk(node))


def _feature_statements(tree):
    """(block, outer): the option's module-level block, from
    `_calibration_n_realizations_initial = ...` through the redraw helper, and the outermost
    statements elsewhere whose own header (not their nested blocks) names the option or its
    helpers."""
    body = tree.body
    first = next(i for i, st in enumerate(body) if isinstance(st, ast.Assign)
                 and ast.unparse(st.targets[0]) == "_calibration_n_realizations_initial")
    last = next(i for i, st in enumerate(body) if isinstance(st, ast.FunctionDef)
                and st.name == "_redraw_calibration_bank_for_point")
    assert first < last
    block = body[first:last + 1]
    lo, hi = block[0].lineno, block[-1].end_lineno

    def header_names_the_feature(st):
        # the statement itself, not the blocks nested in it (an `if` is hit by its test)
        return any(_names_the_feature(v) for f, v in ast.iter_fields(st)
                   if f not in ("body", "orelse", "handlers", "finalbody")
                   for v in (v if isinstance(v, list) else [v]) if isinstance(v, ast.AST))
    hits = [st for st in ast.walk(tree)
            if isinstance(st, ast.stmt) and not isinstance(st, (ast.FunctionDef, ast.ClassDef))
            and not lo <= st.lineno <= hi and header_names_the_feature(st)]
    # outermost hits only: deleting an `if` deletes its body and its else
    outer = [st for st in hits
             if not any(o is not st and any(st is n for n in ast.walk(o)) for o in hits)]
    return block, outer


def _driver_without_the_redraw(dest):
    """The driver with the option's code (see _feature_statements) replaced by `pass`: the
    option-off path as it was before --calibration-bank-per-point."""
    src = DRIVER.read_text()
    block, outer = _feature_statements(ast.parse(src))
    assert any(isinstance(st, ast.If) and ast.unparse(st.test) == _GUARD for st in outer)
    lines = src.splitlines(keepends=True)
    for st in sorted(block + outer, key=lambda n: -n.lineno):
        indent = lines[st.lineno - 1][:st.col_offset]
        lines[st.lineno - 1:st.end_lineno] = [indent + "pass\n"]
    out = dest / "ile_without_redraw"
    out.write_text("".join(lines))
    assert not _names_the_feature(ast.parse(out.read_text()))
    return out


def test_option_off_is_unchanged_and_option_on_reaches_the_likelihood(tmp_path):
    """Off: every output byte-identical to the driver without the option's code, so the
    option costs the default path no RNG draw.  On: lnL differs from off at every point, so
    the likelihood integrates over the redrawn bank, not the startup one (zero data: lnL
    still depends on the bank through <C h|C h>)."""
    import test_calexport_multievent as tcm
    out = {}
    for name, driver, extra in (("off", DRIVER, ()),
                                ("stripped", _driver_without_the_redraw(tmp_path), ()),
                                ("on", DRIVER, ("--calibration-bank-per-point",))):
        (tmp_path / name).mkdir()
        d = tcm._make_fixture(tmp_path / name)
        proc = tcm._run_driver(d, *(_CHEAP + extra), driver=driver)
        assert proc.returncode == 0, proc.stdout[-4000:]
        assert "NOT APPLIED" not in proc.stdout
        assert ("[calmarg bank]" in proc.stdout) == (name == "on")
        out[name] = d["outdir"]
    for i in range(tcm.N_POINTS):
        for suffix in ("_.dat", "_cal.dat"):
            f = "out.xml_%d%s" % (i, suffix)
            assert (out["off"] / f).read_bytes() == (out["stripped"] / f).read_bytes(), f
        lnl_off = np.loadtxt(out["off"] / ("out.xml_%d_.dat" % i))[-4]
        lnl_on = np.loadtxt(out["on"] / ("out.xml_%d_.dat" % i))[-4]
        assert lnl_on != lnl_off, (i, lnl_on, lnl_off)
    # The export reads each point's own bank: the calibration nodes exported with the
    # option on share no row between the two points, nor with the shared startup bank's
    # draws exported with the option off.
    def node_rows(run, i):
        f = out[run] / ("out.xml_%d_cal.dat" % i)
        header = f.read_text().splitlines()[0].lstrip("#").split()
        cols = [j for j, h in enumerate(header) if h.startswith("cal_")]
        assert cols
        return {tuple(r) for r in np.atleast_2d(np.loadtxt(f))[:, cols]}
    on0, on1 = node_rows("on", 0), node_rows("on", 1)
    off = node_rows("off", 0) | node_rows("off", 1)
    assert not (on0 & on1) and not ((on0 | on1) & off)
