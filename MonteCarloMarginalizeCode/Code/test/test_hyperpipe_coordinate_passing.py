"""Coordinate-plugin wiring across the hyperpipe post and puff stages.

Runs the real drivers on a 45-degree rotation plugin, (x, y) -> (u, v):

  * hyperpipe -> puff: when the MC samples the plugin's output basis, the puff
    stage must get the plugin, or it reads columns (u, v) that the grid file
    (x, y) does not have;
  * puffball ranges: the old downselect+reflect idiom, name:[lo,hi] ranges,
    reflection without a downselect, and the get_bounds hook;
  * CEP get_bounds: a rotated-frame box from the plugin recovers the known
    posterior mean, and a failing hook stops the run.

Cost: about ten driver subprocesses, a few seconds each.
"""
import os
import shlex
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
PUFF = os.path.join(CODE, "bin", "util_HyperparameterPuffball.py")
CEP = os.path.join(CODE, "bin", "util_ConstructEOSPosterior.py")
sys.path.insert(0, CODE)
from RIFT.hyperpipe.coords import HyperCoordSpec  # noqa: E402

ROOT2 = np.sqrt(2.0)
PLUGIN = '''
import numpy as np
INPUT_PARAMETERS = ["x", "y"]
_c = _s = np.sqrt(0.5)

def convert_coordinates(x_in, coord_names, low_level_coord_names, **kw):
    X = np.atleast_2d(x_in)
    cols = {n: X[:, i] for i, n in enumerate(low_level_coord_names)}
    cols["u"] = _c * cols["x"] + _s * cols["y"]
    cols["v"] = -_s * cols["x"] + _c * cols["y"]
    return np.column_stack([cols[n] for n in coord_names])

def inverse_convert_coordinates(y_in, coord_names, low_level_coord_names, **kw):
    Y = np.atleast_2d(y_in)
    d = {n: Y[:, i] for i, n in enumerate(coord_names)}
    out = {"x": _c * d["u"] - _s * d["v"], "y": _s * d["u"] + _c * d["v"]}
    return np.column_stack([out[n] for n in low_level_coord_names])

def get_bounds(coord_names, ranges, fail=False, half_width=1.5, **kw):
    if fail:
        raise RuntimeError("get_bounds asked to fail")
    c = 3 * np.sqrt(2.0)
    return {"u": [c - half_width, c + half_width], "v": [-half_width, half_width]}
'''


def _env(tmp):
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(OMP_NUM_THREADS="1", MPLBACKEND="Agg",
               XDG_CACHE_HOME=os.path.join(tmp, "cache"), MPLCONFIGDIR=os.path.join(tmp, "mpl"))
    return env


def _run(tmp, driver, args):
    proc = subprocess.run([sys.executable, driver] + list(args), cwd=tmp, env=_env(tmp),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=900)
    return proc


def _uv(x, y):
    return (x + y) / ROOT2, (y - x) / ROOT2


@pytest.fixture
def work(tmp_path):
    tmp = str(tmp_path)
    plugin = os.path.join(tmp, "rot45_plugin.py")
    with open(plugin, "w") as f:
        f.write(PLUGIN)
    rng = np.random.default_rng(20261003)
    n = 4000
    # Cloud elongated along v, centred at (x, y) = (3, 3), i.e. (u, v) = (3 sqrt2, 0).
    u = 3 * ROOT2 + 0.2 * rng.standard_normal(n)
    v = 0.6 * rng.standard_normal(n)
    x, y = (u - v) / ROOT2, (u + v) / ROOT2
    grid = os.path.join(tmp, "grid.dat")
    np.savetxt(grid, np.column_stack([np.zeros(n), np.zeros(n), x, y]), header="lnL sigma_lnL x y")
    return tmp, plugin, grid


def _out(tmp, name="out.dat"):
    return np.genfromtxt(os.path.join(tmp, name), names=True)


# -------------------------------------------------------------- hyperpipe -> puff

def _spec(plugin, input_parameters="x y"):
    return HyperCoordSpec.from_strings(
        name=plugin, coords_fit="u v",
        coords_sample="u:[3.5,5] v:[-1,1]",
        coord_input_parameters=input_parameters,
    )


def _downselect(spec, names):
    # Same emission as util_RIFT_hyperpipe._build_puff_args.
    bits = []
    for p in names:
        lo, hi = spec.parameter_ranges[p]
        bits += ["--downselect-parameter", p, "--downselect-parameter-range", f"[{lo},{hi}]"]
    return bits


def test_hyperpipe_puff_gets_plugin_and_round_trips(work):
    tmp, plugin, grid = work
    spec = _spec(plugin)
    names, use_plugin = spec.puff_basis("auto")
    assert (names, use_plugin) == (["u", "v"], True)
    args = shlex.split(spec.to_puff_args(force_away=0, puff_factor=0.5)) + _downselect(spec, names)
    assert "--supplementary-coordinate-code" in args
    proc = _run(tmp, PUFF, ["--inj-file", grid, "--inj-file-out", "out.dat"] + args)
    assert proc.returncode == 0, proc.stdout[-3000:]
    out = _out(tmp)
    # The test stage reads the file columns, not the plugin basis.
    assert spec.test_basis() == ["x", "y"]
    assert set(spec.test_basis()) <= set(out.dtype.names)
    u, v = _uv(out["x"], out["y"])
    assert len(out) > 1000
    assert u.min() >= 3.5 - 1e-9 and u.max() <= 5 + 1e-9
    assert v.min() >= -1 - 1e-9 and v.max() <= 1 + 1e-9


def test_puff_without_plugin_cannot_read_plugin_basis(work):
    """The pre-fix emission: --parameter u v with no plugin, against an (x, y) grid."""
    tmp, plugin, grid = work
    proc = _run(tmp, PUFF, ["--inj-file", grid, "--inj-file-out", "out.dat",
                            "--parameter", "u", "--parameter", "v"])
    assert proc.returncode != 0


def test_legacy_spec_emission_unchanged(work):
    """No coord-input-parameters: same args as before, no plugin on the puff."""
    _, plugin, _ = work
    spec = HyperCoordSpec.from_strings(name=plugin, coords_fit="x y",
                                       coords_sample="x:[0,6] y:[0,6]")
    assert spec.to_puff_args(force_away=0.03, puff_factor=0.5).split() == \
        "--force-away 0.03 --puff-factor 0.5 --parameter x --parameter y".split()
    assert spec.to_post_args().count("--supplementary-coordinate-code") == 1


def test_mixed_sampling_basis_refused(work):
    _, plugin, _ = work
    spec = HyperCoordSpec.from_strings(name=plugin, coords_fit="u", coords_nofit="x",
                                       coords_sample="u:[3.5,5] x:[0,6]",
                                       coord_input_parameters="x y")
    with pytest.raises(ValueError, match="mixes"):
        spec.to_puff_args()


def test_input_parameter_in_post_extra_args_refused():
    """CEP has no --supplementary-coordinate-input-parameter; extra-args reaches it verbatim."""
    from RIFT.hyperpipe.coords import coord_spec_from_config_section
    with pytest.raises(ValueError, match="coord-input-parameters"):
        coord_spec_from_config_section({
            "coord-module": "m.py", "coords-fit": "u", "coords-sample": "u:[0,1]",
            "extra-args": "--supplementary-coordinate-input-parameter x"})


def test_plugin_puff_for_implied_nofit_config(work):
    """coord-basis plugin: puff in (u, v) while the MC samples (x, y)."""
    _, plugin, _ = work
    spec = HyperCoordSpec.from_strings(name=plugin, coords_implied="u v", coords_nofit="x y",
                                       coords_sample="x:[0,6] y:[0,6]",
                                       coord_input_parameters="x y")
    assert spec.puff_basis("auto") == (["x", "y"], False)
    assert spec.puff_basis("plugin") == (["u", "v"], True)
    assert spec.test_basis() == ["x", "y"]


# -------------------------------------------------------------- puffball ranges

def _puff(tmp, grid, extra):
    return _run(tmp, PUFF, ["--inj-file", grid, "--inj-file-out", "out.dat",
                            "--parameter", "x", "--parameter", "y", "--puff-factor", "2"] + extra)


@pytest.mark.parametrize("extra", [
    ["--downselect-parameter", "x", "--downselect-parameter-range", "[2.5,3.5]", "--reflect-parameter", "x"],
    ["--parameter-range", "x:[2.5,3.5]", "--reflect-parameter", "x"],
    ["--downselect-parameter", "x", "--downselect-parameter-range", "x:[2.5,3.5]", "--reflect-parameter", "x"],
], ids=["legacy-idiom", "parameter-range", "named-downselect"])
def test_reflection_keeps_all_points_in_range(work, extra):
    tmp, _, grid = work
    proc = _puff(tmp, grid, extra)
    assert proc.returncode == 0, proc.stdout[-3000:]
    out = _out(tmp)
    assert len(out) == 4000                      # reflected, not dropped
    assert out["x"].min() >= 2.5 and out["x"].max() <= 3.5


def test_named_range_overrides_hyperpipe_positional_range(work):
    """hyperpipe emits a positional downselect; a user's named range for the same name wins."""
    tmp, _, grid = work
    proc = _puff(tmp, grid, ["--downselect-parameter", "x", "--downselect-parameter-range", "[-10,10]",
                             "--parameter-range", "x:[2.5,3.5]", "--reflect-parameter", "x"])
    assert proc.returncode == 0, proc.stdout[-3000:]
    out = _out(tmp)
    assert len(out) == 4000 and out["x"].min() >= 2.5 and out["x"].max() <= 3.5


def test_range_for_unknown_parameter_refused(work):
    tmp, _, grid = work
    proc = _puff(tmp, grid, ["--parameter-range", "z:[0,1]"])
    assert proc.returncode != 0
    assert "not a --parameter" in proc.stdout


def test_named_downselect_drops_points(work):
    tmp, _, grid = work
    proc = _puff(tmp, grid, ["--parameter-range", "y:[2.8,3.2]"])
    assert proc.returncode == 0, proc.stdout[-3000:]
    out = _out(tmp)
    assert 0 < len(out) < 4000
    assert out["y"].min() >= 2.8 and out["y"].max() <= 3.2


def test_mismatched_downselect_ranges_refused(work):
    tmp, _, grid = work
    proc = _puff(tmp, grid, ["--downselect-parameter", "x", "--downselect-parameter", "y",
                             "--downselect-parameter-range", "[2,4]"])
    assert proc.returncode != 0
    assert "inconsistent" in proc.stdout


def test_puff_get_bounds(work):
    tmp, plugin, grid = work
    proc = _run(tmp, PUFF, ["--inj-file", grid, "--inj-file-out", "out.dat",
                            "--parameter", "u", "--parameter", "v", "--puff-factor", "2",
                            "--supplementary-coordinate-code", plugin,
                            "--get-range-from-external", "--external-range-args", "half_width=0.5"])
    assert proc.returncode == 0, proc.stdout[-3000:]
    u, v = _uv(_out(tmp)["x"], _out(tmp)["y"])
    assert np.all(np.abs(u - 3 * ROOT2) <= 0.5 + 1e-9) and np.all(np.abs(v) <= 0.5 + 1e-9)


# -------------------------------------------------------------- CEP get_bounds

def _cep_grid(tmp):
    rng = np.random.default_rng(7)
    n = 600
    x = rng.uniform(1.5, 4.5, n)
    y = rng.uniform(1.5, 4.5, n)
    u, v = _uv(x, y)
    lnL = -0.5 * ((u - 3 * ROOT2) / 0.25) ** 2 - 0.5 * (v / 0.6) ** 2 + 10
    path = os.path.join(tmp, "cep_grid.dat")
    np.savetxt(path, np.column_stack([lnL, np.zeros(n), x, y]), header="lnL sigma_lnL x y")
    return path


def _cep(tmp, plugin, extra):
    return _run(tmp, CEP, ["--fname", _cep_grid(tmp), "--parameter", "u", "--parameter", "v",
                           "--supplementary-coordinate-code", plugin, "--get-range-from-external",
                           "--fit-method", "rf", "--n-max", "40000", "--n-step", "4000",
                           "--n-eff", "200", "--n-output-samples", "2000",
                           "--no-plots", "--ignore-errors-in-data"] + extra)


def test_cep_get_bounds_recovers_known_mean(work):
    tmp, plugin, _ = work
    proc = _cep(tmp, plugin, [])
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "from get_bounds" in proc.stdout
    out = np.genfromtxt(os.path.join(tmp, "output-EOS-samples.dat"), names=True)
    assert abs(np.mean(out["x"]) - 3) < 0.15 and abs(np.mean(out["y"]) - 3) < 0.15


def test_cep_get_bounds_failure_stops_the_run(work):
    tmp, plugin, _ = work
    proc = _cep(tmp, plugin, ["--external-range-args", "fail=True"])
    assert proc.returncode != 0
    assert "get_bounds asked to fail" in proc.stdout
