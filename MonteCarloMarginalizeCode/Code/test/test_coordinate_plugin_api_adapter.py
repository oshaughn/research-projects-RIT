"""Coordinate-plugin API adapter: the ``rift`` and ``inplace`` (PR #434) shapes.

  * auto-classification (it never selects inplace), the explicit PLUGIN_API
    attribute and the --supplementary-coordinate-api flag; every review-S1 evasion
    of a #434-style plugin with an inverse is refused, and every production-shaped
    plugin from the 2026-11-02 census (rift_roboto_paper dslice plugins, mmapipe
    natal plugin, in-tree demos and test plugins) classifies rift with no warning;
  * the inplace wrapper: rift call signature, no mutation of the caller's array,
    selection by name, and drivers that support only the rift API refuse it;
  * the fixed rotation demo (demo/hyperpipe/inplace_plugin/demo_conversion.py)
    through CEP's conversion and output-inversion path, and through PUFF; the
    FILE-basis ranges are enforced on PUFF rows and CEP samples; CEP's round-trip
    preflight fails before the fit, passes a tabulated inverse at its default
    tolerance and honours --supplementary-coordinate-roundtrip-tol; PUFF refuses
    --reflect-parameter and an empty downselect; the loader never runs a module
    __getattr__ before classifying;
  * the preflight's tolerance relative to the column spread, its magnitude floor
    (narrow and constant columns), rows outside the forward map's domain
    (skipped, but at least one is needed), a NaN inverse (refused), and the
    plugin ROUNDTRIP_TOL wiring in CEP; inapplicable plugin options warn.

Cost: unit tests are instant; about thirty driver subprocesses, a few seconds each.
"""
import functools
import os
import subprocess
import sys
import types
import warnings

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
CEP = os.path.join(CODE, "bin", "util_ConstructEOSPosterior.py")
PUFF = os.path.join(CODE, "bin", "util_HyperparameterPuffball.py")
DEMO = os.path.join(CODE, "demo", "hyperpipe", "inplace_plugin", "demo_conversion.py")
LINEAR = os.path.join(CODE, "demo", "hyperpipe", "linear_coordinate_convert.py")
LINEAR_INI = os.path.join(CODE, "demo", "hyperpipe", "linear_coordinate_convert.ini")
sys.path.insert(0, CODE)
from RIFT.misc import coordinate_plugin as cp  # noqa: E402

ROOT2 = np.sqrt(2.0)


def _module(**attrs):
    mod = types.ModuleType("_fake_plugin")
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _inplace_fn(X, coord_names, **kw):
    return X


def _rift_fn(x_in, coord_names, low_level_coord_names, **kw):
    return x_in


# ------------------------------------------------------------------ classification

def test_existing_in_tree_plugins_stay_rift():
    for spec in ("rift_default", LINEAR):
        mod = cp._load_module_from_spec(spec)
        assert cp.classify_plugin_api(mod) == "rift"


def _inplace_args(X, coord_names, *args, **kw):
    return X


def _inplace_default(X, coord_names=None, **kw):
    return X


def _inplace_kwonly(X, coord_names, *, theta=0.3, **kw):
    return X


def _rift_low_in_kwargs(x, coord_names, **kw):     # a rift plugin reading low_level from **kw
    return x


def _undecorated(f):
    def wrapper(*a, **k):          # a decorator without functools.wraps
        return f(*a, **k)
    return wrapper


class _CallableInplace:
    def __call__(self, X, coord_names, **kw):
        return X


def test_auto_rift_by_signature():
    assert cp.classify_plugin_api(_module(convert_coordinates=_rift_fn)) == "rift"
    assert cp.classify_plugin_api(_module(convert_coordinates=_rift_fn, inverse_convert_coordinates=_rift_fn)) == "rift"
    assert cp.classify_plugin_api(_module(rot=_rift_fn, inverse_rot=_rift_fn), "rot") == "rift"


def test_auto_never_selects_inplace():
    # An inplace-shaped forward with an inverse is refused in auto mode: PLUGIN_API decides.
    mod = _module(rot=_inplace_fn, inverse_rot=_inplace_fn, get_bounds=lambda p, b, **kw: b)
    with pytest.raises(ValueError, match="cannot tell which plugin API"):
        cp.classify_plugin_api(mod, "rot")
    assert cp.classify_plugin_api(mod, "rot", api="inplace") == "inplace"
    demo = cp._load_module_from_spec(DEMO)
    assert cp.classify_plugin_api(demo, "demo_rotation") == "inplace"      # PLUGIN_API = "inplace"
    del demo.PLUGIN_API                       # PR #434's own demo has no PLUGIN_API
    with pytest.raises(ValueError, match="cannot tell which plugin API"):
        cp.classify_plugin_api(demo, "demo_rotation")


def test_forward_only_stays_rift_with_warning():
    for fn in (_inplace_fn, _inplace_args, _undecorated(_inplace_fn)):
        with pytest.warns(RuntimeWarning, match="does not name low_level_coord_names") as rec:
            assert cp.classify_plugin_api(_module(rot=fn), "rot") == "rift"
        msg = str(rec[0].message)
        assert "WRONG" in msg and "silence" not in msg
    # a forward-only signature that cannot take the rift call at all is refused
    with pytest.raises(ValueError, match="cannot be called with the rift contract"):
        cp.classify_plugin_api(_module(rot=lambda X, coord_names: X), "rot")


def test_explicit_api():
    mod = _module(rot=_inplace_fn, inverse_rot=_inplace_fn)
    assert cp.classify_plugin_api(mod, "rot", api="rift") == "rift"
    assert cp.classify_plugin_api(_module(PLUGIN_API="inplace", convert_coordinates=_rift_fn)) == "inplace"
    assert cp.classify_plugin_api(_module(PLUGIN_API="rift", rot=_inplace_fn, inverse_rot=_inplace_fn),
                                  "rot", api="auto") == "rift"
    for evasive in (_inplace_args, _inplace_default, _undecorated(_inplace_fn)):
        assert cp.classify_plugin_api(_module(PLUGIN_API="inplace", rot=evasive, inverse_rot=evasive),
                                      "rot") == "inplace"
    with pytest.raises(ValueError, match="PLUGIN_API='rift' in the plugin"):
        cp.classify_plugin_api(_module(PLUGIN_API="rift", rot=_inplace_fn), "rot", api="inplace")
    with pytest.raises(ValueError, match="is not one of"):
        cp.classify_plugin_api(_module(PLUGIN_API="same-basis", rot=_inplace_fn), "rot")


_S1_CASES = {
    # review S1: #434-style plugins (forward + inverse_<fn>, no low_level_coord_names
    # parameter) in forms that #517 as first written ran as rift with exit 0; each must refuse
    "plain": dict(rot=_inplace_fn, inverse_rot=_inplace_fn),
    "varargs": dict(rot=_inplace_args, inverse_rot=_inplace_args),
    "defaulted_coord_names": dict(rot=_inplace_default, inverse_rot=_inplace_default),
    "kwonly_default": dict(rot=_inplace_kwonly, inverse_rot=_inplace_fn),
    "decorator_without_wraps": dict(rot=_undecorated(_inplace_fn), inverse_rot=_undecorated(_inplace_fn)),
    "partial": dict(rot=functools.partial(_inplace_fn, theta=0.3), inverse_rot=_inplace_fn),
    "callable_instance": dict(rot=_CallableInplace(), inverse_rot=_CallableInplace()),
    "builtin": dict(rot=np.negative, inverse_rot=np.negative),
    "no_kwargs": dict(rot=lambda X, coord_names: X, inverse_rot=_inplace_fn),
    "helper_prepare": dict(rot=_inplace_fn, inverse_rot=_inplace_fn, prepare=print),
    "helper_jacobian": dict(rot=_inplace_fn, inverse_rot=_inplace_fn, jacobian=np.linalg.det),
    "helper_register_priors": dict(rot=_inplace_fn, inverse_rot=_inplace_fn, register_priors=lambda *a, **k: None),
    "helper_CHARTS": dict(rot=_inplace_fn, inverse_rot=_inplace_fn, CHARTS={"a": {}}),
    "helper_INPUT_PARAMETERS": dict(rot=_inplace_fn, inverse_rot=_inplace_fn, INPUT_PARAMETERS=["x0"]),
    "inverse_rift_shaped": dict(rot=_inplace_fn, inverse_rot=_rift_fn),
    "lowercase_plugin_api_typo": dict(rot=_inplace_args, inverse_rot=_inplace_fn, plugin_api="inplace"),
}


@pytest.mark.parametrize("case", sorted(_S1_CASES))
def test_s1_evasions_refused(case):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="cannot tell which plugin API"):
            cp.classify_plugin_api(_module(**_S1_CASES[case]), "rot")


@pytest.mark.parametrize("fn", [_inplace_fn, _rift_low_in_kwargs])
def test_default_name_without_low_level_and_with_inverse_refused(fn):
    # inverse_convert_coordinates is the rift inverse name too, so neither shape is a tell
    with pytest.raises(ValueError, match="cannot tell which plugin API"):
        cp.classify_plugin_api(_module(convert_coordinates=fn, inverse_convert_coordinates=fn))


def test_module_getattr_is_not_run():
    calls = []

    def mod_getattr(name):
        calls.append(name)
        raise AttributeError(name)
    with pytest.raises(ValueError, match="__getattr__"):
        cp.classify_plugin_api(_module(rot=_inplace_fn, __getattr__=mod_getattr), "rot")
    with pytest.raises(ValueError, match="cannot tell"):
        cp.classify_plugin_api(_module(rot=_inplace_fn, inverse_rot=_inplace_fn, __getattr__=mod_getattr), "rot")
    assert cp.classify_plugin_api(_module(rot=_rift_fn, __getattr__=mod_getattr), "rot") == "rift"
    assert calls == []


def test_loader_never_runs_module_getattr_before_classifying(tmp_path):
    # Review 2 NIT: the loader's hasattr ran __getattr__('rot') twice before the refusal.
    calls = os.path.join(str(tmp_path), "calls.txt")
    plug = os.path.join(str(tmp_path), "ga_plugin.py")
    with open(plug, "w") as f:
        f.write("import numpy as np\n"
                "def __getattr__(name):\n"
                "    open({!r}, 'a').write(name + '\\n')\n"
                "    if name in ('rot', 'inverse_rot'):\n"
                "        return lambda X, coord_names, **kw: np.array(X, copy=True)\n"
                "    raise AttributeError(name)\n".format(calls))
    with pytest.raises(ValueError, match="__getattr__"):
        cp.load_coordinate_converter(plug, "rot", allowed_apis=("rift", "inplace"))
    assert not os.path.exists(calls), open(calls).read()
    # a missing entry point with no __getattr__ is still an AttributeError
    with pytest.raises(AttributeError, match="does not expose"):
        cp.load_coordinate_converter(LINEAR, "no_such_function", ini_path=LINEAR_INI)


def test_roundtrip_tolerance_priority():
    assert cp.roundtrip_tolerance(_module()) == (1e-5, "default")
    assert cp.roundtrip_tolerance(_module(ROUNDTRIP_TOL=1e-3))[0] == 1e-3
    assert cp.roundtrip_tolerance(_module(ROUNDTRIP_TOL=1e-3), 2e-4)[0] == 2e-4
    for bad in (0, -1.0, float("nan"), "x"):
        with pytest.raises(ValueError, match="round-trip tolerance"):
            cp.roundtrip_tolerance(_module(), bad)

    def mod_getattr(name):
        raise AssertionError("module __getattr__ ran")
    assert cp.roundtrip_tolerance(_module(__getattr__=mod_getattr))[0] == 1e-5


def test_classification_never_calls_the_plugin():
    def boom(X, coord_names, **kw):
        raise AssertionError("called during classification")
    with pytest.raises(ValueError, match="cannot tell"):
        cp.classify_plugin_api(_module(rot=boom, inverse_rot=boom), "rot")
    assert cp.classify_plugin_api(_module(PLUGIN_API="inplace", rot=boom, inverse_rot=boom), "rot") == "inplace"


# Production-shaped plugins (census of 2026-11-02): minimal equivalents with the same
# signatures, module attributes and import pattern as the real files.
_CENSUS_PLUGINS = {
    # rift_roboto_paper analyses/rerun_o4ab_distance_export/tools/dslice_distance_coord_plugin.py
    "dslice_distance_coord_plugin.py": """
import numpy as np
def _wf_convert(x_in, src_names, tgt_names):
    return x_in[:, [list(src_names).index(n) for n in tgt_names]]
def convert_coordinates(x_in, coord_names, low_level_coord_names, **kwargs):
    return _wf_convert(np.atleast_2d(x_in), low_level_coord_names, coord_names)
def inverse_convert_coordinates(x_in, coord_names, low_level_coord_names, **kwargs):
    return _wf_convert(np.atleast_2d(x_in), coord_names, low_level_coord_names)
def register_priors(prior_map, prior_range_map=None, coord_names=None, **kwargs):
    return None
""",
    # rift_roboto_paper analyses/distance_gp_interp/tools/dslice_mdist_plugin.py: re-exports the
    # inverse and register_priors of the plugin above, imported by module name
    "dslice_mdist_plugin.py": """
import numpy as np
import dslice_distance_coord_plugin as base
EXTRA = ('mtot_over_dist', 'mc56_over_dist')
inverse_convert_coordinates = base.inverse_convert_coordinates
register_priors = base.register_priors
def convert_coordinates(x_in, coord_names, low_level_coord_names, **kwargs):
    return base.convert_coordinates(x_in, [n for n in coord_names if n not in EXTRA], low_level_coord_names)
""",
    # rift_roboto_paper branch claude/dslice-widening-pr: dslice_coord_plugin_srcframe.py
    "dslice_coord_plugin_srcframe.py": """
def convert_coordinates(x_in, coord_names, low_level_coord_names, **kwargs):
    return x_in
def inverse_convert_coordinates(x_in, coord_names, low_level_coord_names, **kwargs):
    return x_in
def register_priors(prior_map, prior_range_map=None, coord_names=None, **kwargs):
    return None
""",
    # mmapipe mmpipe/coords/natal_coordinate_convert.py
    "natal_coordinate_convert.py": """
NAME = "natal_coordinate_convert"
CHARTS = {"mcq_natal": {"input_parameters": ["m1", "m2"], "output_parameters": ["mc", "q"]}}
DEFAULT_CHART = "mcq_natal"
def prepare(config=None, coord_names=None, low_level_coord_names=None, chart=None, **kwargs):
    return None
def convert_coordinates(x_in, coord_names, low_level_coord_names, chart=None, **kwargs):
    return x_in
def inverse_convert_coordinates(y_in, coord_names, low_level_coord_names, chart=None, **kwargs):
    return y_in
""",
    # the test_hyperpipe_coordinate_passing.py plugin (INPUT_PARAMETERS, get_bounds)
    "hyperpipe_passing_plugin.py": """
INPUT_PARAMETERS = ["x", "y"]
def convert_coordinates(x_in, coord_names, low_level_coord_names, **kw):
    return x_in
def inverse_convert_coordinates(y_in, coord_names, low_level_coord_names, **kw):
    return y_in
def get_bounds(coord_names, ranges, fail=False, half_width=1.5, **kw):
    return {}
""",
    # the test_fit_nonfinite_floor.py / test_eos_gp_matern.py plugins (forward only)
    "forward_only_rift_plugin.py": """
def convert_coordinates(x_in, coord_names, low_level_coord_names, **kwargs):
    return x_in
""",
}


@pytest.mark.parametrize("name", sorted(_CENSUS_PLUGINS) + ["rift_default", "linear_coordinate_convert.py"])
def test_census_production_plugins_classify_rift_without_warning(tmp_path, monkeypatch, name):
    for fname, src in _CENSUS_PLUGINS.items():
        (tmp_path / fname).write_text(src)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "dslice_distance_coord_plugin", raising=False)
    spec = {"rift_default": "rift_default", "linear_coordinate_convert.py": LINEAR}.get(name, str(tmp_path / name))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mod = cp._load_module_from_spec(spec)
        assert cp.classify_plugin_api(mod) == "rift"
        assert cp.classify_plugin_api(mod, None, api="auto") == "rift"
        if name.startswith("dslice"):
            conv, mod2 = cp.load_coordinate_converter(spec, allowed_apis=("rift", "inplace"))
            assert conv.plugin_api == "rift"
            assert conv.inverse is mod2.inverse_convert_coordinates


# ------------------------------------------------------------------ loader / wrapper

def test_rift_driver_default_refuses_inplace():
    with pytest.raises(ValueError, match="does not support"):
        cp.load_coordinate_converter(DEMO, "demo_rotation")


def test_rift_plugin_unchanged_by_loader():
    conv, mod = cp.load_coordinate_converter(LINEAR, ini_path=LINEAR_INI, allowed_apis=("rift", "inplace"),
                                             coord_names=["u", "v", "w"],
                                             low_level_coord_names=["x", "y", "z"])
    assert conv.plugin_api == "rift"
    assert conv.inverse is mod.inverse_convert_coordinates


def test_inplace_wrapper_selects_by_name_and_copies():
    conv, _ = cp.load_coordinate_converter(DEMO, "demo_rotation", allowed_apis=("rift", "inplace"),
                                           external_range_args=["theta=0.3"])
    assert conv.plugin_api == "inplace"
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    X_keep = X.copy()
    names = ["z", "x0", "x1"]
    Y = conv(X, coord_names=["x1", "x0"], low_level_coord_names=names)
    assert np.array_equal(X, X_keep)                     # caller's array untouched
    R = cp._load_module_from_spec(DEMO).rotate(0.3)      # theta reached the forward map
    expect = X[:, 1:3] @ R.T
    assert np.allclose(Y, expect[:, ::-1])
    back = conv.inverse(Y, coord_names=["x1", "x0"], low_level_coord_names=["x0", "x1"])
    assert np.array_equal(Y, conv(X, coord_names=["x1", "x0"], low_level_coord_names=names))
    assert np.allclose(back, X[:, 1:3])
    with pytest.raises(ValueError, match="cannot produce"):
        conv(X, coord_names=["u"], low_level_coord_names=names)


def _mutating_fwd(X, coord_names, **kw):
    X[:, 0] += 1.0                  # PR #434's original wrote into X and returned it
    return X


def _mutating_inv(X, coord_names, **kw):
    X[:, 0] -= 1.0
    return X


def test_inplace_wrapper_never_mutates_caller_array():
    # The plugin writes into its input: only the wrapper's copy keeps the caller's array intact.
    sys.modules["_rift_test_mutating"] = _module(PLUGIN_API="inplace", mut=_mutating_fwd, inverse_mut=_mutating_inv)
    try:
        conv, _ = cp.load_coordinate_converter("_rift_test_mutating", "mut", allowed_apis=("rift", "inplace"))
    finally:
        del sys.modules["_rift_test_mutating"]
    X = np.arange(12, dtype=float).reshape(6, 2)
    X_keep = X.copy()
    Y = conv(X, coord_names=["a", "b"], low_level_coord_names=["a", "b"])
    assert np.array_equal(X, X_keep)
    assert np.allclose(Y[:, 0], X_keep[:, 0] + 1)
    Y_keep = Y.copy()
    back = conv.inverse(Y, coord_names=["a", "b"], low_level_coord_names=["a", "b"])
    assert np.array_equal(Y, Y_keep)
    assert np.allclose(back, X_keep)


def test_explicit_inplace_needs_inverse():
    path_mod = _module(rot=_inplace_fn)
    sys.modules["_rift_test_fwd_only"] = path_mod
    try:
        with pytest.raises(AttributeError, match="inverse_rot"):
            cp.load_coordinate_converter("_rift_test_fwd_only", "rot", api="inplace",
                                         allowed_apis=("rift", "inplace"))
    finally:
        del sys.modules["_rift_test_fwd_only"]


@pytest.mark.parametrize("theta", [np.pi / 4, 0.3])
def test_demo_get_bounds_is_the_rotated_box(theta):
    demo = cp._load_module_from_spec(DEMO)
    file_box = {"x0": [1.0, 4.0], "x1": [-2.0, 0.5], "z": [0, 1]}
    out = demo.get_bounds(["x0", "x1", "z"], file_box, theta=theta)
    g = np.stack(np.meshgrid(np.linspace(1, 4, 41), np.linspace(-2, 0.5, 41)), -1).reshape(-1, 2)
    r = g @ demo.rotate(theta).T
    for k, name in enumerate(["x0", "x1"]):
        assert np.isclose(out[name][0], r[:, k].min()) and np.isclose(out[name][1], r[:, k].max())
    assert out["z"] == [0.0, 1.0]
    with pytest.raises(ValueError, match="needs file-basis ranges"):
        demo.get_bounds(["x0"], {"x0": [0, 1]})


# ------------------------------------------------------------------ drivers

def _run(tmp, driver, args):
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(OMP_NUM_THREADS="1", MPLBACKEND="Agg",
               XDG_CACHE_HOME=os.path.join(tmp, "cache"), MPLCONFIGDIR=os.path.join(tmp, "mpl"))
    return subprocess.run([sys.executable, driver] + list(args), cwd=tmp, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=900)


def _cep_grid(tmp, theta, peak=10.0):
    """lnL is a narrow Gaussian in the ROTATED frame, peaked at r = (c, 0), with
    maximum ``peak``."""
    rng = np.random.default_rng(7)
    n = 600
    R = cp._load_module_from_spec(DEMO).rotate(theta)
    c = 3 * ROOT2
    centre = R.T @ np.array([c, 0.0])                    # the peak in the file basis
    x = centre + rng.uniform(-1.5, 1.5, (n, 2))
    r = x @ R.T
    lnL = -0.5 * ((r[:, 0] - c) / 0.25) ** 2 - 0.5 * (r[:, 1] / 0.6) ** 2 + peak
    path = os.path.join(tmp, "cep_grid.dat")
    np.savetxt(path, np.column_stack([lnL, np.zeros(n), x, rng.uniform(0, 1, n)]),
               header="lnL sigma_lnL x0 x1 z")
    return path, centre


def _cep(tmp, grid, extra):
    return _run(tmp, CEP, ["--fname", grid, "--parameter", "x0", "--parameter", "x1",
                           "--supplementary-coordinate-code", DEMO,
                           "--supplementary-coordinate-function", "demo_rotation",
                           "--fit-method", "rf", "--n-max", "40000", "--n-step", "4000",
                           "--n-eff", "200", "--n-output-samples", "2000",
                           "--no-plots", "--ignore-errors-in-data"] + extra)


@pytest.mark.parametrize("theta,extra", [
    (np.pi / 4, ["--get-range-from-external"]),
    (0.3, ["--get-range-from-external", "--external-range-args", "theta=0.3",
           "--supplementary-coordinate-api", "inplace"]),
    (np.pi / 4, []),                                 # ranges from the transformed grid
])
def test_cep_inplace_round_trip(tmp_path, theta, extra):
    tmp = str(tmp_path)
    grid, centre = _cep_grid(tmp, theta)
    proc = _cep(tmp, grid, extra)
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "api=inplace" in proc.stdout
    out = np.genfromtxt(os.path.join(tmp, "output-EOS-samples.dat"), names=True)
    # Output is in the FILE basis: without the inverse the mean would sit at (3 sqrt2, 0).
    assert abs(np.mean(out["x0"]) - centre[0]) < 0.15, (np.mean(out["x0"]), centre)
    assert abs(np.mean(out["x1"]) - centre[1]) < 0.15, (np.mean(out["x1"]), centre)
    r = np.column_stack([out["x0"], out["x1"]]) @ cp._load_module_from_spec(DEMO).rotate(theta).T
    assert abs(np.mean(r[:, 0]) - 3 * ROOT2) < 0.1 and np.std(r[:, 0]) < 0.4


@pytest.mark.parametrize("extra", [[], ["--sampler-method", "AV", "--internal-use-lnL"]])
def test_cep_inplace_file_basis_box_enforced(tmp_path, extra):
    # Review S2: the rotated sampling box is a strict superset of the file-basis box; before
    # the fix 74 of 2000 output samples fell outside x0:[2,4], x1:[-4,-2].
    # Peak lnL 1, not 10: with --internal-use-lnL the indicator must give -inf outside the
    # box.  A leak that gives lnL = 0 there (val * _ok) is hidden by e^10 but not by e^1
    # (review 2, S3: 1253 of 2000 samples outside the box).
    tmp = str(tmp_path)
    grid, centre = _cep_grid(tmp, np.pi / 4, peak=1.0)      # file-basis peak (3, -3)
    proc = _cep(tmp, grid, ["--get-range-from-external",
                            "--integration-parameter-range", "x0:[2,4]",
                            "--integration-parameter-range", "x1:[-4,-2]"] + extra)
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "File-basis box enforced" in proc.stdout
    out = np.genfromtxt(os.path.join(tmp, "output-EOS-samples.dat"), names=True)
    assert len(out) == 2000
    assert out["x0"].min() >= 2 and out["x0"].max() <= 4, (out["x0"].min(), out["x0"].max())
    assert out["x1"].min() >= -4 and out["x1"].max() <= -2, (out["x1"].min(), out["x1"].max())
    assert abs(np.mean(out["x0"]) - centre[0]) < 0.15 and abs(np.mean(out["x1"]) - centre[1]) < 0.15


def test_cep_inplace_round_trip_preflight_fails_before_fit(tmp_path):
    # Review S3: --parameter x0 alone cannot be inverted (the rotation needs x1).  Fail at
    # load, before the fit, not after the MC run.
    tmp = str(tmp_path)
    grid, _ = _cep_grid(tmp, np.pi / 4)
    proc = _run(tmp, CEP, ["--fname", grid, "--parameter", "x0",
                           "--supplementary-coordinate-code", DEMO,
                           "--supplementary-coordinate-function", "demo_rotation",
                           "--fit-method", "rf", "--n-max", "40000", "--n-step", "4000",
                           "--n-eff", "200", "--no-plots", "--ignore-errors-in-data"])
    assert proc.returncode != 0
    assert "round-trip check failed before the fit" in proc.stdout
    assert "Weight exponent" not in proc.stdout and "n_eff" not in proc.stdout
    # an inverse that does not undo the forward map is caught the same way
    bad = os.path.join(tmp, "bad_inverse.py")
    with open(bad, "w") as f:
        f.write("PLUGIN_API = 'inplace'\n"
                "def rot(X, coord_names, **kw):\n    Y = X.copy(); Y[:, 0] = X[:, 0] + X[:, 1]; return Y\n"
                "def inverse_rot(X, coord_names, **kw):\n    return X.copy()\n")
    proc = _run(tmp, CEP, ["--fname", grid, "--parameter", "x0", "--parameter", "x1",
                           "--supplementary-coordinate-code", bad, "--supplementary-coordinate-function", "rot",
                           "--fit-method", "rf", "--no-plots", "--ignore-errors-in-data"])
    assert proc.returncode != 0
    assert "inverse mismatch: inverse(forward(x)) restricted to the sampled columns ['x0', 'x1'] does not reproduce the data rows (max error / column scale: {'x0':" in proc.stdout


_TAB_PLUGIN = """import numpy as np
PLUGIN_API = "inplace"
_G = np.linspace(-10, 10, 4001); _F = _G + 0.1 * _G**3
def cubic(X, coord_names, **kw):
    Y = np.array(X, dtype=float, copy=True); j = list(coord_names).index("x0")
    Y[:, j] = X[:, j] + 0.1 * X[:, j]**3; return Y
def inverse_cubic(X, coord_names, **kw):
    Y = np.array(X, dtype=float, copy=True); j = list(coord_names).index("x0")
    Y[:, j] = np.interp(X[:, j], _F, _G{wrong}); return Y
"""


@pytest.mark.parametrize("case", ["tabulated", "tight_tol", "wrong_inverse"])
def test_cep_inplace_roundtrip_tolerance(tmp_path, case):
    # Review 2, S1: a tabulated inverse (round-trip error ~1.7e-6) was refused by a fixed
    # 1e-8 absolute tolerance.  The default is now 1e-5 of the column spread; an inverse
    # that is wrong (here off by 0.01) is still refused, and the CLI option overrides.
    tmp = str(tmp_path)
    grid, _ = _cep_grid(tmp, np.pi / 4)
    plug = os.path.join(tmp, "tab_plugin.py")
    with open(plug, "w") as f:
        f.write(_TAB_PLUGIN.format(wrong=" + 0.01" if case == "wrong_inverse" else ""))
    extra = ["--supplementary-coordinate-roundtrip-tol", "1e-9"] if case == "tight_tol" else []
    proc = _run(tmp, CEP, ["--fname", grid, "--parameter", "x0", "--parameter", "x1",
                           "--supplementary-coordinate-code", plug, "--supplementary-coordinate-function", "cubic",
                           "--fit-method", "rf", "--n-max", "20000", "--n-step", "4000", "--n-eff", "100",
                           "--n-output-samples", "200", "--no-plots", "--ignore-errors-in-data"] + extra)
    if case == "tabulated":
        assert proc.returncode == 0, proc.stdout[-3000:]
        assert "round-trip check (inplace): 8 rows OK" in proc.stdout
        assert "tolerance 1e-05 from default" in proc.stdout
    else:
        assert proc.returncode != 0
        assert "round-trip check failed before the fit" in proc.stdout
        assert ("from --supplementary-coordinate-roundtrip-tol" if case == "tight_tol" else "from default") in proc.stdout
        assert "n_eff" not in proc.stdout


# Review 3: preflight cases that pin the tolerance scale, the magnitude floor, the
# non-finite handling and the plugin ROUNDTRIP_TOL wiring in CEP itself.  Each plugin
# doubles every column; the inverse halves it and adds ERR to x0, optionally with NaNs.
_RT_PLUGIN = """import numpy as np
PLUGIN_API = "inplace"
{tol}
def dbl(X, coord_names, **kw):
    Y = 2.0 * np.array(X, dtype=float, copy=True)
    {fwd_nan}
    return Y
def inverse_dbl(X, coord_names, **kw):
    Y = 0.5 * np.array(X, dtype=float, copy=True)
    Y[:, list(coord_names).index("x0")] += {err}
    {inv_nan}
    return Y
"""

_RT_CASES = {
    # name: (x0 column, plugin template fields, extra CLI, expected substring, passes?)
    # x0 spread 1e-3 at magnitude 0.01: an error of 1e-6 is 1e-3 of the spread (refused)
    # but below 1e-5 absolute (an absolute tolerance would pass it).
    "narrow_exact": ("narrow", {}, [], "rows OK", True),
    "narrow_wrong": ("narrow", {"err": "1e-6"}, [], "inverse mismatch", False),
    # spread 1e-2 at magnitude 1e6: an error of 1e-4 (1e-10 of the value) passes via the
    # 1e-3 magnitude floor; scaled by the spread alone it would be refused.
    "large_roundoff": ("large", {"err": "1e-4"}, [], "rows OK", True),
    # constant columns: the floor sets the scale (1 at 1e3, 1e3 at 1e6)
    "const_wrong": ("const1e3", {"err": "5e-3"}, [], "inverse mismatch", False),
    "const_roundoff": ("const1e6", {"err": "1e-4"}, [], "rows OK", None),
    # an inverse that is NaN where the forward map is finite must be refused
    "nan_inverse": ("wide", {"inv_nan": "Y[::3, 1] = np.nan"}, [], "inverse is non-finite", False),
    # rows outside the forward map's domain are skipped (review 3, S1), but not all of them
    "domain": ("wide", {"fwd_nan": "Y[X[:, 1] < -0.5, 1] = np.nan"}, [], "outside the forward map's domain skipped", True),
    "no_domain": ("wide", {"fwd_nan": "Y[:, 1] = np.nan"}, [], "no data row is in the plugin's domain", False),
    # plugin ROUNDTRIP_TOL: an error of 1e-3 of the spread passes at 1e-2; the CLI wins
    "plugin_tol": ("wide", {"err": "2e-3", "tol": "ROUNDTRIP_TOL = 1e-2"}, [], "from plugin ROUNDTRIP_TOL", True),
    "plugin_tol_cli": ("wide", {"err": "2e-3", "tol": "ROUNDTRIP_TOL = 1e-2"},
                       ["--supplementary-coordinate-roundtrip-tol", "1e-4"],
                       "from --supplementary-coordinate-roundtrip-tol", False),
}


@pytest.mark.parametrize("case", sorted(_RT_CASES))
def test_cep_inplace_roundtrip_preflight_cases(tmp_path, case):
    kind, fields, extra, expect, passes = _RT_CASES[case]
    tmp = str(tmp_path)
    rng = np.random.default_rng(11)
    n = 400
    x0 = {"narrow": 0.01 + rng.uniform(0, 1e-3, n), "large": 1e6 + rng.uniform(0, 1e-2, n),
          "const1e3": np.full(n, 1e3), "const1e6": np.full(n, 1e6), "wide": rng.uniform(-1, 1, n)}[kind]
    x1 = rng.uniform(-1, 1, n)
    grid = os.path.join(tmp, "grid.dat")
    np.savetxt(grid, np.column_stack([-0.5 * (x1 / 0.3) ** 2, np.zeros(n), x0, x1]), header="lnL sigma_lnL x0 x1",
               fmt="%.17g")
    plug = os.path.join(tmp, "rt_plugin.py")
    tpl = {"tol": "", "err": "0.0", "fwd_nan": "", "inv_nan": ""}
    tpl.update(fields)
    with open(plug, "w") as f:
        f.write(_RT_PLUGIN.format(**tpl))
    x0_flag = "--parameter-nofit" if kind.startswith("const") else "--parameter"
    proc = _run(tmp, CEP, ["--fname", grid, x0_flag, "x0", "--parameter", "x1",
                           "--supplementary-coordinate-code", plug, "--supplementary-coordinate-function", "dbl",
                           "--fit-method", "rf", "--n-max", "20000", "--n-step", "4000", "--n-eff", "50",
                           "--n-output-samples", "100", "--no-plots", "--ignore-errors-in-data"] + extra)
    assert expect in proc.stdout, proc.stdout[-3000:]
    if passes is None:                      # preflight only: a constant sampled column is degenerate later
        assert "round-trip check (inplace): 8 rows OK" in proc.stdout, proc.stdout[-3000:]
    elif passes:
        assert proc.returncode == 0, proc.stdout[-3000:]
        assert "round-trip check (inplace): 8 rows OK" in proc.stdout
    else:
        assert proc.returncode != 0
        assert "round-trip check failed before the fit" in proc.stdout
        assert "n_eff" not in proc.stdout


def test_cep_puff_warn_on_inapplicable_plugin_options(tmp_path):
    # Review 3, NIT 3: options that cannot act are reported, not silently ignored.
    tmp = str(tmp_path)
    grid, _ = _cep_grid(tmp, np.pi / 4)
    proc = _run(tmp, CEP, ["--fname", grid, "--parameter", "x0", "--parameter", "x1",
                           "--supplementary-coordinate-api", "inplace",
                           "--supplementary-coordinate-roundtrip-tol", "1e-3",
                           "--fit-method", "rf", "--n-max", "10000", "--n-step", "4000", "--n-eff", "20",
                           "--n-output-samples", "50", "--no-plots", "--ignore-errors-in-data"])
    assert "--supplementary-coordinate-api inplace ignored: no --supplementary-coordinate-code" in proc.stdout
    assert "--supplementary-coordinate-roundtrip-tol ignored" in proc.stdout
    proc = _run(tmp, PUFF, ["--inj-file", grid, "--inj-file-out", "out.dat", "--parameter", "x0",
                            "--puff-factor", "1", "--supplementary-coordinate-api", "rift"])
    assert "--supplementary-coordinate-api rift ignored: no --supplementary-coordinate-code" in proc.stdout


def test_cep_inplace_cli_range_without_get_bounds_refused(tmp_path):
    tmp = str(tmp_path)
    grid, _ = _cep_grid(tmp, np.pi / 4)
    proc = _cep(tmp, grid, ["--integration-parameter-range", "x0:[0,6]"])
    assert proc.returncode != 0
    assert "frame of those ranges is ambiguous" in proc.stdout


def test_puff_inplace_round_trip(tmp_path):
    tmp = str(tmp_path)
    rng = np.random.default_rng(3)
    n = 3000
    x = np.column_stack([rng.uniform(1, 4, n), rng.uniform(-2, 0.5, n)])
    grid = os.path.join(tmp, "grid.dat")
    np.savetxt(grid, np.column_stack([np.zeros(n), np.zeros(n), x]), header="lnL sigma_lnL x0 x1")
    proc = _run(tmp, PUFF, ["--inj-file", grid, "--inj-file-out", "out.dat",
                            "--parameter", "x0", "--parameter", "x1", "--puff-factor", "1",
                            "--parameter-range", "x0:[1,4]", "--parameter-range", "x1:[-2,0.5]",
                            "--supplementary-coordinate-code", DEMO,
                            "--supplementary-coordinate-function", "demo_rotation",
                            "--get-range-from-external"])
    assert proc.returncode == 0, proc.stdout[-3000:]
    out = np.genfromtxt(os.path.join(tmp, "out.dat"), names=True)
    assert 100 < len(out) < n
    demo = cp._load_module_from_spec(DEMO)
    # The CLI ranges are FILE-basis ranges and must hold on every output row (review S2:
    # before the fix about 30% of rows fell outside them).  The rotated box holds as well.
    assert out["x0"].min() >= 1 - 1e-9 and out["x0"].max() <= 4 + 1e-9
    assert out["x1"].min() >= -2 - 1e-9 and out["x1"].max() <= 0.5 + 1e-9
    assert "File-basis downselect (inplace plugin)" in proc.stdout
    box = demo.get_bounds(["x0", "x1"], {"x0": [1, 4], "x1": [-2, 0.5]})
    r = np.column_stack([out["x0"], out["x1"]]) @ demo.rotate(np.pi / 4).T
    for k, name in enumerate(["x0", "x1"]):
        assert r[:, k].min() >= box[name][0] - 1e-9 and r[:, k].max() <= box[name][1] + 1e-9
    # without --get-range-from-external the same ranges are refused
    proc = _run(tmp, PUFF, ["--inj-file", grid, "--inj-file-out", "out2.dat",
                            "--parameter", "x0", "--parameter", "x1",
                            "--parameter-range", "x0:[1,4]",
                            "--supplementary-coordinate-code", DEMO,
                            "--supplementary-coordinate-function", "demo_rotation"])
    assert proc.returncode != 0 and "ambiguous" in proc.stdout


def _puff_grid(tmp):
    rng = np.random.default_rng(3)
    n = 3000
    x = np.column_stack([rng.uniform(1, 4, n), rng.uniform(-2, 0.5, n)])
    grid = os.path.join(tmp, "grid.dat")
    np.savetxt(grid, np.column_stack([np.zeros(n), np.zeros(n), x]), header="lnL sigma_lnL x0 x1")
    return grid


def test_puff_inplace_reflect_refused(tmp_path):
    # Review 2, S2: reflection acted in the transformed box, then the file-basis box dropped
    # 1061 of 3000 rows, with exit 0.
    tmp = str(tmp_path)
    proc = _run(tmp, PUFF, ["--inj-file", _puff_grid(tmp), "--inj-file-out", "out.dat",
                            "--parameter", "x0", "--parameter", "x1", "--puff-factor", "1",
                            "--parameter-range", "x0:[1,4]", "--parameter-range", "x1:[-2,0.5]",
                            "--reflect-parameter", "x0", "--reflect-parameter", "x1",
                            "--supplementary-coordinate-code", DEMO,
                            "--supplementary-coordinate-function", "demo_rotation",
                            "--get-range-from-external"])
    assert proc.returncode != 0
    assert "--reflect-parameter is not supported with an inplace" in proc.stdout
    assert not os.path.exists(os.path.join(tmp, "out.dat"))


def test_puff_inplace_empty_downselect_exits_nonzero(tmp_path):
    tmp = str(tmp_path)
    proc = _run(tmp, PUFF, ["--inj-file", _puff_grid(tmp), "--inj-file-out", "out.dat",
                            "--parameter", "x0", "--parameter", "x1", "--puff-factor", "1",
                            "--parameter-range", "x0:[10,11]", "--parameter-range", "x1:[-2,0.5]",
                            "--supplementary-coordinate-code", DEMO,
                            "--supplementary-coordinate-function", "demo_rotation",
                            "--get-range-from-external"])
    assert proc.returncode != 0
    assert "no puffed rows survive the downselect" in proc.stdout
    assert not os.path.exists(os.path.join(tmp, "out.dat"))
