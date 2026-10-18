"""Execute helper_LDG_Events fit-method guards from source, without LAL or event data."""
import ast
import types
from pathlib import Path
import pytest

HELPER = Path(__file__).resolve().parents[2] / "bin/helper_LDG_Events.py"
TREE = ast.parse(HELPER.read_text())


class Parser:
    def error(self, message):
        raise SystemExit(message)


def _run(node, **scope):
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(HELPER), "exec"), scope)
    return scope


def _early_guard():
    return next(n for n in TREE.body if isinstance(n, ast.If) and "use_quadratic_early" in ast.unparse(n.test)
                and "gp-matern" in ast.unparse(n.test))


def _cap_block():
    return next(n for n in ast.walk(TREE) if isinstance(n, ast.If) and ast.unparse(n.test) == "fit_method == 'gp-torch'")


EARLY = ["use_quadratic_early", "use_cov_early", "use_gp_early", "use_gauss_early"]


@pytest.mark.parametrize("flag", EARLY)
def test_gp_matern_rejects_early_surrogate_stage(flag):
    opts = types.SimpleNamespace(**{name: False for name in EARLY})
    setattr(opts, flag, True)
    with pytest.raises(SystemExit, match="gp-matern"):
        _run(_early_guard(), fit_method="gp-matern", opts=opts, parser=Parser())
    _run(_early_guard(), fit_method="gp", opts=opts, parser=Parser())


@pytest.mark.parametrize("method,cap", [("gp-torch", "8000"), ("gp", "12000"), ("gp-matern", None), ("rf", None)])
def test_helper_cap_points_respects_gp_torch_limit(method, cap):
    args = _run(_cap_block(), fit_method=method, helper_cip_args="")["helper_cip_args"].split()
    assert (args[args.index("--cap-points") + 1] if "--cap-points" in args else None) == cap


def test_every_alternate_fit_method_line_is_guarded():
    # Each literal --fit-method stage the gp-matern regex could rewrite is either the
    # main helper line or an early-stage override rejected by the guard.
    src = HELPER.read_text()
    guard = ast.unparse(_early_guard().test)
    for flag in EARLY:
        assert flag in guard
    lines = [l for l in src.splitlines() if "fit-method" in l and not l.lstrip().startswith(("#", "parser.add_argument"))]
    allowed = ("--no-plots --fit-method {}", "'fit-method quadratic '", "'fit-method cov '", "'fit-method gp'",
               "'G2 --fit-method quadratic", "--fit-method\\s+\\S+", "'--fit-method gp-matern'", "--force-fit-method gp-matern")
    for line in lines:
        assert any(a in line for a in allowed), line


@pytest.mark.parametrize("mode,refused", [(None, False), ("off", False), ("auto", True), ("geometric4", True)])
def test_gp_matern_refuses_rf_transverse_modes(mode, refused):
    node = next(n for n in TREE.body if isinstance(n, ast.If) and "rf_transverse_spin_coordinates" in ast.unparse(n.test)
                and "gp-matern" in ast.unparse(n.test))
    opts = types.SimpleNamespace(rf_transverse_spin_coordinates=mode)
    if refused:
        with pytest.raises(SystemExit, match="rf"):
            _run(node, fit_method="gp-matern", opts=opts, parser=Parser())
    else:
        _run(node, fit_method="gp-matern", opts=opts, parser=Parser())
    _run(node, fit_method="rf", opts=opts, parser=Parser())
