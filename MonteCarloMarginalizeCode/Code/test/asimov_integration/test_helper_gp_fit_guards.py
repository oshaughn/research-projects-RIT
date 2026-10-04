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


@pytest.mark.parametrize("flag", ["use_quadratic_early", "use_cov_early", "use_gp_early"])
def test_gp_matern_rejects_early_surrogate_stage(flag):
    opts = types.SimpleNamespace(use_quadratic_early=False, use_cov_early=False, use_gp_early=False)
    setattr(opts, flag, True)
    with pytest.raises(SystemExit, match="gp-matern"):
        _run(_early_guard(), fit_method="gp-matern", opts=opts, parser=Parser())
    _run(_early_guard(), fit_method="gp", opts=opts, parser=Parser())


@pytest.mark.parametrize("method,cap", [("gp-torch", "8000"), ("gp", "12000"), ("gp-matern", None), ("rf", None)])
def test_helper_cap_points_respects_gp_torch_limit(method, cap):
    args = _run(_cap_block(), fit_method=method, helper_cip_args="")["helper_cip_args"].split()
    assert (args[args.index("--cap-points") + 1] if "--cap-points" in args else None) == cap
