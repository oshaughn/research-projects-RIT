"""Keep paired smoke tests independent of differing executable defaults."""
import ast
import importlib.util
from pathlib import Path

import pytest


def test_paired_smokes_pin_the_same_reference_frequency():
    for name in ("run_short_av_ile.py", "run_short_jax_av_ile.py"):
        path = Path(__file__).with_name(name)
        tree = ast.parse(path.read_text(), filename=str(path))
        values = [
            ast.literal_eval(call.args[2])
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "set_option"
            and len(call.args) == 3
            and isinstance(call.args[1], ast.Str)
            and call.args[1].s == "--reference-freq"
        ]
        assert values == [100.0], name


def test_fairdraw_guard_rejects_samples_outside_the_requested_box(tmpdir):
    script = Path(__file__).with_name("run_short_jax_av_ile.py")
    spec = importlib.util.spec_from_file_location("short_jax_smoke_contract", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    boxes = dict(right_ascension=(1.1, 1.3), declination=(0.2, 0.4),
                 inclination=(0.2, 0.6), psi=(0.3, 0.7), distance_mpc=(300., 500.))
    path = Path(str(tmpdir)) / "synthetic_fairdraw.dat"
    header = "# right_ascension declination distance inclination psi phi_orb loglikelihood\n"
    path.write_text(header + "1.2 0.3 400 0.4 0.5 0 10\n")
    module.validate_fairdraw_bounds(path, boxes)
    path.write_text(header + "1.4 0.3 400 0.4 0.5 0 10\n")
    with pytest.raises(RuntimeError, match="right_ascension outside"):
        module.validate_fairdraw_bounds(path, boxes)
