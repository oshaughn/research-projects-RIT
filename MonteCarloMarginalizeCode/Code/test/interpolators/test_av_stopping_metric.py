"""Test AV weight moments without importing optional GPU/LAL dependencies."""
import ast
from pathlib import Path
import numpy as np
import pytest

source = Path(__file__).resolve().parents[2] / "RIFT/integrators/mcsamplerAdaptiveVolume.py"
tree = ast.parse(source.read_text())
helpers = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("_select_av_stopping_statistic", "_av_weight_statistics_from_log")], type_ignores=[])
namespace = {"np": np}
exec(compile(helpers, str(source), "exec"), namespace)
select = namespace["_select_av_stopping_statistic"]
counts = namespace["_av_weight_statistics_from_log"]


def test_known_weight_counts_and_scale_invariance():
    w = np.array([1, 0.5, 0.5])
    result = counts(np.log(w))
    assert result["max_weight"] == 2
    assert result["kish"] == pytest.approx(4 / 1.5)
    assert counts(np.log(w) + 10000) == pytest.approx(result)
    assert counts(np.zeros(2500)) == {"max_weight": 2500, "kish": 2500}


def test_explicit_opt_in_and_default_unchanged():
    c = counts(np.log([1, 0.5, 0.5]))
    assert select(c["max_weight"], c["kish"]) == 2
    assert select(c["max_weight"], c["kish"], "kish") > 2.5
    assert select(c["max_weight"], c["kish"], "max-weight") < 2.5
    with pytest.raises(ValueError):
        select(0, 0, "unsupported")
    for values in ([], [np.nan], [np.inf], [[0, 1]]):
        with pytest.raises(ValueError):
            counts(values)


def test_rare_high_weight_does_not_imply_large_posterior_mass():
    # Smooth narrow peaks can create an extreme weight with little total mass.
    logs = np.r_[np.zeros(1_000_000), np.log(1000)]
    c = counts(logs)
    assert c["max_weight"] == pytest.approx(1001)
    assert c["kish"] == pytest.approx(501000.5)
    assert 1000 / 1_001_000 < 0.001
    assert select(c["max_weight"], c["kish"], "kish") > 100 * c["max_weight"]
