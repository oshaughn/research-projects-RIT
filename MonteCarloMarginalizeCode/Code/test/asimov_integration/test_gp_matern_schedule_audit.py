import importlib.util
from pathlib import Path
import pytest

_spec = importlib.util.spec_from_file_location("gp_contract", Path(__file__).with_name("audit_gp_matern_schedule.py"))
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)

LINE = "1 --internal-use-lnL --fit-method gp-matern --sampler-method AV --gp-predict-backend cupy --av-stop-metric kish --gp-matern-max-train-points 4800 --gp-matern-optimizer-maxiter 25 --gp-matern-seed 25062842 --n-eff 100 --n-output-samples 312 --n-eff 2500 --n-output-samples 2500 --n-max 100000000"


def test_checks_every_stage_and_effective_worker_quota():
    result = module.audit_schedule(LINE + "\n" + LINE.replace("1 --fit-method", "G2 --fit-method"))
    assert len(result) == 2
    assert result[1]["target_per_worker"] == 2500


@pytest.mark.parametrize("bad", [
    LINE.replace("gp-matern --sampler", "rf --sampler"),
    LINE + " --fit-method quadratic",
    LINE + " --cap-points 12000",
    LINE.replace("--n-output-samples 2500", "--n-output-samples 312"),
    LINE.replace("--gp-predict-backend cupy", ""),
    LINE.replace("--internal-use-lnL", ""),
    LINE + " --posterior-unique-draw",
    LINE + " --contingency-unevolved-neff quadpuff",
    LINE + " --internal-bound-factor-if-n-eff-small 1",
])
def test_rejects_silent_method_cap_backend_or_quota_change(bad):
    with pytest.raises(ValueError):
        module.audit_schedule(bad)
