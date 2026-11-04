"""Audit generated args_cip_list.txt before submitting an opt-in all-GP trial.

Run this directly; it imports neither LAL nor RIFT and makes no submissions.
The audit concerns intrinsic schedule lines, not the capped shared-fit producer
or the final extrinsic selection quota.
"""
import argparse
import json
import shlex
from pathlib import Path


def audit_schedule(text, target=2500):
    reports = []
    expected = {
        "--fit-method": "gp-matern", "--sampler-method": "AV",
        "--gp-predict-backend": "cupy", "--av-stop-metric": "kish",
        "--gp-matern-max-train-points": "4800",
        "--gp-matern-optimizer-maxiter": "25",
        "--gp-matern-seed": "25062842",
        "--n-eff": str(target), "--n-output-samples": str(target),
        "--n-max": "100000000",
    }
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tokens = shlex.split(line)
        if "--internal-use-lnL" not in tokens:
            raise ValueError("line {}: Kish AV requires explicit log-likelihood integration".format(number))
        prohibited = {"--posterior-unique-draw", "--contingency-unevolved-neff", "--internal-bound-factor-if-n-eff-small"}
        if prohibited.intersection(tokens):
            raise ValueError("line {}: export/contingency overrides outside this controlled trial".format(number))
        values = {}
        for i, token in enumerate(tokens):
            if token in expected or token == "--cap-points":
                if i + 1 >= len(tokens):
                    raise ValueError("line {}: missing value for {}".format(number, token))
                values.setdefault(token, []).append(tokens[i + 1])
        for flag, value in expected.items():
            seen = values.get(flag, [])
            if not seen or seen[-1] != value:
                raise ValueError("line {}: effective {} must be {}, found {}".format(number, flag, value, seen))
        if any(v != "gp-matern" for v in values["--fit-method"]):
            raise ValueError("line {}: conflicting fit-method overrides".format(number))
        if "--cap-points" in values:
            raise ValueError("line {}: inherited pre-selection cap would alter the GP training pool".format(number))
        reports.append({"line": number, "fit_method": "gp-matern", "target_per_worker": target})
    if not reports:
        raise ValueError("no intrinsic CIP schedule lines found")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", type=Path)
    args = parser.parse_args()
    print(json.dumps({"verified_schedule": str(args.schedule), "stages": audit_schedule(args.schedule.read_text())}, indent=2))
