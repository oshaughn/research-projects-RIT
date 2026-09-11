#!/usr/bin/env python3
"""Validation-only ILE launcher that aggregates band-limited FFT telemetry.

The companion harness requires a clean, commit-pinned RIFT tree before importing
this driver.  This wrapper never catches or converts likelihood exceptions:
failures are counted, then re-raised.
"""

import atexit
import json
import os
import runpy
import sys
from collections import Counter
from pathlib import Path

from RIFT.likelihood import time_marginalization_quadrature as tmq


TELEMETRY_PATH = Path(os.environ["RIFT_FFT_TELEMETRY_FILE"])
REAL_ILE = os.environ["RIFT_REAL_ILE"]
FORCE_FULL = os.environ.get("RIFT_VALIDATION_FORCE_FULL_FFT", "0") == "1"

_sum_keys = (
    "n_rows",
    "n_refined_rows",
    "n_wrap_exposed_rows",
    "n_unmeasurable_rows",
    "n_flat_rows",
    "n_refinements",
    "retained_fft_batches",
    "retained_fft_rows",
    "full_fft_selected_batches",
    "full_fft_selected_rows",
    "full_fft_fallback_batches",
    "full_fft_fallback_rows",
)
_max_keys = (
    "upsample_factor",
    "max_reflected_period",
    "max_dense_factor",
    "max_reference_full_fft_length",
    "max_retained_fft_length",
    "max_retained_grid_length",
    "n_retained_fft_plans",
)
_aggregate = {
    "schema": 1,
    "validation_force_full_fft": FORCE_FULL,
    "successful_calls": 0,
    "failed_calls": 0,
    "failure_types": Counter(),
    "backend_calls": Counter(),
    "strategy_calls": Counter(),
    "factor_rows": Counter(),
    "full_fft_selected_reasons": Counter(),
    "full_fft_fallback_reasons": Counter(),
}
for _key in _sum_keys + _max_keys:
    _aggregate[_key] = 0


if FORCE_FULL:
    def _validation_force_full(x, factor, plan_cache, transform_report, xpy=None):
        if xpy is None:
            xpy = tmq.np
        reason = "validation-only explicit full-FFT control"
        n_rows = int(x.shape[0])
        reasons = transform_report["full_fft_selected_reasons"]
        reasons[reason] = reasons.get(reason, 0) + n_rows
        tmq._record_transform(
            transform_report,
            "full_fft_selected",
            n_rows,
            2 * int(x.shape[-1]),
            int(factor),
        )
        return tmq.reflected_bandlimited_upsample(x, factor, xpy=xpy)

    tmq._reflected_upsample_for_integration = _validation_force_full


_original = tmq.time_marginalize_bandlimited


def _merge_report(report, xpy):
    _aggregate["successful_calls"] += 1
    backend = getattr(xpy, "__name__", type(xpy).__name__)
    _aggregate["backend_calls"][backend] += 1
    _aggregate["strategy_calls"][report.get("bandlimited_fft_strategy", "missing")] += 1
    for key in _sum_keys:
        _aggregate[key] += int(report.get(key, 0) or 0)
    for key in _max_keys:
        _aggregate[key] = max(_aggregate[key], int(report.get(key, 0) or 0))
    for factor, rows in report.get("factor_histogram", {}).items():
        _aggregate["factor_rows"][str(factor)] += int(rows)
    for key in ("full_fft_selected_reasons", "full_fft_fallback_reasons"):
        for reason, rows in report.get(key, {}).items():
            _aggregate[key][reason] += int(rows)


def _instrumented(*args, **kwargs):
    xpy = kwargs.get("xpy", tmq.np)
    try:
        result = _original(*args, **kwargs)
    except BaseException as exc:
        _aggregate["failed_calls"] += 1
        _aggregate["failure_types"][type(exc).__name__] += 1
        raise
    _merge_report(tmq.last_report(), xpy)
    return result


tmq.time_marginalize_bandlimited = _instrumented


def _jsonable():
    return {
        key: dict(value) if isinstance(value, Counter) else value
        for key, value in _aggregate.items()
    }


def _write_telemetry():
    payload = _jsonable()
    TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = TELEMETRY_PATH.with_suffix(TELEMETRY_PATH.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, TELEMETRY_PATH)
    print("RIFT_FFT_TELEMETRY_JSON=" + json.dumps(payload, sort_keys=True), flush=True)


atexit.register(_write_telemetry)
sys.argv[0] = REAL_ILE
runpy.run_path(REAL_ILE, run_name="__main__")
