"""Unit tests for RIFT.hyperpipe.config.validate_config + helpers."""
from __future__ import annotations

import pytest


def test_validate_missing_section(hp_modules):
    with pytest.raises(ValueError, match="missing required section"):
        hp_modules.config.validate_config({})


def test_validate_empty_marg_list(hp_modules):
    cfg = {
        "arch": {"n-iterations": 5, "n-samples-per-job": 1000},
        "post": {"coords-fit": "x"},
        "marg-list": [],
        "puff": {},
        "init": {"file": "x"},
        "general": {},
    }
    with pytest.raises(ValueError, match="at least one entry"):
        hp_modules.config.validate_config(cfg)


def test_validate_bad_arch(hp_modules):
    cfg = {
        "arch": {"n-iterations": 0, "n-samples-per-job": 1000},
        "post": {"coords-fit": "x"},
        "marg-list": [{"exe": "x"}],
        "puff": {},
        "init": {"file": "x"},
        "general": {},
    }
    with pytest.raises(ValueError, match="n-iterations"):
        hp_modules.config.validate_config(cfg)


def test_validate_missing_init(hp_modules):
    cfg = {
        "arch": {"n-iterations": 5, "n-samples-per-job": 1000},
        "post": {"coords-fit": "x"},
        "marg-list": [{"exe": "x"}],
        "puff": {},
        "init": {},
        "general": {},
    }
    with pytest.raises(ValueError, match="init.file"):
        hp_modules.config.validate_config(cfg)


def test_validate_happy_path(hp_modules):
    hp_modules.config.validate_config({
        "arch": {"n-iterations": 5, "n-samples-per-job": 1000},
        "post": {"coords-fit": "x y z"},
        "marg-list": [{"exe": "example.py"}],
        "puff": {},
        "init": {"file": "/tmp/x"},
        "general": {},
    })


def test_truthy_string_handling(hp_modules):
    t = hp_modules.config.truthy
    assert t("true") and t("True") and t("YES") and t("1") and t("on")
    assert not t("false") and not t("no") and not t("") and not t(None)
    assert t(True) and not t(False) and t(1) and not t(0)
