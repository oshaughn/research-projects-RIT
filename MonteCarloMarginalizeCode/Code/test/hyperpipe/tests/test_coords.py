"""
Unit tests for RIFT.hyperpipe.coords.
"""
from __future__ import annotations

import pytest


def test_parse_range_block(hp_modules):
    c = hp_modules.coords
    assert c.parse_range_block("x:[-8,8]") == ("x", (-8.0, 8.0))
    assert c.parse_range_block("g0=[0.2,2]") == ("g0", (0.2, 2.0))


def test_parse_range_string(hp_modules):
    c = hp_modules.coords
    out = c.parse_range_string("x:[-8,8] y:[-1.5,1.5] z:[0,1e3]")
    assert out == {"x": (-8.0, 8.0), "y": (-1.5, 1.5), "z": (0.0, 1e3)}


def test_gaussian_demo_emission(hp_modules):
    """post / puff / test args must match the Gaussian demo verbatim."""
    spec = hp_modules.coords.HyperCoordSpec.from_strings(
        coords_fit="x y z",
        coords_sample="x:[-8,8] y:[-8,8] z:[-8,8]",
    )
    spec.validate(strict_import=False)

    assert sorted(spec.to_post_args().split()) == sorted(
        "--parameter x --parameter y --parameter z "
        "--integration-parameter-range x:[-8,8] "
        "--integration-parameter-range y:[-8,8] "
        "--integration-parameter-range z:[-8,8]".split()
    )
    assert sorted(spec.to_puff_args(force_away=0.03, puff_factor=0.5).split()) == sorted(
        "--parameter x --parameter y --parameter z "
        "--force-away 0.03 --puff-factor 0.5".split()
    )
    assert sorted(spec.to_test_args().split()) == sorted(
        "--parameter x --parameter y --parameter z "
        "--method JS --threshold 0.05".split()
    )


def test_nicer_style_post_args(hp_modules):
    """NICER-style spec with coord module and likelihood-factor trio."""
    spec = hp_modules.coords.HyperCoordSpec.from_strings(
        name="rift_default",
        coords_fit="g0 g1 g2 g3",
        coords_sample="g0:[0.2,2] g1:[-1.6,1.7] g2:[-0.6,0.6] g3:[-0.02,0.02]",
        likelihood_factor=("my_module", "my_factor", "my.ini"),
    )
    post = spec.to_post_args()
    for needle in (
        "--supplementary-coordinate-code rift_default",
        "--supplementary-likelihood-factor-code my_module",
        "--supplementary-likelihood-factor-function my_factor",
        "--supplementary-likelihood-factor-ini my.ini",
        "--integration-parameter-range g1:[-1.6,1.7]",
    ):
        assert needle in post


def test_fmt_num_integer_preserved(hp_modules):
    """Integer-valued floats round-trip as ints (matches demo format)."""
    f = hp_modules.coords.HyperCoordSpec._fmt_num
    assert f(-8.0) == "-8"
    assert f(2.0) == "2"
    assert f(-1.6) == "-1.6"
    assert f(0.02) == "0.02"


def test_validate_rejects_missing_range(hp_modules):
    spec = hp_modules.coords.HyperCoordSpec.from_strings(
        coords_fit="x y",
        coords_sample="x:[-1,1]",
    )
    with pytest.raises(ValueError, match="No integration range supplied"):
        spec.validate(strict_import=False)
