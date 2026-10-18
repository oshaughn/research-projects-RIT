"""Exercise the actual Asimov Liquid policy, including YAML boolean opt-out."""
from pathlib import Path
import configparser
import pytest
liquid = pytest.importorskip("liquid")
ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("mode,expected", [(None,None),("auto","auto"),("off","off"),(False,"off"),(True,"geometric4"),("geometric4","geometric4"),("geometric4-phase-excess","geometric4-phase-excess")])
def test_asimov_transverse_override_render(mode, expected):
    text = (ROOT / "RIFT/asimov/rift.ini").read_text()
    start = text.index('cip-fit-method=')
    end = text.index('cip-sampler-method=', start)
    sampler = {"cip": {}}
    if mode is not None:
        sampler["cip"]["transverse spin coordinates"] = mode
    rendered = liquid.Liquid(text[start:end], from_file=False).render(sampler=sampler)
    parser = configparser.RawConfigParser()
    parser.read_string("[policy]\n" + rendered)
    if expected is None:
        # A silent ledger emits nothing, so pseudo_pipe keeps its default (off).
        assert not parser.has_option("policy", "rf-transverse-spin-coordinates")
    else:
        assert parser.get("policy", "rf-transverse-spin-coordinates").strip('"') == expected
    assert parser.get("policy", "cip-fit-method").strip('"') == "rf"


def test_explicit_other_interpolator_is_not_replaced():
    text = (ROOT / "RIFT/asimov/rift.ini").read_text()
    start = text.index('cip-fit-method=')
    end = text.index('cip-sampler-method=', start)
    rendered = liquid.Liquid(text[start:end], from_file=False).render(sampler={"cip": {"fitting method":"gp", "transverse spin coordinates":"off"}})
    assert 'cip-fit-method="gp"' in rendered
    assert 'rf-transverse-spin-coordinates="off"' in rendered
