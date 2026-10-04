#!/usr/bin/env python3
"""Ledger validation and submit options in RIFT.asimov.rift, driven against a stub production."""

from types import SimpleNamespace

import pytest

pytest.importorskip("asimov")
rift_asimov = pytest.importorskip("RIFT.asimov.rift")
Rift = rift_asimov.Rift


class _Stub:
    _CIP_FIT_METHODS = Rift._CIP_FIT_METHODS
    _validate_transverse_spin_coordinates = Rift._validate_transverse_spin_coordinates
    submit_dag = Rift.submit_dag

    def __init__(self, meta):
        self.production = SimpleNamespace(
            meta=meta, name="Prod0", event=SimpleNamespace(name="GW150914"))
        self.staged = []

    def before_submit(self):
        pass

    def _stage_xml_psds(self, dryrun=False, rundir=None):
        self.staged.append(dryrun)


@pytest.mark.parametrize("value", ["off", "auto", "physics3", True, False])
def test_transverse_spin_ledger_values_accepted(value):
    _Stub({"sampler": {"cip": {"transverse spin coordinates": value}}})._validate_transverse_spin_coordinates()


@pytest.mark.parametrize("meta", [{}, {"sampler": None}, {"sampler": {"cip": None}}, {"sampler": {"cip": {}}}])
def test_transverse_spin_silent_ledger_accepted(meta):
    _Stub(meta)._validate_transverse_spin_coordinates()


@pytest.mark.parametrize("value", ["Auto", "OFF", "physics", "on", None, 1, 0, 20.0])
def test_transverse_spin_ledger_values_rejected(value):
    with pytest.raises(ValueError, match="transverse spin coordinates"):
        _Stub({"sampler": {"cip": {"transverse spin coordinates": value}}})._validate_transverse_spin_coordinates()


def _submit(meta, capsys):
    stub = _Stub(meta)
    stub.submit_dag(dryrun=True)
    assert stub.staged == [True]
    return capsys.readouterr().out.strip().splitlines()[-1].split()


@pytest.mark.parametrize("meta", [{}, {"scheduler": None}, {"scheduler": {}}, {"scheduler": {"priority": None}}])
def test_submit_without_priority(meta, capsys):
    assert "-priority" not in _submit(meta, capsys)


@pytest.mark.parametrize("value", [900, "900", -5])
def test_submit_with_priority(value, capsys):
    command = _submit({"scheduler": {"priority": value}}, capsys)
    assert command[:3] == ["condor_submit_dag", "-priority", str(int(value))]


@pytest.mark.parametrize("value", [True, "900;bad", "1.5"])
def test_submit_rejects_bad_priority(value, capsys):
    with pytest.raises(ValueError, match="priority"):
        _submit({"scheduler": {"priority": value}}, capsys)


@pytest.mark.parametrize("value", ["rf", "gp", "quadratic", "gp-jax-rff"])
def test_cip_fit_method_ledger_values_accepted(value):
    _Stub({"sampler": {"cip": {"fit method": value}}})._validate_transverse_spin_coordinates()


@pytest.mark.parametrize("value", ["RF", "random forest", None, True, ""])
def test_cip_fit_method_ledger_values_rejected(value):
    with pytest.raises(ValueError, match="fit method"):
        _Stub({"sampler": {"cip": {"fit method": value}}})._validate_transverse_spin_coordinates()
