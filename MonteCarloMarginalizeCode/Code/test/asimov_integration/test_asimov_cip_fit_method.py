"""sampler.cip.fitting method is validated before the template renders it."""
from types import SimpleNamespace
import pytest

pytest.importorskip("asimov")
Rift = pytest.importorskip("RIFT.asimov.rift").Rift


class _Stub:
    _CIP_FIT_METHODS = Rift._CIP_FIT_METHODS
    _validate_cip_fit_method = Rift._validate_cip_fit_method

    def __init__(self, meta):
        self.production = SimpleNamespace(meta=meta)


@pytest.mark.parametrize("value", ["rf", "gp", "gp-matern", "gp-torch", "quadratic"])
def test_fit_method_accepted(value):
    _Stub({"sampler": {"cip": {"fitting method": value}}})._validate_cip_fit_method()


@pytest.mark.parametrize("meta", [{}, {"sampler": None}, {"sampler": {"cip": None}}, {"sampler": {"cip": {}}}])
def test_silent_ledger_accepted(meta):
    _Stub(meta)._validate_cip_fit_method()


@pytest.mark.parametrize("value", ["RF", "gp_matern", "random forest", None, 1])
def test_fit_method_rejected(value):
    with pytest.raises(ValueError, match="fitting method"):
        _Stub({"sampler": {"cip": {"fitting method": value}}})._validate_cip_fit_method()
