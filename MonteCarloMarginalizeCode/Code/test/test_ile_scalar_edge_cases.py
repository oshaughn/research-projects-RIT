import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from RIFT.likelihood import factored_likelihood as fl


class _Data:
    def __init__(self, values):
        self.data = np.asarray(values, dtype=np.complex128)


class _TimeSeries:
    def __init__(self, values):
        self.data = _Data(values)
        self.epoch = 0.0
        self.deltaT = 1.0


def test_scalar_time_marginalization_uses_only_retained_modes(monkeypatch):
    """A raw zero mode dropped by InterpolateRholms must not reach Ylms."""
    active = (2, 2)
    dropped = (2, 1)
    raw = {
        "H1": {
            active: _TimeSeries([1.0, 1.0, 1.0, 1.0]),
            dropped: _TimeSeries([0.0, 0.0, 0.0, 0.0]),
        }
    }
    interpolated = {"H1": {active: lambda t: np.ones_like(t)}}
    cross = {"H1": {(active, active): 1.0 + 0.0j}}
    cross_v = {"H1": {(active, active): 0.0 + 0.0j}}
    extrinsic = SimpleNamespace(
        phi=0.0, theta=0.0, tref=0.0, phiref=0.0,
        incl=0.0, psi=0.0, dist=fl.distMpcRef * 1.0e6 * fl.lsu.lsu_PC,
    )

    seen_modes = []

    def fake_ylms(_lmax, _incl, _phase, selected_modes=None):
        seen_modes.extend(selected_modes)
        return {mode: 1.0 + 0.0j for mode in selected_modes}

    monkeypatch.setattr(fl, "ComputeYlms", fake_ylms)
    monkeypatch.setattr(fl, "ComplexAntennaFactor", lambda *args: 1.0 + 0.0j)
    monkeypatch.setattr(fl, "ComputeArrivalTimeAtDetector", lambda *args: 0.0)

    result = fl.FactoredLogLikelihoodTimeMarginalized(
        np.array([0.0, 1.0]), extrinsic, interpolated, raw,
        cross, cross_v, Lmax=2, interpolate=False,
    )

    assert np.isfinite(result)
    assert set(seen_modes) == {active}


def test_ile_tref_prior_uses_backend_portable_sampler_api():
    """ILE must not require uniform_samp_vector, which AV does not export."""
    ile = Path(__file__).parents[1] / "bin" / "integrate_likelihood_extrinsic_batchmode"
    tree = ast.parse(ile.read_text())
    missing_api_uses = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "mcsampler"
        and node.attr == "uniform_samp_vector"
    ]
    assert missing_api_uses == []


def test_ile_scalar_likelihood_calls_supply_raw_and_interpolated_modes():
    """The unmarginalized scalar path must honor FactoredLogLikelihood's API."""
    ile = Path(__file__).parents[1] / "bin" / "integrate_likelihood_extrinsic_batchmode"
    tree = ast.parse(ile.read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "factored_likelihood"
        and node.func.attr == "FactoredLogLikelihood"
    ]
    assert calls
    assert all(len(call.args) >= 6 for call in calls)
