#!/usr/bin/env python
"""
Regression tests for the distance-slice integrand staying on its native backend
(RIFT/misc/distance_slices.py, ``fresh_sample_slices`` -> ``like_at_pinned_d``).

Background (the defect these tests lock down).  ``mcsamplerAdaptiveVolume.integrate_log``
feeds the integrand its DEVICE-NATIVE sample block first and only falls back to a host
copy if that raises::

    if getattr(self, '_integrand_wants_host', False):
        lnL = _eval_integrand(identity_convert(rv))
    else:
        try:
            lnL = _eval_integrand(rv)
        except (TypeError, ValueError):
            self._integrand_wants_host = True
            lnL = _eval_integrand(identity_convert(rv))

``like_at_pinned_d`` used to clip with ``np.clip(np.asarray(arr, float), ...)`` and pin
the distance with ``np.full``.  ``np.asarray`` on a cupy array raises TypeError
("Implicit conversion to a NumPy array is not allowed"), so on a GPU run the device
attempt ALWAYS failed and every block took the host path: a D2H copy of the whole rv
block, host-side clip, and then the device-native ILE likelihood copying ~6 arrays back
H2D via ``xpy_default.asarray`` -- a full PCIe round trip per block for arithmetic that
the GPU was about to redo anyway.  The physics was never wrong; the transfers were pure
waste.

Measured on an RTX PRO 4000 Blackwell (4 slices, n_chunk=10000, cupy 14.1.1 / CUDA 12.9),
before -> after: D2H 3.260 -> 1.886 MB, H2D 3.663 -> 1.832 MB, and lnL bitwise identical.

The eps-inward clip itself is load-bearing and must survive: ``np.random.uniform`` can
return ``rlim - 1ULP``, and the extrinsic likelihood takes ``arccos`` of the
cosine-sampled declination/inclination, which is NaN just outside [-1, 1].  So the
requirement is "clip exactly as before, on whichever backend the samples arrived on".
"""

import numpy as np
import pytest

from RIFT.misc import distance_slices
from RIFT.misc.distance_slices import _array_module, fresh_sample_slices

import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAV


# ---------------------------------------------------------------- fixtures --

BOUNDS = {
    "right_ascension": (0.0, 2 * np.pi),
    "declination": (-1.0, 1.0),     # cosine-sampled -> arccos downstream
    "inclination": (-1.0, 1.0),     # cosine-sampled -> arccos downstream
    "distance": (10.0, 4000.0),
}

OMEGA_PARAMS = [p for p in BOUNDS if p != "distance"]


class _RefSampler(object):
    """Minimal stand-in for the ILE extrinsic sampler.

    ``fresh_sample_slices`` only reads .params_ordered / .pdf / .prior_pdf /
    .llim / .rlim off the reference sampler.
    """

    def __init__(self, bounds=BOUNDS):
        self.params_ordered = list(bounds)
        self.llim = {p: lo for p, (lo, _) in bounds.items()}
        self.rlim = {p: hi for p, (_, hi) in bounds.items()}
        self.pdf = {}
        self.prior_pdf = {}
        for p, (lo, hi) in bounds.items():
            norm = 1.0 / (hi - lo)
            self.pdf[p] = (lambda x, _n=norm: np.full(np.shape(x), _n))
            self.prior_pdf[p] = (lambda x, _n=norm: np.full(np.shape(x), _n))


def _expected_eps(p):
    lo, hi = BOUNDS[p]
    return 1e-12 * max(abs(hi - lo), 1.0)


def _make_likelihood(xp, record):
    """Stand-in for the vectorized ILE likelihood, on backend ``xp``.

    Mirrors the real one's shape: ``xp.asarray`` on every argument, ``arccos`` of the
    cosine-sampled angles, and a rho*rho0 - rho^2/2 peak so AV has something to
    converge on.
    """

    def likelihood_function(right_ascension, declination, inclination, distance):
        record["backends"].add(type(right_ascension).__module__.split(".")[0])
        record["distances"].add(float(distance[0]))
        for name, arr in (("right_ascension", right_ascension),
                          ("declination", declination),
                          ("inclination", inclination)):
            lo, hi = BOUNDS[name]
            eps = _expected_eps(name)
            amin = float(xp.min(arr))
            amax = float(xp.max(arr))
            record["out_of_range"] |= (amin < lo + eps) or (amax > hi - eps)
        ra = xp.asarray(right_ascension, dtype=np.float64)
        dec = np.pi / 2 - xp.arccos(xp.asarray(declination, dtype=np.float64))
        incl = xp.arccos(xp.asarray(inclination, dtype=np.float64))
        d = xp.asarray(distance, dtype=np.float64)
        amp = (400.0 / d) * (0.5 * (1.0 + xp.cos(incl) ** 2)) \
            * (1.0 + 0.3 * xp.cos(dec) * xp.cos(ra))
        rho = 20.0 * amp
        lnL = rho * 20.0 - 0.5 * rho ** 2
        record["saw_nan"] |= bool(xp.any(xp.isnan(lnL)))
        return lnL

    return likelihood_function


def _make_host_only_likelihood(record):
    """A numpy-only integrand (the CI/benchmark contract): rejects cupy input."""
    inner = _make_likelihood(np, record)

    def likelihood_function(right_ascension, declination, inclination, distance):
        for a in (right_ascension, declination, inclination, distance):
            if type(a).__module__.split(".")[0] == "cupy":
                record["n_rejected_device"] += 1
                raise TypeError("host-only integrand: refusing cupy input")
        return inner(right_ascension, declination, inclination, distance)

    return likelihood_function


def _new_record():
    return {"backends": set(), "distances": set(), "out_of_range": False,
            "saw_nan": False, "n_rejected_device": 0}


def _run_slices(monkeypatch, like, d_slices, n_chunk=2000, seed=1234):
    """Run fresh_sample_slices, capturing every MCSampler it builds.

    fresh_sample_slices constructs one sampler per slice and keeps it local, so
    ``_integrand_wants_host`` is only observable by intercepting the constructor.
    """
    made = []
    real_ctor = mcsamplerAV.MCSampler

    class _WatchedMCSampler(real_ctor):
        def __init__(self, *a, **kw):
            super(_WatchedMCSampler, self).__init__(*a, **kw)
            made.append(self)

    monkeypatch.setattr(mcsamplerAV, "MCSampler", _WatchedMCSampler)

    np.random.seed(seed)
    if mcsamplerAV.cupy_ok:
        mcsamplerAV.xpy_default.random.seed(seed)
    out = fresh_sample_slices(_RefSampler(), like, d_slices,
                              n_max=20000, n_eff_target=30, n_chunk=n_chunk,
                              verbose=False)
    return out, made


requires_gpu = pytest.mark.skipif(
    not mcsamplerAV.cupy_ok,
    reason="needs a working cupy/GPU (mcsamplerAdaptiveVolume reports cupy_ok False)")


# ------------------------------------------------------- backend dispatch --

def test_array_module_of_numpy_is_numpy():
    assert _array_module(np.zeros(3)) is np
    assert _array_module(np.float64(1.0)) is np


def test_array_module_of_non_array_is_numpy():
    # lists / python scalars must not send us hunting for cupy
    assert _array_module([1.0, 2.0]) is np
    assert _array_module(1.0) is np


def test_array_module_dispatches_on_the_arrays_own_module(monkeypatch):
    """A cupy-flavoured array must resolve to the cupy module, not numpy.

    Uses a stand-in registered as ``cupy`` so the dispatch rule is exercised on
    hosts with no GPU; the real cupy path is covered by the GPU tests below.
    """
    import sys
    import types

    fake_cupy = types.ModuleType("cupy")

    class _FakeDeviceArray(object):
        pass

    _FakeDeviceArray.__module__ = "cupy"
    fake_cupy.ndarray = _FakeDeviceArray
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    assert _array_module(_FakeDeviceArray()) is fake_cupy


# ------------------------------------------------- backend-agnostic contract --

def test_slices_are_finite_and_stay_inside_the_eps_clip(monkeypatch):
    """The clip/pin contract, on whichever backend the sampler picked.

    Runs on numpy where there is no GPU and on cupy where there is; either way the
    integrand must see values strictly inside (lo, hi) by the eps margin, the
    distance pinned at exactly d_k, and no NaN out of arccos.
    """
    record = _new_record()
    d_slices = np.linspace(200.0, 1200.0, 4)
    (lnL, sigmaL, neff, ntotal), made = _run_slices(
        monkeypatch, _make_likelihood(mcsamplerAV.xpy_default, record), d_slices)

    assert record["backends"] == {"cupy" if mcsamplerAV.cupy_ok else "numpy"}
    assert not record["out_of_range"], \
        "an Omega sample reached (or passed) a bound: the eps-inward clip is gone"
    assert not record["saw_nan"], "arccos saw an out-of-range value"
    # every block is pinned at exactly the requested slice distance
    assert record["distances"] == set(float(d) for d in d_slices)
    assert np.all(np.isfinite(lnL))
    assert np.all(neff > 0)
    assert np.all(ntotal > 0)
    # the integrand matches the sampler's backend, so nothing should ever raise
    assert [getattr(s, "_integrand_wants_host", False) for s in made] == [False] * 4


# -------------------------------------------------------------------- GPU --

@requires_gpu
def test_gpu_integrand_gets_device_arrays_without_a_host_fallback(monkeypatch):
    """The regression: on a GPU run the device attempt must SUCCEED.

    If ``like_at_pinned_d`` forces numpy again, AV catches the TypeError, sets
    ``_integrand_wants_host``, and every subsequent block round-trips over PCIe.
    """
    record = _new_record()
    d_slices = np.linspace(200.0, 1200.0, 4)
    (lnL, _, _, _), made = _run_slices(
        monkeypatch, _make_likelihood(mcsamplerAV.xpy_default, record), d_slices)

    assert record["backends"] == {"cupy"}, \
        "integrand was handed host arrays: like_at_pinned_d forced a D2H copy"
    assert [getattr(s, "_integrand_wants_host", False) for s in made] == [False] * 4, \
        "AV armed its host fallback: the device attempt raised"
    assert not record["out_of_range"]
    assert not record["saw_nan"]
    assert np.all(np.isfinite(lnL))


@requires_gpu
def test_gpu_host_only_integrand_still_falls_back(monkeypatch):
    """A numpy-only integrand on a GPU run must still work, and agree.

    With the device-native clip the TypeError now comes from the integrand rather
    than from the clip, so AV's fallback has to catch it just the same -- exactly
    once per sampler, not once per block.
    """
    d_slices = np.linspace(200.0, 1200.0, 4)

    rec_dev = _new_record()
    (lnL_dev, _, _, _), _ = _run_slices(
        monkeypatch, _make_likelihood(mcsamplerAV.xpy_default, rec_dev), d_slices)

    rec_host = _new_record()
    (lnL_host, _, _, _), made = _run_slices(
        monkeypatch, _make_host_only_likelihood(rec_host), d_slices)

    assert [getattr(s, "_integrand_wants_host", False) for s in made] == [True] * 4
    # one probe per fresh sampler (fresh_sample_slices builds one per slice), not one
    # per block: the flag is what stops it from re-raising every cycle
    assert rec_host["n_rejected_device"] == len(d_slices)
    assert not rec_host["saw_nan"]
    np.testing.assert_array_equal(lnL_dev, lnL_host)


@requires_gpu
def test_gpu_and_cpu_slice_integrals_agree(monkeypatch):
    """Same pinned distances, same clip, same integrand -> same lnL to fp tolerance.

    Draws differ (numpy vs cupy RNG), so this compares the integrals, not samples;
    the tolerance is the sampler's own quoted sigma.
    """
    d_slices = np.linspace(200.0, 1200.0, 4)

    rec_gpu = _new_record()
    (lnL_gpu, sigma_gpu, _, _), _ = _run_slices(
        monkeypatch, _make_likelihood(mcsamplerAV.xpy_default, rec_gpu), d_slices)

    rec_cpu = _new_record()
    (lnL_cpu, sigma_cpu, _, _), _ = _run_slices(
        monkeypatch, _make_likelihood(np, rec_cpu), d_slices)

    tol = 5.0 * np.sqrt(sigma_gpu ** 2 + sigma_cpu ** 2)
    assert np.all(np.abs(lnL_gpu - lnL_cpu) < np.maximum(tol, 0.05)), \
        "GPU and CPU slice integrals disagree by more than the quoted error"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
