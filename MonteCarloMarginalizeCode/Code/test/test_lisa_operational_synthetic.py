#!/usr/bin/env python
"""Operational smoke test for the LISA likelihood path."""

import inspect
import os

import lal
import lalsimulation as lalsim
import numpy as np
import pytest

import RIFT.LISA.lalsimutils_compat as lisa_lalsimutils_compat
import RIFT.lalsimutils as lalsimutils
from RIFT.LISA.response import LISA_response
from RIFT.likelihood import factored_likelihood_LISA


def _synthetic_lisa_params():
    P = lalsimutils.ChooseWaveformParams()
    P.m1 = 1.0e5 * lal.MSUN_SI
    P.m2 = 8.0e4 * lal.MSUN_SI
    P.s1z = 0.1
    P.s2z = -0.05
    P.dist = 1.0e9 * lal.PC_SI
    P.fmin = 1.0e-3
    P.fref = 5.0e-3
    P.fmax = 0.125
    P.deltaT = 4.0
    P.deltaF = 1.0 / 4096.0
    P.approx = lalsim.IMRPhenomD
    P.theta = 0.3
    P.phi = 1.0
    P.psi = 0.2
    P.incl = 0.4
    P.phiref = 0.1
    return P


def _write_cache(tmp_path):
    cache_path = tmp_path / "lisa.cache"
    rows = [
        ("A", "A-fake_strain-1000000-10000.h5"),
        ("E", "E-fake_strain-1000000-10000.h5"),
        ("T", "T-fake_strain-1000000-10000.h5"),
    ]
    cache_path.write_text(
        "".join(
            f"{channel} {channel} 0 1 file://localhost{os.fspath(tmp_path / filename)}\n"
            for channel, filename in rows
        )
    )
    return cache_path


def _flat_psd_like(data_dict):
    psd_dict = {}
    for channel, data in data_dict.items():
        psd = lal.CreateREAL8FrequencySeries(
            "PSD",
            lal.LIGOTimeGPS(0),
            0,
            data.deltaF,
            lalsimutils.lsu_HertzUnit,
            data.data.length,
        )
        psd.data.data[:] = 1.0e-40
        psd_dict[channel] = psd
    return psd_dict


def _evaluate_lisa_lnL(rholms, cross_terms, modes, P, psi=None, inclination=None):
    return factored_likelihood_LISA.FactoredLogLikelihoodAlignedSpinLISA(
        rholms,
        cross_terms,
        P.theta,
        P.phi,
        np.array([P.psi if psi is None else psi]),
        np.array([P.incl if inclination is None else inclination]),
        np.array([P.phiref]),
        np.array([P.dist]),
        modes,
        P.dist,
    )[0]


@pytest.fixture(scope="module")
def lisa_precompute(tmp_path_factory):
    """Build the synthetic TDI data and precompute once for the whole module.

    Module-scoped because the build costs several seconds and every test here
    consumes exactly the same product.
    """
    tmp_path = tmp_path_factory.mktemp("lisa_synthetic")
    P = _synthetic_lisa_params()
    modes = [(2, 2)]

    hlms = lisa_lalsimutils_compat.hlmoff_for_LISA(P, Lmax=2, modes=modes)
    generated_data = LISA_response.create_lisa_injections(
        hlms,
        P.fmax,
        P.fref,
        P.theta,
        P.phi,
        P.psi,
        P.incl,
        P.phiref,
        tref=0.0,
    )
    LISA_response.create_h5_files_from_data_dict(generated_data, os.fspath(tmp_path))

    cache_path = _write_cache(tmp_path)
    data_dict = {
        channel: lisa_lalsimutils_compat.frame_h5_to_hoff(
            os.fspath(cache_path), channel, verbose=False
        )
        for channel in ("A", "E", "T")
    }
    psd_dict = _flat_psd_like(data_dict)

    _, cross_terms, _, rholms, _, _ = factored_likelihood_LISA.PrecomputeAlignedSpinLISA(
        0.0,
        P.fref,
        8.0,
        hlms,
        None,
        data_dict,
        psd_dict,
        P.fmin,
        0.5 / P.deltaT,
        P.fmax,
        P.deltaT,
        P.theta,
        P.phi,
        analyticPSD_Q=False,
        inv_spec_trunc_Q=False,
        T_spec=0.0,
    )

    modes_array = np.array(list(hlms.keys()))
    return P, modes_array, rholms, cross_terms


def test_synthetic_lisa_tdi_precompute_and_likelihood(lisa_precompute):
    P, modes_array, rholms, cross_terms = lisa_precompute

    lnL_at_injection = _evaluate_lisa_lnL(rholms, cross_terms, modes_array, P)
    lnL_offset = _evaluate_lisa_lnL(
        rholms, cross_terms, modes_array, P, psi=P.psi + 0.8, inclination=P.incl + 0.5
    )

    assert np.isfinite(lnL_at_injection)
    assert np.isfinite(lnL_offset)
    assert lnL_at_injection > lnL_offset


def test_lisa_likelihood_names_a_host_backend_for_spherical_harmonics(
    lisa_precompute, monkeypatch
):
    """The LISA likelihood must name numpy when it calls SphericalHarmonicsVectorized.

    `SphericalHarmonicsVectorized` resolves to cupy on any host where cupy merely
    imports, while this routine works entirely in host numpy.  Omitting `xpy`
    here used to raise `TypeError: Unsupported type <class 'numpy.ndarray'>` from
    `cupy._core._kernel._preprocess_args` on every GPU host, and pass everywhere
    else -- so assert on the argument, not on the outcome, and this gate holds on
    a cupy-free CI runner too.
    """
    P, modes_array, rholms, cross_terms = lisa_precompute

    real = factored_likelihood_LISA.SphericalHarmonicsVectorized
    seen = []

    unset = object()
    signature = inspect.signature(real)

    def spy(*args, **kwargs):
        # Bind through the real signature: a call site that passes xpy
        # POSITIONALLY is equally correct and must not be misreported as a
        # failure to name a backend.
        bound = signature.bind(*args, **kwargs)
        seen.append((bound.arguments["theta"], bound.arguments.get("xpy", unset)))
        return real(*args, **kwargs)

    monkeypatch.setattr(
        factored_likelihood_LISA, "SphericalHarmonicsVectorized", spy
    )
    _evaluate_lisa_lnL(rholms, cross_terms, modes_array, P)

    assert seen, "SphericalHarmonicsVectorized was never called"
    for theta, xpy in seen:
        assert xpy is np, (
            "LISA likelihood left xpy unset (or non-numpy: %r) for a %s theta; "
            "host arrays must be computed with numpy"
            % ("unset" if xpy is unset else xpy, type(theta).__name__)
        )
