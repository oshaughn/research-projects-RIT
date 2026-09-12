"""Regression tests for tidal parameters in RIFT's GWSignal interface."""

import numpy as np
import pytest

import lal
import astropy.units as u

import RIFT.lalsimutils as lalsimutils
import RIFT.physics.GWSignal as rift_gws


def _tidal_params(lambda1=500.0, lambda2=800.0, s1z=0.05, s2z=-0.03):
    return lalsimutils.ChooseWaveformParams(
        m1=1.4 * lal.MSUN_SI,
        m2=1.3 * lal.MSUN_SI,
        s1z=s1z,
        s2z=s2z,
        lambda1=lambda1,
        lambda2=lambda2,
        deltaT=1.0 / 4096.0,
        fmin=40.0,
        fref=40.0,
        dist=100.0e6 * lal.PC_SI,
        taper=lalsimutils.lsu_TAPER_NONE,
    )


def _normalized_tail_difference(first, second):
    first = np.asarray(first)
    second = np.asarray(second)
    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(second))

    n_common = min(first.size, second.size)
    first = first[-n_common:]
    second = second[-n_common:]
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    assert first_norm > 0.0
    assert second_norm > 0.0
    return np.linalg.norm(first / first_norm - second / second_norm)


def test_waveform_parameter_dict_propagates_tidal_deformabilities():
    params = rift_gws._waveform_parameter_dict(
        _tidal_params(), taper=0, lmax_nyquist=4
    )

    assert params["lambda1"].unit == u.dimensionless_unscaled
    assert params["lambda2"].unit == u.dimensionless_unscaled
    assert params["lambda1"].value == pytest.approx(500.0)
    assert params["lambda2"].value == pytest.approx(800.0)
    assert params["lmax_nyquist"] == 4


def test_public_tidal_waveform_responds_to_rift_lambdas():
    """Run a public, valid tidal waveform through the GWSignal boundary."""
    if not rift_gws.has_gws:
        pytest.skip("GWSignal is unavailable")

    tidal = rift_gws.hoft(
        _tidal_params(s1z=0.0, s2z=0.0),
        Fp=1.0,
        Fc=0.0,
        approx_string="TaylorT4",
    )
    point_mass = rift_gws.hoft(
        _tidal_params(lambda1=0.0, lambda2=0.0, s1z=0.0, s2z=0.0),
        Fp=1.0,
        Fc=0.0,
        approx_string="TaylorT4",
    )

    assert (
        _normalized_tail_difference(tidal.data.data, point_mass.data.data) > 1.0e-6
    )


def test_seobnrv5thm_gwsignal_modes_respond_to_tides():
    """Exercise a real mode-generating tidal model when its review build exists.

    The test intentionally skips in ordinary environments until SEOBNRv5THM is
    exposed as a GWSignal model.  In the review/production environment it guards
    against the dangerous failure mode where nonzero RIFT lambdas produce the
    same modes as the black-hole limit.
    """
    if not rift_gws.has_gws:
        pytest.skip("GWSignal is unavailable")

    try:
        tidal_generator = rift_gws.gws.models.gwsignal_get_waveform_generator(
            "SEOBNRv5THM"
        )
    except ValueError as exc:
        if "Approximant not implemented in GWSignal" not in str(exc):
            raise
        pytest.skip("SEOBNRv5THM is not exposed through GWSignal: {}".format(exc))

    metadata = tidal_generator.metadata
    if callable(metadata):
        metadata = metadata()
    assert metadata["modes"], "SEOBNRv5THM must expose native modes through GWSignal"

    tidal_dict = rift_gws._waveform_parameter_dict(_tidal_params(), taper=0)
    point_mass_dict = rift_gws._waveform_parameter_dict(
        _tidal_params(lambda1=0.0, lambda2=0.0), taper=0
    )
    # Use distinct generators so the comparison cannot pass or fail because a
    # plugin retains waveform state between calls.
    point_mass_generator = rift_gws.gws.models.gwsignal_get_waveform_generator(
        "SEOBNRv5THM"
    )
    tidal_modes = rift_gws.wfm.GenerateTDModes(tidal_dict, tidal_generator)
    point_mass_modes = rift_gws.wfm.GenerateTDModes(
        point_mass_dict, point_mass_generator
    )

    assert (2, 2) in tidal_modes
    assert (2, 2) in point_mass_modes
    # End-align, where tidal dephasing is largest, and normalize out distance.
    assert (
        _normalized_tail_difference(
            tidal_modes[(2, 2)].value, point_mass_modes[(2, 2)].value
        )
        > 1.0e-6
    )
