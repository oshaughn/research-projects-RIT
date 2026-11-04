"""Fixed-phase Phenom production-mode comparisons; no fitted lag or phase.

Generic spins are the primary waveform control. Aligned inputs exercise the
separate zero-tilt convention branch. Symmetric XPNR uses its native reference
emission axis; raw upstream LAL projection defects are tested separately.
"""
import copy

import lal
import lalsimulation as ls
import numpy as np
import pytest

from RIFT import lalsimutils as lsu
from RIFT.likelihood import factored_likelihood as fl

# RIFT generates modes without multibanding; match both waveform paths.
# Disable the known-broken native antisymmetry option explicitly, including
# XPNR, whose native default otherwise enables it.
PHENOM_TEST_OPTIONS = {"PhenomXAntisymmetricWaveform": 0,
                       "PhenomXPHMMBandVersion": 0,
                       "PhenomXHMThresholdMband": 0.,
                       "PhenomXPHMThresholdMband": 0.}
SPINTAYLOR = {**PHENOM_TEST_OPTIONS, "PhenomXPrecVersion": 320, "PhenomXPFinalSpinMod": 2,
             "PhenomXHMReleaseVersion": 122022}
GENERIC = [
    (70., 20., .45, .3, .8, 1.7, .7, 1.2, 1.1, 1.4),
    (55., 35., .65, .45, 1.1, .6, 1.3, 2.2, .25, .8),
    (80., 15., .6, .35, .6, 2.1, 2., .3, 2., 2.2),
]
FREQUENCIES = np.fft.rfftfreq(16 * 4096, 1 / 4096)
PSD = 1e-46 * ((30 / np.maximum(FREQUENCIES, 1))**4 + 1
               + (FREQUENCIES / 300)**2)
WEIGHT = ((FREQUENCIES >= 20) & (FREQUENCIES <= 512)) / PSD


def parameters(approx, case):
    m1, m2, a1, a2, t1, t2, p12, pjl, phase, theta = case
    spins = ls.SimInspiralTransformPrecessingNewInitialConditions(
        theta, pjl, t1, t2, p12, a1, a2,
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 20., phase)
    p = lsu.ChooseWaveformParams()
    p.m1, p.m2 = m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
    p.incl, p.s1x, p.s1y, p.s1z, p.s2x, p.s2y, p.s2z = map(float, spins)
    p.phiref, p.psi = phase, 0.
    p.dist = 400e6 * lal.PC_SI
    p.fmin = p.fref = 20.
    p.deltaT, p.deltaF = 1 / 4096, 1 / 16
    p.approx = ls.GetApproximantFromString(approx)
    return p


def mode_waveform(p, options):
    options = {**PHENOM_TEST_OPTIONS, **options}
    intrinsic = p.manual_copy()
    intrinsic.phiref = intrinsic.incl = 0.
    modes, _ = fl.internal_hlm_generator(
        intrinsic, 5, verbose=False, quiet=True,
        extra_waveform_kwargs=dict(fd_L_frame=True, fd_alignment_postevent_time=2.,
                                   fd_standoff_factor=.9,
                                   extra_waveform_args=copy.deepcopy(options)))
    time_modes = {lm: lsu.DataInverseFourier(h) for lm, h in modes.items()}
    strain = lsu.hoft_from_hlm(time_modes, p, return_complex=True)
    epoch = np.exp(-2j * np.pi * FREQUENCIES * float(strain.epoch))
    return {"plus": np.fft.rfft(strain.data.data.real) * p.deltaT * epoch,
            "cross": np.fft.rfft(-strain.data.data.imag) * p.deltaT * epoch}


def direct_waveform(p, options):
    options = {**PHENOM_TEST_OPTIONS, **options}
    d = p.to_lal_dict_extended(extra_args_dict=copy.deepcopy(options))
    hp, hc = ls.SimInspiralChooseFDWaveform(
        p.m1, p.m2, p.s1x, p.s1y, p.s1z, p.s2x, p.s2y, p.s2z,
        p.dist, p.incl, p.phiref, 0., 0., 0., p.deltaF, p.fmin,
        1 / (2 * p.deltaT), p.fref, d, p.approx)
    return {k: np.pad(h.data.data, (0, len(FREQUENCIES) - len(h.data.data)))
            for k, h in (("plus", hp), ("cross", hc))}


def assert_fixed_phase_agreement(p, options):
    candidate, reference = mode_waveform(p, options), direct_waveform(p, options)
    for pol in reference:
        a, b = reference[pol], candidate[pol]
        aa = np.sum(abs(a)**2 * WEIGHT)
        bb = np.sum(abs(b)**2 * WEIGHT)
        residual = np.sqrt(np.sum(abs(a - b)**2 * WEIGHT) / aa)
        mismatch = 1 - np.sum(a.conj() * b * WEIGHT).real / np.sqrt(aa * bb)
        assert residual < .02, (pol, residual)
        assert mismatch < 1e-4, (pol, mismatch)


@pytest.mark.parametrize("case", GENERIC, ids=["q3.5", "two-tilted-spins", "q5.3"])
def test_generic_spintaylor_fixed_phase(case):
    p = parameters("IMRPhenomXPHM", case)
    assert np.linalg.norm([p.s1x, p.s1y, p.s2x, p.s2y]) > .1
    assert_fixed_phase_agreement(p, SPINTAYLOR)


@pytest.mark.parametrize("case", GENERIC, ids=["q3.5", "two-tilted-spins", "q5.3"])
@pytest.mark.parametrize("phase", [None, 0.], ids=["fixture-phase", "zero-phase"])
@pytest.mark.parametrize("fref", [0., 20., 35.013])
def test_generic_symmetric_xpnr_fixed_phase(case, phase, fref):
    p = parameters("IMRPhenomXPNR", case)
    if phase is not None:
        # Hold Cartesian spins fixed: compare identical inputs between routes.
        p.phiref = phase
    p.fref = fref
    assert_fixed_phase_agreement(p, PHENOM_TEST_OPTIONS)


def test_xpnr_reference_in_native_cutoff_partial_bin():
    # For this mass pair native f_max_prime is 676.751489 Hz, whereas the
    # public beta samples stop at 676.75 Hz. This is a valid native request.
    p = parameters("IMRPhenomXPNR", GENERIC[0])
    p.fref = 676.751
    assert_fixed_phase_agreement(p, PHENOM_TEST_OPTIONS)


def test_xpnr_reference_mapping_preserves_explicit_angle_selection():
    p = parameters("IMRPhenomXPNR", GENERIC[0])
    d = p.to_lal_dict_extended(extra_args_dict={**PHENOM_TEST_OPTIONS,
                                               "PhenomXPNRUseTunedAngles": 0})
    assert lsu._xpnr_reference_opening_angle(p, d, 18., 2048.) is None
    assert ls.SimInspiralWaveformParamsLookupPhenomXPNRUseTunedAngles(d) == 0


@pytest.mark.parametrize("approx,options", [("IMRPhenomXPHM", SPINTAYLOR),
                                           ("IMRPhenomXPNR", {})])
@pytest.mark.parametrize("case", GENERIC, ids=["q3.5", "two-tilted-spins", "q5.3"])
def test_generic_modes_are_intrinsic_under_polarization(approx, options, case):
    p = parameters(approx, case)
    assert np.linalg.norm([p.s1x, p.s1y, p.s2x, p.s2y]) > .1
    # Undo only the known extrinsic polarization of the reconstructed strain.
    a = mode_waveform(p, options)
    p.psi = .37
    b = mode_waveform(p, options)
    expected_plus = a["plus"] * np.cos(2 * p.psi) - a["cross"] * np.sin(2 * p.psi)
    expected_cross = a["plus"] * np.sin(2 * p.psi) + a["cross"] * np.cos(2 * p.psi)
    for key, expected in (("plus", expected_plus), ("cross", expected_cross)):
        assert np.linalg.norm(b[key] - expected) / np.linalg.norm(expected) < 1e-12


@pytest.mark.parametrize("approx,options", [("IMRPhenomXPHM", {}),
                                           ("IMRPhenomXPHM", SPINTAYLOR),
                                           ("IMRPhenomXO4a", {}),
                                           ("IMRPhenomXPNR", {})],
                         ids=["XPHM", "SpinTaylor", "XO4a", "XPNR"])
@pytest.mark.parametrize("phase,incl,transverse", [(.3, 1.1, 0.), (1.1, 1.4, 1e-12)],
                         ids=["aligned", "below-aligned-threshold"])
@pytest.mark.parametrize("fref", [0., 20., 35.], ids=["default-ref", "start-ref", "higher-ref"])
def test_aligned_boundary_fixed_phase(approx, options, phase, incl, transverse, fref):
    p = parameters(approx, (60., 30., .45, .3, 0., np.pi, .7, .4, phase, incl))
    p.s1x = transverse
    p.fref = fref
    assert np.linalg.norm([p.s1x, p.s1y, p.s2x, p.s2y]) < 1e-10
    assert_fixed_phase_agreement(p, options)


def test_existing_rotation_driver_transports_spintaylor_options(monkeypatch):
    """The existing driver must compare the requested model on both sides."""
    import runpy
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    captured = {}
    dummy = SimpleNamespace(data=SimpleNamespace(data=np.ones(8, dtype=complex)))

    def modes(p, *args, **kwargs):
        captured["modes"] = copy.deepcopy(kwargs["extra_waveform_kwargs"])
        return {(2, 2): dummy}, None

    class Finished(Exception):
        pass

    def direct(p, **kwargs):
        captured["direct"] = copy.deepcopy(kwargs["extra_waveform_args"])
        raise Finished

    monkeypatch.setattr(fl, "internal_hlm_generator", modes)
    monkeypatch.setattr(lsu, "DataInverseFourier", lambda h: dummy)
    monkeypatch.setattr(lsu, "hoft_from_hlm", lambda *a, **k: dummy)
    monkeypatch.setattr(lsu, "complex_hoft", direct)
    script = Path(__file__).parent / "waveform" / "check_waveform_random.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--approximant", "IMRPhenomXPHM",
                                     "--fiducial", "--use-extra-fd-args",
                                     "--use-xphm-spintaylor"])
    with pytest.raises(Finished):
        runpy.run_path(str(script), run_name="__main__")
    assert captured["modes"]["fd_L_frame"] is True
    assert captured["modes"]["extra_waveform_args"] == SPINTAYLOR
    assert captured["direct"] == SPINTAYLOR
