"""Regression guard: RIFT's gwsignal TEOBResumSDALI templates must not carry
a global sign flip relative to the same plugin's polarizations.

Background
----------
RIFT reads TEOBResumSDALI through ``GenerateTDModes``; bilby and dingo read it
through ``GenerateFDWaveform`` (polarizations).  For this one plugin the modes
are *minus* the polarization convention, and dimensionless rather than in
strain units.  ``GWSignal.hlmoft`` rescales by ``nu M G/(c^2 D)``; that factor
must carry the minus sign.

Getting it wrong is invisible in a fit.  The antenna patterns obey
``F+(psi+pi/2) = -F+(psi)`` and ``Fx(psi+pi/2) = -Fx(psi)`` in every detector
for every mode, so a global sign on h is *exactly* ``psi -> psi + pi/2`` and
nothing else: sky location, distance, masses, inclination and the peak
likelihood all still agree.  It displaced psi by a quarter turn in every RIFT
TEOBResumSDALI posterior in the eccentric-PE task force comparison.

What this test asserts, and what it deliberately does not
--------------------------------------------------------
It tests the *physics* (is there a sign error?), not the phase convention.

Fitting a free complex coefficient ``c_m`` per azimuthal index m, RIFT's
templates are correct iff ``c_m = s * exp(i m delta)`` with ``s = +1``: a
common ``delta`` is just a relabelling of ``phi_ref``, but ``s = -1`` is the
psi bug.  For a single m the two are indistinguishable -- ``-exp(i m delta)``
can be reabsorbed into ``delta`` -- which is exactly why the quadrupole alone
cannot see this.  Two m values sharing one ``delta`` separate them:

    arg(c_4) - 2 arg(c_2)   ==  0  if s = +1,   ==  pi  if s = -1

That combination is invariant under any choice of ``delta``, so this test
passes for either TEOB phase convention and fails only on a true sign error.
It is therefore agnostic about whether the ``exp(i m phi_shift)`` in
``hlmoft`` is right for TEOB -- a separate, open question about making
``phase`` line up across approximants.

A whole-waveform scalar fit would NOT work here: the buggy and correct forms
differ per mode by ``-exp(i m pi/2)``, which is ``+1`` at ``m = +-2``, so the
dominant quadrupole agrees either way and carries nearly all the power.

Not run in CI: the runners do not have the plugin packages, so the
backend-availability check skips there.  A missing backend package is the
*only* thing allowed to skip -- once the package is present, every later
failure (building the generator, waveform generation, ``hlmoft``, the fit, the
assertion) is a real failure and is reported as one.
Run it where the plugin is installed:

    python -m pytest -q MonteCarloMarginalizeCode/Code/test/test_gwsignal_teob_mode_sign.py
"""
import importlib.util

import numpy as np
import pytest

lal = pytest.importorskip("lal")
lalsim = pytest.importorskip("lalsimulation")
u = pytest.importorskip("astropy.units")

gws = pytest.importorskip("lalsimulation.gwsignal")
wfm = pytest.importorskip("lalsimulation.gwsignal.core.waveform")

import RIFT.lalsimutils as lsu  # noqa: E402
import RIFT.physics.GWSignal as rgws  # noqa: E402

M1, M2 = 80.0, 40.0          # q = 2, so odd-m modes are alive
DIST, IOTA = 1500.0, np.pi / 4
DELTAT, DELTAF = 1.0 / 4096, 1.0 / 16
F22 = FREF = 20.0


def _pdict():
    return {
        "mass1": M1 * u.solMass, "mass2": M2 * u.solMass,
        "spin1x": 0.0 * u.dimensionless_unscaled,
        "spin1y": 0.0 * u.dimensionless_unscaled,
        "spin1z": 0.0 * u.dimensionless_unscaled,
        "spin2x": 0.0 * u.dimensionless_unscaled,
        "spin2y": 0.0 * u.dimensionless_unscaled,
        "spin2z": 0.0 * u.dimensionless_unscaled,
        "deltaT": DELTAT * u.s, "f22_start": F22 * u.Hz,
        "f22_ref": FREF * u.Hz, "phi_ref": 0.0 * u.rad,
        "distance": DIST * u.Mpc, "inclination": IOTA * u.rad,
        "eccentricity": 0.0 * u.dimensionless_unscaled,
        "longAscNodes": 0.0 * u.rad, "meanPerAno": 0.0 * u.rad,
        "condition": 0,
    }


def _P():
    P = lsu.ChooseWaveformParams()
    P.m1, P.m2 = M1 * lal.MSUN_SI, M2 * lal.MSUN_SI
    P.s1x = P.s1y = P.s1z = P.s2x = P.s2y = P.s2z = 0.0
    P.dist = DIST * 1e6 * lal.PC_SI
    P.incl = IOTA
    P.phiref = P.psi = 0.0
    P.fmin, P.fref = F22, FREF
    P.deltaT, P.deltaF = DELTAT, DELTAF
    P.eccentricity, P.meanPerAno = 0.0, 0.0
    P.taper = lalsim.SIM_INSPIRAL_TAPER_NONE
    P.approx = lalsim.IMRPhenomXPHM      # unused; approx_string drives gwsignal
    return P


def _arr(x):
    return np.asarray(x.value if hasattr(x, "value") else x)


def _times(x):
    t = x.times
    return np.asarray(t.value if hasattr(t, "value") else t)


def _regrid(ts, a, grid):
    return a[np.clip(np.searchsorted(ts, grid), 0, len(a) - 1)]


# Third-party package each model is implemented by; a missing one is the only
# thing here that means "not this host" rather than "broken".  Keyed by
# approximant, so adding an approximant to the parametrization without saying
# what provides it raises KeyError instead of silently skipping.  Both are
# top-level modules, which keeps the ``find_spec`` lookup below simple.
_BACKEND_MODULE = {
    "TEOBResumSDALI": "EOBRun_module",   # pip install teobresums
    "SEOBNRv5EHM": "pyseobnr",
}


def _generator(approx):
    """The gwsignal generator for ``approx``; skip only if its backend is absent.

    Absence is decided on its own terms, before anything is constructed:
    ``find_spec`` locates the plugin package without importing it, so it cannot
    be confused with a failure of the code under test.  The factory call is
    then deliberately unguarded.  On a host that *has* the backend, an API
    incompatibility, a plugin that raises on import, or a regression inside
    ``gwsignal_get_waveform_generator`` is a real failure of this guard;
    catching it would downgrade the very breakage this file exists to detect
    into a green skip that asserts nothing.  A gwsignal too old to know the
    approximant at all fails here for the same reason -- the package is
    installed, so the mismatch is worth reporting.
    """
    module = _BACKEND_MODULE[approx]
    if importlib.util.find_spec(module) is None:
        pytest.skip("%s needs the %s module, which is not installed here"
                    % (approx, module))
    return gws.models.gwsignal_get_waveform_generator(approx)


def _coeffs_per_m(approx, gen):
    """Free complex coefficient per azimuthal index, fitting the gwsignal
    polarizations with RIFT's own modes."""
    hp, hc = wfm.GenerateTDWaveform(_pdict(), gen)
    t_pol = _times(hp)

    hlm = rgws.hlmoft(_P(), Lmax=4, approx_string=approx)
    key0 = sorted(hlm)[0]
    t_rift = (float(hlm[key0].epoch)
              + np.arange(hlm[key0].data.length) * hlm[key0].deltaT)

    grid = np.arange(max(t_rift[0], t_pol[0]), min(t_rift[-1], t_pol[-1]),
                     DELTAT)
    target = (_regrid(t_pol, _arr(hp), grid)
              - 1j * _regrid(t_pol, _arr(hc), grid))

    ms = sorted({k[1] for k in hlm})
    cols = []
    for m in ms:
        b = np.zeros(len(t_rift), dtype=complex)
        for k in hlm:
            if k[1] == m:
                b += hlm[k].data.data * lal.SpinWeightedSphericalHarmonic(
                    IOTA, 0.0, -2, k[0], k[1])
        cols.append(_regrid(t_rift, b, grid))

    A = np.array(cols).T
    keep = np.abs(target) > 0.01 * np.abs(target).max()
    c, *_ = np.linalg.lstsq(A[keep], target[keep], rcond=None)
    return dict(zip(ms, c))


def _sign_invariant(c):
    """arg(c_4) - 2 arg(c_2), in radians, folded to [-pi, pi].

    0 => no global sign error (any phi_ref convention).  +-pi => psi is
    displaced by pi/2.
    """
    z = c[4] * np.conj(c[2]) ** 2
    return np.angle(z / np.abs(z))


@pytest.mark.parametrize("approx", ["TEOBResumSDALI", "SEOBNRv5EHM"])
def test_no_global_sign_error_vs_polarizations(approx):
    c = _coeffs_per_m(approx, _generator(approx))

    for m in (2, 4):
        assert m in c, "need m=%d to separate a sign from a phase" % m

    d = _sign_invariant(c)
    assert abs(d) < np.radians(45), (
        "%s: global sign error -- RIFT's templates are -1 x the gwsignal "
        "polarizations, i.e. psi is displaced by pi/2.\n"
        "  arg(c_4) - 2 arg(c_2) = %+.1f deg (expected ~0, pi means the bug)\n"
        "  per-m coefficients: %s"
        % (approx, np.degrees(d),
           "  ".join("m=%+d: %.3f/%+.1fd" % (m, abs(v), np.degrees(np.angle(v)))
                     for m, v in sorted(c.items()))))
