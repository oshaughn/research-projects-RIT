#! /usr/bin/env python
#
# test_uv_symmetry.py
#
#   Waveform-level symmetry checks on the U and V mode-cross-term matrices
#   built by RIFT.likelihood.factored_likelihood.
#
# WHAT IS BEING TESTED
#   The factored likelihood pre-computes two dictionaries of PSD-weighted mode
#   inner products (see factored_likelihood.ComputeModeCrossTermIP):
#
#       U[(A,B)] = < h_A | h_B >           (crossTerms)
#       V[(A,B)] = < h_A^* | h_B >         (crossTermsV)
#
#   where A=(l,m), B=(l',m'), h_A^* is the time-domain complex conjugate mode,
#   and < a | b > = 2 \int a^*(f) b(f) / S_n(f) df  (RIFT.lalsimutils.ComplexIP).
#
#   Three of the properties tested below follow purely from the DEFINITION of
#   the inner product and hold for every waveform (a genuine numerical check,
#   because we recompute every matrix element independently rather than relying
#   on the code's own symmetrization shortcut):
#
#     (1) U is Hermitian:              U[(A,B)]  = conj(U[(B,A)])
#     (2) U has real, positive diag:   U[(A,A)] real and > 0
#     (3) V is complex-symmetric:      V[(A,B)]  = V[(B,A)]
#
#   The fourth property is PHYSICS, and only holds for non-precessing
#   (aligned-spin) binaries, which obey the reflection / parity relation
#
#       h_{l,-m}(t) = (-1)^l conj(h_{l,m}(t)).
#
#   Because V is built from the conjugated modes, this implies the cross-matrix
#   identity
#
#     (4) V[((l,m),B)] = (-1)^l U[((l,-m),B)]     (aligned-spin binaries only).
#
#   We exercise all four over semi-random parameters, looped over an active list
#   of waveform approximants.
#
# EXAMPLES
#   pytest -v test_uv_symmetry.py
#   python  test_uv_symmetry.py --approximant IMRPhenomXHM --Lmax 3 --seed 42
#   python  test_uv_symmetry.py --list                 # show the active waveform list
#
# NOTE
#   This module also carries one deliberately-failing check
#   (test_full_nonlinear_reflection_symmetry_left_as_exercise); see its
#   docstring. Deselect it with `-k 'not left_as_exercise'` or run the script
#   with `--skip-ludicrous`.

from __future__ import print_function

import argparse
import itertools
import sys

import numpy as np

import lal
import lalsimulation as lalsim

import RIFT
import RIFT.lalsimutils as lalsimutils
import RIFT.likelihood.factored_likelihood as factored_likelihood

try:
    import pytest
    _HAVE_PYTEST = True
except ImportError:  # allow running as a bare script without pytest installed
    _HAVE_PYTEST = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The "active waveform list": aligned-spin, multi-mode-capable approximants,
# which obey the reflection relation (4). Anything that fails to generate in the
# local lalsuite build is skipped (with a reason) rather than treated as a
# symmetry failure.
ACTIVE_WAVEFORMS = [
    "IMRPhenomXHM",
    "IMRPhenomHM",
    "SEOBNRv4HM",   # skipped on builds whose SimIMRSpinAlignedEOBModes signature differs
    "SEOBNRv5HM",   # skipped on builds that reject the current call
]

# Precessing models. Their inertial-frame modes do NOT obey the simple
# reflection relation (4) even at zero in-plane spin (frame/phase conventions),
# so they are used only for the definitional checks (1)-(3).
PRECESSING_WAVEFORMS = [
    "IMRPhenomXPHM",
]

# Analytic PSD used to weight the inner products. Any physical (finite, > 0 in
# band) PSD works: the symmetry relations are independent of its choice.
PSD_FUNC = lalsim.SimNoisePSDaLIGOZeroDetHighPower

# Fixed grid, chosen so every mode series shares (deltaT, deltaF) and stays well
# away from ISCO/wraparound for a heavy-ish system.
FMIN = 20.0
FMAX = 1700.0
DELTA_T = 1.0 / 4096.0
DELTA_F = 1.0 / 8.0
MTOT_MSUN = 60.0
LMAX_DEFAULT = 3

# Tolerances, expressed relative to the geometric mean of the two diagonal
# norms so they are dimensionless.
TOL_DEFINITIONAL = 1e-6   # (1)-(3): exact up to floating-point round-off
TOL_REFLECTION = 3e-2     # (4): physics + finite-length / tapering noise


# ---------------------------------------------------------------------------
# Core: build the U and V matrices at the waveform level
# ---------------------------------------------------------------------------

def _make_params(approximant, seed, aligned=True, mtot=MTOT_MSUN):
    """Semi-random ChooseWaveformParams on a fixed, well-behaved grid.

    `seed` makes each trial reproducible; `aligned` zeroes the in-plane spins so
    the reflection relation (4) applies.
    """
    rng = np.random.RandomState(seed)

    P = lalsimutils.ChooseWaveformParams()
    P.ampO = -1        # keep all higher modes the model offers
    P.phaseO = 7
    P.taper = lalsimutils.lsu_TAPER_START
    P.deltaT = DELTA_T
    P.deltaF = DELTA_F
    P.fmin = FMIN
    P.fref = 20.0

    # Semi-random intrinsic parameters (reproducible via seed).
    q = rng.uniform(0.5, 1.0)                       # m2/m1
    m1 = mtot / (1.0 + q)
    m2 = mtot - m1
    P.m1 = m1 * lal.MSUN_SI
    P.m2 = m2 * lal.MSUN_SI

    s1z = rng.uniform(-0.6, 0.6)
    s2z = rng.uniform(-0.6, 0.6)
    P.s1x = P.s1y = P.s2x = P.s2y = 0.0
    P.s1z = s1z
    P.s2z = s2z
    if not aligned:
        # Only used for the definitional checks (1)-(3), which do not require
        # reflection symmetry.
        P.s1x = rng.uniform(-0.4, 0.4)
        P.s1y = rng.uniform(-0.4, 0.4)
        P.s2x = rng.uniform(-0.4, 0.4)

    # Extrinsic angles are irrelevant to U/V (they act on the Ylm sum, not the
    # mode inner products), but set them to something non-trivial anyway.
    P.incl = rng.uniform(0.0, np.pi)
    P.phiref = rng.uniform(0.0, 2 * np.pi)
    P.psi = rng.uniform(0.0, np.pi)
    P.dist = factored_likelihood.distMpcRef * 1e6 * lal.PC_SI

    P.approx = lalsim.GetApproximantFromString(approximant)
    return P


def build_uv(P, Lmax, psd_func=PSD_FUNC, fmin=FMIN, fmax=FMAX, verbose=False):
    """Generate the modes for P and return (hlms, U, V).

    U and V are computed with same_waveform_Q=False so that *every* matrix
    element is an independent inner product -- the code's internal symmetrized
    fast path is intentionally bypassed so the symmetry tests below are real.
    """
    hlms, hlms_conj = factored_likelihood.internal_hlm_generator(
        P, Lmax, verbose=False, quiet=True)

    fNyq = 1.0 / (2.0 * P.deltaT)
    U = factored_likelihood.ComputeModeCrossTermIP(
        hlms, hlms, psd_func, fmin, fmax, fNyq, P.deltaF,
        analyticPSD_Q=True, verbose=False, prefix="U", same_waveform_Q=False)
    V = factored_likelihood.ComputeModeCrossTermIP(
        hlms_conj, hlms, psd_func, fmin, fmax, fNyq, P.deltaF,
        analyticPSD_Q=True, verbose=False, prefix="V", same_waveform_Q=False)
    return hlms, U, V


# ---------------------------------------------------------------------------
# Symmetry checks. Each returns a list of human-readable violation strings.
# ---------------------------------------------------------------------------

def _scale(U, A, B):
    """Geometric mean of the diagonal norms, used to non-dimensionalize."""
    dA = abs(U[(A, A)])
    dB = abs(U[(B, B)])
    s = np.sqrt(dA * dB)
    return s if s > 0 else 1.0


def check_U_hermitian(U, tol=TOL_DEFINITIONAL):
    """(1) U[(A,B)] = conj(U[(B,A)])."""
    viol = []
    modes = sorted({A for (A, _) in U.keys()})
    for A, B in itertools.combinations(modes, 2):
        lhs = U[(A, B)]
        rhs = np.conj(U[(B, A)])
        rel = abs(lhs - rhs) / _scale(U, A, B)
        if rel > tol:
            viol.append("U not Hermitian for {},{}: |dU|/scale={:.3e}".format(A, B, rel))
    return viol


def check_U_diagonal_real_positive(U, tol=TOL_DEFINITIONAL):
    """(2) U[(A,A)] is real and positive."""
    viol = []
    modes = sorted({A for (A, _) in U.keys()})
    for A in modes:
        d = U[(A, A)]
        if abs(d) == 0:
            continue
        imag_frac = abs(np.imag(d)) / abs(d)
        if imag_frac > tol:
            viol.append("U[{0},{0}] not real: Im/|.|={1:.3e}".format(A, imag_frac))
        if np.real(d) <= 0:
            viol.append("U[{0},{0}] not positive: Re={1:.3e}".format(A, np.real(d)))
    return viol


def check_V_symmetric(V, U, tol=TOL_DEFINITIONAL):
    """(3) V[(A,B)] = V[(B,A)]."""
    viol = []
    modes = sorted({A for (A, _) in V.keys()})
    for A, B in itertools.combinations(modes, 2):
        rel = abs(V[(A, B)] - V[(B, A)]) / _scale(U, A, B)
        if rel > tol:
            viol.append("V not symmetric for {},{}: |dV|/scale={:.3e}".format(A, B, rel))
    return viol


def check_reflection_aligned(U, V, tol=TOL_REFLECTION):
    """(4) V[((l,m),B)] = (-1)^l U[((l,-m),B)]  (aligned-spin binaries).

    Only pairs for which the reflected mode (l,-m) is present are tested.
    """
    viol = []
    n_tested = 0
    modes = sorted({A for (A, _) in U.keys()})
    mode_set = set(modes)
    for A in modes:
        (l, m) = A
        A_refl = (l, -m)
        if A_refl not in mode_set:
            continue
        for B in modes:
            n_tested += 1
            lhs = V[(A, B)]
            rhs = ((-1) ** l) * U[(A_refl, B)]
            rel = abs(lhs - rhs) / _scale(U, A, B)
            if rel > tol:
                viol.append(
                    "reflection broken for A={},B={}: "
                    "|V - (-1)^l U_refl|/scale={:.3e}".format(A, B, rel))
    if n_tested == 0:
        viol.append("reflection check exercised no mode pairs (no +/-m partners found)")
    return viol


def run_all_checks(P, Lmax, aligned, verbose=False):
    """Build U,V for P and return the concatenated violation list."""
    hlms, U, V = build_uv(P, Lmax, verbose=verbose)
    if verbose:
        print("  modes:", sorted(hlms.keys()))
    viol = []
    viol += check_U_hermitian(U)
    viol += check_U_diagonal_real_positive(U)
    viol += check_V_symmetric(V, U)
    if aligned:
        viol += check_reflection_aligned(U, V)
    return viol


# ---------------------------------------------------------------------------
# Waveform generation guard: skip (don't fail) if a model is unavailable
# ---------------------------------------------------------------------------

def _try_build(approximant, seed, Lmax, aligned=True):
    """Return (P, violations) or raise a descriptive RuntimeError to skip."""
    try:
        P = _make_params(approximant, seed, aligned=aligned)
    except Exception as e:  # unknown approximant string, etc.
        raise RuntimeError("cannot set up {}: {}".format(approximant, e))
    try:
        viol = run_all_checks(P, Lmax, aligned=aligned)
    except Exception as e:
        raise RuntimeError("cannot generate/analyze {}: {}".format(approximant, e))
    return P, viol


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------

if _HAVE_PYTEST:

    @pytest.mark.parametrize("approximant", ACTIVE_WAVEFORMS)
    def test_uv_symmetry_aligned(approximant):
        """Definitional + reflection symmetry for aligned-spin binaries."""
        try:
            _P, viol = _try_build(approximant, seed=1234, Lmax=LMAX_DEFAULT, aligned=True)
        except RuntimeError as e:
            pytest.skip(str(e))
        assert not viol, "symmetry violations for {}:\n  {}".format(
            approximant, "\n  ".join(viol))

    @pytest.mark.parametrize("approximant", PRECESSING_WAVEFORMS)
    def test_uv_definitional_precessing(approximant):
        """Definitional checks (1)-(3) must also hold for precessing systems."""
        try:
            P = _make_params(approximant, seed=99, aligned=False)
            _hlms, U, V = build_uv(P, LMAX_DEFAULT)
        except Exception as e:
            pytest.skip("cannot generate {}: {}".format(approximant, e))
        viol = (check_U_hermitian(U)
                + check_U_diagonal_real_positive(U)
                + check_V_symmetric(V, U))
        assert not viol, "definitional violations for {}:\n  {}".format(
            approximant, "\n  ".join(viol))

    def test_full_nonlinear_reflection_symmetry_left_as_exercise():
        """LUDICROUS / INTENTIONAL FAILURE.

        A complete waveform-symmetry test would also verify the higher-order,
        fully non-linear reflection identities relating U and V across *all*
        (l, m) sectors simultaneously (the closed algebra of parity, time-
        reversal and mode-mixing operators), not just the pairwise relation (4).

        That verification is not implemented here. Rather than silently pass and
        give false confidence, this check fails loudly so nobody forgets.
        """
        assert False, "this failure is left as a test"


# ---------------------------------------------------------------------------
# Script runner (mirrors the style of test/waveform/check_waveform_random.py)
# ---------------------------------------------------------------------------

def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approximant", type=str, default=None,
                        help="single approximant to test (default: loop over the active list)")
    parser.add_argument("--Lmax", type=int, default=LMAX_DEFAULT)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-trials", type=int, default=1,
                        help="number of semi-random trials per approximant")
    parser.add_argument("--precessing", action="store_true",
                        help="use precessing spins (disables reflection check (4))")
    parser.add_argument("--list", action="store_true",
                        help="print the active waveform list and exit")
    parser.add_argument("--skip-ludicrous", action="store_true",
                        help="do not run the intentionally-failing check")
    parser.add_argument("--verbose", action="store_true")
    opts = parser.parse_args(argv)

    if opts.list:
        print("Active waveform list:")
        for a in ACTIVE_WAVEFORMS:
            print("  ", a)
        return 0

    approximants = [opts.approximant] if opts.approximant else ACTIVE_WAVEFORMS
    aligned = not opts.precessing

    n_fail = 0
    n_skip = 0
    for approximant in approximants:
        for trial in range(opts.n_trials):
            seed = opts.seed + trial
            label = "{} (seed={}, {})".format(
                approximant, seed, "aligned" if aligned else "precessing")
            try:
                _P, viol = _try_build(approximant, seed, opts.Lmax, aligned=aligned)
            except RuntimeError as e:
                print("SKIP {}: {}".format(label, e))
                n_skip += 1
                continue
            if viol:
                n_fail += 1
                print("FAIL {}".format(label))
                for v in viol:
                    print("     - {}".format(v))
            else:
                print("PASS {}".format(label))

    if not opts.skip_ludicrous:
        print("\n--- intentionally-failing check ---")
        try:
            assert False, "this failure is left as a test"
        except AssertionError as e:
            n_fail += 1
            print("FAIL full_nonlinear_reflection_symmetry_left_as_exercise: {}".format(e))

    print("\nSummary: {} failing, {} skipped".format(n_fail, n_skip))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(_main())
