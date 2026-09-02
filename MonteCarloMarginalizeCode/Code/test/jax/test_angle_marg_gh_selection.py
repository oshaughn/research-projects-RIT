"""`auto` may route to 'laplace' under JAX_ILE_DISTMARG_GH -- but only where the
psi-marginal node placement is actually valid, MEASURED on the data.

Before the GH-for-laplace path existed, choose_angle_marg_scheme returned 'exact'
unconditionally under GH.  It no longer does.  The risk this suite pins is the
opposite one: routing 'auto' to laplace where the placement is NOT valid sends the
run into the raise inside the laplace kernel.

The predicate deliberately does NOT key on mode content alone.  m_max is the
largest |m| in the mode list, so a PRECESSING l=2 system has m_max = 2 and would
pass a mode gate while breaking the aligned-spin symmetry the A0==0/B1==0 identity
has only ever been tested under.  Every measurement of that identity so far is
non-precessing, so the code measures instead of assuming.
"""
import numpy as np
import pytest

import RIFT.likelihood.jax_ile.anglemarg as AM


def _tables(m_max=2, a0=0.0, b1=0.0):
    """Coefficient tables with the identity intact, or deliberately broken."""
    rng = np.random.default_rng(7)
    nphi_A, nphi_B = 2 * m_max + 1, 4 * m_max + 1
    ksA, ksB = 3, 5
    C_A = rng.normal(size=(nphi_A, ksA, 2, 4)) * (1 + 0j)
    C_B = rng.normal(size=(nphi_B, ksB, 2, 4)) * (1 + 0j)
    C_A[:, 1] = a0                      # ks 0  -> A0
    C_A[:, 2] = 1.0                     # ks +1 -> A1
    ks0 = (ksB - 1) // 2
    C_B[:, ks0] = 1.0                   # B0
    C_B[:, ks0 + 1] = b1                # B1
    return C_A, C_B


def test_identity_holds_for_clean_tables():
    ok, info = AM.gh_laplace_supported(*_tables(m_max=2), 2)
    assert ok is True
    assert info["identity_A0_over_A1"] <= AM.GH_PSI_IDENTITY_TOL
    assert info["identity_B1_over_B0"] <= AM.GH_PSI_IDENTITY_TOL
    assert "identity holds" in info["gh_laplace_reason"]


@pytest.mark.parametrize("a0,b1", [(1e-3, 0.0), (0.0, 1e-3), (1e-3, 1e-3)])
def test_identity_check_can_fail(a0, b1):
    """POSITIVE CONTROL: the check must be able to return False at all.

    A predicate that cannot fail is not a check.  Planted harmonics are the only
    way to exercise this -- no mode set tried through m_max = 4 breaks the
    identity naturally, which is precisely why it must be measured rather than
    inferred from the mode list.
    """
    ok, info = AM.gh_laplace_supported(*_tables(m_max=2, a0=a0, b1=b1), 2)
    assert ok is False
    assert "does NOT hold" in info["gh_laplace_reason"]


def test_mode_content_above_validated_is_refused():
    ok, info = AM.gh_laplace_supported(*_tables(m_max=2), 4)
    assert ok is False
    assert "m_max" in info["gh_laplace_reason"]


def test_auto_reaches_laplace_under_gh_when_supported():
    amp = AM.ANGLE_MARG_CROSSOVER_AMPLITUDE * 10.0
    scheme, info = AM.choose_angle_marg_scheme(amp, gh_enabled=True,
                                               gh_laplace_ok=True)
    assert scheme == "laplace", "auto must be able to reach laplace under GH"
    # control: same amplitude, same GH, predicate false -> exact
    scheme_no, info_no = AM.choose_angle_marg_scheme(amp, gh_enabled=True,
                                                     gh_laplace_ok=False)
    assert scheme_no == "exact"
    assert "not " in info_no["reason"]


def test_auto_is_conservative_when_predicate_absent():
    """gh_laplace_ok=None means "caller did not measure it" -> take the safe
    branch.  Guessing here would route auto into the laplace kernel's raise."""
    amp = AM.ANGLE_MARG_CROSSOVER_AMPLITUDE * 10.0
    scheme, _ = AM.choose_angle_marg_scheme(amp, gh_enabled=True)
    assert scheme == "exact"


def test_selector_unchanged_with_gh_off():
    amp_hi = AM.ANGLE_MARG_CROSSOVER_AMPLITUDE * 10.0
    amp_lo = AM.ANGLE_MARG_CROSSOVER_AMPLITUDE / 10.0
    assert AM.choose_angle_marg_scheme(amp_hi, gh_enabled=False)[0] == "laplace"
    assert AM.choose_angle_marg_scheme(amp_lo, gh_enabled=False)[0] == "exact"
