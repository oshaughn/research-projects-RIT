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


def _tables(m_max=2, a0=0.0, b1=0.0, a0_imag_at=None, b1_lower_only=False):
    """Coefficient tables whose RECONSTRUCTED fields satisfy the identity, or not.

    The identity is a statement about the fields
    ``A0(phi) = Re(MA(1))`` and ``B1(phi) = MB(3) + conj(MB(1))`` -- NOT about the
    coefficient slices.  So "clean" here means zeroing every slice that FEEDS those
    two fields (``C_A[:,1]``; ``C_B[:,3]`` AND ``C_B[:,1]``), and a plant means
    making one of them nonzero.  An earlier version of this fixture planted into
    slices while leaving the others random, which is why it could not distinguish
    the two.
    """
    rng = np.random.default_rng(7)
    # Shapes are those angle_coefficient_tables actually returns:
    # C_A (m_max+1, 3, S, npts), C_B (2*m_max+1, 5, S, npts).  An earlier
    # fixture used (2*m_max+1, 3) and (4*m_max+1, 5); the slice-based predicate
    # never indexed the leading axis, so the invalid shape was invisible and the
    # suite passed on it.
    nphi_A, nphi_B = m_max + 1, 2 * m_max + 1
    ksA, ksB = 3, 5
    C_A = (rng.normal(size=(nphi_A, ksA, 2, 4))
           + 1j * rng.normal(size=(nphi_A, ksA, 2, 4)))
    C_B = (rng.normal(size=(nphi_B, ksB, 2, 4))
           + 1j * rng.normal(size=(nphi_B, ksB, 2, 4)))
    ks0 = (ksB - 1) // 2
    C_A[:, 1] = 0.0                       # -> A0(phi) == 0
    C_A[:, 2] = 1.0                       # -> a well-scaled A1
    C_B[:, ks0] = 1.0                     # -> B0
    C_B[:, ks0 + 1] = 0.0                 # \
    C_B[:, ks0 - 1] = 0.0                 # /  both feed B1(phi); zero BOTH
    if a0:
        C_A[:, 1] = a0
    if a0_imag_at is not None:
        # the reviewer's case: a purely IMAGINARY coefficient.  Re(C_A[k,1]) is
        # zero, so a slice-based check sees nothing, but Re(MA(1)) is not.
        C_A[a0_imag_at, 1] = 1j * 1e-3
    if b1:
        C_B[:, ks0 + 1] = b1
    if b1_lower_only:
        # B1(phi) also carries conj(MB(ks0-1)); planting ONLY there must be caught.
        C_B[:, ks0 - 1] = 1e-3
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


def test_imaginary_A0_coefficient_is_detected():
    """A purely imaginary C_A[k,1] gives Re(C_A[k,1]) == 0 but Re(MA(1)) != 0.

    The predicate must measure the RECONSTRUCTED A0(phi).  A slice-based check
    reports this dataset as supported; that was a real defect (external review).
    """
    ok, info = AM.gh_laplace_supported(*_tables(m_max=2, a0_imag_at=1), 2)
    assert ok is False, "imaginary A0 coefficient slipped past the identity check"
    assert "does NOT hold" in info["gh_laplace_reason"]


def test_B1_planted_only_in_the_conjugate_slice_is_detected():
    """B1(phi) = MB(3) + conj(MB(1)); planting only in the LOWER slice must fail.

    A check reading C_B[:,3] alone reports this dataset as supported; that was
    the second half of the same defect.
    """
    ok, info = AM.gh_laplace_supported(*_tables(m_max=2, b1_lower_only=True), 2)
    assert ok is False, "B1 planted in the conjugate slice slipped past the check"
    assert "does NOT hold" in info["gh_laplace_reason"]


def test_explicit_laplace_under_gh_is_gated_too_not_only_auto():
    """The identity guard must cover an EXPLICIT laplace, not only `auto`.

    External review: with the check only inside choose_angle_marg_scheme, `auto`
    was protected while `--angle-marg-scheme laplace` -- and any direct wrapper
    caller -- walked past it, so an m_max == 2 dataset whose identity fails was
    evaluated with a placement derived from that identity.

    It cannot live in the kernel: that runs under jit/grad where the coefficient
    tables are TRACERS and the measurement raises
    TracerArrayConversionError (this is how the first fix was caught).  So the
    enforcement point is the wrapper, on concrete tables, and this pins it there
    for BOTH routes.
    """
    import inspect
    from RIFT.likelihood.jax_ile import wrapper as WR
    src = inspect.getsource(WR.JAXDistPhiPsiMargLikelihood.__init__)
    assert "gh_laplace_supported" in src, (
        "the wrapper does not measure the identity; the GH laplace placement "
        "would be reachable without its premise being checked")
    assert 'angle_marg in ("auto", "laplace")' in src, (
        "the identity gate is not applied to an EXPLICITLY requested laplace")
    assert 'angle_marg == "laplace" and gh_ok is False' in src, (
        "an explicit laplace with a failing identity does not raise")


def test_kernel_enforces_the_response_model_itself_not_only_the_wrapper():
    """The PUBLIC laplace kernel must gate on the response model too.

    External review: this function is in ``anglemarg.__all__`` and is called
    directly by the wrapper and by several test modules, so a wrapper-only gate
    leaves a live bypass -- a direct call with ``data.feature == "rotation"``
    and ``m_max <= 2`` executes the unsupported placement while the wrapper
    correctly refuses it.

    An earlier version of THIS TEST asserted the kernel contained no such check,
    which entrenched the bypass rather than catching it.  What must stay absent
    is only the NUMERICAL A0/B1 measurement, which needs concrete tables and
    raises under jit/grad; ``feature`` is a static Python attribute and is free
    to check.
    """
    import inspect
    src = inspect.getsource(AM.fused_log_likelihood_distphipsimarg_laplace)
    assert "_GH_PSI_STATIC_FEATURES" in src, (
        "the public laplace kernel does not enforce the response model; a "
        "direct call with a banded response would use the placement anyway")
    assert "gh_laplace_supported(" not in src, (
        "the kernel measures the identity numerically; that needs concrete "
        "tables and raises under jax.grad")


def test_kernel_refuses_a_banded_response_on_a_direct_call():
    """Behavioural counterpart: a direct call must RAISE, not merely be
    discouraged.  Uses a stub carrying only what the precondition reads, so it
    exercises the gate rather than a full likelihood build."""
    class _Data:
        lms = [(2, -2), (2, 2)]          # m_max = 2: the mode gate would pass
        feature = "rotation"             # but the response model must not
        npts = 4

    import RIFT.likelihood.jax_ile.core as _core
    saved = _core._DISTMARG_GH_N
    _core._DISTMARG_GH_N = 64            # GH on
    try:
        with pytest.raises(ValueError, match="static detector response"):
            AM.fused_log_likelihood_distphipsimarg_laplace(
                _Data(), np.zeros(1), np.zeros(1), np.zeros(1),
                np.ones(2), np.zeros(2))
    finally:
        _core._DISTMARG_GH_N = saved


def test_global_maxima_cannot_hide_a_locally_invalid_bin():
    """The identity ratio must be POINTWISE, not max|A0| / max|A1|.

    External review: bins (A0,A1) = (1e-3, 1) and (0, 1e6) give a global ratio
    of 1e-9 -- passing -- while the first bin violates by 1e-3.  The GH
    placement computes its centre and width at each bin independently, so one
    invalid bin is enough to invalidate the placement there.
    """
    C_A, C_B = _tables(m_max=2)
    # one bin violates at 1e-3; another carries a denominator 1e6 larger
    C_A[:, 1, 0, 0] = 1e-3
    C_A[:, 2, 0, 0] = 1.0
    C_A[:, 2, 1, 1] = 1e6
    ok, info = AM.gh_laplace_supported(C_A, C_B, 2)
    assert ok is False, (
        "a locally invalid bin was hidden behind a large denominator elsewhere; "
        "the ratio is being taken between unrelated global maxima")
    assert info["identity_A0_over_A1"] > AM.GH_PSI_IDENTITY_TOL


def test_response_model_is_an_angle_independent_precondition():
    """A numerical probe speaks only for the angles it was evaluated at.

    The placement runs at arbitrary sampled angles, so the gate's real guarantee
    is the RESPONSE MODEL: the static path's F+ + i Fx = (F+(0) + i Fx(0))
    e^{-2i psi} is a single u-harmonic at every (ra, dec, incl), which is what
    forces A0 == 0 and B1 == 0.  The banded features do not use that response,
    and an unknown feature must fail closed.
    """
    C_A, C_B = _tables(m_max=2)
    assert AM.gh_laplace_supported(C_A, C_B, 2, feature=None)[0] is True
    for bad in ("freqresponse", "rotation", "something_added_later"):
        ok, info = AM.gh_laplace_supported(C_A, C_B, 2, feature=bad)
        assert ok is False, "feature %r was admitted" % bad
        assert "factorization" in info["gh_laplace_reason"]


def test_wrapper_passes_the_response_feature_through():
    import inspect
    from RIFT.likelihood.jax_ile import wrapper as WR
    src = inspect.getsource(WR.JAXDistPhiPsiMargLikelihood.__init__)
    assert 'feature=getattr(data, "feature", None)' in src, (
        "the wrapper does not forward the response model, so the "
        "angle-independent half of the gate never runs")
