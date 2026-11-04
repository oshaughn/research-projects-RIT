"""Analytic and independent time-domain checks of the direct A/E/T primitive."""
import numpy as np
import pytest

from RIFT.LISA.direct_likelihood import aet_inner_product, aet_log_likelihood


@pytest.mark.parametrize("n", [127, 128])
def test_independent_parseval_normalization(n):
    # Zero-mean sinusoids contain neither DC nor Nyquist; flat PSD=2*dt*sigma^2.
    dt, sigma = 0.25, 2.3
    t = np.arange(n)
    data = np.array([np.cos(2*np.pi*k*t/n) for k in (3, 5, 7)])
    model = 0.6 * data
    df = 1 / (n * dt)
    fd = dt * np.fft.rfft(data)
    fm = dt * np.fft.rfft(model)
    psd = np.full(fd.shape, 2*dt*sigma**2)
    mask = np.ones(fd.shape[1], dtype=bool)
    mask[0] = False
    if n % 2 == 0:
        mask[-1] = False
    expected = -0.5*np.sum((data-model)**2)/sigma**2
    assert aet_log_likelihood(fd, fm, psd, df, mask=mask) == pytest.approx(expected)


def test_analytic_complex_product_and_injection_identity():
    a = np.array([[1+2j, 3j], [2j, 3+1j], [1-1j, 2]])
    b = np.array([[2j, 1], [1, 2j], [3, 1j]])
    psd = np.array([[2., 3.], [4., 5.], [6., 7.]])
    mask = np.array([True, True])
    # Independent scalar arithmetic exercises complex conjugation and channel sum.
    expected = 4*0.2*sum((a[c, k].conjugate()*b[c, k]).real/psd[c, k]
                            for c in range(3) for k in range(2))
    assert aet_inner_product(a, b, psd, 0.2, mask=mask) == pytest.approx(expected)
    assert aet_log_likelihood(a, a, psd, 0.2, mask=mask) == 0
    assert aet_log_likelihood(a, 0*a, psd, 0.2, mask=mask) == pytest.approx(
        -0.5*aet_inner_product(a, a, psd, 0.2, mask=mask))
    offset = aet_log_likelihood(a, b, psd, 0.2, mask=mask)
    assert -offset == pytest.approx(0.5*aet_inner_product(a-b, a-b, psd, 0.2, mask=mask))
    # Simultaneous channel permutation leaves the sum unchanged.
    assert aet_log_likelihood(a[::-1], b[::-1], psd[::-1], 0.2, mask=mask) == offset


def test_explicit_mask_ignores_singular_excluded_bins():
    a = np.ones((3, 4), dtype=complex)
    psd = np.ones((3, 4))
    a[:, 0] = np.nan
    psd[:, 0] = 0
    mask = np.array([False, True, True, False])
    assert aet_inner_product(a, a, psd, 0.25, mask=mask) == 6


@pytest.mark.parametrize('df', [0, -1, np.nan, np.inf, 1j, [1]])
def test_bad_spacing(df):
    a = np.ones((3, 3))
    with pytest.raises(ValueError):
        aet_inner_product(a, a, a, df, mask=np.ones(3, dtype=bool))


@pytest.mark.parametrize('mask', [[1, 1, 0], [True, True], [False]*3])
def test_bad_mask(mask):
    a = np.ones((3, 3))
    with pytest.raises(ValueError):
        aet_inner_product(a, a, a, 1, mask=mask)


@pytest.mark.parametrize('bad', [0, -1, np.nan, np.inf, 1j])
def test_bad_selected_psd(bad):
    a = np.ones((3, 3))
    psd = np.ones((3, 3), dtype=complex if bad == 1j else float)
    psd[0, 1] = bad
    with pytest.raises(ValueError):
        aet_inner_product(a, a, psd, 1, mask=np.ones(3, dtype=bool))


def test_shapes_and_nonfinite_amplitudes():
    a = np.ones((3, 3))
    mask = np.ones(3, dtype=bool)
    for b in [np.ones((1, 3)), np.ones((3, 3, 1))]:
        with pytest.raises(ValueError):
            aet_log_likelihood(a, b, a, 1, mask=mask)
    with pytest.raises(ValueError):
        aet_inner_product(a, a, np.ones((3, 3, 3)), 1, mask=mask)
    a[0, 1] = np.nan
    with pytest.raises(ValueError):
        aet_inner_product(a, a, np.ones((3, 3)), 1, mask=mask)
