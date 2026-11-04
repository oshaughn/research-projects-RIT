import unittest

import lal
import numpy as np

from RIFT.likelihood.noise_evidence import compute_network_log_noise_evidence


class _Data:
    def __init__(self, value, delta_f):
        self.value = value
        self.deltaF = delta_f


class _InnerProduct:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)

    def ip(self, first, second):
        return complex(first.value * second.value * self.kwargs["psd"])


class TestNoiseEvidence(unittest.TestCase):
    def setUp(self):
        _InnerProduct.calls = []

    def test_network_sum_and_bilby_sign(self):
        data = {"L1": _Data(2.0, 0.25), "H1": _Data(3.0, 0.25)}
        psds = {"H1": 2.0, "L1": 4.0}

        total, per_detector = compute_network_log_noise_evidence(
            data, psds, fmin=20.0, fmax=1024.0, fnyq=2048.0,
            inv_spec_trunc_Q=True, T_spec=8.0,
            inner_product_factory=_InnerProduct)

        self.assertEqual(per_detector["H1"]["d_inner_d"], 18.0)
        self.assertEqual(per_detector["L1"]["d_inner_d"], 16.0)
        self.assertEqual(total, -17.0)
        self.assertEqual([call["psd"] for call in _InnerProduct.calls], [2.0, 4.0])
        for call in _InnerProduct.calls:
            self.assertEqual(call["fLow"], 20.0)
            self.assertEqual(call["fMax"], 1024.0)
            self.assertEqual(call["fNyq"], 2048.0)
            self.assertEqual(call["deltaF"], 0.25)
            self.assertTrue(call["inv_spec_trunc_Q"])
            self.assertEqual(call["T_spec"], 8.0)

    def test_detector_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Detector mismatch"):
            compute_network_log_noise_evidence(
                {"H1": _Data(1.0, 0.25)}, {"L1": 1.0},
                fmin=20.0, fmax=1024.0, fnyq=2048.0,
                inner_product_factory=_InnerProduct)


def _two_sided_series(values, delta_f):
    """Pack a COMPLEX16FrequencySeries the way ILE holds conditioned data."""
    series = lal.CreateCOMPLEX16FrequencySeries(
        "d", lal.LIGOTimeGPS(0), 0., delta_f, lal.DimensionlessUnit, len(values))
    series.data.data[:] = values
    return series


class TestAgainstRealInnerProduct(unittest.TestCase):
    """The production path, against the Bilby formula written out by hand.

    The mocked tests above pin the bookkeeping (network sum, sign, which settings
    reach the inner product).  They cannot see the thing the feature actually
    claims -- that lalsimutils.ComplexIP over the two-sided array reproduces
    Bilby's ``4/T sum_{f>0} |d|^2 / S`` over the same band -- because a stub
    returns whatever it likes.  A factor of two here is still a plausible number.
    """

    N = 64
    DELTA_F = 0.5

    def _fixture(self):
        n_one_sided = self.N // 2 + 1
        psd = lal.CreateREAL8FrequencySeries(
            "psd", lal.LIGOTimeGPS(0), 0., self.DELTA_F, lal.SecondUnit,
            n_one_sided)
        psd.data.data[:] = 2.0e-2

        rng = np.random.default_rng(7)
        vals = rng.normal(size=self.N) + 1j*rng.normal(size=self.N)
        # Hermitian packing [-N/2 df, ..., -df, 0, df, ..., (N/2-1) df], so the
        # two-sided sum is real and equals twice the positive-frequency sum.
        positive = vals[self.N//2:].copy()
        vals[1:self.N//2] = np.conj(positive[1:self.N//2][::-1])
        vals[0] = 0.
        return _two_sided_series(vals, self.DELTA_F), psd, positive

    def test_matches_bilby_noise_weighted_inner_product(self):
        data, psd, positive = self._fixture()
        delta_t = 1./(self.N*self.DELTA_F)
        fnyq = 0.5/delta_t
        fmin, fmax = 4.0, 12.0

        total, per_detector = compute_network_log_noise_evidence(
            {"H1": data}, {"H1": psd}, fmin=fmin, fmax=fmax, fnyq=fnyq)

        # bilby.gw.utils.noise_weighted_inner_product: 4/duration * sum(|d|^2/S)
        # over the analysis band, with duration = 1/deltaF.  ComplexIP's band is
        # [round(fmin/df), round(fmax/df)), so fmax's own bin is excluded.
        freqs = self.DELTA_F*np.arange(self.N//2)
        band = (freqs >= fmin) & (freqs < fmax)
        reference = 4.*self.DELTA_F*np.sum(
            np.abs(positive[band])**2 / psd.data.data[:self.N//2][band])

        self.assertGreater(reference, 0.)
        self.assertAlmostEqual(
            per_detector["H1"]["d_inner_d"]/reference, 1.0, places=12)
        self.assertAlmostEqual(total/(-0.5*reference), 1.0, places=12)

    def test_empty_band_is_rejected_rather_than_reported_as_zero(self):
        """fmin above the PSD's support gives (d|d)=0, i.e. a log evidence of 0.

        Nothing downstream can tell that from a real answer, so it has to fail
        here.  The test asserts on the BAND, not on the value: zero-strain data
        also gives 0 and is a legitimate input.
        """
        data, psd, _positive = self._fixture()
        psd.data.data[:] = 0.        # e.g. a --fmin-ifo cut above the whole band
        fnyq = 0.5*self.N*self.DELTA_F

        with self.assertRaisesRegex(ValueError, "No nonzero PSD weight"):
            compute_network_log_noise_evidence(
                {"H1": data}, {"H1": psd}, fmin=4.0, fmax=12.0, fnyq=fnyq)

    def test_zero_strain_is_not_an_error(self):
        data, psd, _positive = self._fixture()
        data.data.data[:] = 0.
        fnyq = 0.5*self.N*self.DELTA_F

        total, per_detector = compute_network_log_noise_evidence(
            {"H1": data}, {"H1": psd}, fmin=4.0, fmax=12.0, fnyq=fnyq)
        self.assertEqual(per_detector["H1"]["d_inner_d"], 0.0)
        self.assertEqual(total, 0.0)

    def test_band_is_respected(self):
        """A narrower band must give a strictly smaller (d|d), not the same one."""
        data, psd, _positive = self._fixture()
        fnyq = 0.5*self.N*self.DELTA_F

        wide, _ = compute_network_log_noise_evidence(
            {"H1": data}, {"H1": psd}, fmin=4.0, fmax=12.0, fnyq=fnyq)
        narrow, _ = compute_network_log_noise_evidence(
            {"H1": data}, {"H1": psd}, fmin=6.0, fmax=10.0, fnyq=fnyq)

        self.assertLess(wide, narrow)
        self.assertLess(wide, 0.)


if __name__ == "__main__":
    unittest.main()
