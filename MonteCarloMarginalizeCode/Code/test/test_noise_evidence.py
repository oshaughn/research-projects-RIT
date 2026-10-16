import unittest

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


if __name__ == "__main__":
    unittest.main()
