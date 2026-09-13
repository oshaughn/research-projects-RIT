"""Fail-closed and Fourier-contract tests for the optional Ripple adapter."""

import unittest
from unittest import mock

import numpy as np

from RIFT.likelihood import gpu_waveform as gw


class TestFourierConvention(unittest.TestCase):
    def test_continuous_roundtrip(self):
        rng = np.random.default_rng(20260912)
        ht = rng.normal(size=32) + 1j * rng.normal(size=32)
        hf = gw._continuous_forward(ht, 1.0 / 16.0, np)
        recovered = gw._continuous_inverse(hf, 1.0 / 16.0, np)
        np.testing.assert_allclose(recovered, ht, rtol=2e-14, atol=2e-14)

    def test_forward_matches_lal_rift_packing(self):
        try:
            import lal
        except ImportError:
            self.skipTest("LAL is optional in lightweight unit environments")
        rng = np.random.default_rng(17)
        n, dt = 32, 1.0 / 16.0
        values = rng.normal(size=n) + 1j * rng.normal(size=n)
        ts = lal.CreateCOMPLEX16TimeSeries(
            "test", lal.LIGOTimeGPS(0), 0, dt, lal.DimensionlessUnit, n
        )
        ts.data.data[:] = values
        fs = lal.CreateCOMPLEX16FrequencySeries(
            "test", ts.epoch, 0, 1.0 / (n * dt), lal.DimensionlessUnit, n
        )
        lal.COMPLEX16TimeFreqFFT(fs, ts, lal.CreateForwardCOMPLEX16FFTPlan(n, 0))
        np.testing.assert_allclose(
            gw._continuous_forward(values, dt, np), fs.data.data,
            rtol=2e-14, atol=2e-14,
        )

    def test_tdfromfd_shift_and_real_ifft_matches_lal(self):
        try:
            import lal
        except ImportError:
            self.skipTest("LAL is optional in lightweight unit environments")
        rng = np.random.default_rng(20260913)
        n, dt = 64, 1.0 / 128.0
        df = 1.0 / (n * dt)
        # A real inverse transform requires real DC and Nyquist coefficients.
        values = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
        values[[0, -1]] = values[[0, -1]].real
        epoch, extra_time = -7.25, 10.5 * dt
        got, got_epoch, shift_samples = gw._lal_tdfromfd_shift_and_irfft(
            values, df, dt, epoch, extra_time, np
        )
        # C round is half away from zero, unlike Python's ties-to-even round.
        self.assertEqual(shift_samples, 11)
        self.assertEqual(got_epoch, epoch + 11 * dt)
        fs = lal.CreateCOMPLEX16FrequencySeries(
            "shifted", lal.LIGOTimeGPS(got_epoch), 0.0, df,
            lal.DimensionlessUnit, len(values),
        )
        k = np.arange(len(values))
        fs.data.data[:] = values * np.exp(2j * np.pi * k * df * 11 * dt)
        ts = lal.CreateREAL8TimeSeries(
            "inverse", fs.epoch, 0.0, dt, lal.DimensionlessUnit, n
        )
        lal.REAL8FreqTimeFFT(ts, fs, lal.CreateReverseREAL8FFTPlan(n, 0))
        np.testing.assert_allclose(got, ts.data.data, rtol=2e-14, atol=2e-14)

    def test_tdfromfd_numpy_jax_agree(self):
        try:
            import jax.numpy as jnp
        except ImportError:
            self.skipTest("JAX is optional in lightweight unit environments")
        n, dt = 32, 1.0 / 64.0
        df = 1.0 / (n * dt)
        values = np.linspace(0.0, 1.0, n // 2 + 1).astype(np.complex128)
        expected = gw._lal_tdfromfd_shift_and_irfft(
            values, df, dt, -1.0, 3.2 * dt, np
        )
        got = gw._lal_tdfromfd_shift_and_irfft(
            jnp.asarray(values), df, dt, -1.0, 3.2 * dt, jnp
        )
        np.testing.assert_allclose(np.asarray(got[0]), expected[0],
                                   rtol=2e-13, atol=2e-13)
        self.assertEqual(got[1:], expected[1:])


class TestRIFTPostprocessing(unittest.TestCase):
    def test_grow_appends_right_and_preserves_epoch(self):
        modes = {(2, 2): np.arange(12, dtype=np.complex128)}
        got, epoch, ntaper = gw._rift_postprocess_td_modes(
            modes, -3.25, 1.0 / 16.0, 20, 2.0, np
        )
        self.assertEqual(epoch, -3.25)
        self.assertEqual(ntaper, 8)
        expected = np.pad(modes[(2, 2)], (0, 8))
        j = np.arange(ntaper)
        expected[:ntaper] *= 0.5 - 0.5 * np.cos(np.pi * j / ntaper)
        np.testing.assert_allclose(got[(2, 2)], expected)

    def test_shrink_discards_left_and_advances_epoch(self):
        source = np.arange(24, dtype=np.float64).astype(np.complex128)
        got, epoch, ntaper = gw._rift_postprocess_td_modes(
            {(2, 2): source, (2, -2): source.conj()},
            -2.0, 1.0 / 16.0, 16, 2.0, np,
        )
        self.assertEqual(epoch, -1.5)
        self.assertEqual(ntaper, 8)
        window = np.ones(16)
        window[:ntaper] = 0.5 - 0.5 * np.cos(np.pi * np.arange(ntaper) / ntaper)
        np.testing.assert_allclose(got[(2, 2)], source[-16:] * window)

    def test_rejects_noncommon_grid(self):
        with self.assertRaisesRegex(gw.WaveformCompatibilityError, "share a grid"):
            gw._rift_postprocess_td_modes(
                {(2, 2): np.zeros(8), (2, -2): np.zeros(10)},
                0.0, 1.0 / 16.0, 16, 2.0, np,
            )

    def test_matches_actual_lalsimutils_post_lal_stage(self):
        try:
            import lal
            import lalsimulation as lalsim
            from RIFT import lalsimutils as lsu
        except ImportError:
            self.skipTest("LAL and RIFT waveform dependencies are optional")
        dt, df, fmin = 1.0 / 512.0, 0.25, 40.0
        p = lsu.ChooseWaveformParams(
            m1=30 * lal.MSUN_SI, m2=25 * lal.MSUN_SI,
            s1z=0.1, s2z=-0.2, fmin=fmin, fref=60.0,
            deltaT=dt, deltaF=df, approx=lalsim.IMRPhenomD,
            dist=200e6 * lal.PC_SI, phiref=0.4, psi=0.0,
        )
        raw_struct = lsu.hlmoft_FromFD_dict(p.manual_copy(), Lmax=2)
        raw = lsu.SphHarmTimeSeries_to_dict(raw_struct, 2)
        expected = lsu.hlmoft(p.manual_copy(), Lmax=2, silent=True)
        source = {label: np.asarray(series.data.data).copy()
                  for label, series in raw.items()}
        got, epoch, ntaper = gw._rift_postprocess_td_modes(
            source, float(raw[(2, 2)].epoch), dt,
            int(1.0 / (dt * df)), fmin, np,
        )
        self.assertEqual(ntaper, max(
            int(0.01 * min(len(source[(2, 2)]), len(got[(2, 2)]))),
            int(1.0 / (fmin * dt)),
        ))
        for label in expected:
            self.assertAlmostEqual(float(expected[label].epoch), epoch, places=12)
            target = np.asarray(expected[label].data.data)
            scale = np.max(np.abs(target))
            if scale == 0:
                np.testing.assert_array_equal(got[label], target)
                continue
            # Physical strain is ~1e-21: a unit-scale absolute tolerance would
            # silently accept a missing or sign-flipped waveform. Normalize
            # both sides, and pin that the comparison rejects those defects.
            np.testing.assert_allclose(got[label] / scale, target / scale,
                                       rtol=2e-14, atol=2e-14)
            for broken in (np.zeros_like(target), -target):
                with self.assertRaises(AssertionError):
                    np.testing.assert_allclose(broken / scale, target / scale,
                                               rtol=2e-14, atol=2e-14)
        self.assertGreater(np.max(np.abs(expected[(2, 2)].data.data)), 0)


class _Params:
    deltaT = 1.0 / 16.0
    deltaF = 0.5
    fmin = 2.0
    fmax = 8.0
    fref = 2.0
    phiref = 0.3
    m1 = 30.0 * 1.9884099021470416e30
    m2 = 25.0 * 1.9884099021470416e30
    s1x = s1y = s2x = s2y = 0.0
    s1z = 0.1
    s2z = -0.2
    dist = 200.0 * 3.085677581491367e22


class TestProviderGuards(unittest.TestCase):
    def setUp(self):
        try:
            import jax.numpy as jnp
            import lalsimulation as lalsim
        except ImportError:
            self.skipTest("JAX and LAL are required for provider contract tests")
        self.jnp = jnp
        self.P = _Params()
        self.P.approx = lalsim.IMRPhenomD

    def test_rift_conditioning_fails_closed(self):
        with self.assertRaisesRegex(gw.WaveformCompatibilityError, "not certified"):
            gw.generate_imrphenomd_fd(self.P, backend=self.jnp)

    def test_rejects_wrong_approximant_before_ripple_import(self):
        import lalsimulation as lalsim
        self.P.approx = lalsim.TaylorF2
        with self.assertRaisesRegex(gw.WaveformCompatibilityError, "only supports"):
            gw.generate_imrphenomd_fd(
                self.P, backend=self.jnp, conditioning="direct_fd"
            )

    def test_direct_fd_is_unconditioned_and_has_exact_symmetries(self):
        jnp = self.jnp

        class FakeRipple:
            @staticmethod
            def gen_IMRPhenomD(f, params, fref):
                return (1.0 + 0.25j) * f ** (-7.0 / 6.0)

        with mock.patch.object(gw, "_load_ripple", return_value=FakeRipple):
            bank = gw.generate_imrphenomd_fd(
                self.P, backend=jnp, conditioning="direct_fd"
            )
        self.assertFalse(bank.conditioned)
        self.assertEqual(bank.epoch, 0.0)
        n = bank.modes[(2, 2)].shape[0]
        reflection = (-np.arange(n)) % n
        h22 = np.asarray(bank.modes[(2, 2)])
        h2m2 = np.asarray(bank.modes[(2, -2)])
        np.testing.assert_allclose(h2m2, np.conj(h22[reflection]))
        for lm, mode in bank.modes.items():
            np.testing.assert_allclose(
                np.asarray(bank.conjugate_modes[lm]),
                np.conj(np.asarray(mode)[reflection]),
            )


if __name__ == "__main__":
    unittest.main()
