"""Conditioning and physical-time contracts of the bilby RIFT source adapters."""
import numpy as np
import pytest
import lal
import lalsimulation as lalsim
from RIFT import lalsimutils as lsu
from RIFT.calmarg import rift_source as source


def lal_polarizations(values, epoch, delta_t):
    """Independent LAL real FFT with its physical epoch, not the adapter helper."""
    result = {}
    for key, samples in [('plus', values.real), ('cross', -values.imag)]:
        ts = lal.CreateREAL8TimeSeries(key, lal.LIGOTimeGPS(epoch), 0, delta_t,
                                      lal.DimensionlessUnit, len(samples))
        ts.data.data[:] = samples
        fs = lal.CreateCOMPLEX16FrequencySeries(key, ts.epoch, 0,
                                               1 / (len(samples) * delta_t),
                                               lal.DimensionlessUnit, len(samples) // 2 + 1)
        lal.REAL8TimeFreqFFT(fs, ts, lal.CreateForwardREAL8FFTPlan(len(samples), 0))
        frequencies = np.arange(fs.data.length) * fs.deltaF
        result[key] = fs.data.data * np.exp(-2j * np.pi * frequencies * float(ts.epoch))
    return result


def complex_series(values, epoch=-0.123456789, delta_t=1 / 128):
    ts = lal.CreateCOMPLEX16TimeSeries('test', lal.LIGOTimeGPS(epoch), 0,
                                     delta_t, lal.DimensionlessUnit, len(values))
    ts.data.data[:] = values
    return ts


@pytest.mark.parametrize('grid', ['full', 'positive_start', 'zero_start_cutoff'])
def test_fft_preserves_fractional_epoch_and_requested_grid(grid):
    t = np.arange(64) / 128
    values = (1 + .3j) * np.sin(2 * np.pi * 10 * t) + .2 * np.cos(2 * np.pi * 24 * t)
    ts = complex_series(values)
    frequencies = np.fft.rfftfreq(len(values), ts.deltaT)
    select = {'full': slice(None), 'positive_start': slice(3, 19),
              'zero_start_cutoff': slice(0, 20)}[grid]
    actual = source._polarizations_from_complex_time_series(ts, frequencies[select])
    expected = lal_polarizations(values, float(ts.epoch), ts.deltaT)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key][select], rtol=1e-13, atol=1e-15)


def test_fft_invariant_under_sample_origin_change():
    rng = np.random.default_rng(145)
    values = rng.normal(size=64) + 1j * rng.normal(size=64)
    a = complex_series(values)
    b = complex_series(np.roll(values, 7), epoch=float(a.epoch) - 7 * a.deltaT)
    frequencies = np.fft.rfftfreq(len(values), a.deltaT)
    first = source._polarizations_from_complex_time_series(a, frequencies)
    second = source._polarizations_from_complex_time_series(b, frequencies)
    for key in first:
        np.testing.assert_allclose(first[key], second[key], rtol=1e-12, atol=1e-14)


def test_fft_rejects_extrapolation():
    with pytest.raises(ValueError, match='outside'):
        source._polarizations_from_complex_time_series(complex_series(np.ones(64)), np.array([0., 65.]))


ADAPTERS = [source.RIFT_lal_binary_black_hole_orig,
            source.RIFT_lal_binary_black_hole,
            source.RIFT_lal_eccentric_binary_black_hole]


@pytest.mark.parametrize('adapter', ADAPTERS)
@pytest.mark.parametrize('method', ['hlmoft', 'gws_hlmoft', 'internal_hlmoft', 'hoft'])
@pytest.mark.parametrize('no_condition', [False, True])
def test_adapter_leaves_conditioning_with_selected_generator(monkeypatch, adapter, method, no_condition):
    # An explicitly unconditioned output has a nonzero first sample. A method
    # name alone must not be taken as evidence that every output is tapered.
    values = (np.linspace(1, 2, 64) + 1j * np.linspace(-.7, .8, 64))
    if not no_condition:
        values[:12] *= np.sin(np.linspace(0, np.pi / 2, 12)) ** 2
    mode = complex_series(values)
    calls = []

    def modes(P, **kwargs):
        assert P.taper == lalsim.SIM_INSPIRAL_TAPER_START
        assert kwargs.get('no_condition') == no_condition
        calls.append('modes')
        return {(2, 2): mode}

    monkeypatch.setattr(source.lalsimutils, 'hlmoft', modes)
    monkeypatch.setattr(source.rgws, 'hlmoft', modes)

    def internal(P, **kwargs):
        assert kwargs['extra_waveform_kwargs']['no_condition'] == no_condition
        calls.append('internal')
        return {(2, 2): lsu.DataFourier(mode)}, {}

    monkeypatch.setattr(source, 'internal_hlm_generator', internal)
    monkeypatch.setattr(source.lalsimutils, 'complex_hoft', lambda P: (calls.append('hoft') or mode))

    def unexpected_taper(*args):
        raise AssertionError('Adapter added a polarization taper after the selected generator')

    monkeypatch.setattr(source.lalsim, 'SimInspiralREAL8WaveTaper', unexpected_taper)
    frequencies = np.fft.rfftfreq(64, mode.deltaT)
    options = dict(waveform_approximant='IMRPhenomTPHM', sampling_frequency=128.,
                   minimum_frequency=20., reference_frequency=20., Lmax=2,
                   h_method=method, extra_waveform_kwargs={'no_condition': no_condition})
    extra = {'lambda_1': 0., 'lambda_2': 0.} if adapter is ADAPTERS[0] else {}
    if adapter is ADAPTERS[2]:
        extra.update(eccentricity=.1, mean_per_ano=.2)
    actual = adapter(frequencies, 60., 50., 400., 0., 0., .1, 0., 0., -.1,
                     iota=.9, phase=.3, **extra, **options)
    mode_route = method != 'hoft' and not (adapter is ADAPTERS[0] and method == 'internal_hlmoft')
    projected = values * lal.SpinWeightedSphericalHarmonic(.9, -.3, -2, 2, 2) if mode_route else values
    expected = lal_polarizations(projected, float(mode.epoch), mode.deltaT)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-14)
    assert calls == (['internal'] if method == 'internal_hlmoft' and mode_route
                     else ['modes'] if mode_route else ['hoft'])


@pytest.mark.parametrize('method,approximant,no_condition', [
    ('gws_hlmoft', 'SEOBNRv5PHM', False),
    ('hlmoft', 'SEOBNRv4', False),
    ('hlmoft', 'SEOBNRv4', True),
    ('hlmoft', 'IMRPhenomTPHM', False),
    ('hlmoft', 'IMRPhenomTPHM', True),
    ('hlmoft', 'IMRPhenomXPHM', False),
    ('hlmoft', 'IMRPhenomXPHM', True),
    ('internal_hlmoft', 'IMRPhenomTPHM', False),
    ('internal_hlmoft', 'IMRPhenomTPHM', True),
    ('hoft', 'TaylorT4', False),
])
def test_real_generator_conditioning_and_epoch(monkeypatch, method, approximant, no_condition):
    """Actual generators: compare with captured TD output and count polarization tapers."""
    if approximant == 'SEOBNRv5PHM':
        pytest.importorskip('pyseobnr')
    seen = []
    tapers = []
    original_taper = source.lalsim.SimInspiralREAL8WaveTaper

    def taper(data, flag):
        tapers.append(flag)
        return original_taper(data, flag)

    monkeypatch.setattr(source.lalsim, 'SimInspiralREAL8WaveTaper', taper)
    original_combine = source.lalsimutils.hoft_from_hlm
    original_hoft = source.lalsimutils.complex_hoft

    def combine(*args, **kwargs):
        ht = original_combine(*args, **kwargs)
        seen.append((ht.data.data.copy(), float(ht.epoch), ht.deltaT))
        return ht

    def hoft(*args, **kwargs):
        ht = original_hoft(*args, **kwargs)
        seen.append((ht.data.data.copy(), float(ht.epoch), ht.deltaT))
        return ht

    monkeypatch.setattr(source.lalsimutils, 'hoft_from_hlm', combine)
    monkeypatch.setattr(source.lalsimutils, 'complex_hoft', hoft)
    frequencies = np.fft.rfftfreq(4096, 1 / 1024)
    spin = 0. if approximant == 'TaylorT4' else .1
    output = source.RIFT_lal_binary_black_hole(
        frequencies, 60., 50., 400., 0., 0., spin, 0., 0., -spin, .9, .3,
        waveform_approximant=approximant, sampling_frequency=1024., Lmax=2,
        minimum_frequency=20., reference_frequency=20., h_method=method,
        extra_waveform_kwargs={'no_condition': no_condition, 'fd_L_frame': True})
    assert seen
    values, epoch, delta_t = seen[-1]
    expected = lal_polarizations(values, epoch, delta_t)
    for key in expected:
        scale = np.max(np.abs(expected[key]))
        np.testing.assert_allclose(output[key], expected[key], rtol=1e-11, atol=scale * 1e-13)
    # The fallback still receives its requested taper through complex_hoft.
    # The other real paths add no LAL polarization taper in the adapter.
    assert tapers == ([lalsim.SIM_INSPIRAL_TAPER_START] * 2
                      if method == 'hoft' or approximant == 'SEOBNRv4' else [])
    # no_condition does not override the legacy SEOB branch's own taper policy.
    if not no_condition or approximant == 'SEOBNRv4':
        assert abs(values[0]) < np.max(abs(values)) * 1e-13
    elif approximant == 'IMRPhenomTPHM':
        assert abs(values[0]) > np.max(abs(values)) * 1e-8
