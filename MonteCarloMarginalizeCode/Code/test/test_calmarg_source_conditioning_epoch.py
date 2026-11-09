"""Conditioning and physical-time contracts of the bilby RIFT source adapters."""
from pathlib import Path
import subprocess
import sys

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


@pytest.mark.parametrize("extended", [False, True])
def test_tidal_batch_combine_matches_ranges(tmp_path, extended):
    posterior = tmp_path / "posterior.dat"
    batches = tmp_path / "weight_files"
    batches.mkdir()
    rows = ["m1 m2 lnL ps lambda1 lambda2 row"] + [
        f"5 1.3 0 1 0 {400 + 100 * i} {i}" for i in range(3)]
    posterior.write_text("\n".join(rows) + "\n")

    # Create the last batch first. Its shorter length must not affect pairing.
    np.savetxt(batches / "tidal_weights_s2e3.dat", [1.])
    np.savetxt(batches / "tidal_weights_s0e2.dat", [0., 0.])
    if extended:
        for start, end in ((2, 3), (0, 2)):
            (batches / f"tidal_weights_s{start}e{end}.extended_posterior").write_text(
                "\n".join([rows[0], *rows[1 + start:1 + end]]) + "\n")

    # The old range overlaps corrected output and must never contribute weights.
    np.savetxt(batches / "weights_s0e3.dat", [1.e6, 1.e6, 1.e6])
    combiner = Path(__file__).resolve().parents[1] / "bin/combine_weights_and_rejection_sample.py"
    completed = subprocess.run(
        [sys.executable, str(combiner), str(posterior), str(batches)],
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    combined = np.genfromtxt(tmp_path / "reweighted_posterior_samples.dat", names=True)
    assert combined["row"] == 2.
    assert combined["lambda2"] == 600.


@pytest.mark.parametrize("have_corrected_prefix", [False, True])
def test_tidal_batch_combine_refuses_unrecomputed_legacy_ranges(tmp_path, have_corrected_prefix):
    posterior = tmp_path / "posterior.dat"
    posterior.write_text("m1 m2 lnL ps lambda1 lambda2\n5 1.3 0 1 0 400\n5 1.3 0 1 0 600\n")
    batches = tmp_path / "weight_files"
    batches.mkdir()
    np.savetxt(batches / "weights_s1e2.dat", [1.])
    if have_corrected_prefix:
        np.savetxt(batches / "tidal_weights_s0e1.dat", [1.])

    combiner = Path(__file__).resolve().parents[1] / "bin/combine_weights_and_rejection_sample.py"
    completed = subprocess.run(
        [sys.executable, str(combiner), str(posterior), str(batches)],
        capture_output=True, text=True)
    assert completed.returncode != 0
    assert "tidal" in completed.stderr
    assert not (tmp_path / "reweighted_posterior_samples.dat").exists()



@pytest.mark.parametrize("start,end,reason", [
    (1, 2, "contiguous prefix"),
    (0, 3, "range exceeds the input posterior"),
])
def test_tidal_batch_combine_checks_extended_ranges(tmp_path, start, end, reason):
    posterior = tmp_path / "posterior.dat"
    header = "m1 m2 lnL ps lambda1 lambda2\n"
    row = "5 1.3 0 1 0 400\n"
    posterior.write_text(header + row * 2)
    batches = tmp_path / "weight_files"
    batches.mkdir()
    stem = batches / f"tidal_weights_s{start}e{end}"
    np.savetxt(stem.with_suffix(".dat"), np.ones(end - start))
    stem.with_suffix(".extended_posterior").write_text(header + row * (end - start))

    combiner = Path(__file__).resolve().parents[1] / "bin/combine_weights_and_rejection_sample.py"
    completed = subprocess.run(
        [sys.executable, str(combiner), str(posterior), str(batches)],
        capture_output=True, text=True)
    assert completed.returncode != 0
    assert reason in completed.stderr
    assert not (tmp_path / "reweighted_posterior_samples.dat").exists()



@pytest.mark.parametrize("rift_mode,legacy,corrected,have_tides,expected_row", [
    (False, True, False, True, 0),
    (False, True, True, True, 0),
    (False, False, True, True, None),
    (True, True, False, False, 0),
    (True, True, False, True, None),
    (True, True, True, True, 1),
])
def test_tidal_json_batch_combine_uses_declared_waveform_source(
        tmp_path, rift_mode, legacy, corrected, have_tides, expected_row):
    from bilby.core.result import Result
    import pandas as pd

    posterior = tmp_path / "posterior.json"
    samples = {"mass_1": [5., 5.], "mass_2": [1.3, 1.3], "row": [0, 1]}
    if have_tides:
        samples.update(lambda_1=[0., 0.], lambda_2=[400., 600.])
    result = Result(label="fixture", outdir=str(tmp_path), priors={},
                    search_parameter_keys=[], posterior=pd.DataFrame(samples))
    result.save_to_file(filename=str(posterior))

    batches = tmp_path / "weight_files"
    batches.mkdir()
    if legacy:
        np.savetxt(batches / "weights_s0e2.dat", [1., 0.])
    if corrected:
        np.savetxt(batches / "tidal_weights_s0e2.dat", [0., 1.])

    combiner = Path(__file__).resolve().parents[1] / "bin/combine_weights_and_rejection_sample.py"
    cmd = [sys.executable, str(combiner), str(posterior), str(batches)]
    if rift_mode:
        cmd.append("--use-rift-samples")
    completed = subprocess.run(cmd, capture_output=True, text=True)
    output = tmp_path / "reweighted_posterior_samples.dat"
    if expected_row is None:
        assert completed.returncode != 0
        reason = ("No corrected tidal weights found" if rift_mode
                  else "Corrected RIFT JSON weights require --use-rift-samples")
        assert reason in completed.stderr
        assert not output.exists()
        if rift_mode:
            # A worker-style flag must fail instead of selecting native JSON.
            typo = subprocess.run([*cmd[:-1], "--use_rift_samples", "True"],
                                  capture_output=True, text=True)
            assert typo.returncode != 0
            assert "unrecognized arguments: --use_rift_samples" in typo.stderr
            assert not output.exists()
    else:
        assert completed.returncode == 0, completed.stderr
        combined = np.genfromtxt(output, names=True)
        assert combined["row"] == expected_row
        if have_tides:
            assert combined["lambda_2"] == [400., 600.][expected_row]
