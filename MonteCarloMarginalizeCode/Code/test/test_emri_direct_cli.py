"""Independent FFT/white-noise checks of the direct fixture calculation."""

import numpy as np
import pytest

from RIFT.LISA.emri_direct_cli import evaluate_channel_trials, channel_digest


@pytest.mark.parametrize('n', [127, 128])
def test_target_snr_and_time_domain_residual(n):
    t = np.arange(n)
    data = np.asarray([np.cos(2*np.pi*k*t/n) for k in (2, 4, 7)])
    levels = np.array([2., 3., 5.])
    result = evaluate_channel_trials(
        data, [('same', data), ('half', data/2), ('null', data*0)],
        dt=2., psd_levels=levels, target_snr=12., fmin=0, fmax=.25,
    )
    assert result['snr_squared'] == pytest.approx(144.)
    assert result['injection_log_likelihood'] == 0
    assert result['null_log_likelihood'] == pytest.approx(-72.)
    assert [r['log_likelihood'] for r in result['trials']] == pytest.approx([0, -18., -72.])
    # Independently derived discrete white-noise variance S/(2*dt).
    variance = np.asarray(result['noise']['one_sided_psd']) / 4.
    expected = -.5*np.sum((data-data/2)**2/variance[:, None])
    assert result['trials'][1]['log_likelihood'] == pytest.approx(expected)
    assert result['selected_bins'] == (n-1)//2


def test_noise_is_fixed_across_trials_and_changes_have_effect():
    n = 96
    t = np.arange(n)
    data = np.asarray([np.cos(2*np.pi*k*t/n) for k in (2, 4, 7)])
    kwargs = dict(dt=1., psd_levels=[1, 1, 1], target_snr=10, fmin=0, fmax=.5)
    a = evaluate_channel_trials(data, [('offset', data*.9)], **kwargs)
    b = evaluate_channel_trials(data, [('offset', data*1.1)], **kwargs)
    assert a['noise'] == b['noise']
    assert a['trials'][0]['log_likelihood'] == pytest.approx(-.5)
    assert b['trials'][0]['log_likelihood'] == pytest.approx(-.5)
    assert channel_digest(data) != channel_digest(data+.1)


@pytest.mark.parametrize('change', [
    {'dt': 0}, {'dt': np.nan}, {'target_snr': 0}, {'target_snr': np.inf},
    {'psd_levels': [1, -1, 1]}, {'psd_levels': [1, 1]},
    {'fmin': .2, 'fmax': .1}, {'fmin': 10, 'fmax': 11},
])
def test_invalid_noise_and_sampling(change):
    rng = np.random.default_rng(19)
    data = rng.normal(size=(3, 64))
    kwargs = dict(dt=1., psd_levels=[1, 1, 1], target_snr=10, fmin=0, fmax=.5)
    kwargs.update(change)
    with pytest.raises(ValueError):
        evaluate_channel_trials(data, [('same', data)], **kwargs)


def test_invalid_channels_and_no_band_power():
    kwargs = dict(dt=1., psd_levels=[1, 1, 1], target_snr=10, fmin=0, fmax=.5)
    for data in (np.ones((2, 64)), np.ones((3, 64), dtype=complex),
                 np.zeros((3, 64)), np.full((3, 64), np.nan)):
        with pytest.raises(ValueError):
            evaluate_channel_trials(data, [], **kwargs)
    rng = np.random.default_rng(41)
    data = rng.normal(size=(3, 64))
    with pytest.raises(ValueError):
        evaluate_channel_trials(data, [('bad', data[:, :-1])], **kwargs)


def test_absolute_psd_is_frozen_across_fixture_changes():
    n, dt = 128, 2.
    t = np.arange(n)
    data = np.asarray([np.cos(2*np.pi*k*t/n) for k in (2, 4, 7)])
    kwargs = dict(dt=dt, psd_levels=[2, 3, 5], target_snr=None, fmin=0, fmax=.25)
    base = evaluate_channel_trials(data, [('offset', data/2)], **kwargs)
    changed = evaluate_channel_trials(data*2, [('offset', data)], **kwargs)
    assert base['noise']['normalization'] == 'fixed_supplied_psd'
    assert base['noise']['one_sided_psd'] == [2., 3., 5.]
    assert changed['noise'] == base['noise']
    assert changed['snr_squared'] == pytest.approx(4*base['snr_squared'])
    assert changed['trials'][0]['log_likelihood'] == pytest.approx(
        4*base['trials'][0]['log_likelihood'])


@pytest.mark.parametrize('n',[127,128])
def test_periodic_hann_fourier_parseval_and_fixed_psd_identities(n):
    from RIFT.LISA.emri_direct_cli import apply_analysis_window
    dt=2.;epoch=10000.;k=7
    phase=2*np.pi*np.arange(n)/n
    raw=np.tile(np.cos(k*phase),(3,1))
    window={'kind':'periodic_hann'}
    data,metadata=apply_analysis_window(raw,dt=dt,epoch_seconds=epoch,window=window)
    transform=np.fft.rfft(data[0])
    expected=np.zeros_like(transform)
    expected[k]=n/4.;expected[k-1]=expected[k+1]=-n/8.
    np.testing.assert_allclose(transform,expected,rtol=0,atol=1e-12)
    # Parseval: all support lies strictly inside positive frequency bins.
    assert 2*np.sum(np.abs(transform)**2)/n == pytest.approx(np.sum(data[0]**2))
    assert np.sum(data[0]**2)==pytest.approx(3*n/16.)
    assert metadata['retained_start_seconds']==epoch
    assert metadata['retained_duration_seconds']==n*dt
    assert metadata['raw_channel_sha256']==channel_digest(raw)
    assert metadata['windowed_channel_sha256']==channel_digest(data)
    assert data[0,0]==0 and data[0,-1]!=0  # periodic, not symmetric np.hanning
    kwargs=dict(dt=dt,psd_levels=[2,3,5],target_snr=None,fmin=0,fmax=.25,
                analysis_window=window,retained_epoch_seconds=epoch)
    result=evaluate_channel_trials(raw,[('same',raw),('half',raw/2),('null',raw*0)],**kwargs)
    rho2=2*dt*np.sum(data**2/np.array([2,3,5])[:,None])
    assert result['snr_squared']==pytest.approx(rho2)
    assert result['injection_log_likelihood']==0
    assert result['null_log_likelihood']==pytest.approx(-rho2/2)
    assert [r['log_likelihood'] for r in result['trials']]==pytest.approx([0,-rho2/8,-rho2/2])
    assert result['injection_sha256']==channel_digest(data)
    assert result['window']==metadata
    assert result['trials'][1]['window']['raw_channel_sha256']==channel_digest(raw/2)
    changed=evaluate_channel_trials(raw*2,[('half',raw)],**kwargs)
    assert changed['noise']==result['noise']
    assert changed['snr_squared']==pytest.approx(4*rho2)
    assert result['noise']['one_sided_psd']==[2.,3.,5.]
    assert result['noise']['kind']=='synthetic_independent_white_after_window'


def schema_fixture(version=1):
    return {'schema_version':version,'waveform':{},'response':{},'injection':{},'trials':[],
            'noise':{'kind':'synthetic_independent_white','target_snr':None}}


def test_schema_one_unchanged_fields_and_explicit_no_window_bytes():
    import json
    from RIFT.LISA.emri_direct_cli import validate_fixture_schema
    assert validate_fixture_schema(schema_fixture())==1
    raw=np.tile(np.cos(2*np.pi*7*np.arange(128)/128),(3,1))
    kwargs=dict(dt=2,psd_levels=[1,2,3],target_snr=None,fmin=0,fmax=.25)
    a=evaluate_channel_trials(raw,[('same',raw)],**kwargs)
    b=evaluate_channel_trials(raw,[('same',raw)],analysis_window=None,**kwargs)
    assert json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
    assert 'window' not in a and 'window' not in a['trials'][0]
    assert a['noise']['kind']=='synthetic_independent_white'


@pytest.mark.parametrize('mutation',[
    'wrong_schema','bool_schema','extra_field','window_schema1','wrong_noise',
    'target','symmetric_hann','window_extra','missing_window'])
def test_schema_window_noise_refusals(mutation):
    from RIFT.LISA.emri_direct_cli import validate_fixture_schema
    value=schema_fixture(2);value['window']={'kind':'periodic_hann'}
    value['noise']['kind']='synthetic_independent_white_after_window'
    assert validate_fixture_schema(value)==2
    if mutation=='wrong_schema':value['schema_version']=3
    elif mutation=='bool_schema':value['schema_version']=True
    elif mutation=='extra_field':value['extra']=1
    elif mutation=='window_schema1':value['schema_version']=1
    elif mutation=='wrong_noise':value['noise']['kind']='synthetic_independent_white'
    elif mutation=='target':value['noise']['target_snr']=20
    elif mutation=='symmetric_hann':value['window']['kind']='hann'
    elif mutation=='window_extra':value['window']['alpha']=.5
    else:del value['window']
    with pytest.raises(ValueError):validate_fixture_schema(value)


def test_window_refuses_rescaling_invalid_arrays_and_metadata():
    from RIFT.LISA.emri_direct_cli import apply_analysis_window
    raw=np.ones((3,64));window={'kind':'periodic_hann'}
    with pytest.raises(ValueError,match='fixed absolute PSD'):
        evaluate_channel_trials(raw,[],dt=1,psd_levels=[1,1,1],target_snr=20,
                                fmin=0,fmax=.5,analysis_window=window)
    for bad in (raw[:2],raw.astype(complex),raw*np.nan):
        with pytest.raises(ValueError):apply_analysis_window(bad,dt=1,epoch_seconds=0,window=window)
    with pytest.raises(ValueError):apply_analysis_window(raw,dt=1,epoch_seconds=np.nan,window=window)
