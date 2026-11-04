"""Opt-in dispatch and native-provider rejection before expensive work."""
from types import SimpleNamespace

import numpy as np
import pytest


def test_opt_in_dispatch_preserves_arguments(monkeypatch):
    from RIFT.likelihood import gpu_precompute as gpu
    from RIFT.likelihood import factored_likelihood_rotating_freqresponse as fr
    seen = {}

    def replacement(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return 'replacement sentinel'

    monkeypatch.setenv('RIFT_GPU_PRECOMPUTE','1')
    monkeypatch.setenv('RIFT_GPU_WAVEFORM','lal')
    monkeypatch.setattr(gpu,'PrecomputeLikelihoodTermsRotatingFreqResponseGPU',replacement)
    result = fr.PrecomputeLikelihoodTermsRotatingFreqResponse(
        100.,0.2,None,{}, {},2,512.,Qmax=1,p_max=1,skip_interpolation=True)
    assert result == 'replacement sentinel'
    assert seen['args'] == (100.,0.2,None,{}, {},2,512.)
    assert seen['kwargs']['Qmax'] == 1 and seen['kwargs']['p_max'] == 1
    assert seen['kwargs']['skip_interpolation'] is True


def test_unvalidated_waveform_env_fails_closed(monkeypatch):
    from RIFT.likelihood import factored_likelihood_rotating_freqresponse as fr
    monkeypatch.setenv('RIFT_GPU_PRECOMPUTE','1')
    monkeypatch.setenv('RIFT_GPU_WAVEFORM','ripple')
    with pytest.raises(ValueError,match='not yet validated'):
        fr.PrecomputeLikelihoodTermsRotatingFreqResponse(100.,0.2,None,{}, {},2,512.)


@pytest.mark.parametrize('mismatch', ['epoch','delta_t','delta_f'])
def test_native_provider_grid_mismatch_rejected(mismatch):
    from RIFT.likelihood.gpu_precompute import PrecomputeLikelihoodTermsRotatingFreqResponseGPU
    n,df,dt = 32,0.125,0.25
    common = dict(modes={(2,2):np.ones(n,dtype=complex)},delta_f=df,
                  delta_t=dt,epoch=1e9,conditioned=True)
    main = SimpleNamespace(**common)
    conjugate = SimpleNamespace(**common)
    if mismatch == 'epoch':
        conjugate.epoch += 1.  # default np.isclose used to accept this at GPS epochs
    elif mismatch == 'delta_t':
        conjugate.delta_t *= 1.000001
    else:
        conjugate.delta_f *= 1.000001
    P = SimpleNamespace(deltaT=dt,dist=1.)
    data = {'H1':SimpleNamespace(deltaF=df)}
    with pytest.raises(ValueError,match='different grids or modes'):
        PrecomputeLikelihoodTermsRotatingFreqResponseGPU(
            1e9,0.1,P,data,{'H1':None},2,2.,backend=np,
            waveform_provider=lambda *a,**k:(main,conjugate))
