"""Run the actual helper parser and stage builder, with no external work or likelihoods."""
import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]

class StageBuilt(Exception):
    pass

def generate(monkeypatch,tmp_path, mode, mc=10, force_method=None, fref=35, extra=()):
    lal=pytest.importorskip('lal')
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setenv('GW_SURROGATE','')
    from RIFT import lalsimutils
    import RIFT.misc.dag_utils_generic as dag
    monkeypatch.setattr(dag,'which',lambda name:'/usr/bin/true')
    commands=[]
    monkeypatch.setattr(os,'system',lambda command:commands.append(command) or 0)
    monkeypatch.chdir(tmp_path)
    script=ROOT/'bin/helper_LDG_Events.py'
    args=['--event-time','1400000000','--fake-data','--manual-ifo-list','H1',
          '--psd-file','H1=fake-psd.xml','--working-directory',str(tmp_path),
          '--fmin','20','--fmin-template',str(fref),'--no-enforce-duration-bound',
          '--force-notune-initial-grid','--propose-fit-strategy','--assume-precessing-spin']
    if mode is not None: args+=['--rf-transverse-spin-coordinates',mode]
    if mc is not None:
        p=lalsimutils.ChooseWaveformParams();p.m1,p.m2=lalsimutils.m1m2(mc*lal.MSUN_SI,.24)
        p.s1x=.2;p.s1z=.1;p.s2y=.3;p.s2z=-.1;p.fref=fref
        filename=str(tmp_path/'synthetic')
        lalsimutils.ChooseWaveformParams_array_to_xml([p],fname=filename,fref=fref)
        args+=['--sim-xml',filename+'.xml.gz','--event','0']
    if force_method is not None:args+=['--force-fit-method',force_method]
    args+=list(extra)
    monkeypatch.setattr(sys,'argv',[str(script)]+args)
    stop=next(i for i,line in enumerate(script.read_text().splitlines(),1)
              if line=='with open("helper_cip_arg_list.txt",\'w+\') as f:')
    result={}
    def trace(frame,event,arg):
        if frame.f_code.co_filename==str(script) and event=='line' and frame.f_lineno==stop:
            g=frame.f_globals
            result.update(lines=list(g['helper_cip_arg_list']),fit_method=g['fit_method'],
                          mass=g['event_dict'].get('MChirp'),placeholder=g['event_dict'].get('rf_mass_is_placeholder',False),
                          commands=commands,ile=g['helper_ile_args'])
            raise StageBuilt
        return trace
    previous=sys.gettrace();sys.settrace(trace)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            runpy.run_path(str(script),run_name='__main__')
    except StageBuilt:
        pass
    finally:sys.settrace(previous)
    assert result,'Helper did not reach generated stage boundary'
    return result

@pytest.mark.parametrize('mode,mc,activated',[('physics3',10,True),('auto',10,True),('auto',25,False),('off',10,False),('auto',None,False)])
def test_actual_helper_top_level_policy(monkeypatch,tmp_path,mode,mc,activated):
    result=generate(monkeypatch,tmp_path,mode,mc)
    active=[line for line in result['lines'] if '--rf-transverse-spin-coordinates physics3' in line]
    assert bool(active)==activated
    if activated:
        assert result['fit_method']=='rf'
        for line in active:
            for name in ['delta_mc','mu1','mu2','chiMinus','s1x','s1y','s2x','s2y']:
                assert name in line
            assert '--fref 35.0' in line
        assert '--reference-freq 35.0' in result['ile']
    if mc is None:assert result['placeholder']

def test_actual_helper_explicit_gp_is_preserved(monkeypatch,tmp_path):
    result=generate(monkeypatch,tmp_path,'auto',10,'gp')
    assert result['fit_method']=='gp'
    assert not any('--rf-transverse-spin-coordinates physics3' in line for line in result['lines'])


def test_actual_off_has_unchanged_generated_stages(monkeypatch,tmp_path):
    default=generate(monkeypatch,tmp_path,None,10)
    off=generate(monkeypatch,tmp_path,'off',10)
    assert off['lines']==default['lines']
    assert off['ile']==default['ile']
    assert off['fit_method']==default['fit_method']=='gp'

def test_actual_explicit_physics_rejects_explicit_gp(monkeypatch,tmp_path):
    with pytest.raises(ValueError,match='No complete two-spin RF stage'):
        generate(monkeypatch,tmp_path,'physics3',10,'gp')


ACTIVATION = ' --rf-transverse-spin-coordinates physics3 --fref 35.0'

def test_auto_effect_in_a_bare_helper_run(monkeypatch,tmp_path):
    # Without rf or phase options, auto switches every stage to rf in the mu1/mu2 basis,
    # cuts the first stage 3->2 iterations, and activates only the complete stage.
    default=generate(monkeypatch,tmp_path,None,10)
    off=generate(monkeypatch,tmp_path,'off',10)
    auto=generate(monkeypatch,tmp_path,'auto',10)
    assert off['lines']==default['lines']
    assert [line.split()[0] for line in default['lines']]==['3','2','3']
    assert [line.split()[0] for line in auto['lines']]==['2','2','3']
    for line in default['lines']:
        assert '--fit-method gp' in line and '--parameter mc ' in line and 'mu1' not in line
    for line in auto['lines']:
        assert '--fit-method rf' in line
        assert '--parameter-implied mu1 --parameter-implied mu2 --parameter-nofit mc' in line
    assert [line.endswith(ACTIVATION) for line in auto['lines']]==[False,False,True]
    assert auto['ile']==default['ile']

def test_auto_effect_in_the_asimov_configuration(monkeypatch,tmp_path):
    # The Asimov template already forces rf and the phase basis: auto only appends the
    # activation and the ILE reference frequency to the complete stage.
    extra=['--internal-use-aligned-phase-coordinates']
    default=generate(monkeypatch,tmp_path,None,10,'rf',extra=extra)
    off=generate(monkeypatch,tmp_path,'off',10,'rf',extra=extra)
    auto=generate(monkeypatch,tmp_path,'auto',10,'rf',extra=extra)
    assert off['lines']==default['lines']
    assert auto['lines'][:-1]==default['lines'][:-1]
    assert auto['lines'][-1]==default['lines'][-1]+ACTIVATION
    assert auto['ile']==default['ile']
