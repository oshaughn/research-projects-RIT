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

def generate(monkeypatch,tmp_path, mode, mc=10, force_method=None, fref=35):
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

@pytest.mark.parametrize('mode,mc,expected',[('geometric4',10,'geometric4'),('auto',10,'geometric4'),
    ('auto',19.9,'geometric4'),('auto',20.1,None),('auto',25,None),('off',10,None),('auto',None,None)])
def test_actual_helper_top_level_policy(monkeypatch,tmp_path,mode,mc,expected):
    # The exact 20 boundary is checked directly in test_auto_conservative_mass;
    # XML mass roundtrips can put nominal 20 infinitesimally below the boundary.
    result=generate(monkeypatch,tmp_path,mode,mc)
    import shlex
    active=[line for line in result['lines'] if '--rf-transverse-spin-coordinates ' in line]
    assert bool(active)==(expected is not None)
    if expected is not None:
        assert result['fit_method']=='rf'
        for line in active:
            tokens=shlex.split(line)
            assert tokens[tokens.index('--rf-transverse-spin-coordinates')+1]==expected
            for name in ['delta_mc','mu1','mu2','chiMinus','s1x','s1y','s2x','s2y']:
                assert name in line
            assert '--fref 35.0' in line
        assert '--reference-freq 35.0' in result['ile']
    if mc is None:assert result['placeholder']

def test_actual_helper_explicit_gp_is_preserved(monkeypatch,tmp_path):
    result=generate(monkeypatch,tmp_path,'auto',10,'gp')
    assert result['fit_method']=='gp'
    assert not any('--rf-transverse-spin-coordinates ' in line for line in result['lines'])


def test_actual_off_has_unchanged_generated_stages(monkeypatch,tmp_path):
    default=generate(monkeypatch,tmp_path,None,10)
    off=generate(monkeypatch,tmp_path,'off',10)
    assert off['lines']==default['lines']
    assert off['ile']==default['ile']
    assert off['fit_method']==default['fit_method']=='gp'

def test_actual_explicit_geometric4_rejects_explicit_gp(monkeypatch,tmp_path):
    with pytest.raises(ValueError,match='No complete two-spin RF stage'):
        generate(monkeypatch,tmp_path,'geometric4',10,'gp')

def test_actual_helper_refuses_physics3(monkeypatch,tmp_path):
    with pytest.raises(SystemExit):
        generate(monkeypatch,tmp_path,'physics3',10)


@pytest.mark.parametrize('mode',['geometric4','geometric4-phase-excess'])
def test_actual_geometric4_helper(monkeypatch,tmp_path,mode):
    import shlex
    result=generate(monkeypatch,tmp_path,mode,25)
    active=[shlex.split(line) for line in result['lines'] if ('--rf-transverse-spin-coordinates '+mode) in line]
    assert active, 'Explicit geometric4 must survive actual helper generation'
    for tokens in active:
        i=tokens.index('--rf-transverse-spin-coordinates')
        assert tokens[i+1]==mode
        assert float(tokens[tokens.index('--fref')+1])==35.
    assert result['fit_method']=='rf'


@pytest.mark.parametrize('mode',['geometric4','geometric4-phase-excess'])
def test_actual_geometric4_rejects_gp(monkeypatch,tmp_path,mode):
    with pytest.raises(ValueError,match='No complete two-spin RF stage'):
        generate(monkeypatch,tmp_path,mode,10,'gp')


def test_actual_asimov_default_generates_geometric4(monkeypatch,tmp_path):
    liquid=pytest.importorskip('liquid')
    import configparser
    template=(ROOT/'RIFT/asimov/rift.ini').read_text()
    start=template.index('cip-fit-method=')
    end=template.index('cip-sampler-method=',start)
    rendered=liquid.Liquid(template[start:end],from_file=False).render(sampler={'cip':{}})
    parser=configparser.RawConfigParser()
    parser.read_string('[policy]\n'+rendered)
    mode=parser.get('policy','rf-transverse-spin-coordinates').strip('"')
    result=generate(monkeypatch,tmp_path,mode,10)
    active=[line for line in result['lines'] if '--rf-transverse-spin-coordinates ' in line]
    assert active
    assert all('--rf-transverse-spin-coordinates geometric4 ' in line for line in active)
