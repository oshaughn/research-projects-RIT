"""Regression checks for inverse puffing and the circular exclusion metric."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from RIFT.misc import eccentric_phase_puff as puff
from RIFT.misc import eccentric_phase_pipeline as pipeline


def test_circular_covariance_chart_joins_the_seam():
    chart, center = puff.centered_angles([.01,2*np.pi-.01])
    assert np.ptp(chart) == pytest.approx(.02)
    assert abs(center) < 1e-12


def test_periodic_mahalanobis_minimizes_all_lifts_and_is_invariant():
    x=np.array([[20.,0.],[0.,.01]])
    y=np.array([[0.,0.],[0.,2*np.pi-.01]])
    vi=np.array([[100.,9.],[9.,1.]])
    actual=puff.minimum_periodic_distances(x,y,vi,1,block_size=1)
    expected=[]
    for row in y:
        lifts=np.arange(-100,101)*2*np.pi
        d=x[:,None,:]-row;d=np.broadcast_to(d,(len(x),len(lifts),2)).copy()
        d[:,:,1]+=lifts
        expected.append(np.sqrt(np.min(np.einsum('...i,ij,...j->...',d,vi,d))))
    assert np.allclose(actual,expected,atol=1e-9)
    shifted=y.copy();shifted[:,1]+=12*np.pi
    assert np.allclose(actual,puff.minimum_periodic_distances(x,shifted,vi,1),atol=1e-9)


def test_pipeline_preserves_nofit_stage_and_all_prior_arguments():
    early='2 --use-eccentricity --use-meanPerAno --parameter mc --parameter eccentricity --parameter-nofit meanPerAno --eccentricity-prior log_uniform'
    later=early.replace('--parameter-nofit meanPerAno','--parameter meanPerAno')
    assert pipeline.stage_arguments(early)==early
    transformed=pipeline.stage_arguments(later,10,20)
    assert transformed.startswith(later)
    assert '--eccentric-phase-coordinates' in transformed
    assert pipeline.stage_arguments(transformed,10,20)==transformed
    with pytest.raises(ValueError):pipeline.stage_arguments(transformed,15,20)


def test_pipeline_recognizes_equals_options_and_rejects_frequency_conflicts():
    stage='--use-eccentricity --use-meanPerAno --parameter=eccentricity --parameter=meanPerAno'
    transformed=pipeline.stage_arguments(stage)
    assert pipeline.FLAG in transformed
    configured=stage+' --eccentric-phase-target-frequency=30'
    with pytest.raises(ValueError):pipeline.stage_arguments(configured,10,20)
    matching=pipeline.stage_arguments(configured,10,30)
    assert matching.count(pipeline.TARGET)==1
    assert pipeline.FLAG in pipeline.puff_arguments('--parameter=eccentricity --parameter=meanPerAno')


def test_inverse_uses_new_masses_and_e_even_if_angle_is_first():
    lal=pytest.importorskip('lal')
    from RIFT import lalsimutils as lsu
    from RIFT.misc import eccentric_phase
    old=lsu.ChooseWaveformParams();old.m1=18*lal.MSUN_SI;old.m2=17*lal.MSUN_SI
    old.eccentricity=.1;old.meanPerAno=1.;old.fmin=old.fref=10.
    names=['meanPerAno','mc','delta_mc','eccentricity']
    values=np.array([[.7,16.,.1,.13]])
    ps,native=puff.native_proposals([old],values,names,10.,20.)
    p=ps[0]
    theta=eccentric_phase.forward(p.m1/lal.MSUN_SI,p.m2/lal.MSUN_SI,p.eccentricity,p.meanPerAno)
    assert abs(np.angle(np.exp(1j*(theta-.7))))<1e-10
    assert p.extract_param('mc')/lal.MSUN_SI==pytest.approx(16.)
    assert p.eccentricity==.13 and p.fmin==p.fref==10.
    assert old.eccentricity==.1 and old.meanPerAno==1.
    assert native[0,0]==p.meanPerAno


def test_zero_puff_roundtrips_xml_without_frequency_redefinition(tmp_path):
    lal=pytest.importorskip('lal')
    from RIFT import lalsimutils as lsu
    rng=np.random.default_rng(173);points=[]
    for _ in range(80):
        p=lsu.ChooseWaveformParams();p.m1=rng.uniform(19,21)*lal.MSUN_SI;p.m2=rng.uniform(16,18)*lal.MSUN_SI
        p.eccentricity=rng.uniform(.05,.15);p.meanPerAno=rng.uniform(0,2*np.pi)
        p.fmin=p.fref=10.;p.s1z=.3;p.s2z=.2;points.append(p)
    lsu.ChooseWaveformParams_array_to_xml(points,fname=str(tmp_path/'input'))
    code=Path(__file__).resolve().parents[1]
    command=[sys.executable,str(code/'bin/util_ParameterPuffball.py'),
             '--inj-file',str(tmp_path/'input.xml.gz'),'--inj-file-out',str(tmp_path/'output'),
             '--puff-factor','0','--eccentric-phase-coordinates','--eccentric-phase-reference-frequency','15',
             '--eccentric-phase-target-frequency','30','--random-seed','17']
    for name in ['meanPerAno','mc','delta_mc','eccentricity']:command+=['--parameter',name]
    env=dict(os.environ,PYTHONPATH=str(code)+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1')
    done=subprocess.run(command,cwd=tmp_path,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
    assert done.returncode==0,done.stdout[-4000:]
    a=lsu.xml_to_ChooseWaveformParams_array(str(tmp_path/'input.xml.gz'))
    b=lsu.xml_to_ChooseWaveformParams_array(str(tmp_path/'output.xml.gz'))
    assert len(a)==len(b)==80
    for x,y in zip(a,b):
        assert x.fmin==y.fmin and x.fref==y.fref
        assert x.eccentricity==y.eccentricity
        assert abs(np.angle(np.exp(1j*(x.meanPerAno-y.meanPerAno))))<1e-8
        assert x.m1==pytest.approx(y.m1,rel=1e-7)
        assert x.m2==pytest.approx(y.m2,rel=1e-7)
    audit=json.loads((tmp_path/'output_eccentric_phase.json').read_text())
    assert audit['reference_frequency']==15. and not audit['prior_change']


def test_cli_puff_follows_a_curved_effective_phase_band(tmp_path):
    lal=pytest.importorskip('lal')
    from RIFT import lalsimutils as lsu
    from RIFT.misc import eccentric_phase
    rng=np.random.default_rng(812);points=[]
    for _ in range(200):
        p=lsu.ChooseWaveformParams();p.m1=rng.uniform(19,21)*lal.MSUN_SI;p.m2=rng.uniform(16,18)*lal.MSUN_SI
        p.eccentricity=rng.uniform(.07,.14);p.fmin=p.fref=10.;p.s1z=.3;p.s2z=.2
        p.meanPerAno=eccentric_phase.inverse(p.m1/lal.MSUN_SI,p.m2/lal.MSUN_SI,p.eccentricity,.3+rng.normal(0,.015))
        points.append(p)
    lsu.ChooseWaveformParams_array_to_xml(points,fname=str(tmp_path/'band'))
    expected_frequencies={(p.fmin,p.fref) for p in lsu.xml_to_ChooseWaveformParams_array(str(tmp_path/'band.xml.gz'))}
    code=Path(__file__).resolve().parents[1];spread={}
    for enabled in [False,True]:
        name='effective' if enabled else 'native'
        cmd=[sys.executable,str(code/'bin/util_ParameterPuffball.py'),'--inj-file',str(tmp_path/'band.xml.gz'),
             '--inj-file-out',str(tmp_path/name),'--puff-factor','.5','--random-seed','42']
        for param in ['meanPerAno','mc','delta_mc','eccentricity']:cmd+=['--parameter',param]
        if enabled:cmd+=['--eccentric-phase-coordinates']
        done=subprocess.run(cmd,cwd=tmp_path,env=dict(os.environ,PYTHONPATH=str(code)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
        assert done.returncode==0,done.stdout[-4000:]
        ps=lsu.xml_to_ChooseWaveformParams_array(str(tmp_path/(name+'.xml.gz')))
        assert len(ps)>160
        theta=eccentric_phase.forward(np.array([p.m1/lal.MSUN_SI for p in ps]),np.array([p.m2/lal.MSUN_SI for p in ps]),
            np.array([p.eccentricity for p in ps]),np.array([p.meanPerAno for p in ps]))
        spread[name]=np.sqrt(np.mean(np.angle(np.exp(1j*(theta-.3)))**2))
        assert all((p.fmin,p.fref) in expected_frequencies for p in ps)
    assert spread['effective']<.06
    assert spread['native']>.5


@pytest.mark.parametrize('ending', ['\n', '\r\n'])
def test_stage_options_stay_on_one_line(ending):
    stage = '--use-eccentricity --use-meanPerAno --parameter eccentricity --parameter meanPerAno'
    actual = pipeline.stage_arguments(stage + ending, 15, 30)
    assert len(actual.splitlines()) == 1
    assert actual == pipeline.stage_arguments(stage, 15, 30)


@pytest.mark.parametrize('mode', ['valid', 'amr', 'periodic', 'missing-anomaly'])
def test_pipeline_checks_resolved_ini_configuration(monkeypatch, tmp_path, capsys, mode):
    import runpy
    code = Path(__file__).resolve().parents[1]
    script = code / 'bin/util_RIFT_pseudo_pipe.py'
    options = ['assume-eccentric=True']
    if mode != 'missing-anomaly':
        options += ['use-meanPerAno=True']
    if mode == 'amr':
        options += ['internal-use-amr=True']
    if mode == 'periodic':
        options += ['internal-cip-use-periodic-ecc-vars=True']
    if mode != 'valid':
        options += ['eccentric-phase-coordinates=True']
    ini = tmp_path / 'config.ini'
    ini.write_text('[rift-pseudo-pipe]\n' + '\n'.join(options))
    argv = [str(script), '--use-ini', str(ini)]
    if mode == 'valid':
        argv += ['--eccentric-phase-coordinates']
    monkeypatch.setattr(sys, 'argv', argv)
    boundary = next(i for i, line in enumerate(script.read_text().splitlines(), 1)
                    if line == "if opts.rf_transverse_spin_coordinates == 'physics3':")
    class Resolved(Exception):
        pass
    def trace(frame, event, arg):
        if frame.f_code.co_filename == str(script) and event == 'line' and frame.f_lineno == boundary:
            assert frame.f_globals['opts'].eccentric_phase_coordinates
            raise Resolved
        return trace
    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        if mode == 'valid':
            with pytest.raises(Resolved):
                runpy.run_path(str(script), run_name='__main__')
        else:
            with pytest.raises(SystemExit) as error:
                runpy.run_path(str(script), run_name='__main__')
            assert error.value.code == 2
            assert 'eccentric-phase' in capsys.readouterr().err
    finally:
        sys.settrace(previous)


def test_effective_puff_refuses_native_angle_reflection(tmp_path):
    code = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(code/'bin/util_ParameterPuffball.py'),
           '--inj-file', str(tmp_path/'absent.xml.gz'),
           '--parameter', 'eccentricity', '--parameter', 'meanPerAno',
           '--eccentric-phase-coordinates', '--reflect-parameter', 'meanPerAno',
           '--downselect-parameter', 'meanPerAno', '--downselect-parameter-range', '[0,0.5]']
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert done.returncode != 0
    message = done.stdout + done.stderr
    assert ('cannot reflect native meanPerAno' in message
            or 'unrecognized arguments: --reflect-parameter meanPerAno' in message)


def test_pipeline_refuses_quadratic_only_when_effective_fit_is_active():
    early = '--fit-method quadratic --use-eccentricity --use-meanPerAno --parameter eccentricity --parameter-nofit meanPerAno'
    assert pipeline.stage_arguments(early) == early
    fitting = early.replace('--parameter-nofit meanPerAno', '--parameter meanPerAno')
    with pytest.raises(ValueError, match='quadratic.*constrained circle'):
        pipeline.stage_arguments(fitting)
