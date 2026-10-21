"""Scientific measure, moving-intrinsic inverse, and sampling/export contracts."""
import argparse
import ast
import configparser
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from RIFT import lalsimutils
from RIFT.misc import eccentric_phase as ep, eccentric_phase_models as models
from RIFT.misc import eccentric_phase_pipeline as pipeline
from RIFT.misc.eccentric_phase_sampling import ANGLE, SamplingChart, validate_options
from test_eccentric_phase_models import map_record

ROOT = Path(__file__).resolve().parents[1]


def chart(alias='eccentricity', procedure='peters-full-e', bounds=(0., 2*np.pi), redshift=0.):
    names = ['m1', 'm2', 's1z', 's2z', alias, 'meanPerAno']
    return SamplingChart(names, lalsimutils.convert_waveform_coordinates,
        lambda x: np.ones_like(x)/(bounds[1]-bounds[0]), bounds, 15., 20.,
        procedure=procedure, map_data=models.encode_map(map_record()) if procedure == 'eob-tabulated' else None,
        source_redshift=redshift, enforce_kerr=True)


@pytest.mark.parametrize('alias', ep.ECCENTRICITY_NAMES)
@pytest.mark.parametrize('procedure', models.PROCEDURES)
def test_moving_parameters_redshift_inverse_and_row_isolation(alias, procedure):
    ch = chart(alias, procedure, redshift=.1)
    rng = np.random.default_rng(85)
    e = rng.uniform(.18, .24, 80)
    value = {'eccentricity':e, 'eccentricity_ln':np.log(e), 'eccentricity_squared':e*e}[alias]
    theta = np.r_[1e-9, 2*np.pi-1e-9, rng.uniform(0, 2*np.pi, 78)]
    x = np.c_[rng.uniform(18, 20, 80), rng.uniform(14, 16, 80),
              rng.uniform(.3, .7, 80), rng.uniform(.1, .3, 80), value, theta]
    original = x.copy()
    native, logj, valid = ch.to_native(x)
    assert valid.all() and ch.names[-1] == ANGLE
    recovered = ep.forward(1.1*x[:,0], 1.1*x[:,1], e, native[:,-1], 15, 20,
                           s1z=x[:,2], s2z=x[:,3], **ch.model)
    assert np.max(abs(np.angle(np.exp(1j*(recovered-theta))))) < 2e-11
    np.testing.assert_array_equal(x, original)
    if procedure != 'eob-tabulated':np.testing.assert_array_equal(logj, 0.)
    bad = x.copy();bad[0,4] = np.inf
    native, _, valid = ch.to_native(bad)
    assert not valid[0] and valid[1:].all() and np.isneginf(native[0]).all()


@pytest.mark.parametrize('procedure', models.PROCEDURES)
@pytest.mark.parametrize('bounds', [(0., 2*np.pi), (.4, 2.)])
def test_native_prior_normalization_and_angular_moment(procedure, bounds):
    ch = chart(procedure=procedure, bounds=bounds)
    n = 16384
    theta = (np.arange(n)+.5)*2*np.pi/n
    x = np.c_[np.full(n,20.), np.full(n,16.), np.full(n,.7), np.full(n,.2), np.full(n,.21), theta]
    native, logj, valid = ch.to_native(x)
    weights = np.exp(ch.log_correction(native, logj, valid))
    assert weights.mean() == pytest.approx(1., abs=7e-4)
    expected = (np.sin(bounds[1])-np.sin(bounds[0]))/(bounds[1]-bounds[0])
    assert np.mean(weights*np.cos(native[:,-1])) == pytest.approx(expected, abs=7e-4)
    if procedure == 'eob-tabulated':
        assert np.ptp(logj) > .01
        # A flat sheared prior without the inverse Jacobian fails this moment.
        if bounds[0] == 0:assert abs(np.cos(native[:,-1]).mean()) > .01


@pytest.mark.parametrize('log_space', [False, True])
@pytest.mark.parametrize('procedure', models.PROCEDURES)
def test_export_preserves_importance_weights_and_native_likelihood(log_space, procedure):
    ch = chart(procedure=procedure)
    n = 200
    x = np.c_[np.full(n,20.), np.full(n,16.), np.full(n,.7), np.full(n,.2),
              np.linspace(.19,.23,n), np.linspace(0,2*np.pi,n,endpoint=False)]
    native, logj, valid = ch.to_native(x)
    correction = ch.log_correction(native, logj, valid)
    lnL = -.5*((native[:,4]-.21)/.01)**2 + np.cos(native[:,-1])
    raw = dict(zip(ch.names, x.T))
    raw.update(log_integrand=lnL+correction,
               integrand=lnL+correction if log_space else np.exp(lnL+correction),
               log_joint_prior=np.full(n,-2.), log_joint_s_prior=np.full(n,-3.),
               joint_prior=np.full(n,np.exp(-2)), joint_s_prior=np.full(n,np.exp(-3)))
    log_weights = raw['log_integrand']+raw['log_joint_prior']-raw['log_joint_s_prior']
    ch.restore_samples(raw, log_space)
    assert ANGLE not in raw
    np.testing.assert_allclose(raw['meanPerAno'], native[:,-1])
    np.testing.assert_allclose(raw['log_integrand'], lnL, atol=1e-12)
    np.testing.assert_allclose(raw['integrand'], lnL if log_space else np.exp(lnL), atol=1e-12)
    np.testing.assert_allclose(raw['log_integrand']+raw['log_joint_prior']-raw['log_joint_s_prior'], log_weights, atol=1e-12)
    np.testing.assert_allclose(np.log(raw['joint_prior']), raw['log_joint_prior'])
    np.testing.assert_allclose(np.log(raw['joint_s_prior']), raw['log_joint_s_prior'])


@pytest.mark.parametrize('log_space',[False,True])
def test_invalid_retained_draw_has_zero_weight_without_nan(log_space):
    ch=chart()
    x=np.array([[20,16,.7,.2,np.inf,.1],[20,16,.7,.2,.21,.3]])
    raw=dict(zip(ch.names,x.T))
    raw.update(log_integrand=np.array([-np.inf,0.]),
               integrand=np.array([-np.inf,0.]) if log_space else np.array([0.,1.]),
               log_joint_prior=np.full(2,-2.),log_joint_s_prior=np.full(2,-3.),
               joint_prior=np.full(2,np.exp(-2)),joint_s_prior=np.full(2,np.exp(-3)))
    ch.restore_samples(raw,log_space)
    log_weights=raw['log_integrand']+raw['log_joint_prior']-raw['log_joint_s_prior']
    assert np.isneginf(log_weights[0]) and np.isfinite(log_weights[1])
    assert raw['joint_prior'][0]==0 and raw['joint_s_prior'][0]>0
    assert not np.isnan(np.exp(log_weights)).any()


def test_wrapped_narrow_band_and_partial_native_support():
    ch = chart(bounds=(.4,2.))
    theta = np.linspace(0,2*np.pi,2000,endpoint=False)
    x = np.c_[np.full(2000,20.), np.full(2000,16.), np.full(2000,.7), np.full(2000,.2),
              np.full(2000,.21),theta]
    native, logj, valid = ch.to_native(x)
    def native_fn(*columns):
        assert np.all(columns[-1] >= .4) and np.all(columns[-1] < 2.)
        return np.cos(columns[-1])
    value = ch.wrap(native_fn, log_space=True)(*x.T)
    mask = (native[:,-1]>=.4)&(native[:,-1]<2.)
    assert np.isneginf(value[~mask]).all()
    np.testing.assert_allclose(value[mask], np.cos(native[mask,-1])+ch.log_correction(native,logj,valid)[mask])
    # A seam-crossing Gaussian in the sheared angle remains resolved through inverse/forward.
    full = chart();n = 6000;rng = np.random.default_rng(19)
    draws = x[:1].repeat(n,axis=0);draws[:,-1]=rng.normal(0,.025,n)%(2*np.pi)
    physical,_,_=full.to_native(draws)
    recovered=ep.forward(20,16,.21,physical[:,-1],15,20)
    assert np.std(np.angle(np.exp(1j*recovered))) == pytest.approx(.025,abs=.001)


def test_helper_and_asimov_propagate_sampling_only_to_eligible_cip(tmp_path):
    liquid = pytest.importorskip('liquid')
    text=(ROOT/'RIFT/asimov/rift.ini').read_text();a=text.index('{% assign assume_size = likelihood');b=text.index('# Marginalisations',a)
    extra={'eccentric-phase-sampling':True,'eccentric-phase-reference-frequency':15,'eccentric-phase-target-frequency':20}
    rendered=liquid.Liquid(text[a:b],from_file=False).render(likelihood={'assume':['eccentric']},sampler={'extra eccentric arguments':extra})
    ini=configparser.RawConfigParser();ini.read_string('[rift-pseudo-pipe]\n'+rendered)
    parser=argparse.ArgumentParser();pipeline.add_arguments(parser);opts=parser.parse_args([])
    for k,v in ini['rift-pseudo-pipe'].items():
        if hasattr(opts,k.replace('-','_')):setattr(opts,k.replace('-','_'),ast.literal_eval(v))
    assert opts.eccentric_phase_sampling
    assert '--eccentric-phase-sampling' in pipeline.build_arguments(opts)
    stage='--use-eccentricity --use-meanPerAno --parameter eccentricity_ln --parameter meanPerAno'
    emitted=pipeline.stage_arguments(stage,15,20,sampling=True)
    assert '--eccentric-phase-sampling' in emitted and '--eccentric-phase-coordinates' in emitted
    assert ANGLE not in emitted  # external parameter declarations stay native
    early=stage.replace('--parameter meanPerAno','--parameter-nofit meanPerAno')
    assert pipeline.stage_arguments(early,15,20,sampling=True)==early
    puff=pipeline.puff_arguments('--parameter eccentricity_ln --parameter meanPerAno',15,20)
    assert '--eccentric-phase-sampling' not in puff


def test_unsupported_native_only_paths_fail_closed():
    defaults=dict(eccentric_phase_sampling=True,sampler_method='AV',meanPerAno_min=0.,meanPerAno_max=2*np.pi)
    validate_options(SimpleNamespace(**defaults))
    for bad in [dict(sampler_method='adaptive_cartesian_gpu'),dict(sampler_oracle=['RS']),
                dict(import_prior_dictionary_file='old.pkl'),dict(meanPerAno_max=9.)]:
        with pytest.raises(ValueError):validate_options(SimpleNamespace(**dict(defaults,**bad)))


def test_sampling_host_backend_gate_and_captured_device_defaults():
    """GPU-equipped workers must not pass device arrays to the host chart."""
    from types import ModuleType
    tree=ast.parse((ROOT/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py').read_text())
    block=next(n for n in tree.body if isinstance(n,ast.If)
               and 'opts.eccentric_phase_sampling' in ast.unparse(n.test)
               and 'opts.sampler_method' in ast.unparse(n.test))
    program=compile(ast.Module(body=[block],type_ignores=[]),'sampling-host-gate','exec')
    for enabled in [False, True]:
        fake_device=SimpleNamespace(__name__='cupy')
        av=ModuleType('fake_av')
        def forbidden(x):raise AssertionError('device converter reached')
        ns=dict(__name__=av.__name__,xpy_default=fake_device,identity_convert_togpu=forbidden,cupy_ok=True)
        exec('def draw(values,xpy=xpy_default):\n return identity_convert_togpu(xpy.asarray(values))\nclass MCSampler:\n pass\n',ns)
        av.draw=ns['draw'];av.MCSampler=ns['MCSampler'];av.xpy_default=fake_device
        exec(program,dict(opts=SimpleNamespace(eccentric_phase_sampling=enabled,sampler_method='AV'),mcsamplerAdaptiveVolume=av))
        if enabled:
            np.testing.assert_equal(av.draw([1.,2.]),[1.,2.])
            assert not ns['cupy_ok']
        else:assert av.xpy_default is fake_device and ns['cupy_ok']


def test_actual_cip_sampling_fit_evidence_and_native_xml(monkeypatch, tmp_path):
    """Recover a narrow seam-crossing posterior through the real CIP and AV."""
    import runpy
    import sys
    import lal
    from scipy.special import i0e
    rng=np.random.default_rng(528);n=800;kappa=400.
    mc=rng.uniform(15,17,n);eta=rng.uniform(.2,.249,n)
    total=mc/eta**.6;delta=np.sqrt(1-4*eta)
    m1=total*(1+delta)/2;m2=total*(1-delta)/2
    e=rng.uniform(.07,.14,n);theta=rng.uniform(0,2*np.pi,n)
    ell=ep.inverse(m1,m2,e,theta,15,20)
    table=np.c_[np.arange(n),m1,m2,np.zeros((n,6)),e,ell,
                kappa*(np.cos(theta)-1),np.full(n,.01),np.full(n,100),np.full(n,1000)]
    monkeypatch.chdir(tmp_path);np.savetxt('grid.dat',table)
    script=ROOT/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py'
    args=[str(script),'--fname','grid.dat','--fit-method','polynomial','--fit-order','1',
          '--no-plots','--use-eccentricity','--use-meanPerAno','--eccentric-phase-sampling',
          '--eccentric-phase-reference-frequency','15','--eccentric-phase-target-frequency','20',
          '--mc-range','[15,17]','--eta-range','[.2,.249]','--ecc-min','.07','--ecc-max','.14',
          '--eccentricity-prior','uniform','--sampler-method','AV','--n-chunk','10000',
          '--n-max','100000','--n-eff','1000','--n-output-samples','1000','--tripwire-fraction','2']
    for name in ['mc','eta','eccentricity','meanPerAno']:args+=['--parameter',name]
    monkeypatch.setattr(sys,'argv',args)
    state=np.random.get_state();np.random.seed(371)
    try:
        with pytest.raises(SystemExit) as exit_info:runpy.run_path(str(script),run_name='__main__')
    finally:np.random.set_state(state)
    assert exit_info.value.code == 0
    pts=lalsimutils.xml_to_ChooseWaveformParams_array('output-ILE-samples.xml.gz')
    assert len(pts)==1000
    transported=ep.forward(np.array([p.m1/lal.MSUN_SI for p in pts]),np.array([p.m2/lal.MSUN_SI for p in pts]),
                          np.array([p.eccentricity for p in pts]),np.array([p.meanPerAno for p in pts]),15,20)
    residual=np.angle(np.exp(1j*transported))
    assert np.std(residual)==pytest.approx(.05,abs=.012)
    assert abs(np.angle(np.exp(1j*transported).mean()))<.012
    annotation=np.loadtxt('integral_result+annotation.dat')
    assert annotation[0]==pytest.approx(np.log(i0e(kappa)),abs=.10)
    # Legacy CIP writes an oversupply of lnL rows; XML uses its first n_output rows.
    saved=np.loadtxt('output-ILE-samples_lnL.dat')[:len(pts)]
    np.testing.assert_allclose(saved,kappa*(np.cos(transported)-1),atol=.02)


def test_actual_generated_submit_files_route_sampling_to_cip_only(tmp_path):
    """Build a real DAG with a tiny manual grid; never submit or evaluate ILE."""
    import os
    import re
    import subprocess
    import sys
    import lal
    repo=ROOT.parents[1]
    text=(repo/'.travis/ref_ini/GW150914.ini').read_text()
    text=text.replace('[engine]','[engine]\necc_min=0.01\necc_max=0.3')
    text=re.sub(r'(?m)^internal-truncate-files-for-osg-file-transfer=.*$',
                'internal-truncate-files-for-osg-file-transfer=False',text)
    text=text.replace('ile-force-gpu=True','ile-force-gpu=False')
    text+='\nassume-eccentric=True\nuse-meanPerAno=True\neccentric-phase-sampling=True\n'
    text+='eccentric-phase-reference-frequency=15\neccentric-phase-target-frequency=30\n'
    text+='force-ecc-min=0.01\nforce-ecc-max=0.3\nile-no-gpu=True\n'
    ini=tmp_path/'sampling.ini';ini.write_text(text)
    p=lalsimutils.ChooseWaveformParams();p.m1=35*lal.MSUN_SI;p.m2=30*lal.MSUN_SI
    p.eccentricity=.1;p.meanPerAno=.2;p.fref=20;p.fmin=20
    lalsimutils.ChooseWaveformParams_array_to_xml([p,p.manual_copy()],fname=str(tmp_path/'seed'),fref=20)
    (tmp_path/'foo.cache').write_text('');run=tmp_path/'run'
    env=dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''),
             PATH=os.pathsep.join([str(ROOT/'bin'),os.path.dirname(sys.executable),os.environ.get('PATH','')]),
             OMP_NUM_THREADS='1',RIFT_LOWLATENCY='True',SINGULARITY_RIFT_IMAGE='foo',SINGULARITY_BASE_EXE_DIR='/usr/bin/')
    cmd=[sys.executable,str(ROOT/'bin/util_RIFT_pseudo_pipe.py'),'--use-ini',str(ini),
         '--use-coinc',str(repo/'.travis/ref_ini/coinc.xml'),'--use-rundir',str(run),
         '--fake-data-cache',str(tmp_path/'foo.cache'),'--manual-initial-grid',str(tmp_path/'seed.xml.gz'),'--skip-reproducibility']
    proc=subprocess.run(cmd,cwd=tmp_path,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=240)
    (tmp_path/'build.log').write_text(proc.stdout)
    assert proc.returncode==0,proc.stdout[-12000:]
    stages=(run/'args_cip_list.txt').read_text().splitlines()
    eligible=[line for line in stages if pipeline.eligible(line)]
    assert eligible and all(pipeline.SAMPLING in pipeline.option_tokens(line) for line in eligible)
    for line in stages:
        if not pipeline.eligible(line):assert pipeline.SAMPLING not in pipeline.option_tokens(line)
    cip=[];puff=[];ile=[]
    for path in run.rglob('*.sub'):
        content=path.read_text()
        if 'util_ConstructIntrinsicPosterior_GenericCoordinates' in content:cip.append(content)
        if path.name.upper().startswith('PUFF'):puff.append(content)
        # Container jobs name a wrapper executable rather than the ILE binary.
        if path.name.startswith('ILE'):ile.append(content)
    assert cip and any(pipeline.SAMPLING in line for line in cip)
    assert puff and all(pipeline.FLAG in line and pipeline.SAMPLING not in line for line in puff)
    assert ile and all(pipeline.SAMPLING not in line for line in ile)
    args=(run/'args_ile.txt').read_text()
    assert all(flag in args for flag in ['--time-marginalization','--vectorized','--gpu','--force-xpy'])
    assert list(run.glob('*.dag'))
