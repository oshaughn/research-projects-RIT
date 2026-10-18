"""Coordinate acceptance: inverse, seams, spin association, native priors, wiring."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import pytest
from RIFT.misc import eccentric_phase as ep,eccentric_phase_models as models,eccentric_phase_pipeline as pipe


def map_record():
    return dict(schema='rift-eccentric-map-v1',features=['M','delta','e','chi_eff','chi_a'],
        center=[36,.1,.21,.5,.1],scale=[10,.2,.2,.5,.5],bounds=[[15,0,.001,-1,-1],[80,.9,.9,1,1]],
        powers=[[0,0,0,0,0],[0,0,0,1,0],[0,0,0,0,1]],
        coefficients=[[100,.03,-.01],[4,.01,0],[2,0,.01]],reference_frequency=15.,target_frequency=20.,
        provenance={'source':'synthetic map exercises coordinate contract, not physical validation'})


@pytest.mark.parametrize('procedure',models.PROCEDURES)
def test_roundtrip_spins_seams_scalars_and_batches(procedure):
    rng=np.random.default_rng(8301);m1=rng.uniform(8,50,80);m2=rng.uniform(5,30,80)
    e=rng.uniform(0,.95,80);angle=rng.uniform(-20,20,80);s1,s2=rng.uniform(-1,1,(2,80))
    options=dict(procedure=procedure,s1z=s1,s2z=s2)
    if procedure=='eob-tabulated':options['map_data']=models.encode_map(map_record())
    theta=ep.forward(m1,m2,e,angle,15,20,**options)
    recovered=ep.inverse(m1,m2,e,theta,15,20,**options)
    assert abs(np.angle(np.exp(1j*(recovered-angle)))).max()<2e-11
    swapped={**options,'s1z':s2,'s2z':s1}
    np.testing.assert_allclose(theta,ep.forward(m2,m1,e,angle,15,20,**swapped),atol=1e-11)
    np.testing.assert_allclose(ep.forward(m1,m2,e,angle,15,15,**options),angle%(2*np.pi),atol=1e-14)
    scalar={k:(v[0] if isinstance(v,np.ndarray) else v) for k,v in options.items()}
    assert ep.forward(m1[0],m2[0],e[0],angle[0],15,20,**scalar)==pytest.approx(theta[0],abs=1e-11)
    np.testing.assert_allclose(ep.features(m1,m2,e,angle,15,20,**options),ep.features(m1,m2,e,angle+2*np.pi,15,20,**options),atol=1e-11)


def test_full_e_against_ode_and_small_e_limit():
    from scipy.integrate import solve_ivp
    mass=36.;eta=.25;x0=(np.pi*mass*ep.MTSUN*15)**(2/3);length=(2/3)*np.log(20/15)
    for e in [0.,1e-4,.21,.8,.97]:
        def rhs(t,y):
            ee=y[0];e2=ee*ee;x=x0*np.exp(t)
            return [-(19/12)*ee*(1-e2)*(1+121/304*e2)/(1+73/24*e2+37/96*e2**2),5/(64*eta)*x**(-2.5)*(1-e2)**3.5/(1+73/24*e2+37/96*e2**2)]
        sol=solve_ivp(rhs,[0,length],[e,0],rtol=1e-11,atol=1e-12)
        actual=models.full_e_advance(18,18,e,15,20)
        assert actual==pytest.approx(sol.y[1,-1],rel=2e-9,abs=1e-9)
    assert abs(models.full_e_advance(18,18,1e-4,15,20)-ep.phase_advance(18,18,1e-4,15,20))<1e-9


def test_map_guard_invertible_even_with_bad_harmonics_and_global_exterior():
    record=map_record();record['coefficients'][0][1:]=[30.,-10.]
    data=models.encode_map(record);angle=np.linspace(0,2*np.pi,1001)
    theta=np.unwrap(ep.forward(20,16,.21,angle,15,20,procedure='eob-tabulated',map_data=data,s1z=.7,s2z=.25))
    assert np.min(np.diff(theta)/np.diff(angle))>=.2-1e-9
    assert ep.forward(100,80,.21,.4,15,20,procedure='eob-tabulated',map_data=data)==pytest.approx(ep.forward(100,80,.21,.4,15,20,procedure='peters-full-e'),abs=1e-11)
    assert ep.forward(20,16,.21,.4,15,20,procedure='eob-tabulated',map_data=data,s1z=.7,s2z=.2)!=ep.forward(20,16,.21,.4,15,20,procedure='eob-tabulated',map_data=data,s1z=.2,s2z=.7)


def test_freeze_and_emission_conflicts(tmp_path):
    from RIFT.misc import eccentric_phase_policy as policy
    parser=argparse.ArgumentParser();pipe.add_arguments(parser)
    path=tmp_path/'map with spaces.json';path.write_text(json.dumps(map_record()))
    opts=parser.parse_args(['--eccentric-phase-procedure','eob-tabulated','--eccentric-phase-map',str(path)])
    frozen=models.freeze(opts,15,20);assert frozen['coordinate_map_sha256']
    # Worker args contain the immutable data, requiring no map-file transfer.
    line=pipe.stage_arguments('--use-eccentricity --use-meanPerAno --parameter eccentricity --parameter meanPerAno\n',15,20,**models.worker_options(opts))
    tokens=pipe.option_tokens(line);assert '--eccentric-phase-map-data' in tokens
    assert str(path) not in line and '\n' not in line
    assert pipe.puff_arguments('--parameter eccentricity --parameter meanPerAno',15,20,**models.worker_options(opts)).endswith(opts.eccentric_phase_map_data)
    with pytest.raises(ValueError):pipe.stage_arguments(line,15,20,procedure='peters-full-e')
    with pytest.raises(ValueError):models.freeze(opts,15,21)  # duplicate file/data refused
    opts.eccentric_phase_map=None
    with pytest.raises(ValueError):models.freeze(opts,15,21)  # endpoint conflict refused
    assert policy.mass_anchor(120,15)['identity_fallback']


def test_vector_projection_and_puff_use_proposed_component_spins():
    lal=pytest.importorskip('lal');from RIFT import lalsimutils
    from RIFT.misc.eccentric_phase_puff import native_proposals
    opts=dict(procedure='eob-tabulated',map_data=models.encode_map(map_record()))
    names=['m1','m2','s1z','s2z','eccentricity','meanPerAno']
    x=np.array([[20.,16.,.7,.2,.21,.4],[21.,17.,.3,.6,.22,1.2]])
    fit=ep.fitting_names(names,names)
    result=ep.convert(x,fit,names,lalsimutils.convert_waveform_coordinates,reference_frequency=15,target_frequency=20,**opts)
    np.testing.assert_allclose(result[:,-2:],ep.features(x[:,0],x[:,1],x[:,4],x[:,5],15,20,s1z=x[:,2],s2z=x[:,3],**opts),atol=1e-11)
    p=lalsimutils.ChooseWaveformParams();p.m1=18*lal.MSUN_SI;p.m2=18*lal.MSUN_SI;p.eccentricity=.2
    proposals=x.copy();proposals[:,5]=[.7,1.3]
    new,physical=native_proposals([p,p],proposals,names,15,20,**opts)
    np.testing.assert_allclose(ep.forward(physical[:,0],physical[:,1],physical[:,4],physical[:,5],15,20,s1z=physical[:,2],s2z=physical[:,3],**opts),proposals[:,5],atol=1e-11)
    bad=proposals.copy();bad[0,2]=np.nan
    _,physical=native_proposals([p,p],bad,names,15,20,**opts)
    assert np.isnan(physical[0,5]) and np.isfinite(physical[1,5])


@pytest.mark.parametrize('mutation',[lambda r:r.update(schema='unknown'),lambda r:r.update(scale=[0]*5),lambda r:r.update(powers=[[1.5,0,0,0,0]]),lambda r:r.update(provenance={})])
def test_invalid_map_refused(mutation):
    record=copy.deepcopy(map_record());mutation(record)
    with pytest.raises(ValueError):models.encode_map(record)


def test_asimov_preserves_model_selection_and_map_path():
    import ast,configparser
    liquid=pytest.importorskip('liquid')
    root=Path(__file__).resolve().parents[1]
    text=(root/'RIFT/asimov/rift.ini').read_text();a=text.index('{% assign assume_size = likelihood');b=text.index('# Marginalisations',a)
    extra={'eccentric-phase-coordinates':True,'eccentric-phase-procedure':'eob-tabulated','eccentric-phase-map':'/tmp/map with spaces.json'}
    rendered=liquid.Liquid(text[a:b],from_file=False).render(likelihood={'assume':['eccentric']},sampler={'extra eccentric arguments':extra})
    ini=configparser.RawConfigParser();ini.read_string('[rift-pseudo-pipe]\n'+rendered)
    assert ast.literal_eval(ini['rift-pseudo-pipe']['eccentric-phase-procedure'])=='eob-tabulated'
    assert ast.literal_eval(ini['rift-pseudo-pipe']['eccentric-phase-map'])==extra['eccentric-phase-map']
    assert ast.literal_eval(ini['rift-pseudo-pipe']['eccentric-phase-frequency-policy'])=='lvk-mass'


def test_full_e_is_default_and_no_event_map_is_packaged():
    parser=argparse.ArgumentParser();pipe.add_arguments(parser)
    assert parser.parse_args([]).eccentric_phase_procedure=='peters-full-e'
    assert models.worker_options(argparse.Namespace())['procedure']=='peters-full-e'
    assert not list((Path(models.__file__).parent/'eccentric_phase_maps').glob('*.json'))
    with pytest.raises(ValueError):models.map_path('builtin:danilo-public-spin-v1')


@pytest.mark.parametrize('payload',[[1,2],dict(schema='rift-eccentric-map-v1')])
def test_incomplete_map_payload_is_a_value_error(payload):
    import base64
    data=base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    with pytest.raises(ValueError):models.load_map(data)
