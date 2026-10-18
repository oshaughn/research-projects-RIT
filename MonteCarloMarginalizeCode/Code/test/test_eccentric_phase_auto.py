"""Automatic build input, cache, global inverse and public-Hamiltonian checks."""
import argparse,json
import numpy as np
import pytest
from RIFT.misc import eccentric_phase_build as build,eccentric_phase_models as models,eccentric_phase as ep

LINE='--use-eccentricity --use-meanPerAno --parameter eccentricity --parameter meanPerAno --mc-range [12,18] --eta-range [.15,.25] --mtot-range [25,50] --ecc-min .02 --ecc-max .4 --chi-max .8 --chi-small-max .6'
SEED=dict(m1=18.,m2=18.,s1z=.5,s2z=.5)
def synthetic(point,reference,target):
    return np.r_[.3+.01*point[0],np.arange(1,9)*.001],dict(synthetic=True)

def test_final_prior_overrides_redshift_and_zero_spin():
    p=build.prior_specification([LINE+' --mc-range [10,16] --ecc-max .3 --source-redshift .2'])
    assert p['mc_detector']==[12.,19.2] and p['e']==[.02,.3]
    assert p['mtot_detector']==[30.,60.] and p['s2z']==[-.5993999999999999,.5993999999999999]
    assert build.prior_specification([LINE],zero_spin=True)['s1z']==[0.,0.]
    assert build.prior_specification([LINE+' --ecc-min 0 --eccentricity-prior log_uniform'])['e'][0]==.001
    with pytest.raises(ValueError):build.interval("__import__('os').system('false')",'mc')

def test_generation_cache_and_global_inverse(tmp_path):
    priors=build.prior_specification([LINE]);cache=tmp_path/'cache.json'
    data,report=build.generate(SEED,priors,15.,20.,cache,evaluator=synthetic)
    assert report['state']=='generated' and report['validation_profiles']==5
    assert report['sampled_phase_error_rad']<1e-10
    def forbidden(*args):raise AssertionError('cache should avoid recomputation')
    again,reused=build.generate(SEED,priors,15.,20.,cache,evaluator=forbidden)
    assert data==again and reused['cache_reused']
    changed=dict(priors,e=[.03,.3]);_,fresh=build.generate(SEED,changed,15.,20.,cache,evaluator=synthetic)
    assert not fresh['cache_reused'] and fresh['signature_sha256']!=report['signature_sha256']
    saved=json.loads(cache.read_text());saved['record']['center'][0]+=1;cache.write_text(json.dumps(saved))
    with pytest.raises(ValueError):build.generate(SEED,changed,15.,20.,cache,evaluator=synthetic)
    rng=np.random.default_rng(752);m1,m2=rng.uniform(3,100,(2,300));e=rng.uniform(0,.98,300);a=rng.uniform(-10,20,300)
    kw=dict(procedure='eob-tabulated',map_data=data,s1z=.5,s2z=.5)
    theta=ep.forward(m1,m2,e,a,15.,20.,**kw)
    back=ep.inverse(m1,m2,e,theta,15.,20.,**kw)
    assert np.max(abs(np.angle(np.exp(1j*(back-a)))))<3e-11
    assert ep.forward(18,18,.2,.4,15,20,**kw)!=ep.forward(18,18,.2,.4,15,20)

def test_identity_and_unavailable_chart_do_not_evaluate():
    priors=build.prior_specification([LINE])
    def forbidden(*args):raise AssertionError('identity must skip EOB')
    _,report=build.generate(SEED,priors,15.,15.,None,evaluator=forbidden)
    assert report['state']=='identity-frequency-fallback' and report['attempts']==0
    _,report=build.generate(SEED,dict(priors,e=[0.,.001]),15.,20.,None,evaluator=forbidden)
    assert report['state']=='peters-no-EOB-chart' and report['attempts']==0

def test_no_returning_orbits_has_explicit_peters_fallback():
    from RIFT.misc.eccentric_phase_eob.radial_core import DomainError
    def unavailable(*args):raise DomainError('no returning orbit')
    data,report=build.generate(SEED,build.prior_specification([LINE]),15.,20.,None,evaluator=unavailable)
    assert report['successful_profiles']==0 and report['failed_profiles']<=build.MAX_PROFILES
    assert ep.forward(18,18,.2,.4,15,20,procedure='eob-tabulated',map_data=data)==pytest.approx(ep.forward(18,18,.2,.4,15,20))

@pytest.mark.parametrize('spins',[(.5,.5),(.8,-.2),(-.4,-.7),(0.,0.)])
def test_public_spin_hamiltonian_derivatives_and_label_exchange(spins):
    from RIFT.misc.eccentric_phase_eob.aligned_spin import AlignedSpinEOB
    eta=.22;d=np.sqrt(1-4*eta);model=AlignedSpinEOB(eta,*spins);swapped=AlignedSpinEOB(eta,*spins[::-1],delta=-d)
    r,p,j,h=12.,.1,4.,1e-5;H=model.hamiltonian
    for func,expected in [(model.h_r,(H(r+h,p,j)-H(r-h,p,j))/(2*h)),(model.h_prstar,(H(r,p+h,j)-H(r,p-h,j))/(2*h)),(model.omega,(H(r,p,j+h)-H(r,p,j-h))/(2*h))]:
        assert func(r,p,j)==pytest.approx(expected,rel=2e-7)
    assert model.momentum_from_energy(r,H(r,p,j),j,1)==pytest.approx(p,abs=1e-11)
    assert H(r,p,j)==pytest.approx(swapped.hamiltonian(r,p,j),abs=1e-14)
    np.testing.assert_allclose(model.forces(r,p,j),swapped.forces(r,p,j),rtol=1e-13,atol=1e-15)

def test_cache_signature_covers_peters_baseline_and_libraries():
    keys=build.sources()
    assert {'eccentric_phase','eccentric_phase_models','numpy_version','scipy_version','cumulative_simpson'}<=set(keys)

def test_map_taper_ends_at_sampled_box_and_runtime_errors_fall_back():
    priors=build.prior_specification([LINE])
    data,report=build.generate(SEED,priors,15.,20.,None,evaluator=synthetic)
    record=models.load_map(data)[0];bounds=np.array(report['local_domain'])
    active=bounds[1]>bounds[0]
    center=np.array(record['center'])[active]
    np.testing.assert_allclose(center+.75*(np.array(record['bounds'])[:,active]-center),bounds[:,active],atol=1e-12)
    def stalled(*args):raise RuntimeError('brentq failed to converge')
    _,report=build.generate(SEED,priors,15.,20.,None,evaluator=stalled)
    assert report['successful_profiles']==0 and report['failed_profiles']>0

def test_initial_orbit_survives_a_bracket_end_outside_the_bound_domain():
    from RIFT.misc.eccentric_phase_eob.aligned_spin import AlignedSpinEOB
    from RIFT.misc.eccentric_phase_eob.public_transport import initial_orbit
    w0=np.pi*100*ep.MTSUN*20;guess=(1-.85**2)/w0**(2/3)
    p,_=initial_orbit(.2,.85,w0,n=32,model_factory=lambda nu:AlignedSpinEOB(nu,.9,.9))
    assert 1.1<p/guess<1.3
