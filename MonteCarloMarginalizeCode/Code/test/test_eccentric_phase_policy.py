"""Mass policy and actual Asimov -> helper -> saved CIP/puff build contract."""
import argparse
import ast
import configparser
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from RIFT.misc import eccentric_phase_policy as policy
from RIFT.misc import eccentric_phase_pipeline as pipeline

ROOT=Path(__file__).resolve().parents[1]
ILE='X --fmin-template 10 --reference-freq 10 --fmin-ifo H1=20 --fmin-ifo L1=20 --fmax 512'


def options(**kw):
    parser=argparse.ArgumentParser();pipeline.add_arguments(parser)
    result=parser.parse_args(['--eccentric-phase-coordinates'])
    for key,val in kw.items():setattr(result,key,val)
    return result


@pytest.mark.parametrize('mass,expected,reason',[(10,40,'lvk-ceiling-for-low-mass'),(36,20.0698,'none'),(70,10,'mass-landmark-below-band-identity'),(200,10,'mass-landmark-below-band-identity')])
def test_mass_scaling_and_clips(mass,expected,reason):
    record=policy.mass_anchor(mass,10)
    assert record['target_frequency_hz']==pytest.approx(expected,rel=2e-4)
    assert record['fallback']==reason
    assert record['raw_mass_frequency_hz']*mass==pytest.approx(policy.ANCHOR_X**1.5/(policy.MTSUN*3.141592653589793))


def test_high_mass_at_native_band_edge_falls_back_to_identity():
    r=policy.mass_anchor(200,20)
    assert r['target_frequency_hz']==20 and r['identity_fallback']
    assert r['fallback']=='mass-landmark-below-band-identity'
    assert policy.mass_anchor(200,35)['target_frequency_hz']==35


@pytest.mark.parametrize('mass',[None,0,-1,float('nan'),float('inf')])
def test_bad_mass_fails_without_changing_native_inputs(mass):
    with pytest.raises((ValueError,TypeError)):policy.mass_anchor(mass,10)


def test_no_fake_event_mass_and_explicit_target_override():
    opts=options()
    with pytest.raises(ValueError,match='placeholder'):
        policy.resolve(opts,{'m1':50,'m2':50,'rf_mass_is_placeholder':True},ILE,'SEOBNRv5EHM')
    opts=options(eccentric_phase_target_frequency=30)
    r=policy.resolve(opts,{},ILE,'SEOBNRv5EHM')
    assert r['policy_version']=='explicit-v1' and opts.eccentric_phase_target_frequency==30
    assert not r['waveform_frequency_change'] and not r['physical_prior_change']
    hinted=options(eccentric_phase_detector_mass=36)
    assert policy.resolve(hinted,{},ILE,'SEOBNRv5EHM')['mass_source']=='explicit-detector-total-mass'


def test_native_epoch_and_late_override_guards():
    different=ILE.replace('--reference-freq 10','--reference-freq 20')
    r=policy.resolve(options(),{'m1':18,'m2':18},different,'SEOBNRv5EHM')
    assert r['reference_frequency_hz']==10 and r['waveform_reference_hz']==20
    with pytest.raises(ValueError,match='Ambiguous'):
        policy.resolve(options(),{'m1':18,'m2':18},different,'TEOBResumSDALI')
    explicit=options(eccentric_phase_reference_frequency=10)
    assert policy.resolve(explicit,{'m1':18,'m2':18},different,'TEOBResumSDALI')['native_epoch_source']=='explicit'
    with pytest.raises(ValueError,match='override'):
        policy.check_final_ile(r,different+' --fmin-template 15')
    policy.check_final_ile(r,different+' --fmax 1024')
    with pytest.raises(ValueError,match='No LVK'):
        policy.mass_anchor(36,10,20,15)


def test_build_receipt_freezes_one_value_for_both_stages(tmp_path):
    event={'m1':18.,'m2':18.};saved=event.copy();opts=options()
    r=policy.resolve(opts,event,ILE,'SEOBNRv5EHM');assert event==saved
    path=tmp_path/policy.RECEIPT;policy.write(r,path)
    outer=options();policy.read_into(outer,path)
    target=outer.eccentric_phase_target_frequency;ref=outer.eccentric_phase_reference_frequency
    stage='--fit-method rf --use-eccentricity --use-meanPerAno --parameter eccentricity_ln --parameter meanPerAno\n'
    cip=pipeline.stage_arguments(stage,ref,target)
    puff=pipeline.puff_arguments('--parameter eccentricity_ln --parameter meanPerAno --parameter mc',ref,target)
    assert '\n' not in cip
    assert str(target) in cip and str(target) in puff
    assert 'lvk-mass' not in cip+puff
    early=stage.replace('--parameter meanPerAno','--parameter-nofit meanPerAno')
    assert pipeline.stage_arguments(early,ref,target)==early
    with pytest.raises(ValueError,match='conflicts'):
        policy.read_into(options(eccentric_phase_target_frequency=40),path)


@pytest.mark.parametrize('extra',[{'eccentric-phase-coordinates':True},{'eccentric-phase-coordinates':True,'eccentric-phase-target-frequency':30},{'eccentric-phase-coordinates':True,'eccentric-phase-frequency-policy':'fixed'}])
def test_actual_asimov_render_defaults_and_overrides(extra,tmp_path):
    liquid=pytest.importorskip('liquid')
    text=(ROOT/'RIFT/asimov/rift.ini').read_text();a=text.index('{% assign assume_size = likelihood');b=text.index('# Marginalisations',a)
    rendered=liquid.Liquid(text[a:b],from_file=False).render(likelihood={'assume':['eccentric']},sampler={'extra eccentric arguments':extra})
    ini=configparser.RawConfigParser();ini.optionxform=str;ini.read_string('[rift-pseudo-pipe]\n'+rendered)
    opts=options()
    for k,v in ini['rift-pseudo-pipe'].items():
        key=k.replace('-','_')
        if hasattr(opts,key):setattr(opts,key,ast.literal_eval(v))
    pipeline.validate_request(opts)
    assert opts.eccentric_phase_frequency_policy==extra.get('eccentric-phase-frequency-policy','lvk-mass')
    helper_args=pipeline.build_arguments(opts)
    parser=argparse.ArgumentParser();pipeline.add_arguments(parser)
    import shlex
    helper=parser.parse_args(shlex.split(helper_args))
    r=policy.resolve(helper,{'m1':18,'m2':18},ILE,'SEOBNRv5EHM')
    expected=extra.get('eccentric-phase-target-frequency',20 if extra.get('eccentric-phase-frequency-policy')=='fixed' else policy.mass_anchor(36,10)['target_frequency_hz'])
    assert r['target_frequency_hz']==pytest.approx(expected)
    path=tmp_path/policy.RECEIPT;policy.write(r,path);policy.read_into(opts,path)
    assert opts.eccentric_phase_target_frequency==pytest.approx(expected)


def test_actual_build_hooks_execute_policy_not_only_accept_options(tmp_path):
    # Execute the real helper's opt-in block after its ILE/mass setup.
    tree=ast.parse((ROOT/'bin/helper_LDG_Events.py').read_text())
    block=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=='opts.eccentric_phase_coordinates' and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='phase_decision' for t in x.targets) for x in n.body))
    stage='--fit-method rf --use-eccentricity --use-meanPerAno --parameter eccentricity_ln --parameter meanPerAno'
    ns=dict(opts=options(),event_dict={'m1':18.,'m2':18.},helper_ile_args=ILE,helper_cip_args=stage,helper_cip_arg_list=[stage],helper_puff_args='--parameter eccentricity_ln --parameter meanPerAno',eccentric_phase_pipeline=pipeline,use_ini=False,approx_str=None)
    ns['opts'].lowlatency_propose_approximant=False
    import os
    before=os.getcwd()
    try:
        os.chdir(tmp_path);exec(compile(ast.Module(body=[block],type_ignores=[]),'real-helper-hook','exec'),ns)
    finally:os.chdir(before)
    r=json.loads((tmp_path/policy.RECEIPT).read_text())
    assert r['target_frequency_hz']==pytest.approx(policy.mass_anchor(36,10)['target_frequency_hz'])
    assert '--eccentric-phase-target-frequency' in ns['helper_cip_arg_list'][0]
    assert '--eccentric-phase-target-frequency' in ns['helper_puff_args']
    parent=(ROOT/'bin/util_RIFT_pseudo_pipe.py').read_text()
    assert parent.index('eccentric_phase_policy.read_into(opts)')<parent.index('# Final CIP lines')
    assert parent.index('eccentric_phase_policy.check_final_ile')<parent.index("with open('args_ile.txt'")


def test_ini_policy_is_revalidated_before_shell_propagation():
    with pytest.raises(ValueError,match='Unknown'):
        pipeline.build_arguments(options(eccentric_phase_frequency_policy='unsupported-policy'))


def test_high_mass_fallback_is_exact_identity_in_the_real_coordinate():
    from RIFT.misc import eccentric_phase
    record=policy.mass_anchor(120,10)
    theta=eccentric_phase.forward(60.,60.,.1,1.2,10,record['target_frequency_hz'])
    assert float(theta)==pytest.approx(1.2,abs=1e-14)
    assert record['achieved_x']>record['anchor_x']  # no PN transport is applied here


def test_network_floor_and_analysis_ceiling_are_respected():
    assert policy.mass_anchor(10,10,20,35)['target_frequency_hz']==35
    r=policy.resolve(options(),{'m1':18,'m2':18},ILE.replace('H1=20','H1=30').replace('L1=20','L1=40'),'SEOBNRv5EHM')
    assert r['allowed_band_hz']==[30,40] and r['identity_fallback']


def test_late_detector_band_changes_cannot_bypass_the_frozen_policy():
    r=policy.resolve(options(),{'m1':18,'m2':18},ILE,'SEOBNRv5EHM')
    with pytest.raises(ValueError,match='detector-band override'):
        policy.check_final_ile(r,ILE+' --fmin-ifo H1=30 --fmin-ifo L1=30')
    with pytest.raises(ValueError,match='detector-band override'):
        policy.check_final_ile(r,ILE+' --fmax 15')


def test_last_detector_override_sets_the_actual_network_floor():
    r=policy.resolve(options(),{'m1':18,'m2':18},ILE+' --fmin-ifo H1=30 --fmin-ifo=L1=40','SEOBNRv5EHM')
    assert r['allowed_band_hz']==[30,40] and r['identity_fallback']
