"""Deployment contracts for redundant L-frame RF features; no likelihood calls."""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rf_transverse_spin', ROOT/'RIFT/misc/rf_transverse_spin.py')
rf=importlib.util.module_from_spec(spec);spec.loader.exec_module(rf)
FULL='3 --fit-method rf --use-precessing --parameter mc --parameter delta_mc --parameter-implied mu1 --parameter-implied mu2 --parameter-implied chiMinus --parameter-implied s1x --parameter-implied s1y --parameter-implied s2x --parameter-implied s2y --parameter-nofit chi1 --parameter-nofit chi2 --parameter-nofit cos_theta1 --parameter-nofit cos_theta2 --parameter-nofit phi1 --parameter-nofit phi2'

@pytest.mark.parametrize('mc',[None,float('nan'),float('inf'),-1,0,20,21,'not a mass'])
def test_auto_unknown_and_boundary_off(mc):
    assert rf.stage_arguments(FULL,'auto',mc,True,20)==FULL

@pytest.mark.parametrize('mc',[1,19.999,'10'])
def test_auto_lowmass_fullstage(mc):
    line=rf.stage_arguments(FULL,'auto',mc,True,40)
    assert line==FULL+' --rf-transverse-spin-coordinates physics3 --fref 40.0'
    assert '--parameter-nofit phi2' in line

@pytest.mark.parametrize('line',[FULL.replace('fit-method rf','fit-method gp'),FULL.replace('fit-method rf','fit-method quadratic'),FULL.replace('--parameter-implied s2y','--parameter-nofit s2y')])
def test_reduced_or_other_interpolator_unchanged(line):
    assert rf.stage_arguments(line,'physics3',10,True,20)==line

def test_override_and_applicability():
    assert rf.stage_arguments(FULL,'off',10,True,20)==FULL
    assert rf.stage_arguments(FULL,'auto',10,False,20)==FULL
    with pytest.raises(ValueError):rf.stage_arguments(FULL,'physics3',10,False,20)
    assert 'physics3' in rf.stage_arguments(FULL,'physics3',25,True,20)

def test_frequency_replaces_existing_without_mixing_sampling():
    out=rf.stage_arguments(FULL+' --fref 11','physics3',10,True,30)
    assert out.count('--fref')==1 and '--fref 30.0' in out
    with pytest.raises(ValueError):rf.stage_arguments(FULL,'physics3',10,True,float('nan'))

def test_exact_frozen_scalar_formulas_and_zero_axis():
    rng=np.random.default_rng(131)
    m1=rng.uniform(10,50,500);m2=m1*rng.uniform(.1,1,500)
    s1=rng.uniform(-.5,.5,(500,3));s2=rng.uniform(-.5,.5,(500,3))
    x=rf.scalar_features(m1,m2,s1,s2,20)
    q=m2/m1;eta=q/(1+q)**2;v=(np.pi*(m1+m2)*rf.MTSUN*20)**(1/3)
    w1=1/(1+q)**2;w2=q*q*w1;L=eta/v;D=L+w1*s1[:,2]+w2*s2[:,2]
    T=w1[:,None]*s1[:,:2]+w2[:,None]*s2[:,:2];G=(2+1.5*q)[:,None]*w1[:,None]*s1[:,:2]+(2+1.5/q)[:,None]*w2[:,None]*s2[:,:2]
    T2=(T*T).sum(axis=1);J=np.sqrt(D*D+T2)
    expected=np.stack([T2/(D*D+(.1*L)**2),T2/(J+D)/L,(G*G).sum(axis=1)/(eta*v*v)**2],axis=1)
    np.testing.assert_allclose(x,expected,rtol=1e-14,atol=1e-15)
    np.testing.assert_array_equal(rf.scalar_features(m1,m2,np.zeros_like(s1),np.zeros_like(s2)),0)
    assert np.isfinite(x).all()
    assert not np.array_equal(x,rf.scalar_features(m1,m2,s1,s2,40))

def test_vector_conversion_native_columns_preserved():
    physical=np.array([[20,10,.3,.2,-.1,.1,-.2,.4],[20,20,0,0,.2,0,0,.3]])
    names=['m1','m2','s1x','s1y','s1z','s2x','s2y','s2z']
    def converter(x,coord_names,low_level_coord_names,**kw):
        return x[:,[low_level_coord_names.index(n) for n in coord_names]]
    out=rf.convert(physical,names+list(rf.FEATURE_NAMES),names,35,converter)
    np.testing.assert_array_equal(out[:,:8],physical)
    np.testing.assert_array_equal(out[:,8:],rf.scalar_features(physical[:,0],physical[:,1],physical[:,2:5],physical[:,5:8],35))

def test_cli_and_pipeline_forwarding_source_contract():
    for file in ['helper_LDG_Events.py','util_RIFT_pseudo_pipe.py','util_ConstructIntrinsicPosterior_GenericCoordinates.py']:
        src=(ROOT/'bin'/file).read_text();ast.parse(src)
        assert '"--rf-transverse-spin-coordinates"' in src
    pseudo=(ROOT/'bin/util_RIFT_pseudo_pipe.py').read_text()
    assert "cmd += ' --rf-transverse-spin-coordinates {} '" in pseudo
    helper=(ROOT/'bin/helper_LDG_Events.py').read_text()
    assert helper.index('stage_arguments(line')<helper.index('with open("helper_cip_arg_list.txt"')
    cip=(ROOT/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py').read_text()
    assert 'opts.fit_load_gp' in cip[cip.index('if opts.rf_transverse_spin_coordinates:'):cip.index('# SANITY COMPATIBILITY CHECK')]
    assert 'extract_fit_param(P_list[indx_line], coord_names[indx])' in cip
    template=(ROOT/'RIFT/asimov/rift.ini').read_text()
    assert "['cip'] contains 'transverse spin coordinates'" in template
    assert "['transverse spin coordinates'] == false" in template

def test_component_mirror_is_symmetric():
    s1=np.array([.2,.3,-.5]);s2=np.array([-.3,.1,.4])
    np.testing.assert_allclose(rf.scalar_features(30,10,s1,s2),rf.scalar_features(10,30,s2,s1),rtol=2e-15)

def test_invalid_physical_rows_do_not_abort_valid_batch():
    physical=np.array([[20,10,.3,.2,-.1,.1,-.2,.4],[-np.inf]*8])
    names=['m1','m2','s1x','s1y','s1z','s2x','s2y','s2z']
    def converter(x,coord_names,low_level_coord_names,**kw):
        return x[:,[low_level_coord_names.index(n) for n in coord_names]]
    out=rf.convert(physical,names+list(rf.FEATURE_NAMES),names,20,converter,enforce_kerr=True)
    assert np.isfinite(out[0]).all() and np.isneginf(out[1]).all()

def test_no_command_before_initialization():
    src=(ROOT/'bin/util_RIFT_pseudo_pipe.py').read_text()
    assert src.index("cmd = \" helper_LDG_Events.py")<src.index("cmd += ' --rf-transverse-spin-coordinates")

@pytest.mark.parametrize('frequency',[float('nan'),float('inf'),0,-1])
def test_scalar_frequency_rejects_nonphysical_values(frequency):
    with pytest.raises(ValueError):rf.scalar_features(20,10,np.zeros(3),np.zeros(3),frequency)

def test_actual_native_spherical_conversion_and_source_redshift():
    # LAL is optional for lightweight CI; deployment validation runs this on real LAL.
    pytest.importorskip('lal')
    import sys
    sys.path.insert(0,str(ROOT))
    from RIFT import lalsimutils
    names=['mc','delta_mc','chi1','chi2','cos_theta1','cos_theta2','phi1','phi2']
    low=np.array([[10,.2,.5,.7,-.2,.4,1,2],[15,.1,0,0,1,-1,0,0]])
    physical_names=['m1','m2','s1x','s1y','s1z','s2x','s2y','s2z']
    for redshift in [0,.4]:
        physical=lalsimutils.convert_waveform_coordinates(low,coord_names=physical_names,low_level_coord_names=names,source_redshift=redshift)
        out=rf.convert(low,physical_names+list(rf.FEATURE_NAMES),names,35,lalsimutils.convert_waveform_coordinates,source_redshift=redshift)
        np.testing.assert_allclose(out[:,:8],physical,rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(out[:,8:],rf.scalar_features(physical[:,0],physical[:,1],physical[:,2:5],physical[:,5:8],35),rtol=1e-14,atol=1e-14)
        import lal
        p=lalsimutils.ChooseWaveformParams();p.fref=35;p.m1=physical[0,0]*lal.MSUN_SI;p.m2=physical[0,1]*lal.MSUN_SI
        p.s1x,p.s1y,p.s1z=physical[0,2:5];p.s2x,p.s2y,p.s2z=physical[0,5:8]
        np.testing.assert_allclose([rf.extract(p,n) for n in rf.FEATURE_NAMES],out[0,8:],rtol=1e-14)

def test_bool_mass_does_not_activate_auto():
    assert rf.stage_arguments(FULL,'auto',True,True,20)==FULL

def test_actual_native_kerr_rejection_mixed_batch():
    pytest.importorskip('lal')
    import sys
    sys.path.insert(0,str(ROOT))
    from RIFT import lalsimutils
    names=['mc','delta_mc','chi1','chi2','cos_theta1','cos_theta2','phi1','phi2']
    low=np.array([[10,.2,.5,.7,-.2,.4,1,2],[10,.2,1.1,.7,-.2,.4,1,2]])
    fit=['delta_mc','mu1','mu2','chiMinus','s1x','s1y','s2x','s2y']+list(rf.FEATURE_NAMES)
    out=rf.convert(low,fit,names,35,lalsimutils.convert_waveform_coordinates,enforce_kerr=True)
    assert np.isfinite(out[0]).all() and np.isneginf(out[1]).all()
