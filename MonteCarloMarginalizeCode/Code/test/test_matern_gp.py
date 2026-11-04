"""Fast fresh-fit recipe/selection/cache checks independent of lalsuite imports."""
import importlib.util
from pathlib import Path
import tempfile
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parents[1]/'RIFT'/'interpolators'
def load(name):
 s=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
fit=load('matern_gp');cache=load('cached_matern_gp')

def test_strata_reproducible_and_aligned():
 y=np.linspace(-25,0,101);rho=np.linspace(0,.99,101)
 a,mode=fit.select_training_rows(y,30,19,rho);b,_=fit.select_training_rows(y,30,19,rho)
 assert np.array_equal(a,b) and len(set(a))==30 and 'rho1' in mode
 assert all(np.any((rho[a]>=lo)&(rho[a]<hi)) for lo,hi in [(0,.1),(.1,.3),(.3,.5),(.5,.7),(.7,1)])
 assert np.array_equal(a,fit.select_training_rows(y+117,30,19,rho)[0])
 c,mode=fit.select_training_rows(y,30,19);assert len(c)==30 and 'unavailable' in mode


def test_fresh_stage_models_and_exact_cache(tmp_path):
 for dim in [3,8]:
  rng=np.random.default_rng(83+dim);x=rng.normal(size=(45,dim));x[:,0]*=1e-4;x[:,1]+=1e4
  y=17+np.sin(x[:,-1])-((x[:,0]/1e-4)**2)/2;err=np.linspace(.05,.4,len(y))
  p,r=fit.fit_matern_gp(x,y,err,max_train_points=30,optimizer_maxiter=3,seed=28,feature_names=['f'+str(i) for i in range(dim)],provenance={'lnL_shift':50.})
  ix=np.array(r['selected_indices']);g=p.named_steps['gp'];s=p.named_steps['scale']
  assert r['training_rows']==30 and r['dimensions']==dim and r['optimizer_restarts']==0
  assert np.array_equal(g.alpha,np.maximum(err[ix]**2/np.std(y[ix])**2,1e-10))
  assert np.allclose(s.mean_,x[ix].mean(0)) and g.normalize_y
  assert np.allclose(g.kernel_.k1.k2.nu,2.5)
  assert g.optimizer is None and r['provenance']['lnL_shift']==50.
  c=cache.from_sklearn(p,backend='numpy',batch_size=7)
  q=rng.normal(size=(31,dim));q[:,0]*=1e-4;q[:,1]+=1e4
  assert np.max(np.abs(c.predict(q)-p.predict(q)))<1e-9
  path=tmp_path/('stage'+str(dim)+'.pkl');joblib.dump(p,path);restored=joblib.load(path)
  assert np.array_equal(restored.predict(q),p.predict(q))
  assert restored.rift_matern_provenance['selected_indices']==r['selected_indices']


def test_bad_inputs_rejected():
 import pytest
 x=np.ones((8,2));y=np.arange(8.)
 with pytest.raises(ValueError):fit.fit_matern_gp(x,y,-np.ones(8))
 with pytest.raises(ValueError):fit.fit_matern_gp(x,y,np.ones(8),feature_names=['one'])
 with pytest.raises(ValueError):fit.fit_matern_gp(x,y,np.ones(8),rho1=np.ones(7))
 with pytest.raises(ValueError):fit.fit_matern_gp(x,y,np.ones(8),optimizer_maxiter=0)
 with pytest.raises(ValueError):fit.fit_matern_gp(x,np.ones(8),np.ones(8))


def test_native_geometry_is_stage_aware():
 assert fit.native_rho1(np.ones((3,2)),['mc','s1z']) is None
 spherical=np.array([[0,np.nan],[.8,.6]])
 assert np.allclose(fit.native_rho1(spherical,['chi1','cos_theta1']),[0,.64])
 assert np.allclose(fit.native_rho1(np.array([[.3,.4]]),['s1x','s1y']),[.5])
 assert np.allclose(fit.native_rho1(np.array([[.25]]),['chi1_perp']),[.25])
