import os,sys,runpy,tempfile
# Run as a standalone LAL-equipped CLI proof; exits before any integration.
from pathlib import Path
import numpy as np
try:
    import lal  # noqa: F401
except ImportError:
    print('SKIP rf_transverse_cli_smoke: lal unavailable')
    sys.exit(0)
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root));os.environ['GW_SURROGATE']='';os.environ['OMP_NUM_THREADS']='1'
rng=np.random.default_rng(1);n=160
m1=rng.uniform(20,25,n);m2=rng.uniform(10,15,n);spins=rng.uniform(-.4,.4,(n,6))
dat=np.c_[np.arange(n),m1,m2,spins,100-rng.uniform(0,10,n),np.full(n,.1),np.full(n,100),np.full(n,1000)]
folder=Path(tempfile.mkdtemp(prefix='rf-cip-smoke-'));os.chdir(folder);np.savetxt('grid.dat',dat)
args=['--fname','grid.dat','--fit-method','rf','--use-precessing','--no-plots','--fref','35','--rf-transverse-spin-coordinates','physics3','--parameter','delta_mc']
for p in ['mu1','mu2','chiMinus','s1x','s1y','s2x','s2y']:args += ['--parameter-implied',p]
for p in ['mc','chi1','chi2','cos_theta1','cos_theta2','phi1','phi2']:args+=['--parameter-nofit',p]
sys.argv=[str(root/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py')]+args
class Complete(Exception):pass
script=root/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py';stop=next(i for i,s in enumerate(script.read_text().splitlines(),1) if s=='oracle_realizations = None')
def tracer(frame,event,arg):
 if event=='line' and frame.f_code.co_filename==str(script) and frame.f_lineno==stop:
  g=frame.f_globals;x=g['dat_out_low_level_coord_names'];xp=g['convert_coords'](x)
  np.testing.assert_allclose(xp,g['X'],rtol=2e-5,atol=3e-6)
  assert xp.shape[1]==11 and x.shape[1]==8
  assert np.isfinite(g['my_fit'](xp)).all()
  print('ACTUAL_CIP_TRAINING_PREDICTION_SMOKE_PASS',xp.shape,x.shape)
  raise Complete
 return tracer
completed = False
sys.settrace(tracer)
try:runpy.run_path(str(script),run_name='__main__')
except Complete:completed = True
except SystemExit as exc:
 raise RuntimeError("CIP exited before the training/prediction proof boundary") from exc
finally:sys.settrace(None)

assert completed, "CIP did not reach the training/prediction proof boundary"
