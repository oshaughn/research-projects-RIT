import os,sys,runpy,tempfile
# Run as a standalone LAL-equipped CLI proof; exits before any integration.
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root));os.environ['GW_SURROGATE']='';os.environ['OMP_NUM_THREADS']='1'
# argv 'redshift': also check CIP wires --source-redshift and --downselect-enforce-kerr into the converter
# argv 'flag': use the deprecated --rf-transverse-spin-coordinates alias instead of the implied names
# argv 'extra': add a ninth fit coordinate (redundant bases are the caller's choice)
redshift='redshift' in sys.argv[1:]
mode=next((m for m in ('geometric4','geometric4-phase-excess') if m in sys.argv[1:]),'geometric4')
extra='extra' in sys.argv[1:]
flag='flag' in sys.argv[1:]
chart=['rf_phase_excess' if mode=='geometric4-phase-excess' else 'rf_total_perp','rf_total_azimuth','rf_residual_parallel','rf_residual_perpendicular']
rng=np.random.default_rng(1);n=160
m1=rng.uniform(20,25,n);m2=rng.uniform(10,15,n);spins=rng.uniform(-.4,.4,(n,6))
dat=np.c_[np.arange(n),m1,m2,spins,100-rng.uniform(0,10,n),np.full(n,.1),np.full(n,100),np.full(n,1000)]
folder=Path(tempfile.mkdtemp(prefix='rf-cip-smoke-'));os.chdir(folder);np.savetxt('grid.dat',dat)
args=['--fname','grid.dat','--fit-method','rf','--use-precessing','--no-plots','--fref','35','--parameter','delta_mc']
if flag:args+=['--rf-transverse-spin-coordinates',mode]
for p in ['mu1','mu2','chiMinus']+(['s1x','s1y','s2x','s2y'] if flag else chart):args += ['--parameter-implied',p]
for p in ['mc','chi1','chi2','cos_theta1','cos_theta2','phi1','phi2']:args+=['--parameter-nofit',p]
if extra:args+=['--parameter-implied','phi1']
if redshift:args+=['--source-redshift','0.3','--downselect-enforce-kerr']
sys.argv=[str(root/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py')]+args
class Complete(Exception):pass
script=root/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py';stop=next(i for i,s in enumerate(script.read_text().splitlines(),1) if s=='oracle_realizations = None')
def tracer(frame,event,arg):
 if event=='line' and frame.f_code.co_filename==str(script) and frame.f_lineno==stop:
  g=frame.f_globals;x=g['dat_out_low_level_coord_names'];xp=g['convert_coords'](x)
  assert g['coord_names']==['delta_mc','mu1','mu2','chiMinus']+chart+(['phi1'] if extra else []), g['coord_names']
  if redshift:
   from RIFT import lalsimutils
   if mode in ('geometric4','geometric4-phase-excess'):
    # Native training rows are detector frame; predictions sample source masses.
    x=x.copy();x[:,g['low_level_coord_names'].index('mc')]/=1.3
   x=np.r_[x,x[:1]];x[-1,g['low_level_coord_names'].index('chi1')]=1.2  # Kerr-violating proposal
   xp=g['convert_coords'](x)
   expected=lalsimutils.convert_waveform_coordinates(x,coord_names=g['coord_names'],low_level_coord_names=g['low_level_coord_names'],
       fref=35.,source_redshift=0.3,enforce_kerr=True)
   # Same conversion via two call paths; allow float64 round-off (seen at 4e-16 on py3.9).
   np.testing.assert_allclose(xp,expected,rtol=1e-12,atol=1e-13)
   assert np.isneginf(xp[-1]).all() and np.isfinite(xp[:-1]).all()
   unshifted=lalsimutils.convert_waveform_coordinates(x[:-1],coord_names=g['coord_names'],low_level_coord_names=g['low_level_coord_names'],fref=35.)
   assert not np.allclose(xp[:-1],unshifted)
   xp=xp[:-1]
   if mode in ('geometric4','geometric4-phase-excess'):np.testing.assert_allclose(xp,g['X'],rtol=2e-5,atol=3e-6)
  else:
   np.testing.assert_allclose(xp,g['X'],rtol=2e-5,atol=3e-6)
  assert xp.shape[1]==8+extra and x.shape[1]==8
  assert np.isfinite(g['my_fit'](xp)).all()
  print('ACTUAL_CIP_TRAINING_PREDICTION_SMOKE_PASS',xp.shape,x.shape)
  raise Complete
 return tracer
completed = False
sys.settrace(tracer)
try:runpy.run_path(str(script),run_name='__main__')
except Complete:
 completed = True
except SystemExit as exc:
 raise RuntimeError("CIP exited before the training/prediction proof boundary") from exc
finally:sys.settrace(None)

assert completed, "CIP did not reach the training/prediction proof boundary"
