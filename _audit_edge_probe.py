from __future__ import print_function, division
import numpy as np
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.test_noloop_gpu_stencils as T

banks = T._setup()
Pv = T._P_vec()
tvals = np.arange(int(2*T.T_HALFWIDTH/T.deltaT))*T.deltaT - T.T_HALFWIDTH
npts = len(tvals)
print("npts (window) =", npts, " deltaT =", T.deltaT)

orig = fl._q_window_numpy_interp
rec = []
def spy(Q_block, si, fo, npts_, ti, xpy=np, _o=orig):
    si = np.asarray(si)
    rec.append((Q_block.shape[0], int(si.min()), int(si.max()), npts_))
    return _o(Q_block, si, fo, npts_, ti, xpy=xpy)
fl._q_window_numpy_interp = spy
for key,n_cal in (('plain',1),('cal',T.N_CAL)):
    rec[:] = []
    lookupNKDict, rholmArrayDict, ctU, ctV, epochDict = banks[key]
    fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, Pv, lookupNKDict, rholmArrayDict, ctU, ctV, epochDict,
        Lmax=T.Lmax, xpy=np, n_cal=n_cal, cal_method='loop', time_interp='sinc')
    print("--- %s (n_cal=%d): %d dispatches" % (key,n_cal,len(rec)))
    for n_time,lo,hi,np_ in set(rec):
        print("    Q buffer n_time=%-6d ifirst in [%d,%d]  -> left margin=%d, right margin=%d"
              % (n_time,lo,hi,lo, n_time-(hi+np_)))
fl._q_window_numpy_interp = orig
