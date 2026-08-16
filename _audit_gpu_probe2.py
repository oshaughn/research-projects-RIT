from __future__ import print_function, division
import os
import numpy as np, cupy
import RIFT.likelihood.factored_likelihood as FL
from RIFT.likelihood import Q_inner_product as QIP

rng = np.random.RandomState(1)
n_time, n_lm = 512, 3
Q = rng.randn(n_time,n_lm)+1j*rng.randn(n_time,n_lm)
starts = np.arange(100,110,dtype=np.int32); fracs = rng.rand(len(starts))
A = rng.randn(len(starts),n_lm)+1j*rng.randn(len(starts),n_lm)

print("=== pathological launch shapes: loud or silent? ===")
for tx,ty,note in [(1024,1,"shared=131072B > 48KB"),(384,1,"shared=49152B == 48KB"),
                   (383,1,"shared 49024B"),(64,64,"4096 threads > 1024"),
                   (0,128,"THREADS_X=0"),(4,0,"THREADS_Y=0")]:
    os.environ["RIFT_Q_SINC_THREADS_X"]=str(tx); os.environ["RIFT_Q_SINC_THREADS_Y"]=str(ty)
    try:
        r = QIP.Q_inner_product_sinc_cupy(cupy.asarray(Q),cupy.asarray(A),
              cupy.asarray(starts),cupy.asarray(fracs),16)
        cupy.cuda.Stream.null.synchronize()
        ref = np.einsum("ej,etj->et",A,FL._sinc_Q_window_numpy(Q,starts,fracs,16))
        d=float(np.max(np.abs(cupy.asnumpy(r)-ref)))
        print("  tx=%-5d ty=%-5d %-26s -> ran, maxdiff=%.3e %s"%(tx,ty,note,d,"OK" if d<1e-12 else "*** SILENTLY WRONG ***"))
    except Exception as e:
        print("  tx=%-5d ty=%-5d %-26s -> %s: %s"%(tx,ty,note,type(e).__name__,str(e)[:120]))
    # reset device state
    try: cupy.cuda.Device().synchronize()
    except Exception as e: print("     (device left in error state: %s)"%type(e).__name__)
