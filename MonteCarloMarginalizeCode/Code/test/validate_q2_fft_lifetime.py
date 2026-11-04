"""Development patch lifetime gate, in a fresh assigned-GPU process."""
import gc,json,time
from pathlib import Path
import cupy as cp
from RIFT.likelihood import time_marginalization_quadrature as tq
pool=cp.get_default_memory_pool(); cache=cp.fft.config.get_plan_cache()
def clean():
 cp.cuda.get_current_stream().synchronize();cache.clear();gc.collect();pool.free_all_blocks()
def state(tag):
 free,total=cp.cuda.runtime.memGetInfo()
 s=dict(tag=tag,time=time.time(),device_used=total-free,pool_total=pool.total_bytes(),pool_used=pool.used_bytes(),plans=cache.get_curr_size(),plan_bytes=cache.get_curr_memsize())
 print(json.dumps(s),flush=True);return s
def cycle():
 for count in range(64,144):
  x=cp.ones((count,819),dtype=cp.complex128)
  y=tq._reflected_bandlimited_upsample_retained(x,8,xpy=cp)
  cp.cuda.get_current_stream().synchronize();del x,y
# Warm libraries, then measure six identical cycles with the implemented helper.
cycle();clean();baseline=state('baseline')
rows=[]
for i in range(6):
 cycle();rows.append(state('cycle-'+str(i)))
clean();final=state('final')
summary={'baseline':baseline,'cycles':rows,'final':final,'cupy':cp.__version__,
 'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
 'runtime':cp.cuda.runtime.runtimeGetVersion(),
 'pool_growth_after_first':rows[-1]['pool_total']-rows[0]['pool_total'],
 'device_growth_after_first':rows[-1]['device_used']-rows[0]['device_used'],
 'residual_after_clean':final['device_used']-baseline['device_used']}
Path('lifetime-summary.json').write_text(json.dumps(summary,indent=2))
assert summary['pool_growth_after_first']==0,summary
assert summary['device_growth_after_first']==0,summary
assert summary['residual_after_clean']==0,summary
print('LIFETIME_PASS',flush=True)
