import sys, os
sys.path.insert(0, os.path.join(os.environ["PYTHONPATH"], "..", "..", "test", "jax"))
sys.path.insert(0, os.path.expanduser("~/rift_ghlaplace_20260902/MonteCarloMarginalizeCode/Code/test/jax"))
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from test_angle_marg_exact import make_synth, _dist_grid, RA, DEC, INCL, INTERP
from RIFT.likelihood.jax_ile import anglemarg as AM
from RIFT.likelihood.jax_ile import core as core_mod
import RIFT; assert "rift_ghlaplace" in RIFT.__file__, RIFT.__file__
data = make_synth(scale=float(sys.argv[1]) if len(sys.argv) > 1 else 6.0)
amp = AM.ANGLE_MARG_CROSSOVER_AMPLITUDE
def run(fn, n_grid, gh):
    xg, lw = _dist_grid(data, n=n_grid)
    core_mod._DISTMARG_GH_N = gh
    r = float(np.asarray(fn(data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
                            xg, lw, interp=INTERP, amp_sizing=amp))[0])
    core_mod._DISTMARG_GH_N = 0
    return r
EX = AM.fused_log_likelihood_distphipsimarg_exact
LP = AM.fused_log_likelihood_distphipsimarg_laplace
print("uniform-grid convergence:", flush=True)
for n in (64, 128, 512, 2048, 8192):
    print("   n=%6d  exact %.10f   laplace %.10f" % (n, run(EX,n,0), run(LP,n,0)), flush=True)
print("GH node convergence (n_grid=128 supplies only the support):", flush=True)
for g in (17, 33, 65, 129):
    print("   gh=%4d  exact %.10f   laplace %.10f" % (g, run(EX,128,g), run(LP,128,g)), flush=True)
