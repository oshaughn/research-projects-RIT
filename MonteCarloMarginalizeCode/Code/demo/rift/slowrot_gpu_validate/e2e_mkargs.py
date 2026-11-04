"""
e2e_mkargs.py -- build integrate_likelihood_extrinsic_batchmode arg lists for the GPU<->CPU
end-to-end consistency check, from a finite-size ILE case.json (frames/PSD/grid + ile_common).

Usage:  python3 e2e_mkargs.py <mode> <output-file-basename>
  mode in {baseline_cpu, baseline_gpu, rotation_cpu, rotation_gpu, finite_cpu, finite_gpu}
    *_gpu     -> add --gpu (hits the xpy/cupy likelihood_function; needs cupy + a GPU)
    rotation* -> add --rotation-slow (Path A/B)      [mutually exclusive with finite*]
    finite*   -> add case.json ile_finite_extra (Path D --freqresponse)

Env overrides (for a quick check): E2E_NEFF (default 200), E2E_NMAX (600000), E2E_NCHUNK (20000),
E2E_SEED (1234).  --force-xpy is stripped from ile_common: it is INERT for branch selection
(opts.gpu stays False unless --gpu is ALSO passed and cupy is absent) -- you must pass --gpu to
actually exercise the GPU code path.
"""
import json, os, sys

c = json.load(open('case.json'))
common = list(c['ile_common'])
fin = list(c.get('ile_finite_extra', []))
while '--force-xpy' in common:            # inert for branch selection; --gpu is what matters
    common.remove('--force-xpy')

def override(args, k, v):
    if k in args:
        args[args.index(k) + 1] = v
    else:
        args += [k, v]

for k, env, dflt in [('--n-eff', 'E2E_NEFF', '200'),
                     ('--n-max', 'E2E_NMAX', '600000'),
                     ('--n-chunk', 'E2E_NCHUNK', '20000')]:
    override(common, k, os.environ.get(env, dflt))
common += ['--seed', os.environ.get('E2E_SEED', '1234'), '--internal-hard-fail-on-error']

mode, out = sys.argv[1], sys.argv[2]
args = list(common)
if 'finite' in mode:
    args += fin
if 'rotation' in mode:
    args += ['--rotation-slow', '--rotation-n-harmonics', '2', '--rotation-p-max', '1']
if 'gpu' in mode:
    args += ['--gpu']
args += ['--output-file', out]
print(' '.join(args))
