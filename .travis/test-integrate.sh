#! /bin/bash

# Unit/regression tests for the sampler helpers themselves (fast, no data needed).
# The extrinsic "zoom box" limits under the cosine samplers live here: they are pure
# coordinate-transform + prior-mass identities, so they belong with the integrator gate
# rather than with the end-to-end run tests.  The adaptive-volume empty-live-volume
# regression is here for the same reason (it is a threshold identity, not a run test).
python -m pytest -q MonteCarloMarginalizeCode/Code/test/test_limit_cosine_samplers.py
python -m pytest -q MonteCarloMarginalizeCode/Code/test/test_av_empty_live_volume.py

python MonteCarloMarginalizeCode/Code/test/test_mcsamplerEnsemble_extended.py --as-test --n-max 100000

python MonteCarloMarginalizeCode/Code/test/test_mcsamplerEnsemble_extended.py --as-test --n-max 100000 --use-lnL
