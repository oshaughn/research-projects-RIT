#!/usr/bin/env bash
#
# (a) CREATE one of these GP likelihoods.
#
# Package an ILE `.net` file into a small, self-contained, *differentiable*
# lnL(theta) bundle (<base>.npz + <base>.meta.json).  The bundle reconstructs a
# pure-JAX, jax.grad-able likelihood with NO RIFT/lalsimutils dependency at load
# time -- this is the artifact you hand downstream (population inference,
# differentiable samplers, ...).  See ../../../RIFT/interpolators/jax_gp/ARTIFACT.md.
#
# Usage:
#     source config.sh && ./01_create_likelihood.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -n "${RIFT_CODE_DIR:-}" ] || source "${HERE}/config.sh"

echo "=== [1] building differentiable lnL artifact from ${NET} ==="
"${PY}" -m RIFT.interpolators.jax_gp.applications.export_artifact \
  --net "${NET}" \
  --out "${ARTIFACT}" \
  --coords bns \
  --method "${METHOD}" \
  --quadgp-residual "${QUADGP_RESIDUAL}" \
  --cap-points "${CAP_POINTS}" \
  --n-opt-steps "${N_OPT_STEPS}" \
  --n-features "${N_FEATURES}"

echo
echo "=== [2] sanity-check: reload the bundle cold + confirm it differentiates ==="
# This mimics a downstream user with ONLY the .npz/.meta.json (no fit context).
"${PY}" - "${ARTIFACT}" <<'PY'
import sys, json
import numpy as np, jax
from RIFT.interpolators.jax_gp import export

base = sys.argv[1]
model = export.load(base)                       # pure-JAX, differentiable
meta = json.load(open(base + ".meta.json"))
print("  method        :", meta["method"],
      "(residual:", meta.get("resid_meta", {}).get("method"), ")")
print("  fit coords    :", model.coord_names)
theta = np.asarray(model.x_mean)                # a point in fit coordinates
v, g = model.lnL_and_grad(theta)
gj = jax.grad(model.lnL_physical)(jax.numpy.asarray(theta))
ok = np.allclose(np.asarray(gj), g, atol=1e-6) and np.all(np.isfinite(g))
print("  lnL(theta0)   : %.4f" % v)
print("  grad finite + jax.grad matches lnL_and_grad:", bool(ok))
assert ok, "exported likelihood failed the differentiability check"
print("  OK: portable differentiable lnL artifact written to", base + ".{npz,meta.json}")
PY
