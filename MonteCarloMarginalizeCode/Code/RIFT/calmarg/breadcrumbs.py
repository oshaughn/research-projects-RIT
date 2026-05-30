"""
Breadcrumbs: a portable, integrator-agnostic save/load object for a LEARNED proposal
distribution (Option B in DESIGN_adaptive_driver.md).

The point of this module is the *interface and schema*, not the model.  Today it carries
a Gaussian over calibration spline-node parameters (mean/cov) plus the prior; the schema
reserves an `extrinsic` slot so the SAME object can later carry the extrinsic proposal
(the decade-old "save the extrinsic distribution" goal), and a `kind` field so a
normalizing flow can drop in behind the same load()/sample() interface.

Stored as a single .npz (arrays + a JSON metadata sidecar string).  Keep the schema
STABLE: add fields, do not repurpose them; bump SCHEMA_VERSION on incompatible changes.
"""
from __future__ import division

import json
import numpy as np

SCHEMA_VERSION = 1


def save(path, cal=None, extrinsic=None, kind="gaussian", meta=None):
    """Write a breadcrumb file.

    cal : dict or None -- the learned calibration proposal, with keys
        proposal_mean (dim,), proposal_cov (dim,dim), prior_mean (dim,), prior_sigma (dim,),
        node_log_f (n_nodes,), n_nodes_amp (int), dets (list[str]).
        Node-vector layout per detector: [amp_0..amp_{N-1}, phase_0..phase_{N-1}],
        concatenated over `dets` in order.
    extrinsic : reserved for the future (must be None for now).
    kind : 'gaussian' now; 'nflow' etc later (same load/sample interface).
    meta : json-able dict (iteration, n_pilot_points, neff_cal, source composite, ...).
    """
    if extrinsic is not None:
        raise NotImplementedError("extrinsic breadcrumb slot is reserved (Option B); not wired yet")
    d = dict(schema_version=np.int64(SCHEMA_VERSION), kind=str(kind),
             has_cal=np.bool_(cal is not None), has_extrinsic=np.bool_(False),
             meta_json=json.dumps(meta or {}))
    if cal is not None:
        d.update(
            cal_proposal_mean=np.asarray(cal["proposal_mean"], dtype=float),
            cal_proposal_cov=np.asarray(cal["proposal_cov"], dtype=float),
            cal_prior_mean=np.asarray(cal["prior_mean"], dtype=float),
            cal_prior_sigma=np.asarray(cal["prior_sigma"], dtype=float),
            cal_node_log_f=np.asarray(cal["node_log_f"], dtype=float),
            cal_n_nodes_amp=np.int64(cal["n_nodes_amp"]),
            cal_dets=np.array(list(cal["dets"]), dtype=object),
        )
    np.savez(path, **d)
    return path


def load(path):
    """Read a breadcrumb file -> dict {schema_version, kind, cal, extrinsic, meta}."""
    z = np.load(path, allow_pickle=True)
    ver = int(z["schema_version"])
    if ver > SCHEMA_VERSION:
        raise ValueError("breadcrumb schema_version %d newer than supported %d"
                         % (ver, SCHEMA_VERSION))
    out = dict(schema_version=ver, kind=str(z["kind"]),
               meta=json.loads(str(z["meta_json"])), cal=None, extrinsic=None)
    if bool(z["has_cal"]):
        out["cal"] = dict(
            proposal_mean=z["cal_proposal_mean"], proposal_cov=z["cal_proposal_cov"],
            prior_mean=z["cal_prior_mean"], prior_sigma=z["cal_prior_sigma"],
            node_log_f=z["cal_node_log_f"], n_nodes_amp=int(z["cal_n_nodes_amp"]),
            dets=[str(x) for x in z["cal_dets"]],
        )
    return out


if __name__ == "__main__":
    # round-trip smoke test
    dim = 6
    cal = dict(proposal_mean=np.arange(dim, dtype=float),
               proposal_cov=np.eye(dim) * 0.1,
               prior_mean=np.zeros(dim), prior_sigma=np.ones(dim),
               node_log_f=np.linspace(1, 3, dim // 2), n_nodes_amp=dim // 2,
               dets=["H1", "L1", "V1"])
    import tempfile, os
    p = os.path.join(tempfile.mkdtemp(), "bc.npz")
    save(p, cal=cal, meta=dict(iteration=2, neff_cal=87.3))
    g = load(p)
    assert g["kind"] == "gaussian" and g["cal"]["dets"] == ["H1", "L1", "V1"]
    assert np.allclose(g["cal"]["proposal_mean"], cal["proposal_mean"])
    assert g["meta"]["iteration"] == 2
    print("PASS: breadcrumb save/load round-trips (cal Gaussian; extrinsic slot reserved).")
