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

EXTRINSIC slot (schema v2): a portable learned proposal over the EXTRINSIC parameters,
the decade-old "save the extrinsic distribution to inform the next iteration" goal.  It is
a list of parameter GROUPS (matching the ILE GMM sampler's gmm_dict structure -- e.g.
(right_ascension, declination), (distance, inclination), (phi_orb, psi)); each group holds
a Gaussian mixture (means/covariances/weights) over its parameters, plus the parameter
NAMES (so the indices reconstruct against the next run's params_ordered) and the sampling
bounds.  kind='gmm' now; a normalizing flow drops in later behind the same interface.
"""
from __future__ import division

import json
import numpy as np

SCHEMA_VERSION = 2


def save(path, cal=None, extrinsic=None, kind="gaussian", meta=None):
    """Write a breadcrumb file.

    cal : dict or None -- the learned calibration proposal, with keys
        proposal_mean (dim,), proposal_cov (dim,dim), prior_mean (dim,), prior_sigma (dim,),
        node_log_f (n_nodes,), n_nodes_amp (int), dets (list[str]).
        Node-vector layout per detector: [amp_0..amp_{N-1}, phase_0..phase_{N-1}],
        concatenated over `dets` in order.
    extrinsic : dict or None -- the learned EXTRINSIC proposal:
        {'kind': 'gmm',
         'groups': [ {'params': [name,...], 'means': (K,d), 'covariances': (K,d,d),
                      'weights': (K,), 'bounds': (d,2)}, ... ]}.
        One group per gmm_dict block; the GMM is over the group's params in `params` order.
    kind : top-level kind tag ('gaussian' for the cal Gaussian; extrinsic kind is its own).
    meta : json-able dict (iteration, n_pilot_points, neff_cal, source composite, ...).
    """
    d = dict(schema_version=np.int64(SCHEMA_VERSION), kind=str(kind),
             has_cal=np.bool_(cal is not None), has_extrinsic=np.bool_(extrinsic is not None),
             meta_json=json.dumps(meta or {}))
    if cal is not None:
        d.update(
            cal_proposal_mean=np.asarray(cal["proposal_mean"], dtype=float),
            cal_proposal_cov=np.asarray(cal["proposal_cov"], dtype=float),
            cal_prior_mean=np.asarray(cal["prior_mean"], dtype=float),
            cal_prior_sigma=np.asarray(cal["prior_sigma"], dtype=float),
            cal_node_log_f=np.asarray(cal["node_log_f"], dtype=float),
            cal_n_nodes_amp=np.int64(cal["n_nodes_amp"]),
            cal_dets=np.array([str(x) for x in cal["dets"]]),   # string dtype (NOT object): no
        )
    if extrinsic is not None:
        groups = extrinsic["groups"]
        d["ext_kind"] = str(extrinsic.get("kind", "gmm"))
        d["ext_n_groups"] = np.int64(len(groups))
        for i, g in enumerate(groups):
            d["ext_g%d_params" % i] = np.array([str(x) for x in g["params"]])   # string, not object
            d["ext_g%d_means" % i] = np.asarray(g["means"], dtype=float)
            d["ext_g%d_covs" % i] = np.asarray(g["covariances"], dtype=float)
            d["ext_g%d_weights" % i] = np.asarray(g["weights"], dtype=float)
            d["ext_g%d_bounds" % i] = np.asarray(g["bounds"], dtype=float)
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
    if "has_extrinsic" in z and bool(z["has_extrinsic"]):
        groups = []
        for i in range(int(z["ext_n_groups"])):
            groups.append(dict(
                params=[str(x) for x in z["ext_g%d_params" % i]],
                means=z["ext_g%d_means" % i], covariances=z["ext_g%d_covs" % i],
                weights=z["ext_g%d_weights" % i], bounds=z["ext_g%d_bounds" % i],
            ))
        out["extrinsic"] = dict(kind=str(z["ext_kind"]), groups=groups)
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
    # PORTABILITY: the file must contain NO pickled objects, so a breadcrumb written by
    # one numpy (e.g. the container's 2.x) loads under any other (e.g. the host's 1.x).
    # np.load(allow_pickle=False) raises if any array is object-dtype -- guards the dets/
    # params regression where dtype=object silently pickled and broke cross-version load.
    with np.load(p, allow_pickle=False) as _z:
        assert _z["cal_dets"].dtype.kind in ("U", "S"), _z["cal_dets"].dtype

    # extrinsic (GMM) round-trip
    ext = dict(kind="gmm", groups=[
        dict(params=["right_ascension", "declination"],
             means=np.array([[1.0, 0.2], [4.0, -0.3]]),
             covariances=np.array([np.eye(2) * 0.05, np.eye(2) * 0.1]),
             weights=np.array([0.6, 0.4]),
             bounds=np.array([[0.0, 2 * np.pi], [-np.pi / 2, np.pi / 2]])),
        dict(params=["distance", "inclination"],
             means=np.array([[500.0, 1.0]]), covariances=np.array([np.diag([1e4, 0.1])]),
             weights=np.array([1.0]), bounds=np.array([[1.0, 1000.0], [0.0, np.pi]])),
    ])
    p2 = os.path.join(tempfile.mkdtemp(), "bc2.npz")
    save(p2, cal=cal, extrinsic=ext, meta=dict(iteration=3))
    g2 = load(p2)
    assert g2["extrinsic"]["kind"] == "gmm" and len(g2["extrinsic"]["groups"]) == 2
    assert g2["extrinsic"]["groups"][0]["params"] == ["right_ascension", "declination"]
    assert np.allclose(g2["extrinsic"]["groups"][0]["means"], ext["groups"][0]["means"])
    assert np.allclose(g2["extrinsic"]["groups"][1]["covariances"], ext["groups"][1]["covariances"])
    assert g2["cal"] is not None   # cal + extrinsic coexist in one breadcrumb
    with np.load(p2, allow_pickle=False) as _z2:   # portability: no pickled object arrays
        assert _z2["ext_g0_params"].dtype.kind in ("U", "S"), _z2["ext_g0_params"].dtype
    print("PASS: breadcrumb save/load round-trips (cal Gaussian + extrinsic GMM, schema v%d), "
          "pickle-free (portable across numpy versions)." % SCHEMA_VERSION)
