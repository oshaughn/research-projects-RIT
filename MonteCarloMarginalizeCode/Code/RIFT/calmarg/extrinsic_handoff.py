"""
Extrinsic handoff: learn a portable proposal over the EXTRINSIC parameters from one
iteration's posterior samples, and seed the NEXT iteration's extrinsic sampler from it.

This is the decade-old "save the extrinsic distribution to inform the next iteration"
goal, generalized from the calibration pilot's breadcrumb (RIFT.calmarg.breadcrumbs).
The extrinsic posterior barely changes from iteration to iteration (it is set by the data
+ best-fit template, not by the small intrinsic-grid moves), so carrying it forward lets
the sampler start near the answer instead of cold each time.

GMM-first (this module): RIFT's ensemble sampler (mcsamplerEnsemble) is already seedable --
its `gmm_dict` maps parameter GROUPS (tuples of indices into params_ordered) to a fitted
`gaussian_mixture_model.gmm`, and a non-None entry is used as the starting proposal.  So
the handoff is:
    fit_extrinsic_proposal(samples, log_weights, groups, bounds)  # per group, RIFT gmm.fit
        -> a portable breadcrumb 'extrinsic' dict (means/covs/weights/bounds + param names)
    gmm_dict_from_breadcrumb(extrinsic, params_ordered)
        -> reconstruct gmm objects, keyed by the dim-group indices, for the next sampler.

We use RIFT's OWN gmm.fit (the same fitter the sampler uses in update_sampling_prior), so
the stored means/covariances are in exactly the model's internal (normalized) frame and
restore to a byte-identical model -- no coordinate guesswork.  A normalizing flow can later
drop in behind the same fit/seed interface (breadcrumb kind != 'gmm').

The standard extrinsic groups (matching the ILE GMM gmm_dict) are
    (right_ascension, declination), (distance, inclination), (phi_orb, psi).
"""
from __future__ import division

import numpy as np

# RIFT's ensemble-sampler GMM (the one gmm_dict expects).  Imported lazily so this module
# is importable without the integrator stack (e.g. for breadcrumb round-trip tests).
def _gmm_module():
    import RIFT.integrators.gaussian_mixture_model as GMM
    return GMM


STANDARD_GROUPS = [
    ["right_ascension", "declination"],
    ["distance", "inclination"],
    ["phi_orb", "psi"],
]


def fit_extrinsic_proposal(samples, log_weights, groups=None, bounds=None,
                           n_comp=4, max_iters=1000):
    """Fit a per-group Gaussian mixture to extrinsic POSTERIOR samples.

    samples : dict {param_name: 1-D array (n,)}  -- the extrinsic samples of one run.
    log_weights : (n,) importance log-weights (log L + log prior - log sampling_prior),
        the same weights the sampler's update_sampling_prior uses.  None -> uniform.
    groups : list of param-name lists (default STANDARD_GROUPS); only groups whose params
        are ALL present in `samples` are fit.
    bounds : dict {param_name: (lo, hi)} sampling bounds (required for the GMM frame).
    n_comp : mixture components per group.

    Returns the breadcrumb 'extrinsic' dict:
        {'kind': 'gmm', 'groups': [ {'params', 'means'(K,d), 'covariances'(K,d,d),
                                      'weights'(K,), 'bounds'(d,2)}, ... ]}.
    """
    GMM = _gmm_module()
    if groups is None:
        groups = STANDARD_GROUPS
    if bounds is None:
        raise ValueError("fit_extrinsic_proposal needs per-parameter sampling bounds")
    n = len(next(iter(samples.values())))
    lw = np.zeros(n) if log_weights is None else np.asarray(log_weights, dtype=float)

    out_groups = []
    for grp in groups:
        if not all(p in samples for p in grp):
            continue
        d = len(grp)
        sample_array = np.column_stack([np.asarray(samples[p], dtype=float) for p in grp])
        grp_bounds = np.array([list(bounds[p]) for p in grp], dtype=float)   # (d, 2)
        k = min(n_comp, max(1, sample_array.shape[0]))
        model = GMM.gmm(k, grp_bounds, max_iters=max_iters)
        # the model may run on cupy (GPU); move inputs onto its device first.
        model.fit(model.identity_convert_togpu(sample_array),
                  log_sample_weights=model.identity_convert_togpu(lw))
        # model.means/.covariances are lists (length k) in the model's internal frame.
        means = np.array([np.asarray(model.identity_convert(m)) for m in model.means])      # (k, d)
        covs = np.array([np.asarray(model.identity_convert(c)) for c in model.covariances])  # (k, d, d)
        weights = np.asarray(model.identity_convert(model.weights), dtype=float).reshape(-1)  # (k,)
        out_groups.append(dict(params=list(grp), means=means, covariances=covs,
                               weights=weights, bounds=grp_bounds))
    return dict(kind="gmm", groups=out_groups)


def reconstruct_gmm(group, max_iters=1000, adapt=True, cov_inflate=2.0):
    """Rebuild a RIFT gaussian_mixture_model.gmm from a stored breadcrumb group.
    adapt=True -> the seeded components keep adapting in the next run (extrinsics drift a
    little); adapt=False freezes them.

    cov_inflate (>=1) widens the seeded covariances (in the model's normalized frame) before
    handing off.  A warm-start proposal should be a bit BROADER than the source posterior: the
    ensemble sampler can contract a too-wide proposal but a too-tight one starves (all first-
    batch samples land in one spot -> zero effective weight -> the sampler discards the seed
    via _reset()).  This matters most when the source iteration was under-converged.  2.0 (in
    covariance, ~1.4x in width) is a safe default.

    All model arrays (means/covariances/weights AND bounds) are moved onto the model's device
    (cupy on GPU): the sampler's score()/_normalize write into an xpy.empty array, so a
    leftover numpy `self.bounds` raises 'non-scalar numpy.ndarray cannot be used for fill'."""
    GMM = _gmm_module()
    means = np.asarray(group["means"]); covs = np.asarray(group["covariances"], dtype=float) * float(cov_inflate)
    weights = np.asarray(group["weights"], dtype=float); bounds = np.asarray(group["bounds"], dtype=float)
    k = means.shape[0]
    model = GMM.gmm(k, bounds, max_iters=max_iters)
    model.bounds = model.identity_convert_togpu(bounds)          # must match self.xpy (GPU)
    model.means = [model.identity_convert_togpu(means[i]) for i in range(k)]
    model.covariances = [model.identity_convert_togpu(covs[i]) for i in range(k)]
    model.weights = model.identity_convert_togpu(weights)
    model.adapt = [bool(adapt)] * k
    model.d = means.shape[1]
    return model


def _permute_group(group, perm):
    """Return a copy of a breadcrumb group with its parameter columns reordered by `perm`
    (perm[j] = source column index that should land at output position j)."""
    means = np.asarray(group["means"])[:, perm]
    covs = np.asarray(group["covariances"])[:, perm][:, :, perm]
    bounds = np.asarray(group["bounds"])[perm]
    return dict(params=[group["params"][j] for j in perm], means=means,
                covariances=covs, weights=group["weights"], bounds=bounds)


def gmm_dict_from_breadcrumb(extrinsic, params_ordered, adapt=True, existing_keys=None, cov_inflate=2.0):
    """Build a gmm_dict {dim_group_tuple: gmm} to SEED mcsamplerEnsemble, from a breadcrumb
    'extrinsic' dict.  dim_group_tuple are indices into `params_ordered` (the sampler's
    parameter order this run), looked up by parameter NAME -- so the handoff is robust to a
    different parameter ordering between runs.  Groups whose params are not all present in
    params_ordered this run are skipped (with no error).

    `existing_keys` (the sampler's actual gmm_dict keys this run) makes the seed robust to the
    WITHIN-group parameter ORDER: the sampler may pair, e.g., (psi, phi_orb) while the
    breadcrumb stored (phi_orb, psi).  We match each breadcrumb group to the existing key with
    the same dim SET, then permute the stored means/covariances/bounds columns into that key's
    dim order -- so the seeded model lines up with how the sampler will draw/score it.  Without
    existing_keys the key is just the breadcrumb's own param order."""
    if extrinsic is None or extrinsic.get("kind") != "gmm":
        return {}
    name_to_idx = {p: i for i, p in enumerate(params_ordered)}
    key_by_set = {frozenset(k): tuple(k) for k in existing_keys} if existing_keys is not None else None
    gmm_dict = {}
    for group in extrinsic["groups"]:
        if not all(p in name_to_idx for p in group["params"]):
            continue
        grp_idx = [name_to_idx[p] for p in group["params"]]   # dim index of each stored column
        if key_by_set is not None:
            target = key_by_set.get(frozenset(grp_idx))
            if target is None:
                continue                                      # sampler has no matching group
        else:
            target = tuple(grp_idx)
        perm = [grp_idx.index(dim) for dim in target]         # reorder stored cols -> target order
        g = group if perm == list(range(len(perm))) else _permute_group(group, perm)
        gmm_dict[target] = reconstruct_gmm(g, adapt=adapt, cov_inflate=cov_inflate)
    return gmm_dict


# ---------------------------------------------------------------------------
# Proof-of-concept: fit a synthetic multi-cluster extrinsic posterior, round-trip it through
# a breadcrumb, and show a seeded GMM starts on the posterior (vs a cold wide prior).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from RIFT.calmarg import breadcrumbs
    import tempfile, os

    rng = np.random.default_rng(0)
    # A bimodal sky posterior (two sky modes) + a unimodal distance/inclination blob.
    bounds = {"right_ascension": (0.0, 2 * np.pi), "declination": (-np.pi / 2, np.pi / 2),
              "distance": (1.0, 1000.0), "inclination": (0.0, np.pi)}
    n = 4000
    mode = rng.random(n) < 0.6
    ra = np.where(mode, rng.normal(1.0, 0.10, n), rng.normal(4.2, 0.15, n)) % (2 * np.pi)
    dec = np.where(mode, rng.normal(0.2, 0.08, n), rng.normal(-0.4, 0.10, n))
    dist = np.clip(rng.normal(450.0, 60.0, n), 1, 1000)
    incl = np.clip(rng.normal(1.1, 0.2, n), 0, np.pi)
    samples = {"right_ascension": ra, "declination": dec, "distance": dist, "inclination": incl}

    ext = fit_extrinsic_proposal(samples, log_weights=None, bounds=bounds, n_comp=3)
    print("fit %d groups: %s" % (len(ext["groups"]), [g["params"] for g in ext["groups"]]))

    # round-trip through a breadcrumb
    p = os.path.join(tempfile.mkdtemp(), "ext.npz")
    breadcrumbs.save(p, extrinsic=ext, meta=dict(iteration=1))
    g = breadcrumbs.load(p)
    assert g["extrinsic"]["kind"] == "gmm"
    assert np.allclose(g["extrinsic"]["groups"][0]["means"], ext["groups"][0]["means"])

    # seed: reconstruct the gmm_dict against a (shuffled) params_ordered, draw from the
    # seeded sky GMM, and check the draws land on the bimodal posterior (means recovered).
    params_ordered = ["distance", "psi", "right_ascension", "phi_orb", "declination", "inclination"]
    gmm_dict = gmm_dict_from_breadcrumb(g["extrinsic"], params_ordered)
    sky_key = (params_ordered.index("right_ascension"), params_ordered.index("declination"))
    assert sky_key in gmm_dict, "sky group not seeded"
    sky = gmm_dict[sky_key]
    draws = np.asarray(sky.identity_convert(sky.sample(3000)))
    # nearest-mode recovery: each true mode should have draws clustered around it
    for true_ra in (1.0, 4.2):
        near = np.min(np.abs((draws[:, 0] - true_ra + np.pi) % (2 * np.pi) - np.pi))
        assert near < 0.5, "seeded GMM draws miss the sky mode at ra=%.1f" % true_ra
    frac_mode1 = np.mean(np.abs(((draws[:, 0] - 1.0 + np.pi) % (2 * np.pi)) - np.pi) < 1.0)
    print("seeded sky GMM: draws recover both modes; ~%.0f%% near mode-1 (true ~60%%)" % (100 * frac_mode1))
    print("PASS: extrinsic posterior -> breadcrumb -> seeded GMM reproduces the (bimodal) "
          "sky distribution; ready to seed mcsamplerEnsemble's gmm_dict.")
