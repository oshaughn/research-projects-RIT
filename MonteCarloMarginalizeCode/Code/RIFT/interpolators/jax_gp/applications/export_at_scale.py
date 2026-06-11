"""
export_at_scale -- point at a real RIFT run directory, ship a differentiable lnL
artifact for its ``all.net``, and validate it against the run's own posterior.

This is the "do it for real, do it for many" wrapper around the jax_gp primitives.
``export_artifact``/``jax_cip`` already turn a *single* ``all.net`` into a pure-JAX
``lnL(theta)`` surrogate; this tool adds the three things a production sweep needs:

  1. **Discovery.** Read a run directory the way RIFT left it -- detect the
     ``all.net`` column layout (tides / no-tides / precessing all differ), parse the
     active ``args_cip_list.txt`` for the fit coordinates and the prior box
     (``--mc-range`` / ``--eta-range`` / ``--chi-max``), and locate the run's own
     final CIP posterior (``posterior_samples-<N>.dat``) to validate against.

  2. **Export + validate the *deliverable*.** Fit the surrogate in dimension-agnostic
     physical fit coordinates -- ``[mc, delta_mc]`` plus whichever spin/tidal columns
     actually vary, so it covers aligned, BNS *and* precessing runs without needing a
     hand-written coordinate transform -- export it, **reload the saved artifact**,
     draw a posterior *from the reloaded bytes* (importance sampling over the run's
     prior box), and report the Jensen-Shannon divergence of the ``mc``/``q``/
     ``chi_eff`` marginals against the run's CIP posterior.

  3. **Scale.** ``batch`` discovers many run directories and either runs them locally
     or emits an HTCondor DAG -- one node per run, the submit file templated from the
     run's *own* ``CIP.sub`` (accounting group, singularity image, requirements) so
     the jobs land in the same place the run itself ran.

Nothing is written back into the run directory: every artifact + report goes under a
separate ``--workroot`` (default ``./export_at_scale_out``), one subdirectory per run.

At this stage we interpolate **only ``all.net``** (the existing intrinsic ILE
deliverable). The distance-grid export is a separate track.

CLI::

    # one run, immediately
    python -m RIFT.interpolators.jax_gp.applications.export_at_scale one \\
        --run-dir /path/to/rundir --workroot /path/to/out

    # inspect what discovery found, without doing any work
    python -m RIFT.interpolators.jax_gp.applications.export_at_scale discover \\
        --run-dir /path/to/rundir

    # many runs -> a condor DAG (submit it with condor_submit_dag)
    python -m RIFT.interpolators.jax_gp.applications.export_at_scale batch \\
        --runs '/data/*/S*/rift*/' --workroot /path/to/out --condor
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import os
import shlex
import sys
import time

import numpy as np


# --------------------------------------------------------------------------- #
# 1. all.net column-layout detection
# --------------------------------------------------------------------------- #
#
# RIFT's util_CleanILE writes a *variable-width* composite. The intrinsic block
# grows with the physics (aligned -> precessing adds in-plane spins; BNS adds
# tides), and the trailing diagnostic columns are always, in order,
# ``lnL  sigma_lnL  ntot [neff]``.  We therefore key off the *tail*: the intrinsic
# block is everything between the leading index column and those diagnostics.

#: candidate names for the (1- or 2-mass + spin [+ tidal]) intrinsic block, by size
_INTRINSIC_BY_SIZE = {
    2: ["m1", "m2"],
    4: ["m1", "m2", "s1z", "s2z"],
    8: ["m1", "m2", "s1x", "s1y", "s1z", "s2x", "s2y", "s2z"],
    6: ["m1", "m2", "s1z", "s2z", "lambda1", "lambda2"],
    10: ["m1", "m2", "s1x", "s1y", "s1z", "s2x", "s2y", "s2z",
         "lambda1", "lambda2"],
}


def detect_net_layout(path, max_probe=200):
    """Infer the column map of a RIFT ``all.net`` / ``.composite`` file.

    Returns ``(cols, intrinsic_names, has_neff)`` where ``cols`` is the
    name->index dict :func:`load_ile_net` consumes and ``intrinsic_names`` are the
    physical columns of the intrinsic block (e.g. ``['m1','m2','s1x',...]``).

    The detection is tail-anchored (``... lnL sigma_lnL ntot [neff]``) and
    sanity-checked on the candidate ``sigma_lnL`` column (a small positive MC error),
    so it is robust to the aligned/precessing/tidal width differences without a
    hard-coded table.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())
            if len(rows) >= max_probe:
                break
    if not rows:
        raise ValueError("no data rows in {}".format(path))
    ncols = len(rows[0])
    data = np.array([[float(x) for x in r] for r in rows if len(r) == ncols],
                    dtype=float)

    def _try(has_neff):
        tail = 4 if has_neff else 3
        lnL_i = ncols - tail
        sig_i = lnL_i + 1
        if lnL_i < 3:                       # need indx + at least m1,m2
            return None
        n_intr = lnL_i - 1                  # columns 1..lnL_i-1 (0 is index)
        sig = data[:, sig_i]
        # sigma_lnL is a small, finite, non-negative MC error
        if not np.all(np.isfinite(sig)) or np.any(sig < 0) or np.median(sig) > 10:
            return None
        names = _INTRINSIC_BY_SIZE.get(n_intr)
        if names is None:
            # unknown block size: name masses + generic spin/tidal slots so the
            # fit still runs (we just lose the pretty chi_eff label).
            names = ["m1", "m2"] + ["x{}".format(k) for k in range(n_intr - 2)]
        cols = {"indx": 0}
        for j, nm in enumerate(names):
            cols[nm] = 1 + j
        cols["lnL"] = lnL_i
        cols["sigma_lnL"] = sig_i
        cols["ntot"] = sig_i + 1
        return cols, names, has_neff

    # Prefer the 4-trailing (with neff) form; fall back to 3-trailing.
    for has_neff in (True, False):
        got = _try(has_neff)
        if got is not None:
            return got
    raise ValueError(
        "could not infer column layout of {} (ncols={})".format(path, ncols))


# --------------------------------------------------------------------------- #
# 2. run-directory discovery
# --------------------------------------------------------------------------- #

def _parse_pair(s):
    """'[a,b]' -> [float(a), float(b)] (or None)."""
    if not s:
        return None
    try:
        v = ast.literal_eval(s)
        return [float(v[0]), float(v[1])]
    except Exception:
        return None


def _last_cip_arg_line(run_dir):
    """The active (last non-empty) line of args_cip_list.txt, minus its leading
    iteration-label token (e.g. ``3``/``Z``/``1``)."""
    p = os.path.join(run_dir, "args_cip_list.txt")
    if not os.path.exists(p):
        return None
    lines = [ln.strip() for ln in open(p) if ln.strip()]
    if not lines:
        return None
    toks = lines[-1].split(None, 1)
    return toks[1] if len(toks) == 2 else lines[-1]


def _cip_opt(tokens, name, multi=False):
    """Pull ``--name VALUE`` (repeatable if ``multi``) out of a token list."""
    out = []
    i = 0
    while i < len(tokens):
        if tokens[i] == name:
            if i + 1 < len(tokens):
                out.append(tokens[i + 1])
            i += 2
        else:
            i += 1
    if multi:
        return out
    return out[-1] if out else None


def _find_latest_posterior(run_dir):
    """Most-recent CIP intrinsic posterior; falls back to the extrinsic fairdraw.

    Prefers ``posterior_samples-<N>.dat`` (highest N) in the run dir, then in the
    highest ``iteration_<k>_cip/``, then ``extrinsic_posterior_samples.dat``.
    """
    cands = []
    for p in glob.glob(os.path.join(run_dir, "posterior_samples-*.dat")):
        m = os.path.basename(p)[len("posterior_samples-"):-len(".dat")]
        try:
            cands.append((int(m), p))
        except ValueError:
            pass
    if cands:
        return max(cands)[1]
    it_dirs = sorted(glob.glob(os.path.join(run_dir, "iteration_*_cip")),
                     key=lambda d: int(d.split("_")[-2]) if d.split("_")[-2].isdigit()
                     else -1)
    for d in reversed(it_dirs):
        sub = glob.glob(os.path.join(d, "posterior_samples-*.dat"))
        if sub:
            return sorted(sub)[-1]
    extr = os.path.join(run_dir, "extrinsic_posterior_samples.dat")
    return extr if os.path.exists(extr) else None


def _parse_condor_env(run_dir):
    """Lift accounting / singularity / requirements from the run's own CIP.sub
    (or ILE.sub) so a fan-out job can land where the run itself ran."""
    env = {}
    for name in ("CIP.sub", "CIP_0.sub", "ILE.sub"):
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            continue
        for ln in open(p):
            ln = ln.strip()
            for key in ("accounting_group", "accounting_group_user",
                        "request_memory", "request_disk", "request_cpus",
                        "requirements"):
                if ln.lower().startswith(key.lower()) and "=" in ln:
                    env.setdefault(key, ln.split("=", 1)[1].strip())
            if "SingularityImage" in ln and "=" in ln:
                env.setdefault("singularity_image",
                               ln.split("=", 1)[1].strip().strip('"'))
        break
    return env


def discover_run(run_dir):
    """Inspect a RIFT run directory and return a dict describing how to export +
    validate it. Raises if there is no usable ``all.net``."""
    run_dir = os.path.abspath(run_dir)
    net = os.path.join(run_dir, "all.net")
    if not os.path.exists(net) or os.path.getsize(net) == 0:
        raise FileNotFoundError("no usable all.net in {}".format(run_dir))

    cols, intrinsic_names, has_neff = detect_net_layout(net)

    cip_line = _last_cip_arg_line(run_dir)
    toks = shlex.split(cip_line) if cip_line else []
    mc_range = _parse_pair(_cip_opt(toks, "--mc-range"))
    eta_range = _parse_pair(_cip_opt(toks, "--eta-range"))
    chi_max = _cip_opt(toks, "--chi-max")
    chi_max = float(chi_max) if chi_max else None
    chi_small_max = _cip_opt(toks, "--chi-small-max")
    chi_small_max = float(chi_small_max) if chi_small_max else None
    aligned_prior = _cip_opt(toks, "--aligned-prior") or "uniform"
    precessing = "--use-precessing" in toks
    params = (_cip_opt(toks, "--parameter", multi=True)
              + _cip_opt(toks, "--parameter-implied", multi=True)
              + _cip_opt(toks, "--parameter-nofit", multi=True))
    n_out = _cip_opt(toks, "--n-output-samples")

    # event / pipeline tags for a readable workdir name
    parts = run_dir.split(os.sep)
    event = next((p for p in reversed(parts) if p.startswith(("S", "G", "GW"))),
                 parts[-2] if len(parts) > 1 else "event")
    tag = parts[-1]

    # distance-export ("dgrid") detection. When the run marginalised the distance grid,
    # it leaves per-job *.dgrid files and/or a consolidated all_dgrid.dat: the lnL is
    # then a function of (intrinsic, distance). That higher-dim export is a SEPARATE
    # track (still under development); here we record its presence so the caller can
    # route it. The intrinsic all.net export below is unaffected.
    dgrid_consolidated = next(
        (p for p in (os.path.join(run_dir, "all_dgrid.dat"),
                     os.path.join(run_dir, "consolidated_dgrid.dat"))
         if os.path.exists(p)), None)
    has_dgrid = bool(dgrid_consolidated) or bool(
        glob.glob(os.path.join(run_dir, "*.dgrid"))
        or glob.glob(os.path.join(run_dir, "iteration_*_ile", "*.dgrid")))

    return {
        "run_dir": run_dir,
        "net": net,
        "ncols": len(cols) + (1 if has_neff else 0),
        "cols": cols,
        "intrinsic_names": intrinsic_names,
        "has_spins": any(n.startswith("s") for n in intrinsic_names),
        "has_tides": any(n.startswith("lambda") for n in intrinsic_names),
        "precessing": precessing,
        "cip_parameters": params,
        "mc_range": mc_range,
        "eta_range": eta_range,
        "chi_max": chi_max if chi_max is not None else 0.99,
        "chi_small_max": (chi_small_max if chi_small_max is not None
                          else (chi_max if chi_max is not None else 0.99)),
        "aligned_prior": aligned_prior,
        "n_output_samples": int(n_out) if n_out else 2000,
        "posterior": _find_latest_posterior(run_dir),
        "condor_env": _parse_condor_env(run_dir),
        "event": event,
        "tag": tag,
        "label": "{}__{}".format(event, tag),
    }


# --------------------------------------------------------------------------- #
# 3. dimension-agnostic fit coordinates
# --------------------------------------------------------------------------- #
#
# We fit lnL in [mc, delta_mc] + (varying spin/tidal columns). mc/delta_mc remove
# the dominant curved chirp-mass ridge; raw spin components keep the rest fully
# general (aligned -> just s1z,s2z survive as non-constant; precessing -> all six).
# Constant columns (e.g. all spins zero in a no-spin run) are dropped from the fit
# and recorded so we can reconstruct full physical vectors for chi_eff / writing.

_CONST_TOL = 1e-6


def raw_to_fit(X_raw, raw_names, mass_coord="eta", spin_coord="aligned_eff"):
    """``(m1,m2,spin...) -> (mc, <mass2>, <aligned spin>, <in-plane spin>)`` columns.

    The surrogate must be fit in the coordinates the lnL Fisher is quadratic in *and*
    axis-aligned with (the quadratic core whitens per-dimension and the GP residual
    uses axis-aligned ARD lengthscales -- neither can represent a sharp ridge along a
    diagonal direction).

    ``mass_coord``: ``"eta"`` (default) or ``"delta_mc"``.  eta is the Fisher-quadratic
    variable; delta_mc hides that curvature near equal mass (it is quadratic in
    delta_mc), so eta fixes the mass-ratio (q) marginal.

    ``spin_coord``: ``"aligned_eff"`` (default) replaces the aligned components
    ``(s1z, s2z)`` with ``(chi_eff, chiMinus)`` -- the principal axes of the
    aligned-spin Fisher.  In ``(s1z, s2z)`` the well-measured chi_eff is a *diagonal*
    ridge an axis-aligned ARD GP over-smooths (the low-mass aligned-spin failure mode);
    rotating to ``(chi_eff, chiMinus)`` makes it axis-aligned and resolvable.  The
    in-plane components ``(s1x, s1y, s2x, s2y)`` are kept as-is (weakly constrained).
    ``"cartesian"`` keeps the raw components.
    """
    idx = {n: i for i, n in enumerate(raw_names)}
    m1 = X_raw[:, idx["m1"]]
    m2 = X_raw[:, idx["m2"]]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    if mass_coord == "eta":
        second, sname = m1 * m2 / (m1 + m2) ** 2, "eta"
    else:
        second, sname = (m1 - m2) / (m1 + m2), "delta_mc"
    cols = [mc, second]
    names = ["mc", sname]
    spin_names = [n for n in raw_names if n not in ("m1", "m2")]
    if spin_coord == "aligned_eff" and "s1z" in idx and "s2z" in idx:
        M = m1 + m2
        s1z, s2z = X_raw[:, idx["s1z"]], X_raw[:, idx["s2z"]]
        cols += [(m1 * s1z + m2 * s2z) / M, (m1 * s1z - m2 * s2z) / M]
        names += ["chi_eff", "chiMinus"]
        for n in spin_names:                       # keep the in-plane components
            if n in ("s1z", "s2z"):
                continue
            cols.append(X_raw[:, idx[n]]); names.append(n)
    else:
        for n in spin_names:
            cols.append(X_raw[:, idx[n]]); names.append(n)
    return np.column_stack(cols), names


def fit_to_physical(Xfit, fit_names):
    """Inverse of the mass part + spin passthrough: fit coords -> physical dict
    with arrays m1, m2, q, chi_eff and any spin components present."""
    idx = {n: i for i, n in enumerate(fit_names)}
    mc = Xfit[:, idx["mc"]]
    dmc = Xfit[:, idx["delta_mc"]]
    eta = 0.25 * (1.0 - dmc ** 2)
    mtot = mc * eta ** (-3.0 / 5.0)
    m1 = 0.5 * mtot * (1.0 + dmc)
    m2 = 0.5 * mtot * (1.0 - dmc)
    out = {"m1": m1, "m2": m2, "mc": mc, "eta": eta, "q": m2 / m1}
    for n in fit_names:
        if n not in ("mc", "delta_mc"):
            out[n] = Xfit[:, idx[n]]
    s1z = out.get("s1z", np.zeros_like(m1))
    s2z = out.get("s2z", np.zeros_like(m1))
    out["chi_eff"] = (m1 * s1z + m2 * s2z) / (m1 + m2)
    return out


def fit_prior_box(fit_names, spec):
    """Per-coordinate uniform-prior box [lo, hi] in fit coords, from the run's CIP
    ranges (mc, eta->delta_mc, chi-max per spin component)."""
    mc_r = spec["mc_range"] or [0.9, 250.0]
    eta_r = spec["eta_range"] or [0.01, 0.2499999]
    d_hi = math.sqrt(max(0.0, 1.0 - 4.0 * eta_r[0]))
    d_lo = math.sqrt(max(0.0, 1.0 - 4.0 * min(eta_r[1], 0.25)))
    chi = spec["chi_max"]
    lo, hi = [], []
    for n in fit_names:
        if n == "mc":
            lo.append(mc_r[0]); hi.append(mc_r[1])
        elif n == "delta_mc":
            lo.append(max(0.0, d_lo)); hi.append(min(0.999, d_hi))
        elif n.startswith("s"):                       # spin component
            lo.append(-chi); hi.append(chi)
        elif n.startswith("lambda"):
            lo.append(0.0); hi.append(5000.0)
        else:
            lo.append(-chi); hi.append(chi)
    return np.array(lo), np.array(hi)


# --------------------------------------------------------------------------- #
# 3b. RIFT-prior-correct sampling coordinates (apples-to-apples validation)
# --------------------------------------------------------------------------- #
#
# CIP does NOT sample exp(lnL) flat: it applies its prior_map in the *sampled*
# coordinates.  To compare apples-to-apples we reproduce that measure.  The trick
# (read straight out of CIP's prior_map) is to sample spins in spherical
# coordinates (chi, cos_theta, phi), where RIFT's default precessing prior is
# *separable and flat* -- chi: uniform magnitude (s_magnitude_uniform_prior=1/R),
# cos_theta: uniform, phi: uniform -- so the only non-constant prior factor left is
# the MASS prior (mc_prior ~ mc; delta_mc_prior ~ eta^-6/5).  The Cartesian-spin
# 1/chi^2 "singularity" is just the Jacobian of this flat spherical prior, so
# sampling in (chi,cos_theta,phi) both matches RIFT exactly and avoids the singular
# geometry.  Aligned runs sample s1z,s2z in Cartesian with the s_component_zprior
# shape when --aligned-prior alignedspin-zprior was used.

def build_sampling(spec, fit_names, spin_modes=None):
    """Return the RIFT-measure sampling spec for the artifact's ``fit_names``.

    ``spin_modes`` (per body ``{"1": m, "2": m}`` with ``m`` in
    ``{"sph", "cart_z", None}``) fixes how each body's spin is *sampled*, derived from
    the raw physics (which spin columns vary), NOT from ``fit_names`` -- because the
    fit may use ``chi_eff``/``chiMinus`` instead of ``s1z``/``s2z``.  When omitted, it
    is inferred from ``fit_names`` (back-compat: Cartesian fit coords).

    Returns a dict with ``names``/``lo``/``hi`` (prior box), ``ln_prior(theta)``,
    ``to_fit(theta)`` (sampling coords -> the artifact's ``fit_names`` vector),
    ``raw_to_sample`` (ILE data -> sampling coords) and ``to_compare`` (samples ->
    every physical comparison parameter).
    """
    import jax.numpy as jnp

    fit_set = set(fit_names)
    eta_r = spec["eta_range"] or [0.01, 0.2499999]
    mc_r = spec["mc_range"] or [0.9, 250.0]
    d_hi = math.sqrt(max(0.0, 1.0 - 4.0 * eta_r[0]))
    d_lo = math.sqrt(max(0.0, 1.0 - 4.0 * min(eta_r[1], 0.25)))
    precessing = spec["precessing"]
    zprior = spec.get("aligned_prior") == "alignedspin-zprior"
    Rbody = {"1": spec["chi_max"], "2": spec["chi_small_max"]}

    # DECOUPLE fit vs sampling coordinate. The artifact may be fit in eta (the
    # Fisher-quadratic variable, which the quadratic core can capture), but we always
    # SAMPLE in delta_mc: its prior is smooth (eta^-6/5, no equal-mass singularity)
    # and its geometry is better-conditioned for NUTS -- exactly RIFT's own choice.
    # to_fit() maps the sampled delta_mc to whatever mass coordinate the artifact uses.
    mass2_fit = fit_names[1] if len(fit_names) > 1 and fit_names[1] in ("eta", "delta_mc") \
        else "delta_mc"
    names = ["mc", "delta_mc"]
    lo = [mc_r[0], max(0.0, d_lo)]
    hi = [mc_r[1], min(0.999, d_hi)]
    # how each body's spin is SAMPLED -- from the raw physics (spin_modes) if given,
    # else inferred from the fit Cartesian components (back-compat).
    if spin_modes is None:
        spin_modes = {}
        for b in ("1", "2"):
            present = [c for c in ("s%sx" % b, "s%sy" % b, "s%sz" % b) if c in fit_set]
            spin_modes[b] = ("sph" if len(present) == 3
                             else "cart_z" if present else None)
    body_mode = {}
    for b in ("1", "2"):
        R = Rbody[b]
        m = spin_modes.get(b)
        if m == "sph":                               # precessing -> spherical sampling
            body_mode[b] = ("sph", ["s%sx" % b, "s%sy" % b, "s%sz" % b])
            names += ["chi%s" % b, "cos_theta%s" % b, "phi%s" % b]
            lo += [0.0, -1.0, 0.0]; hi += [R, 1.0, 2.0 * math.pi]
        elif m == "cart_z":                          # aligned -> Cartesian s{b}z
            body_mode[b] = ("cart", ["s%sz" % b])
            names.append("s%sz" % b); lo.append(-R); hi.append(R)
        else:
            body_mode[b] = (None, [])
    lo = np.array(lo); hi = np.array(hi)
    nidx = {n: i for i, n in enumerate(names)}

    mass_prior = spec.get("_mass_prior", "m1m2")
    def ln_prior(theta):
        mc = theta[nidx["mc"]]
        dmc = theta[nidx["delta_mc"]]
        eta = 0.25 * (1.0 - dmc * dmc)
        if mass_prior == "flat":
            lp = mc * 0.0
        else:
            # uniform-in-(m1,m2) in the SAMPLED (delta_mc) coordinate: the
            # (1-4eta)^-1/2 factor cancels with the d eta/d delta_mc Jacobian, leaving
            # the smooth p(mc, delta_mc) ~ mc * eta^-6/5 (no equal-mass singularity).
            lp = jnp.log(mc) - 1.2 * jnp.log(eta)
        if zprior:                                   # s_component_zprior on aligned comps
            for b in ("1", "2"):
                mode, comps = body_mode[b]
                if mode == "cart":
                    R = Rbody[b]
                    for c in comps:
                        if c.endswith("z"):
                            s = theta[nidx[c]]
                            lp = lp + jnp.log(-jnp.log(jnp.abs(s) / R + 1e-7))
        return lp

    def to_fit(theta):
        # map sampled (mc, delta_mc) to the artifact's mass coordinate (eta or delta_mc)
        dmc = theta[nidx["delta_mc"]]
        mass_val = 0.25 * (1.0 - dmc * dmc) if mass2_fit == "eta" else dmc
        vals = {"mc": theta[nidx["mc"]], mass2_fit: mass_val}
        for b in ("1", "2"):
            mode, comps = body_mode[b]
            if mode == "sph":
                chi = theta[nidx["chi%s" % b]]
                ct = theta[nidx["cos_theta%s" % b]]
                ph = theta[nidx["phi%s" % b]]
                st = jnp.sqrt(jnp.clip(1.0 - ct * ct, 0.0, 1.0))
                vals["s%sz" % b] = chi * ct
                vals["s%sx" % b] = chi * st * jnp.cos(ph)
                vals["s%sy" % b] = chi * st * jnp.sin(ph)
            elif mode == "cart":
                for c in comps:
                    vals[c] = theta[nidx[c]]
        # aligned-spin principal axes, if the artifact was fit in them
        if "chi_eff" in fit_set or "chiMinus" in fit_set:
            eta_v = 0.25 * (1.0 - dmc * dmc)
            mtot = vals["mc"] * eta_v ** (-3.0 / 5.0)
            m1 = 0.5 * mtot * (1.0 + dmc); m2 = 0.5 * mtot * (1.0 - dmc)
            s1z = vals.get("s1z", 0.0 * dmc); s2z = vals.get("s2z", 0.0 * dmc)
            M = m1 + m2
            vals["chi_eff"] = (m1 * s1z + m2 * s2z) / M
            vals["chiMinus"] = (m1 * s1z - m2 * s2z) / M
        return jnp.stack([vals[n] for n in fit_names])

    def raw_to_sample(X_raw, raw_names):
        ridx = {n: i for i, n in enumerate(raw_names)}
        m1 = X_raw[:, ridx["m1"]]; m2 = X_raw[:, ridx["m2"]]
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        dmc = (m1 - m2) / (m1 + m2)             # sampling coordinate is delta_mc
        cols = [mc, dmc]
        for b in ("1", "2"):
            mode, comps = body_mode[b]
            if mode == "sph":
                sx = X_raw[:, ridx["s%sx" % b]]; sy = X_raw[:, ridx["s%sy" % b]]
                sz = X_raw[:, ridx["s%sz" % b]]
                chi = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
                ct = np.where(chi > 1e-12, sz / np.clip(chi, 1e-12, None), 0.0)
                ph = np.mod(np.arctan2(sy, sx), 2.0 * np.pi)
                cols += [chi, ct, ph]
            elif mode == "cart":
                for c in comps:
                    cols.append(X_raw[:, ridx[c]])
        return np.column_stack(cols)

    def to_compare(S):
        """Samples (in sampling coords) -> every physical comparison parameter:
        masses (mc, eta, q), aligned spin combos (chi_eff, chiMinus), and the
        cylindrical-polar spin of each body (s{b}z, chi{b}_perp, phi{b})."""
        mc = S[:, nidx["mc"]]; dmc = S[:, nidx["delta_mc"]]
        eta = 0.25 * (1.0 - dmc ** 2)
        mtot = mc * eta ** (-3.0 / 5.0)
        m1 = 0.5 * mtot * (1.0 + dmc); m2 = 0.5 * mtot * (1.0 - dmc)
        out = {"mc": mc, "q": m2 / m1, "eta": eta, "m1": m1, "m2": m2}
        for b in ("1", "2"):
            mode, _ = body_mode[b]
            if mode == "sph":
                chi = S[:, nidx["chi%s" % b]]; ct = S[:, nidx["cos_theta%s" % b]]
                out["s%sz" % b] = chi * ct
                out["chi%s_perp" % b] = chi * np.sqrt(np.clip(1.0 - ct ** 2, 0.0, 1.0))
                out["phi%s" % b] = S[:, nidx["phi%s" % b]]
            elif mode == "cart" and ("s%sz" % b) in nidx:
                out["s%sz" % b] = S[:, nidx["s%sz" % b]]
            else:
                out["s%sz" % b] = np.zeros_like(m1)
        out["chi_eff"] = (m1 * out["s1z"] + m2 * out["s2z"]) / (m1 + m2)
        out["chiMinus"] = (m1 * out["s1z"] - m2 * out["s2z"]) / (m1 + m2)
        return out

    return {"names": names, "lo": lo, "hi": hi, "ln_prior": ln_prior,
            "to_fit": to_fit, "raw_to_sample": raw_to_sample,
            "to_compare": to_compare}


# --------------------------------------------------------------------------- #
# 4. fit + export the artifact (the deliverable)
# --------------------------------------------------------------------------- #

def fit_and_export(spec, out_base, method="svgp", sigma_cut=0.6, lnL_offset=40.0,
                   cap_points=8000, n_features=256, n_opt_steps=300, seed=0,
                   quadgp_residual="svgp", keep_curv_frac=0.01,
                   ls_lo_frac=0.2, ls_hi_frac=1.0, mass_coord="eta",
                   spin_coord="auto"):
    """Build, persist and cold-reload-verify a differentiable lnL artifact for the
    run's ``all.net``. Returns a metadata dict (also the fit-coord arrays needed by
    validation, under private keys).

    ``spin_coord="auto"`` (default) fits BOTH ``aligned_eff`` (chi_eff/chiMinus
    principal axes -- fixes the sharp low-mass aligned spin) and ``cartesian``
    (s1z/s2z) and keeps whichever has the lower peak-region holdout RMSE.  This is the
    "never regress" guard: aligned_eff is a large net win but worse on a minority of
    events, and holdout RMSE reliably picks the better of the two (it selects cartesian
    exactly where aligned_eff would regress).  Pass an explicit ``aligned_eff`` /
    ``cartesian`` to skip the selection (half the fit cost)."""
    import shutil
    if spin_coord == "auto":
        cands = []
        for sc in ("aligned_eff", "cartesian"):
            m = fit_and_export(
                spec, out_base + "__" + sc, method=method, sigma_cut=sigma_cut,
                lnL_offset=lnL_offset, cap_points=cap_points, n_features=n_features,
                n_opt_steps=n_opt_steps, seed=seed, quadgp_residual=quadgp_residual,
                keep_curv_frac=keep_curv_frac, ls_lo_frac=ls_lo_frac,
                ls_hi_frac=ls_hi_frac, mass_coord=mass_coord, spin_coord=sc)
            cands.append((m["holdout_rmse"], sc, m))
        cands.sort(key=lambda t: t[0])
        best = cands[0][2]
        for ext in (".npz", ".meta.json"):           # promote the winner to out_base
            shutil.copyfile(best["out_base"] + ext, out_base + ext)
        for _, sc, _ in cands:                        # clean up the candidate exports
            for ext in (".npz", ".meta.json"):
                try:
                    os.remove(out_base + "__" + sc + ext)
                except OSError:
                    pass
        best["out_base"] = out_base
        best["spin_coord"] = cands[0][1]
        best["spin_coord_auto"] = {sc: round(r, 3) for r, sc, _ in cands}
        return best

    from RIFT.interpolators.jax_gp import get_interpolator, export
    from RIFT.interpolators.jax_gp.benchmark.datasets import load_ile_net
    from RIFT.interpolators.jax_gp.applications.jax_cip import _tree_ring_select

    # 1. load the intrinsic block with the *detected* layout, sigma-cut + dedupe
    X_raw, y, yerr, _ = load_ile_net(
        spec["net"], fit_params=tuple(spec["intrinsic_names"]),
        cols=spec["cols"], sigma_cut=sigma_cut, return_errors=True)

    # how each body's spin is SAMPLED is set by the raw physics (which spin columns
    # vary), independent of the fit representation (Cartesian vs chi_eff/chiMinus).
    raw_names = list(spec["intrinsic_names"])
    ridx = {n: i for i, n in enumerate(raw_names)}
    spin_modes = {}
    for b in ("1", "2"):
        comps = [c for c in ("s%sx" % b, "s%sy" % b, "s%sz" % b) if c in ridx]
        varies = [c for c in comps if X_raw[:, ridx[c]].std() > _CONST_TOL]
        inplane = any(c.endswith(("x", "y")) for c in varies)
        spin_modes[b] = ("sph" if inplane else "cart_z" if varies else None)

    # 2. physical fit coordinates; drop columns that do not vary (record them)
    Xfit_all, fit_names_all = raw_to_fit(X_raw, raw_names,
                                         mass_coord=mass_coord, spin_coord=spin_coord)
    spread = Xfit_all.std(axis=0)
    keep = spread > _CONST_TOL
    keep[0] = keep[1] = True                          # always keep mc, delta_mc
    fit_names = [n for n, k in zip(fit_names_all, keep) if k]
    constants = {n: float(Xfit_all[:, i].mean())
                 for i, (n, k) in enumerate(zip(fit_names_all, keep)) if not k}
    Xfit = Xfit_all[:, keep]

    # 3. high-lnL region + stratified ("tree-ring") downselect to bound fit cost
    #    (carry the raw intrinsic columns through the same masks: validation needs
    #    them to build the proposal in RIFT's sampling coordinates)
    ok = np.all(np.isfinite(Xfit), axis=1) & np.isfinite(y) & np.isfinite(yerr)
    Xfit, y, yerr, X_raw = Xfit[ok], y[ok], yerr[ok], X_raw[ok]
    lnL_max = float(np.max(y))
    band = y > lnL_max - lnL_offset
    Xfit, y, yerr, X_raw = Xfit[band], y[band], yerr[band], X_raw[band]
    if cap_points and len(y) > cap_points:
        sel = _tree_ring_select(y, cap_points, seed=seed)
        Xfit, y, yerr, X_raw = Xfit[sel], y[sel], yerr[sel], X_raw[sel]

    # 4. honest 15% holdout
    rng = np.random.default_rng(seed)
    n = len(y)
    perm = rng.permutation(n)
    n_hold = max(1, int(round(0.15 * n)))
    ho, tr = perm[:n_hold], perm[n_hold:]
    Xtr, ytr, etr, Xho, yho = Xfit[tr], y[tr], yerr[tr], Xfit[ho], y[ho]

    # 5. fit the interpolator with the per-point MC errors
    cls = get_interpolator(method)
    if method in ("rff", "gp-jax-rff"):
        model = cls(n_features=n_features, n_opt_steps=n_opt_steps, seed=seed)
    elif method in ("svgp", "gp-jax-svgp"):
        model = cls(n_inducing=n_features, n_opt_steps=n_opt_steps, seed=seed)
    elif method == "quadgp":
        # forward only the kwargs the chosen residual backend accepts (via **gp_kwargs)
        if quadgp_residual == "svgp":
            gpkw = dict(n_inducing=n_features, seed=seed,
                        ls_lo_frac=ls_lo_frac, ls_hi_frac=ls_hi_frac)
        elif quadgp_residual == "rff":
            gpkw = dict(n_features=n_features, seed=seed)
        else:                                   # exact: no inducing/seed kwargs
            gpkw = {}
        model = cls(gp_method=quadgp_residual, n_opt_steps=n_opt_steps,
                    keep_curv_frac=keep_curv_frac, **gpkw)
    else:
        model = cls(n_opt_steps=n_opt_steps)
    model = model.fit(Xtr, ytr, y_errors=etr)
    model.coord_names = list(fit_names)

    # 6. export, reload, and prove the saved bytes are faithful + differentiable
    export.save(model, out_base, coord_names=fit_names,
                extra_meta={"constants": constants, "event": spec["event"],
                            "tag": spec["tag"], "net": spec["net"]})
    reloaded = export.load(out_base)
    p_reload = reloaded.predict(Xho)
    if not np.allclose(model.predict(Xho), p_reload, rtol=1e-5, atol=1e-4):
        raise AssertionError("reloaded predict() disagrees with the fitted model")
    import jax
    import jax.numpy as jnp
    g = np.asarray(jax.grad(reloaded.lnL_physical)(jnp.asarray(Xtr[0])))
    if not np.all(np.isfinite(g)):
        raise AssertionError("jax.grad of reloaded lnL is not finite")
    # Headline holdout RMSE is over the PE-relevant peak region (within 15 nats of the
    # peak); the eta quadratic core extrapolates steeply in the deep low-lnL tail
    # (~zero posterior weight), which would otherwise dominate a plain RMSE.
    rmse_all = float(np.sqrt(np.mean((p_reload - yho) ** 2)))
    peak = yho > (lnL_max - 15.0)
    holdout_rmse = float(np.sqrt(np.mean((p_reload[peak] - yho[peak]) ** 2))) \
        if peak.any() else rmse_all

    meta = {
        "out_base": out_base, "method": method, "coord_names": fit_names,
        "constants": constants, "n_train": int(len(ytr)),
        "n_holdout": int(len(yho)), "lnL_max": lnL_max,
        "holdout_rmse": holdout_rmse, "holdout_rmse_all": rmse_all,
        "mass_coord": mass_coord, "spin_coord": spin_coord,
        "keep_curv_frac": keep_curv_frac,
        "grad_finite": True, "n_intrinsic_dims": len(fit_names),
    }
    # private handoff to validation (not serialised in the public report verbatim)
    meta["_fit_names"] = fit_names
    meta["_Xfit"] = Xfit
    meta["_y"] = y
    meta["_X_raw"] = X_raw
    meta["_raw_names"] = list(spec["intrinsic_names"])
    meta["_spin_modes"] = spin_modes
    return meta


# --------------------------------------------------------------------------- #
# 5. validation: sample the reloaded artifact, compare to the CIP posterior
# --------------------------------------------------------------------------- #

def _load_posterior_dat(path):
    """Load a RIFT ``posterior_samples-*.dat`` / ``extrinsic_posterior_samples.dat``
    into a name->array dict using its ``# ...`` header, and derive the full intrinsic
    comparison set (mc, eta, q, chi_eff, chiMinus, and cylindrical-polar spins
    s{b}z, chi{b}_perp, phi{b}) so it lines up with :func:`build_sampling`'s
    ``to_compare``."""
    header = None
    with open(path) as fh:
        for ln in fh:
            if ln.strip().startswith("#"):
                header = ln.lstrip("#").split()
                break
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    if header is None or len(header) != data.shape[1]:
        # headerless / mismatched: fall back to the canonical CIP column order
        header = ["m1", "m2", "a1x", "a1y", "a1z", "a2x", "a2y", "a2z",
                  "mc", "eta", "indx", "Npts", "ra", "dec", "tref", "phiorb",
                  "incl", "psi", "dist", "p", "ps", "lnL", "mtotal", "q"]
        header = header[:data.shape[1]]
    cols = {n: data[:, i] for i, n in enumerate(header)}
    if {"m1", "m2"} <= cols.keys():
        m1, m2 = cols["m1"], cols["m2"]
        cols.setdefault("mc", (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)
        cols.setdefault("q", np.minimum(m1, m2) / np.maximum(m1, m2))
        cols.setdefault("eta", m1 * m2 / (m1 + m2) ** 2)
        # cylindrical-polar spins from the Cartesian a{b}{x,y,z} columns
        for b in ("1", "2"):
            ax, ay, az = "a%sx" % b, "a%sy" % b, "a%sz" % b
            if {ax, ay, az} <= cols.keys():
                cols.setdefault("s%sz" % b, cols[az])
                cols.setdefault("chi%s_perp" % b,
                                np.sqrt(cols[ax] ** 2 + cols[ay] ** 2))
                cols.setdefault("phi%s" % b,
                                np.mod(np.arctan2(cols[ay], cols[ax]), 2 * np.pi))
        if {"a1z", "a2z"} <= cols.keys():
            cols.setdefault("chi_eff", (m1 * cols["a1z"] + m2 * cols["a2z"]) / (m1 + m2))
            cols.setdefault("chiMinus", (m1 * cols["a1z"] - m2 * cols["a2z"]) / (m1 + m2))
    return cols


def _sample_flow_v060(target, lo, hi, init_theta=None, n_samples=8000, n_chains=30,
                      n_train_loops=6, n_prod_loops=2, n_epochs=12, seed=0):
    """flowMC (>=0.6.0) normalizing-flow importance sampling on the box [lo,hi].

    A self-contained replacement for the legacy ``jax_cip.sample_flow_is`` (whose
    positional ``Sampler(...)`` call broke under flowMC 0.6.0's keyword-only API).
    Trains an RQ-spline flow over a sigmoid-into-box latent, then i.i.d. draws +
    importance weights ``exp(target + log_jac - log_q)``. Use in ``rift_ad_export``.
    """
    import jax
    import jax.numpy as jnp
    from flowMC.Sampler import Sampler
    from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle

    lo = jnp.asarray(lo); hi = jnp.asarray(hi); span = hi - lo
    d = int(lo.shape[0])

    def theta_of_u(u):
        return lo + span * jax.nn.sigmoid(u)

    def log_jac(u):
        return jnp.sum(jnp.log(span) + jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u))

    def u_logpdf(u, data=None):
        return target(theta_of_u(u)) + log_jac(u)

    key = jax.random.PRNGKey(seed)
    key, kb, ks, ki, kd = jax.random.split(key, 5)
    bundle = RQSpline_MALA_Bundle(
        rng_key=kb, n_chains=n_chains, n_dims=d, logpdf=u_logpdf,
        n_local_steps=50, n_global_steps=50, n_training_loops=n_train_loops,
        n_production_loops=n_prod_loops, n_epochs=n_epochs)
    sampler = Sampler(n_dim=d, n_chains=n_chains, rng_key=ks,
                      resource_strategy_bundles=bundle)
    if init_theta is not None:
        frac = np.clip((np.asarray(init_theta, float) - np.asarray(lo))
                       / np.asarray(span), 1e-3, 1 - 1e-3)
        u0 = np.log(frac / (1 - frac))
        init = jnp.asarray(u0)[None, :] + 0.3 * jax.random.normal(ki, (n_chains, d))
    else:
        init = 0.3 * jax.random.normal(ki, (n_chains, d))
    sampler.sample(init, {})

    flow = sampler.resources["model"]
    u = jnp.asarray(flow.sample(kd, n_samples))
    theta = np.asarray(jax.vmap(theta_of_u)(u))
    log_q = np.asarray(jax.vmap(flow.log_prob)(u))    # log_prob is per-sample in 0.6.0
    log_p = np.asarray(jax.jit(jax.vmap(u_logpdf))(u))
    log_w = np.array(log_p - log_q, dtype=np.float64)
    m = np.max(log_w)
    logZ = float(m + np.log(np.mean(np.exp(log_w - m))))
    w = np.exp(log_w - m); w = w / w.sum()
    ess = float(1.0 / np.sum(w ** 2))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_samples, size=min(8000, n_samples), replace=True, p=w)
    samples = theta[idx]
    return {"samples": samples, "ess": ess, "ess_frac": ess / n_samples,
            "logZ": logZ, "mean": samples.mean(0), "std": samples.std(0)}


#: every intrinsic parameter the validation reports a JS on. Masses, the aligned-spin
#: combinations (chi_eff, chiMinus), and the cylindrical-polar spin of each body
#: (aligned component s{b}z, in-plane magnitude chi{b}_perp, azimuth phi{b}).
ALL_COMPARE_PARAMS = (
    "mc", "eta", "q", "chi_eff", "chiMinus",
    "s1z", "chi1_perp", "phi1", "s2z", "chi2_perp", "phi2",
)


def validate_artifact(spec, fit_meta, out_dir, n_samples=40000, inflate=1.2,
                      seed=0, sampler="auto", compare_params=ALL_COMPARE_PARAMS):
    """Draw a posterior from the *reloaded* artifact and JS-compare its marginals to
    the run's CIP posterior. Writes ``posterior_interp.dat`` and returns the report.

    The target is the run's actual posterior ``lnL(theta) + ln prior(theta)`` -- NOT a
    flat-prior caricature -- so the comparison is apples-to-apples with CIP.  Sampling
    is done in RIFT's own coordinates (:func:`build_sampling`): spins in
    ``(chi, cos_theta, phi)``, where RIFT's isotropic prior is flat, plus the
    non-uniform mass-prior shape (``mc_prior`` ~ mc, ``eta_prior`` ~ eta^-6/5).  This
    both matches CIP's measure and avoids the Cartesian-spin 1/chi^2 singularity.

    ``sampler``: ``"gaussian"`` -- fast mu-matched importance sampling (great in low
    dimension); ``"nuts"`` -- gradient-based NUTS preconditioned with the data
    covariance, which exploits the artifact's AD gradients and explores the curved,
    high-dimensional precessing posterior far better than a single Gaussian proposal;
    ``"auto"`` (default) picks ``nuts`` when there are >3 sampling dimensions, else
    ``gaussian``.
    """
    from RIFT.interpolators.jax_gp import export
    from RIFT.interpolators.jax_gp.applications.jax_cip import (
        sample_gaussian_is, sample_nuts_muframe)
    from RIFT.interpolators.jax_gp.applications.compare import js_with_stderr

    import jax.numpy as jnp
    fit_names = fit_meta["_fit_names"]
    X_raw, raw_names = fit_meta["_X_raw"], fit_meta["_raw_names"]
    y = fit_meta["_y"]
    reloaded = export.load(fit_meta["out_base"])

    # Sample in RIFT's OWN coordinates + measure (apples-to-apples): spins in
    # (chi,cos_theta,phi) where the isotropic prior is flat, plus the mass-prior
    # shape. The NUTS/IS target is lnL(theta) + ln prior(theta) -- exactly the CIP
    # posterior, not a flat-prior caricature.
    smp = build_sampling(spec, fit_names, spin_modes=fit_meta.get("_spin_modes"))
    names, lo, hi = smp["names"], smp["lo"], smp["hi"]
    Xs = smp["raw_to_sample"](X_raw, raw_names)        # ILE data in sampling coords

    def target(theta):
        return reloaded.lnL_physical(smp["to_fit"](theta)) + smp["ln_prior"](theta)

    # proposal/preconditioner: lnL-weighted mean+cov of the data in sampling coords,
    # restricted to the prior box
    inb = np.all((Xs >= lo) & (Xs <= hi), axis=1)
    Xp, yp = (Xs[inb], y[inb]) if inb.sum() >= 10 else (Xs, y)
    w = np.exp(yp - yp.max()); w /= w.sum()
    gmean = (Xp * w[:, None]).sum(0)
    gcov = np.atleast_2d(np.cov(Xp.T, aweights=w))
    if gcov.shape[0] == 1:                       # 1-D: cov() returns a scalar
        gcov = gcov.reshape(1, 1)

    if sampler == "auto":
        sampler = "nuts" if len(names) > 3 else "gaussian"
    if sampler == "flow":
        # normalizing-flow IS (flowMC >=0.6.0); needs the rift_ad_export env.
        res = _sample_flow_v060(target, lo, hi, init_theta=gmean,
                                n_samples=max(8000, spec["n_output_samples"]),
                                seed=seed)
    elif sampler == "nuts":
        # Gradient-based NUTS, dense mass matrix seeded from the data covariance.
        # Not proposal-limited -> explores the curved, weakly-constrained precessing
        # directions; uses the artifact's jax.grad lnL (+ prior) directly.
        ndraw = max(2000, spec["n_output_samples"])
        res = sample_nuts_muframe(target, gmean, gcov, lo, hi, num_warmup=1000,
                                  num_samples=ndraw, num_chains=2, seed=seed)
    else:
        res = sample_gaussian_is(target, gmean, gcov, lo, hi,
                                 n_samples=n_samples, inflate=inflate, seed=seed)
    samples = res["samples"]                      # [n, d] in sampling coords

    # spin-magnitude constraint is automatic: chi in [0,R] by the box, and the
    # spherical map keeps |spin| = chi <= R. Map to physical comparison params.
    phys = smp["to_compare"](samples)

    # write the interpolated posterior (RIFT-ish .dat, every intrinsic column we have)
    os.makedirs(out_dir, exist_ok=True)
    post_path = os.path.join(out_dir, "posterior_interp.dat")
    out_names = [n for n in ("m1", "m2", "mc", "eta", "q", "chi_eff", "chiMinus",
                             "s1z", "chi1_perp", "phi1", "s2z", "chi2_perp", "phi2")
                 if n in phys]
    np.savetxt(post_path, np.column_stack([phys[n] for n in out_names]),
               header=" ".join(out_names))

    # JS divergence vs the run's own CIP posterior
    js = {}
    ref = None
    if spec["posterior"] and os.path.exists(spec["posterior"]):
        ref = _load_posterior_dat(spec["posterior"])
        for prm in compare_params:
            if prm in phys and prm in ref and len(phys[prm]) > 20:
                a, b = phys[prm], ref[prm]
                if np.std(a) < 1e-9 and np.std(b) < 1e-9:
                    js[prm] = {"js_bits": 0.0, "js_stderr": 0.0, "degenerate": True,
                               "interp_mean": float(np.mean(a)), "interp_std": 0.0,
                               "ref_mean": float(np.mean(b)), "ref_std": 0.0,
                               "n_interp": int(len(a)), "n_ref": int(len(b))}
                    continue
                val, se = js_with_stderr(a, b)
                js[prm] = {
                    "js_bits": val, "js_stderr": se,
                    "interp_mean": float(np.mean(a)), "ref_mean": float(np.mean(b)),
                    "interp_std": float(np.std(a)), "ref_std": float(np.std(b)),
                    "n_interp": int(len(a)), "n_ref": int(len(b)),
                }
    # Honesty about sampling: a JS computed from few *independent* draws is noisy
    # regardless of the (resampled-with-replacement) bootstrap stderr. Flag it.
    ess = float(res.get("ess", float("nan")))
    if not np.isfinite(ess) or ess < 100:
        quality = "sampling-limited"
    elif ess < 400:
        quality = "marginal"
    else:
        quality = "ok"
    return {
        "posterior_interp": post_path,
        "reference_posterior": spec["posterior"],
        "n_posterior_samples": int(len(samples)),
        "sampler": sampler,
        "is_ess": ess,
        "is_ess_frac": float(res.get("ess_frac", float("nan"))),
        "logZ": res.get("logZ"),
        "quality": quality,
        "js": js,
    }


# --------------------------------------------------------------------------- #
# 6. orchestration: one run end-to-end
# --------------------------------------------------------------------------- #

def run_one(run_dir, workroot, method="quadgp", n_samples=40000, seed=0,
            cap_points=8000, n_features=256, n_opt_steps=300, lnL_offset=40.0,
            sigma_cut=0.6, sampler="auto", keep_curv_frac=0.01,
            ls_lo_frac=0.2, ls_hi_frac=1.0, mass_coord="eta",
            spin_coord="auto", write_plot=True):
    """Discover -> fit+export -> validate one run; write all artifacts under
    ``workroot/<label>/`` and return the full report."""
    import RIFT.interpolators.jax_gp  # noqa: F401  (enables float64)

    spec = discover_run(run_dir)
    out_dir = os.path.join(os.path.abspath(workroot), spec["label"])
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, "lnL_artifact")

    t0 = time.time()
    fit_meta = fit_and_export(
        spec, out_base, method=method, sigma_cut=sigma_cut,
        lnL_offset=lnL_offset, cap_points=cap_points, n_features=n_features,
        n_opt_steps=n_opt_steps, seed=seed, keep_curv_frac=keep_curv_frac,
        ls_lo_frac=ls_lo_frac, ls_hi_frac=ls_hi_frac, mass_coord=mass_coord,
        spin_coord=spin_coord)
    t_fit = time.time() - t0

    t1 = time.time()
    val = validate_artifact(spec, fit_meta, out_dir, n_samples=n_samples,
                            seed=seed, sampler=sampler)
    t_val = time.time() - t1

    public_fit = {k: v for k, v in fit_meta.items() if not k.startswith("_")}
    report = {
        "run_dir": spec["run_dir"], "event": spec["event"], "tag": spec["tag"],
        "net": spec["net"], "ncols": spec["ncols"],
        "intrinsic_names": spec["intrinsic_names"], "precessing": spec["precessing"],
        "mc_range": spec["mc_range"], "eta_range": spec["eta_range"],
        "chi_max": spec["chi_max"], "out_dir": out_dir,
        # use case (c): flag distance-export runs. This export is intrinsic-only
        # (all.net); the (intrinsic + distance) "dgrid" surrogate is a separate,
        # forthcoming track -- recorded here so a run isn't silently half-covered.
        "has_dgrid": spec.get("has_dgrid", False),
        "fit": public_fit, "validation": val,
        "timing_sec": {"fit": t_fit, "validate": t_val},
    }
    if spec.get("has_dgrid"):
        report["dgrid_note"] = ("distance-grid files detected; the intrinsic all.net "
                                "export is complete, the (intrinsic+distance) export "
                                "is a separate track (not yet implemented)")
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    _write_summary_md(report, os.path.join(out_dir, "summary.md"))
    if write_plot:
        try:
            _write_corner(spec, fit_meta, val, out_dir)
        except Exception as e:        # plotting must never fail the run
            report["plot_error"] = str(e)
    return report


def _write_summary_md(report, path):
    v = report["validation"]
    lines = [
        "# lnL export + validation: {} / {}".format(report["event"], report["tag"]),
        "",
        "- run dir: `{}`".format(report["run_dir"]),
        "- all.net columns: {}  intrinsic: {}".format(
            report["ncols"], ", ".join(report["intrinsic_names"])),
        "- precessing: {}   fit dims: {} ({})".format(
            report["precessing"], report["fit"]["n_intrinsic_dims"],
            ", ".join(report["fit"]["coord_names"])),
        "- method: {}   train pts: {}   holdout RMSE: {:.3f} nats".format(
            report["fit"]["method"], report["fit"]["n_train"],
            report["fit"]["holdout_rmse"]),
        "- artifact: `{}.npz` (+ .meta.json)".format(report["fit"]["out_base"]),
        ("- distance-export (dgrid) detected: intrinsic all.net export done; "
         "(intrinsic+distance) export is a separate forthcoming track."
         if report.get("has_dgrid") else ""),
        "- reference posterior: `{}`".format(v["reference_posterior"]),
        "- interpolated posterior: `{}`  ({} samples)".format(
            v["posterior_interp"], v["n_posterior_samples"]),
        "- sampler: {}   IS ESS: {:.0f}   quality: **{}**".format(
            v.get("sampler", "?"), v.get("is_ess", 0), v.get("quality", "?")),
        ("" if v.get("quality") == "ok" else
         "  > NOTE: low effective sample count -- the JS values below are "
         "sampling-limited (noisy/upper bounds), not necessarily surrogate error. "
         "Re-run with `--sampler nuts`, more `--n-samples`, or `--method quadgp`."),
        "",
        "## Jensen-Shannon divergence vs CIP posterior (bits; 0 = identical)",
        "",
        "| param | JS (bits) | interp mean±std | ref mean±std |",
        "|---|---|---|---|",
    ]
    for prm, d in v["js"].items():
        lines.append("| {} | {:.4f} ± {:.4f} | {:.4g} ± {:.3g} | {:.4g} ± {:.3g} |"
                      .format(prm, d["js_bits"], d["js_stderr"],
                              d["interp_mean"], d["interp_std"],
                              d["ref_mean"], d["ref_std"]))
    if not v["js"]:
        lines.append("| (no reference posterior found) | | | |")
    open(path, "w").write("\n".join(lines) + "\n")


def _write_corner(spec, fit_meta, val, out_dir):
    """Overlay interpolated-vs-reference 1D marginals (matplotlib only; optional)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    phys_ref = (_load_posterior_dat(spec["posterior"])
                if spec["posterior"] and os.path.exists(spec["posterior"]) else {})
    # every reported parameter, in a tidy grid (skip degenerate/constant directions)
    params = [p for p in ALL_COMPARE_PARAMS
              if p in val["js"] and not val["js"][p].get("degenerate")]
    if not params:
        return
    interp = np.loadtxt(val["posterior_interp"])
    hdr = open(val["posterior_interp"]).readline().lstrip("#").split()
    icols = {n: interp[:, i] for i, n in enumerate(hdr)}
    ncol = min(4, len(params))
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.8 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, prm in zip(axes, params):
        if prm in icols:
            ax.hist(icols[prm], bins=50, density=True, histtype="step",
                    label="interp artifact", lw=2)
        if prm in phys_ref:
            ax.hist(phys_ref[prm], bins=50, density=True, histtype="step",
                    label="CIP posterior", lw=2)
        ax.set_xlabel(prm)
        ax.set_title("JS={:.4f}".format(val["js"][prm]["js_bits"]), fontsize=9)
    for ax in axes[len(params):]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle("{} / {}".format(spec["event"], spec["tag"]))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "marginals.png"), dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 7. batch / condor fan-out
# --------------------------------------------------------------------------- #

def expand_runs(patterns):
    """Expand globs/paths to run directories that actually contain a usable all.net."""
    runs = []
    for pat in patterns:
        hits = glob.glob(pat) or [pat]
        for h in hits:
            h = h.rstrip(os.sep)
            if os.path.isdir(h) and os.path.exists(os.path.join(h, "all.net")):
                runs.append(os.path.abspath(h))
            elif os.path.basename(h) == "all.net" and os.path.exists(h):
                runs.append(os.path.dirname(os.path.abspath(h)))
    return sorted(set(runs))


def write_condor_batch(run_dirs, workroot, method="quadgp", n_samples=40000,
                       sampler="auto", python=None, pythonpath=None,
                       accounting_group=None):
    """Emit a one-node-per-run HTCondor DAG under ``workroot/condor/``.

    The submit file is templated from the *first run's own* CIP.sub (accounting
    group / singularity image / requirements) so the validation jobs land in the
    same pool the run itself used. Returns the DAG path.
    """
    workroot = os.path.abspath(workroot)
    cdir = os.path.join(workroot, "condor")
    os.makedirs(cdir, exist_ok=True)
    python = python or sys.executable
    pythonpath = pythonpath or os.environ.get("PYTHONPATH", "")
    env0 = _parse_condor_env(run_dirs[0]) if run_dirs else {}
    acct = accounting_group or env0.get("accounting_group", "ligo.dev.o4.cbc.pe.rift")
    mod = "RIFT.interpolators.jax_gp.applications.export_at_scale"

    sub = os.path.join(cdir, "export_at_scale.sub")
    with open(sub, "w") as fh:
        fh.write(
            "universe = vanilla\n"
            "executable = {py}\n"
            "arguments = \"-m {mod} one --run-dir $(rundir) "
            "--workroot {wr} --method {m} --n-samples {ns} --sampler {smp} "
            "--no-plot\"\n"
            # NOTE: do NOT use `getenv = True` -- many pools (e.g. CIT) set
            # SUBMIT_ALLOW_GETENV=false and reject it. Set the environment explicitly;
            # the absolute conda python (executable) + PYTHONPATH are what's needed.
            # Thread caps mirror the ulimit -u thread-spawn lesson (see memory note).
            "environment = \"HOME={home} PYTHONPATH={pp} JAX_PLATFORMS=cpu "
            "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
            "XLA_FLAGS=--xla_cpu_multi_thread_eigen=false\"\n"
            "output = {cdir}/$(label).out\n"
            "error  = {cdir}/$(label).err\n"
            "log    = {cdir}/export_at_scale.log\n"
            "request_memory = {mem}\n"
            "request_disk = {disk}\n"
            "request_cpus = 2\n"
            "accounting_group = {acct}\n"
            "accounting_group_user = {user}\n"
            "queue\n".format(
                py=python, mod=mod, wr=workroot, m=method, ns=n_samples,
                smp=sampler, pp=pythonpath, cdir=cdir,
                home=os.path.expanduser("~"),
                mem=env0.get("request_memory", "8000M"),
                disk=env0.get("request_disk", "2000M"), acct=acct,
                user=env0.get("accounting_group_user",
                              os.environ.get("USER", "user"))))

    dag = os.path.join(cdir, "export_at_scale.dag")
    with open(dag, "w") as fh:
        for i, rd in enumerate(run_dirs):
            spec_label = "{}__{}".format(
                os.path.basename(os.path.dirname(rd)), os.path.basename(rd))
            jid = "export_{}".format(i)
            fh.write('JOB {jid} {sub}\n'.format(jid=jid, sub=sub))
            fh.write('VARS {jid} rundir="{rd}" label="{lab}"\n'.format(
                jid=jid, rd=rd, lab=spec_label))
    return dag


# --------------------------------------------------------------------------- #
# 8. CLI
# --------------------------------------------------------------------------- #

def _add_common(p):
    p.add_argument("--method", default="quadgp",
                   choices=["svgp", "rff", "exact", "quadgp"],
                   help="surrogate interpolator (default: quadgp -- PE-grade Fisher "
                        "quadratic core + GP residual; svgp is faster for easy/low-D)")
    p.add_argument("--n-samples", type=int, default=40000,
                   help="importance-sampling proposal draws (default: 40000)")
    p.add_argument("--sampler", default="auto",
                   choices=["auto", "gaussian", "nuts", "flow"],
                   help="validation sampler: auto (nuts if >3 fit dims, else "
                        "gaussian) | gaussian (fast IS, low-D) | nuts (gradient "
                        "NUTS, robust high-D) | flow (flowMC IS; needs rift_ad_export)")
    p.add_argument("--cap-points", type=int, default=8000)
    p.add_argument("--n-features", type=int, default=256,
                   help="SVGP inducing points / RFF features (default: 256)")
    p.add_argument("--mass-coord", default="eta", choices=("eta", "delta_mc"),
                   help="second mass coordinate the surrogate fits in (default eta). "
                        "The lnL Fisher is quadratic in (mc, eta); eta is quadratic in "
                        "delta_mc, so delta_mc hides that curvature near equal mass and "
                        "the quadratic core can't capture q. eta fixes the q marginal.")
    p.add_argument("--spin-coord", default="auto",
                   choices=("auto", "aligned_eff", "cartesian"),
                   help="aligned-spin fit coordinates. aligned_eff = chi_eff,chiMinus "
                        "(principal axes; fixes the sharp low-mass aligned spin that an "
                        "axis-aligned ARD GP over-smooths in s1z,s2z). auto (default) "
                        "fits both and keeps the lower-holdout-RMSE one -> never worse "
                        "than cartesian. cartesian = raw s1z,s2z.")
    p.add_argument("--keep-curv-frac", type=float, default=0.01,
                   help="quadgp: keep eigen-curvature directions above this fraction of "
                        "the max in the exact Fisher core (default 0.01). With "
                        "--mass-coord eta this must be small enough to RETAIN the "
                        "(gentle) eta curvature; 0.05 leaves it to the GP residual.")
    p.add_argument("--ls-hi-frac", type=float, default=1.0,
                   help="quadgp/svgp: upper bound on the ARD lengthscale as a fraction "
                        "of the peak-region width (default 1.0). LOWER it (0.3-0.6) to "
                        "force shorter lengthscales = less GP smoothing = sharper "
                        "marginals (the CIP smoothing-length analog).")
    p.add_argument("--ls-lo-frac", type=float, default=0.2,
                   help="quadgp/svgp: lower bound on the ARD lengthscale fraction "
                        "(default 0.2).")
    p.add_argument("--n-opt-steps", type=int, default=300)
    p.add_argument("--lnL-offset", type=float, default=40.0)
    p.add_argument("--sigma-cut", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="inspect a run dir; print what was found")
    d.add_argument("--run-dir", required=True)

    o = sub.add_parser("one", help="export + validate a single run")
    o.add_argument("--run-dir", required=True)
    o.add_argument("--workroot", default="./export_at_scale_out")
    o.add_argument("--no-plot", action="store_true")
    _add_common(o)

    b = sub.add_parser("batch", help="many runs: local loop or a condor DAG")
    b.add_argument("--runs", nargs="+", required=True,
                   help="run dirs or globs (e.g. '/data/*/S*/rift*/')")
    b.add_argument("--workroot", default="./export_at_scale_out")
    b.add_argument("--condor", action="store_true",
                   help="emit a condor DAG instead of running locally")
    b.add_argument("--accounting-group", default=None)
    b.add_argument("--no-plot", action="store_true")
    _add_common(b)

    args = p.parse_args(argv)

    if args.cmd == "discover":
        spec = discover_run(args.run_dir)
        spec = {k: v for k, v in spec.items() if k != "cols"}
        print(json.dumps(spec, indent=2, default=str))
        return spec

    if args.cmd == "one":
        rep = run_one(args.run_dir, args.workroot, method=args.method,
                      n_samples=args.n_samples, seed=args.seed,
                      cap_points=args.cap_points, n_features=args.n_features,
                      n_opt_steps=args.n_opt_steps, lnL_offset=args.lnL_offset,
                      sigma_cut=args.sigma_cut, sampler=args.sampler,
                      keep_curv_frac=args.keep_curv_frac,
                      ls_lo_frac=args.ls_lo_frac, ls_hi_frac=args.ls_hi_frac,
                      mass_coord=args.mass_coord, spin_coord=args.spin_coord,
                      write_plot=not args.no_plot)
        print(json.dumps({"out_dir": rep["out_dir"],
                          "holdout_rmse": rep["fit"]["holdout_rmse"],
                          "quality": rep["validation"]["quality"],
                          "is_ess": rep["validation"]["is_ess"],
                          "js": rep["validation"]["js"]}, indent=2))
        return rep

    if args.cmd == "batch":
        runs = expand_runs(args.runs)
        if not runs:
            raise SystemExit("no run dirs with all.net matched {}".format(args.runs))
        print("[batch] {} run dirs".format(len(runs)), file=sys.stderr)
        if args.condor:
            dag = write_condor_batch(runs, args.workroot, method=args.method,
                                     n_samples=args.n_samples, sampler=args.sampler,
                                     accounting_group=args.accounting_group)
            print("wrote DAG: {}\nsubmit with: condor_submit_dag {}".format(dag, dag))
            return dag
        results = []
        for rd in runs:
            try:
                rep = run_one(rd, args.workroot, method=args.method,
                              n_samples=args.n_samples, seed=args.seed,
                              cap_points=args.cap_points, n_features=args.n_features,
                              n_opt_steps=args.n_opt_steps,
                              lnL_offset=args.lnL_offset, sigma_cut=args.sigma_cut,
                              sampler=args.sampler, keep_curv_frac=args.keep_curv_frac,
                              ls_lo_frac=args.ls_lo_frac, ls_hi_frac=args.ls_hi_frac,
                              mass_coord=args.mass_coord, spin_coord=args.spin_coord,
                              write_plot=not args.no_plot)
                results.append({"run": rd, "quality": rep["validation"]["quality"],
                                "is_ess": rep["validation"]["is_ess"],
                                "js": rep["validation"]["js"]})
                print("[ok] {}".format(rd), file=sys.stderr)
            except Exception as e:
                results.append({"run": rd, "error": str(e)})
                print("[FAIL] {}: {}".format(rd, e), file=sys.stderr)
        summ = os.path.join(os.path.abspath(args.workroot), "batch_summary.json")
        os.makedirs(os.path.dirname(summ), exist_ok=True)
        json.dump(results, open(summ, "w"), indent=2)
        print("wrote {}".format(summ))
        return results


if __name__ == "__main__":
    main()
