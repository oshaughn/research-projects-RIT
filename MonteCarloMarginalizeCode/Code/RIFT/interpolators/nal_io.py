"""nal_io -- generic reader/evaluator for Normal-Approximate-Likelihood (NAL) artifacts.

Motivation.  RIFT can already CONSUME a stored quadratic, but only through ad-hoc h5py reads inside
`util_ConstructIntrinsicPosterior_GenericCoordinates.fit_quadratic_stored` (`--fit-load-quadratic`):
the coordinates are implicit in the run configuration, nothing declares a frame or a cosmology, and
the reader is not importable from anywhere else.  This module is the generic form -- a NAL is just
a bounded multivariate normal in a NAMED chart, so it should be loadable, evaluable and summable
without going through a driver.

The intended entry point is RIFT's existing plugin hook, which both
`util_ConstructIntrinsicPosterior_GenericCoordinates.py` and `util_ConstructEOSPosterior.py`
expose identically:

    --supplementary-likelihood-factor-code   RIFT.interpolators.nal_io
    --supplementary-likelihood-factor-function  nal_lnL
    --supplementary-likelihood-factor-ini    my_nal.ini

The driver calls `prepare_nal_lnL(config=<ConfigParser>, coords=<coord_names>)` once, then adds
`nal_lnL(*x)` to its own lnL, where `x` is one array per sampling coordinate in `coord_names`
order.  Because the contribution is purely additive, a run whose own data file is a DUMMY (a
placeholder net-marg grid) and whose entire likelihood comes from this plugin is a legitimate and
already-used configuration -- that is how the hook itself has been exercised inside CIP.

NOTE the prepare hook is only reached on RIFT >= the fix in `test_supplementary_likelihood_hook.py`;
before that commit both drivers misspelt the assignment and `--supplementary-likelihood-factor-ini`
raised `TypeError: 'NoneType' object is not callable`.  A plugin with no `prepare_` function was
unaffected, so `nal_lnL` still works without an ini on older RIFT if configured by environment.

WHAT A NAL IS HERE, precisely
    ln L(theta) = lnL_peak - 1/2 (theta - mu)^T Gamma (theta - mu),   theta in B (bounds box)
    zero outside B.
Gamma is the Fisher matrix (-Hessian of lnL), matching `BayesianLeastSquares.fit_quadratic` and the
`lnL_gamma.dat` sidecar.  Equivalently Sigma = Gamma^-1.

TRUNCATION.  A NAL is used as a TRUNCATED normal: the consumer only ever evaluates it at physically
realisable parameters.  Two consequences that are easy to get wrong and are handled explicitly here:
  * the truncation normalisation is a per-event CONSTANT (it depends only on that event's own mu and
    Gamma, not on the hyperparameters being sampled), so it cancels in any hyper-posterior.  It is
    therefore NOT applied by default; `renormalize=True` is available for standalone use where the
    absolute scale matters.
  * the fitted `mu` may legitimately lie OUTSIDE the physical domain.  That is the standard device
    for representing a likelihood that rails against a boundary (e.g. eta -> 1/4), not a sign of a
    broken fit, and `mu` must never be read as "the measured value".  Nothing here rejects an
    artifact on that basis.

Environment: pure numpy; h5py only if the gwalk view is used.
"""
import glob as _glob
import json
import os

import numpy as np

def _rng(seed):
    """numpy Generator when available, else a legacy RandomState.

    `default_rng` needs numpy >= 1.17; RIFT is deployed into a range of environments and this
    module is otherwise dependency-free, so it should not be the thing that breaks on an old one.
    Only used for the truncation-mass Monte Carlo, where stream equivalence does not matter.
    """
    try:
        return np.random.default_rng(seed)
    except AttributeError:                                # pragma: no cover - old numpy
        return np.random.RandomState(seed)


__all__ = ["NAL", "NALSet", "load_nal", "load_nal_dir", "write_nal",
           "nal_lnL", "prepare_nal_lnL", "write_gwalk_view", "SCHEMA_VERSION"]

SCHEMA_VERSION = 2

# Charts this module knows how to build from RIFT's native parameters.  Definitions are taken from
# RIFT/lalsimutils.py, NOT from any design document:
#   xi       (:961-966)  dot(Lhat, m1*chi1Vec + m2*chi2Vec)/(m1+m2), Lhat = zhat
#   chiMinus (:971-976)  dot(Lhat, m1*chi1Vec - m2*chi2Vec)/(m1+m2)   <- MASS WEIGHTED
#   delta_mc (:587-590)  eta = (1 - delta^2)/4, i.e. delta = sqrt(1-4 eta)
KNOWN_COORDS = ("mc", "eta", "delta_mc", "xi", "chiMinus",
                "s1z", "s2z", "s1x_bar", "s1y_bar", "s2x_bar", "s2y_bar", "u_d", "dist")


def _derive(name, have):
    """Best-effort derivation of one chart coordinate from a dict of available arrays.

    Deliberately narrow: it covers the mass/aligned-spin identities CIP actually samples in, and
    raises a NAMED error otherwise rather than silently returning something plausible.
    """
    if name in have:
        return have[name]
    g = have.get
    if name == "delta_mc" and "eta" in have:
        return np.sqrt(np.maximum(1.0 - 4.0 * np.asarray(g("eta"), float), 0.0))
    if name == "eta" and "delta_mc" in have:
        return 0.25 * (1.0 - np.asarray(g("delta_mc"), float) ** 2)
    if name in ("xi", "chiMinus") and all(k in have for k in ("s1z", "s2z")) \
            and ("q" in have or "delta_mc" in have or "eta" in have):
        if "q" in have:
            q = np.asarray(g("q"), float)
        else:
            d = _derive("delta_mc", have)
            q = (1.0 - d) / (1.0 + d)
        s1z, s2z = np.asarray(g("s1z"), float), np.asarray(g("s2z"), float)
        # m1/M = 1/(1+q), m2/M = q/(1+q)
        return (s1z + q * s2z) / (1.0 + q) if name == "xi" else (s1z - q * s2z) / (1.0 + q)
    if name == "u_d" and "dist" in have:
        return 1.0 / np.asarray(g("dist"), float)
    if name == "dist" and "u_d" in have:
        return 1.0 / np.asarray(g("u_d"), float)
    raise KeyError(
        "nal_io: cannot build chart coordinate %r from the sampler coordinates %s. Either sample "
        "in a chart that contains it, or extend _derive()." % (name, sorted(have)))


class NAL(object):
    """One event's bounded multivariate-normal likelihood in a declared chart."""

    def __init__(self, mu, gamma, coord_names, lnL_peak=0.0, bounds=None, meta=None):
        self.mu = np.asarray(mu, float).ravel()
        self.gamma = np.asarray(gamma, float)
        self.gamma = 0.5 * (self.gamma + self.gamma.T)
        self.coord_names = list(coord_names)
        self.lnL_peak = float(lnL_peak)
        self.bounds = None if bounds is None else np.asarray(bounds, float)
        self.meta = dict(meta or {})
        d = len(self.mu)
        if self.gamma.shape != (d, d):
            raise ValueError("nal_io: gamma shape %s does not match mu (%d)"
                             % (self.gamma.shape, d))
        if len(self.coord_names) != d:
            raise ValueError("nal_io: %d coord_names for %d-dimensional NAL"
                             % (len(self.coord_names), d))

    @property
    def ndim(self):
        return len(self.mu)

    def cov(self):
        return np.linalg.inv(self.gamma)

    def marginal(self, keep):
        """Marginal NAL over a subset of coordinates, by name or index.

        Uses Sigma = Gamma^-1 and takes the SUB-BLOCK -- equivalently the Schur complement
        Gamma_AA - Gamma_AB Gamma_BB^-1 Gamma_BA.  This is the MARGINAL.  Taking `Gamma_AA`
        instead would give the CONDITIONAL (nuisance held fixed), which is systematically too
        narrow; they are easy to confuse and are not the same object.
        """
        idx = [self.coord_names.index(k) if isinstance(k, str) else int(k) for k in keep]
        S = self.cov()[np.ix_(idx, idx)]
        b = None if self.bounds is None else self.bounds[idx]
        return NAL(self.mu[idx], np.linalg.inv(S), [self.coord_names[i] for i in idx],
                   lnL_peak=self.lnL_peak, bounds=b, meta=self.meta)

    def lnL(self, theta, renormalize=False):
        """ln L at theta, shape (N, ndim) or (ndim,).  Outside `bounds` returns -inf."""
        X = np.atleast_2d(np.asarray(theta, float))
        d = X - self.mu
        out = self.lnL_peak - 0.5 * np.einsum("ij,jk,ik->i", d, self.gamma, d)
        if self.bounds is not None:
            inside = np.all((X >= self.bounds[:, 0]) & (X <= self.bounds[:, 1]), axis=1)
            out = np.where(inside, out, -np.inf)
        if renormalize:
            out = out - self._log_mass()
        return out

    def _log_mass(self, n=200000, seed=0):
        """log of the Gaussian mass inside `bounds`, by Monte Carlo.

        Deliberately NOT a product of 1-D marginal masses.  That factorisation ignores correlations
        and is badly biased for a correlated fit -- measured against brute force on a 3-D Gaussian
        with rho = 0.9 pairwise, the factorised value is 42% low.  This is a per-event CONSTANT and
        cancels in any hyper-posterior, which is why `renormalize` defaults to False.
        """
        if self.bounds is None:
            return 0.0
        rng = _rng(seed)
        G = rng.multivariate_normal(self.mu, self.cov(), size=n)
        f = np.all((G >= self.bounds[:, 0]) & (G <= self.bounds[:, 1]), axis=1).mean()
        return float(np.log(max(f, 1.0 / n)))


class NALSet(object):
    """A catalogue of NALs, summed.  Each event contributes additively in lnL."""

    def __init__(self, nals):
        self.nals = list(nals)
        if not self.nals:
            raise ValueError("nal_io: empty NALSet")
        ch = {tuple(n.coord_names) for n in self.nals}
        if len(ch) != 1:
            raise ValueError("nal_io: NALSet requires one common chart, got %s" % sorted(ch))
        self.coord_names = list(self.nals[0].coord_names)

    def lnL(self, theta, renormalize=False):
        X = np.atleast_2d(np.asarray(theta, float))
        tot = np.zeros(len(X))
        for n in self.nals:
            tot = tot + n.lnL(X, renormalize=renormalize)
        return tot


def load_nal(path):
    """Load one artifact.  Accepts <base>.npz (with sidecar <base>.meta.json) or the .meta.json."""
    base = path[:-len(".meta.json")] if path.endswith(".meta.json") else \
        (path[:-4] if path.endswith(".npz") else path)
    meta = {}
    if os.path.exists(base + ".meta.json"):
        meta = json.load(open(base + ".meta.json"))
    d = np.load(base + ".npz", allow_pickle=False)
    names = meta.get("coord_names")
    if names is None:
        raise KeyError("nal_io: %s.meta.json must declare coord_names -- a NAL without a named "
                       "chart is not interpretable" % base)
    g = d["gamma"] if "gamma" in d else np.linalg.inv(d["cov"])
    return NAL(d["theta_star"], g, names,
               lnL_peak=float(meta.get("lnL_peak", 0.0)),
               bounds=d["bounds"] if "bounds" in d else None, meta=meta)


def load_nal_dir(pattern):
    """Load every artifact matching a glob (e.g. '/path/*.npz'), sorted for reproducibility."""
    out = [load_nal(p) for p in sorted(_glob.glob(pattern))]
    if not out:
        raise IOError("nal_io: no artifacts matched %r" % pattern)
    return out


def _sha256(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _git_sha(path):
    """Short git sha of the tree `path` lives in, or None.  Best-effort provenance only."""
    import subprocess
    try:
        d = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path)) or "."
        out = subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def check_frame_invariant(coord_names, frame, cosmology=None, d_prior=None):
    """An artifact may carry EITHER the distance coordinate OR source-frame masses, never both.

    A distance-marginalised DETECTOR-frame quadratic is not reusable: it has silently integrated
    the mass-redshift degeneracy against one particular distance prior, and nothing downstream can
    undo that or even detect it.  So:

      * u_d present  =>  frame must be 'detector' (distance has not been marginalised yet)
      * frame source =>  u_d must be absent, AND the cosmology and the distance prior that was
                         integrated must both be recorded, since the source-frame masses are
                         meaningless without them.

    Raises ValueError rather than warning: an artifact that cannot state its own frame honestly
    should not be written at all.  (The shipped O3/O4 NAL catalogue records none of this -- its
    npz carries only names/labels/centers/sigs/covs/ess/method/kl -- so a consumer cannot tell
    which cosmology produced it.)
    """
    has_ud = "u_d" in coord_names
    if frame not in ("detector", "source"):
        raise ValueError("nal_io: frame must be 'detector' or 'source', got %r" % (frame,))
    if has_ud and frame != "detector":
        raise ValueError("nal_io: chart carries u_d, so masses are detector-frame; got frame=%r"
                         % (frame,))
    if frame == "source":
        if has_ud:
            raise ValueError("nal_io: frame='source' must not also carry u_d")
        if not cosmology:
            raise ValueError("nal_io: frame='source' requires a declared cosmology")
        if not d_prior:
            raise ValueError("nal_io: frame='source' requires the distance prior that was "
                             "integrated out (name and range)")


def write_nal(base, nal, chart=None, frame="detector", cosmology=None, d_prior=None,
              symmetry=None, unconstrained_dirs=None, validation=None, parents=None,
              run_id=None, lnL_ref="noise_hypothesis_ratio", extra=None):
    """Write <base>.npz + <base>.meta.json.

    `parents` should be the source files the fit consumed (all.net / all_dslice.dat); their
    sha256 is recorded so an artifact can always be traced to the grid that produced it.
    """
    coord_names = list(nal.coord_names)
    check_frame_invariant(coord_names, frame, cosmology, d_prior)
    bounds = nal.bounds
    if bounds is None:
        sd = np.sqrt(np.diag(nal.cov()))
        bounds = np.stack([nal.mu - 10 * sd, nal.mu + 10 * sd], 1)
    np.savez(base + ".npz", theta_star=nal.mu, gamma=nal.gamma, bounds=np.asarray(bounds))
    meta = {
        "schema": SCHEMA_VERSION, "method": "nal",
        "chart": chart or nal.meta.get("chart"), "coord_names": coord_names,
        "frame": frame, "cosmology": cosmology, "d_prior": d_prior,
        "lnL_peak": float(nal.lnL_peak), "lnL_ref": lnL_ref,
        "symmetry": symmetry, "unconstrained_dirs": list(unconstrained_dirs or []),
        "parents": [{"path": p, "sha256": _sha256(p)} for p in (parents or [])
                    if os.path.exists(p)],
        "run_id": run_id, "git_sha": _git_sha(__file__),
        "validation": dict(validation or {}),
    }
    if extra:
        meta.update(extra)
    with open(base + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return base + ".npz", base + ".meta.json"


def write_gwalk_view(path, nal, label, scale_max=None):
    """Emit the gwalk HDF5 view for one NAL.

    Two gwalk behaviours this deliberately works around, both verified against gwalk 2.3.0:
      * `offset` is asserted into [-scale_max, scale_max] with scale_max defaulting to 500, and a
        loud event's lnL_peak-derived offset exceeds that (lnL_peak ~ SNR^2/2, so SNR > ~32 trips
        it).  scale_max is therefore written explicitly, sized to the value being stored.
      * gwalk's own `normalize()` recomputes `offset` from a product of 1-D marginal masses that
        ignores correlations, and OVERWRITES whatever was stored.  Consumers must not call it;
        the exact lnL_peak is carried in `offset` here instead.
    """
    import h5py
    Sig = nal.cov()
    sd = np.sqrt(np.diag(Sig))
    cor = Sig / np.outer(sd, sd)
    D = nal.ndim
    sign, logdet = np.linalg.slogdet(nal.gamma)
    offset = nal.lnL_peak + 0.5 * D * np.log(2 * np.pi) - 0.5 * logdet
    if scale_max is None:
        scale_max = max(500.0, 2.0 * abs(offset))
    with h5py.File(path, "a") as f:
        grp = f.require_group(label)
        for k, v in (("mu", nal.mu), ("std", sd), ("cor", cor), ("cov", Sig),
                     ("limits", nal.bounds if nal.bounds is not None
                      else np.stack([nal.mu - 10 * sd, nal.mu + 10 * sd], 1)),
                     ("offset", np.array([offset])), ("scale", np.array([1.0]))):
            if k in grp:
                del grp[k]
            grp.create_dataset(k, data=np.asarray(v))
        grp.attrs["ndim"] = D
        grp.attrs["scale_max"] = scale_max
        grp.attrs["coord_names"] = json.dumps(nal.coord_names)
    return offset


# ----------------------------------------------------------------- RIFT plugin hook entry points
_STATE = {"set": None, "coords": None, "renormalize": False}


def prepare_nal_lnL(config=None, coords=None):
    """Called once by CIP / EOSPosterior with the parsed ini and the run's coordinate names.

    ini section:
        [nal]
        artifacts = /path/to/*.npz      ; glob, or a comma-separated list
        renormalize = false             ; per-event constant, cancels in a hyper-posterior
    Falls back to the environment variable RIFT_NAL_ARTIFACTS when no ini is supplied, so the
    plugin also works on RIFT versions predating the prepare-hook fix.
    """
    pat = os.environ.get("RIFT_NAL_ARTIFACTS")
    if config is not None and config.has_section("nal"):
        if config.has_option("nal", "artifacts"):
            pat = config.get("nal", "artifacts")
        if config.has_option("nal", "renormalize"):
            _STATE["renormalize"] = config.get("nal", "renormalize").strip().lower() \
                in ("1", "true", "yes")
    if not pat:
        raise ValueError("nal_io: no artifacts configured -- set [nal] artifacts in the ini or "
                         "the RIFT_NAL_ARTIFACTS environment variable")
    nals = []
    for part in pat.split(","):
        part = part.strip()
        nals += load_nal_dir(part) if any(c in part for c in "*?[") else [load_nal(part)]
    _STATE["set"] = NALSet(nals)
    _STATE["coords"] = list(coords) if coords is not None else None
    print("nal_io: loaded %d NAL artifact(s), chart %s; sampler coords %s"
          % (len(nals), _STATE["set"].coord_names, _STATE["coords"]))


def nal_lnL(*x):
    """Additive lnL contribution.  `x` is one array per sampling coordinate, in coord_names order.

    Matches the calling convention in util_ConstructIntrinsicPosterior_GenericCoordinates.py:2952
    and util_ConstructEOSPosterior.py:946 (`log_likelihood_function(*x) + supplemental(*x)`).
    """
    if _STATE["set"] is None:
        prepare_nal_lnL(config=None, coords=None)
    S = _STATE["set"]
    names = _STATE["coords"]
    arrs = [np.atleast_1d(np.asarray(a, float)).ravel() for a in x]
    if names is None:
        if len(arrs) != len(S.coord_names):
            raise ValueError("nal_io: called with %d coordinates but the artifact chart has %d, "
                             "and no coord_names were supplied to prepare_nal_lnL"
                             % (len(arrs), len(S.coord_names)))
        have = dict(zip(S.coord_names, arrs))
    else:
        have = dict(zip(names, arrs))
    theta = np.stack([_derive(k, have) for k in S.coord_names], 1)
    return S.lnL(theta, renormalize=_STATE["renormalize"])
