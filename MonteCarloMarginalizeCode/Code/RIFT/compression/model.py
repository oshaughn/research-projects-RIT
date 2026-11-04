"""Weighted native-posterior mixtures with an explicit retained-prior quotient.

Likelihood amplitudes have one fixed, arbitrary event constant. Hidden nuisance
priors remain those of the generating analysis.
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from .cosmology import CosmologyContext

SCHEMA = "rift-posterior-compression-v1"


def normalized(weights):
    w = np.asarray(weights, float)
    if w.ndim != 1 or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid posterior weights")
    return w/w.sum()


def weighted_ks(a, wa, b, wb):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 1 or b.ndim != 1 or not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("KS samples must be nonempty finite one-dimensional arrays")
    if len(a) != len(wa) or len(b) != len(wb):
        raise ValueError("KS sample/weight lengths differ")
    ia, ib = np.argsort(a), np.argsort(b)
    grid = np.sort(np.r_[a, b])
    ca = np.r_[0., np.cumsum(normalized(wa)[ia])]
    cb = np.r_[0., np.cumsum(normalized(wb)[ib])]
    return float(np.max(abs(ca[np.searchsorted(a[ia], grid, side="right")]-
                            cb[np.searchsorted(b[ib], grid, side="right")])))


def split_groups(groups, seed):
    """Reproduce disjoint original-group training, tuning and audit partitions."""
    ids = np.random.default_rng(seed).permutation(np.unique(groups))
    train, tune, held = [np.isin(groups, g) for g in np.split(ids, [int(.6*len(ids)), int(.8*len(ids))])]
    return ids, train, tune, held


def fit_mixture(x, w, k, seed, maxiter):
    """Weighted EM; row multiplicity never replaces original information weights."""
    w = normalized(w)
    rng = np.random.default_rng(seed)
    center = w@x
    scale = np.sqrt(w@((x-center)**2))
    if np.any(scale < 1e-12):
        raise ValueError("Constant coordinate: request a lower-dimensional target")
    y = (x-center)/scale
    n, d = y.shape
    means = [y[rng.choice(n, p=w)]]
    for _ in range(1, k):
        distances = np.min([np.sum((y-m)**2, axis=1) for m in means], axis=0)*w
        means.append(y[rng.choice(n, p=normalized(distances))])
    means = np.array(means)
    covs = np.repeat(np.eye(d)[None], k, axis=0)
    weights = np.full(k, 1./k)
    previous = -np.inf
    converged = False
    for iteration in range(maxiter):
        terms = []
        for weight, mean, cov in zip(weights, means, covs):
            factor = np.linalg.cholesky(cov)
            z = solve_triangular(factor, (y-mean).T, lower=True).T
            terms.append(np.log(weight)-.5*np.sum(z*z, axis=1)-np.log(np.diag(factor)).sum()-.5*d*np.log(2*np.pi))
        terms = np.array(terms).T
        total = logsumexp(terms, axis=1)
        objective = w@total
        if iteration > 4 and abs(objective-previous) < 1e-6:
            converged = True
            break
        previous = objective
        r = np.exp(terms-total[:, None])*w[:, None]
        counts = r.sum(axis=0)
        if np.any(counts < 1e-10):
            raise ValueError("Collapsed mixture component")
        weights = counts/counts.sum()
        means = r.T@y/counts[:, None]
        for j in range(k):
            dx = y-means[j]
            covs[j] = (dx*r[:, j, None]).T@dx/counts[j]+np.eye(d)*1e-6
    return dict(weights=weights.tolist(), means=(means*scale+center).tolist(),
                covariances=(covs*scale[None, :, None]*scale[None, None, :]).tolist(),
                converged=converged, iterations=iteration+1)


class CompressedFit:
    def __init__(self, artifact):
        if artifact.get("schema") != SCHEMA:
            raise ValueError("Unsupported compression schema")
        self.artifact = artifact
        self.contract = artifact["contract"]
        self.target = artifact["target"]
        if self.target not in ("aligned_distance", "precessing_source_redshift"):
            raise ValueError("Unknown target")
        self.dimension = 5 if self.target == "aligned_distance" else 9
        self.cosmology = CosmologyContext.from_yaml(artifact["cosmology_yaml"],
                                                   self.contract["bounds"]["distance"][1])
        g = artifact["mixture"]
        self.weights = np.asarray(g["weights"], float)
        self.means = np.asarray(g["means"], float)
        self.covs = np.asarray(g["covariances"], float)
        k, d = len(self.weights), self.dimension
        if self.means.shape != (k, d) or self.covs.shape != (k, d, d) or not np.isfinite(self.means).all() or not np.isfinite(self.covs).all():
            raise ValueError("Invalid mixture arrays")
        if not np.isfinite(self.weights).all() or np.any(self.weights <= 0) or not np.isclose(self.weights.sum(), 1.):
            raise ValueError("Invalid mixture weights")
        if not np.allclose(self.covs, self.covs.transpose(0, 2, 1)):
            raise ValueError("Covariances must be symmetric")
        self.factors = [np.linalg.cholesky(c) for c in self.covs]
        self.constants = np.log(self.weights)-.5*d*np.log(2*np.pi)-np.array([np.log(np.diag(c)).sum() for c in self.factors])

    @classmethod
    def load(cls, path, *, allow_failed=False):
        artifact = json.loads(Path(path).read_text())
        if not allow_failed and artifact.get("validation", {}).get("status") != "pass_native_recovery":
            raise ValueError("Candidate failed or lacks native recovery; allow_failed=True is diagnostic access only")
        return cls(artifact)

    def _native(self, points, mass_frame, radial_coordinate, cosmology):
        a = np.asarray(points, float)
        scalar = a.ndim == 1
        if a.ndim not in (1, 2) or a.shape[-1] != self.dimension:
            raise ValueError("Expected points with final dimension " + str(self.dimension))
        x = a.reshape(-1, self.dimension).copy()
        ctx = cosmology or self.cosmology
        if radial_coordinate == "redshift":
            z = x[:, -1].copy()
            x[:, -1] = ctx.luminosity_distance(z)
        elif radial_coordinate == "luminosity_distance":
            z = ctx.redshift(x[:, -1]) if mass_frame == "source" else None
        else:
            raise ValueError("Unknown radial coordinate")
        if mass_frame == "source":
            x[:, 0] *= 1+z
        elif mass_frame != "detector":
            raise ValueError("Unknown mass frame")
        # Native chart is Mc_detector, delta=sqrt(1-4 eta), spins, dL.
        eta = x[:, 1].copy()
        x[:, 1] = np.sqrt(np.maximum(0., 1-4*eta))
        physical = np.isfinite(x).all(axis=1) & (eta > 0) & (eta <= .25)
        return x, physical, scalar

    def support(self, x):
        b = self.contract["bounds"]
        eta = (1-x[:, 1]**2)/4
        good = np.isfinite(x).all(1) & (x[:, 1] >= 0) & (x[:, 1] < 1)
        for value, limits in ((x[:, 0], b["mc"]), (eta, b["eta"]), (x[:, -1], b["distance"])):
            good &= (value >= limits[0]) & (value <= limits[1])
        if self.dimension == 9:
            radii = np.column_stack([np.linalg.norm(x[:, 2:5], axis=1), np.linalg.norm(x[:, 5:8], axis=1)])
            good &= (radii <= self.contract["spin_caps"]).all(1)
            sz = x[:, [4, 7]]
        else:
            sz = x[:, 2:4]
            good &= (abs(sz) <= self.contract["spin_caps"]).all(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            total = x[:, 0]*eta**(-.6)
            m1, m2 = total*(1+x[:, 1])/2, total*(1-x[:, 1])/2
            q = m2/m1
        values = dict(m1=m1, m2=m2, mtot=total, mc=x[:, 0], eta=eta, q=q, s1z=sz[:, 0], s2z=sz[:, 1])
        if self.dimension == 9:
            values.update(chi1=radii[:, 0], chi2=radii[:, 1])
        elif not self.contract["precessing"]:
            values.update(chi1=abs(sz[:, 0]), chi2=abs(sz[:, 1]))
        for key, limits in self.contract.get("cuts", {}).items():
            if self.dimension == 5 and self.contract["precessing"] and key in ("chi1", "chi2"):
                good &= abs(sz[:, int(key[-1])-1]) <= limits[1]
                continue
            if key not in values:
                raise ValueError("Cannot marginalize a coupled spin-magnitude cut in an aligned product")
            good &= (values[key] >= limits[0]) & (values[key] <= limits[1])
        return good

    def log_chart_density(self, x):
        out = []
        for start in range(0, len(x), 8192):
            a = x[start:start+8192]
            terms = [c-.5*np.sum(solve_triangular(L, (a-m).T, lower=True).T**2, axis=1)
                     for c, m, L in zip(self.constants, self.means, self.factors)]
            out.append(logsumexp(terms, axis=0))
        return np.concatenate(out) if out else np.empty(0)

    def log_native_prior_chart(self, x):
        eta = (1-x[:, 1]**2)/4
        # |d(m1,m2)/d(Mc,delta)|, finite at equal mass.
        lp = np.log(x[:, 0])-1.2*np.log(eta)-np.log(2.)
        prior = self.contract["spin_prior"]
        caps = self.contract["spin_caps"]
        if self.dimension == 9:
            if prior == "uniform_mag_prec":
                r2 = np.column_stack([np.sum(x[:, 2:5]**2, 1), np.sum(x[:, 5:8]**2, 1)])
                if np.any(r2 == 0):
                    raise ValueError("Exact zero-spin limit of this density chart is uncertified")
                lp -= np.log(r2).sum(1)
            elif prior != "volumetric":
                raise ValueError("Full-spin target requires a precessing native spin prior")
        else:
            sz = x[:, 2:4]
            radial = np.array([self.contract.get("cuts", {}).get("chi"+str(i+1), [0., cap]) for i, cap in enumerate(caps)])
            if self.contract["precessing"]:
                radial[:, 1] = np.minimum(radial[:, 1], caps)
                if np.any(radial[:, 0] < 0) or np.any(radial[:, 0] >= radial[:, 1]):
                    raise ValueError("Invalid native spin-shell support")
            if prior == "volumetric":
                density = radial[:, 1]**2-np.maximum(sz**2, radial[:, 0]**2)
            elif prior == "uniform_mag_prec":
                if np.any((sz == 0) & (radial[:, 0] == 0)):
                    raise ValueError("Exact projected-spin prior cusp is uncertified")
                density = np.log(radial[:, 1]/np.maximum(abs(sz), radial[:, 0]))
            elif prior == "zprior_aligned":
                density = -np.log(np.maximum(abs(sz)/np.asarray(caps), 1e-7))
            elif prior == "uniform_aligned":
                density = np.ones_like(sz)
            else:
                raise ValueError("Unsupported spin prior")
            if np.any(density <= 0):
                raise ValueError("Spin prior boundary limit is uncertified")
            lp += np.log(density).sum(1)
        return lp+self.cosmology.distance_logprior(x[:, -1], self.contract["distance_prior"])

    def log_likelihood(self, points, *, mass_frame="source", radial_coordinate=None, cosmology=None):
        if radial_coordinate is None:
            radial_coordinate = "redshift" if self.dimension == 9 else "luminosity_distance"
        x, physical, scalar = self._native(points, mass_frame, radial_coordinate, cosmology)
        ok = physical & self.support(x)
        out = np.full(len(x), -np.inf)
        if ok.any():
            out[ok] = self.log_chart_density(x[ok])-self.log_native_prior_chart(x[ok])
        return float(out[0]) if scalar else out

    def sample_native(self, n, seed=73):
        """Native posterior draws in detector Mc, eta, spins, dL."""
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        rng = np.random.default_rng(seed)
        pieces, count = [], 0
        for _ in range(100):
            labels = rng.choice(len(self.weights), max(n-count, 1024), p=self.weights)
            x = np.empty((len(labels), self.dimension))
            for j, (mean, factor) in enumerate(zip(self.means, self.factors)):
                mask = labels == j
                x[mask] = mean+rng.normal(size=(mask.sum(), self.dimension))@factor.T
            x = x[self.support(x)]
            pieces.append(x)
            count += len(x)
            if count >= n:
                result = np.concatenate(pieces)[:n]
                result[:, 1] = (1-result[:, 1]**2)/4
                return result
        raise ValueError("Mixture has inadequate physical support for posterior draws")

    def integration_bank(self, n=65536, seed=73, cosmology=None):
        """Fixed proposal bank in source component masses, spins and redshift.

        Unconditioned proposal draws retain the original denominator n. Removing
        unsupported points does not renormalize the integral or event amplitude.
        """
        if n < 1:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        labels = rng.choice(len(self.weights), n, p=self.weights)
        x = np.empty((n, self.dimension))
        for j, (mean, factor) in enumerate(zip(self.means, self.factors)):
            mask = labels == j
            x[mask] = mean+rng.normal(size=(mask.sum(), self.dimension))@factor.T
        x = x[self.support(x)]
        ctx = cosmology or self.cosmology
        z = ctx.redshift(x[:, -1])
        eta = (1-x[:, 1]**2)/4
        total = x[:, 0]*eta**(-.6)/(1+z)
        y = x.copy()
        y[:, 0], y[:, 1], y[:, -1] = total*(1+x[:, 1])/2, total*(1-x[:, 1])/2, z
        log_jac = np.log(x[:, 0])-1.2*np.log(eta)-np.log(2.)-2*np.log1p(z)-np.log(ctx.distance_derivative(z))
        logw = -self.log_native_prior_chart(x)+log_jac-np.log(n)
        if not np.isfinite(logw).all():
            raise ValueError("Unresolved integration-bank transform")
        return IntegrationBank(y, logw, dict(proposal_draws=n, supported_draws=len(x), seed=seed,
            measure="dm1_source dm2_source " + ("ds1z ds2z" if self.dimension == 5 else "d6_cartesian_spin") + " dz",
            amplitude="Fixed arbitrary event constant; no absolute evidence"))


class IntegrationBank:
    def __init__(self, points, log_weights, metadata):
        self.points, self.log_weights, self.metadata = points, log_weights, metadata

    def integrate_log_density(self, log_density):
        values = np.asarray(log_density(self.points), float)
        if values.shape != self.log_weights.shape or np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError("Population function must return one finite or -inf log density per row")
        terms = values+self.log_weights
        log_integral = float(logsumexp(terms))
        ess = float(np.exp(2*log_integral-logsumexp(2*terms))) if np.isfinite(log_integral) else 0.
        return log_integral, dict(self.metadata, importance_ess=ess)


def fit_posterior(points, weights, groups, contract, target, *, components=(1, 2, 4, 8), seed=711, maxiter=200, tolerance=.08):
    points = np.asarray(points, float)
    weights = normalized(weights)
    groups = np.asarray(groups)
    if groups.shape != weights.shape or len(points) != len(weights):
        raise ValueError("Inconsistent row/weight/group lengths")
    ctx = CosmologyContext(contract["cosmology"], contract["bounds"]["distance"][1])
    template = dict(schema=SCHEMA, target=target, contract=contract, cosmology_yaml=ctx.serialized,
                    mixture=dict(weights=[1.], means=[np.zeros(points.shape[1]).tolist()],
                                 covariances=[np.eye(points.shape[1]).tolist()]))
    guard = CompressedFit(template)
    x, physical, _ = guard._native(points, "detector", "luminosity_distance", None)
    if not np.all(physical & guard.support(x)):
        raise ValueError("Posterior rows violate declared physical analysis support")
    ids, train, tune, held = split_groups(groups, seed)
    if len(ids) < 100:
        raise ValueError("At least 100 original intrinsic groups are required")
    candidates, best = [], None
    for k in components:
        if train.sum() < 10*k*points.shape[1]:
            continue
        mixture = fit_mixture(x[train], weights[train], k, seed+k, maxiter)
        model = CompressedFit(dict(template, mixture=mixture))
        score = float(normalized(weights[tune])@model.log_chart_density(x[tune]))
        candidates.append(dict(components=k, tuning_log_density=score, converged=mixture["converged"]))
        if mixture["converged"] and (best is None or score > best[0]):
            best = score, model
    if best is None:
        raise ValueError("No converged mixture within the configured component/iteration budget")
    model = best[1]
    draw = model.sample_native(max(10000, int(held.sum())), seed+1000)
    reference = points[held]
    rng = np.random.default_rng(seed+1001)
    scale = np.std(reference, axis=0)
    scale = np.where(scale > 0, scale, 1.)
    projections = np.r_[np.eye(points.shape[1]), rng.normal(size=(16, points.shape[1]))]
    scores = [weighted_ks((reference/scale)@v, weights[held], (draw/scale)@v, np.ones(len(draw))) for v in projections]
    group_weights = np.array([weights[held & (groups == group)].sum() for group in ids[int(.8*len(ids)):]])
    ess = float(1/np.sum(normalized(group_weights)**2))
    report = dict(status="pass_native_recovery" if max(scores) <= tolerance and ess >= 100 else "failed_validation",
                  scope="Native posterior marginals and random joint projections on disjoint original groups; arbitrary-prior tails remain unvalidated",
                  maximum_ks=max(scores), ks=scores, threshold=tolerance,
                  heldout_group_ess=ess, candidates=candidates, seed=seed,
                  groups=dict(training=int(len(ids)*.6), tuning=int(len(ids)*.2), heldout=len(group_weights)))
    model.artifact["validation"] = report
    return model, report
