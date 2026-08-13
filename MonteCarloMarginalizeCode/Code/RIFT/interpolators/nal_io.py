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
unaffected.  On such a RIFT the artifacts can still be supplied through RIFT_NAL_ARTIFACTS, but the
SAMPLING BASIS must then be declared too, as RIFT_NAL_SAMPLER_COORDS='mc,eta,...': without it the
plugin does not know what the arrays it is handed are called, and refuses to guess (see
`nal_lnL`).  Guessing is not conservative -- an artifact in (mc, delta_mc) fed a sampler in
(mc, eta) passes every dimension check while evaluating eta as delta_mc.

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

THE RUN'S OWN CHART.  Artifact metadata cannot say what the SAMPLER is walking in, and the
coordinate names do not distinguish it: 'mc' and 'delta_mc' are spelt identically in a
detector-frame and a source-frame chart.  Comparing artifacts with each other (`check_set_compatible`)
therefore does not protect a run, and a single artifact is not compared with anything at all.  The
run must declare its own frame and chart -- `[nal] sampler_frame` / `sampler_chart`, or
RIFT_NAL_SAMPLER_FRAME / RIFT_NAL_SAMPLER_CHART -- and they are checked against EVERY artifact,
including a lone one (see `check_sampler_compatible`).  No conversion is attempted: mapping between
frames needs a redshift per sample, which the plugin is never handed.

SCALE OF THE CONTRIBUTION.  `nal_lnL` returns the artifacts' lnL with their summed peak SUBTRACTED,
so it is never positive.  Both drivers' default path evaluates
`likelihood_function(*x) * np.exp(supplemental_ln_likelihood(*x))`, and float64 exp overflows above
~709: a real loud-event artifact (lnL_peak ~ SNR^2/2, e.g. 3386 for SNR ~ 82) would silently become
inf there, and the drivers' own `lnL_shift` does not reach this separate exponentiation.  The
subtracted constant is a fixed multiplicative factor on the likelihood: it cancels in any posterior
but it does NOT cancel in an absolute lnL or an evidence, which is what `integral_result.dat` and
the `_lnL.dat` sidecar are read as -- an odds ratio against a run without this factor would be
wrong by exp(offset).  It is reported at preparation and exposed as `nal_lnL_offset()`, following
the same `<function>_...` naming convention as the `prepare_<function>` hook; both drivers look for
that function and ADD the constant back into every absolute likelihood and evidence they write, so
the centring stays inside the sampler where it is needed.

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
           "nal_lnL", "nal_lnL_offset", "prepare_nal_lnL", "write_gwalk_view",
           "check_frame_invariant", "check_artifact_frame_invariant",
           "check_set_compatible", "check_sampler_compatible",
           "SCHEMA_VERSION"]

SCHEMA_VERSION = 2

# A dropped coordinate whose bounds sit at least this many marginal sigma from mu on BOTH sides is
# treated as unbounded when marginalising: the Gaussian tail beyond 5 sigma is 2.9e-7 per side.
_UNBOUNDED_SIGMA = 5.0
# Correlation below this counts as none: the truncation factor is then a constant, not theta-dependent.
_CORR_TOL = 1e-8
# An eigenvalue of gamma at or below this fraction of the largest one counts as zero (see
# NAL._check_positive_definite).  Relative, so it is invariant under a rescaling of lnL.
_PD_RTOL = 1e-12

# Charts this module knows how to build from RIFT's native parameters.  Definitions are taken from
# RIFT/lalsimutils.py, NOT from any design document:
#   xi       (:961-966)  dot(Lhat, m1*chi1Vec + m2*chi2Vec)/(m1+m2), Lhat = zhat
#   chiMinus (:971-976)  dot(Lhat, m1*chi1Vec - m2*chi2Vec)/(m1+m2)   <- MASS WEIGHTED
#   delta_mc (:587-590)  eta = (1 - delta^2)/4, i.e. delta = sqrt(1-4 eta)
KNOWN_COORDS = ("mc", "eta", "delta_mc", "xi", "chiMinus",
                "s1z", "s2z", "s1x_bar", "s1y_bar", "s2x_bar", "s2y_bar", "u_d", "dist")

# Names that carry the luminosity distance.  BOTH of them: `_derive` treats u_d = 1/dist as
# interchangeable, so a chart carrying `dist` says exactly as much about the distance as one
# carrying `u_d`, and the frame invariant must recognise either.
_DISTANCE_COORDS = ("u_d", "dist")

# Metadata keys `write_nal` owns -- either validated here (frame/cosmology/d_prior, through
# check_frame_invariant) or derived from the artifact itself.  `extra` may not overwrite them: the
# value that was checked and the value that is recorded must be the same value.
_RESERVED_META_KEYS = ("schema", "method", "chart", "coord_names", "frame", "cosmology", "d_prior",
                       "lnL_peak", "lnL_ref", "symmetry", "unconstrained_dirs", "parents",
                       "run_id", "git_sha", "validation")


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
        self.source = None                                # set by load_nal(); for error messages
        self._log_mass_cache = None                       # (settings, value); see log_mass()
        d = len(self.mu)
        if self.gamma.shape != (d, d):
            raise ValueError("nal_io: gamma shape %s does not match mu (%d)"
                             % (self.gamma.shape, d))
        if len(self.coord_names) != d:
            raise ValueError("nal_io: %d coord_names for %d-dimensional NAL"
                             % (len(self.coord_names), d))
        self._check_positive_definite()

    def _check_positive_definite(self):
        """`gamma` must be finite and positive definite, or this is not a peaked likelihood.

        Not a formality.  A negative eigenvalue makes lnL INCREASE away from mu along that
        direction: the object is a saddle, `lnL_peak` is not the peak, and `_peak_offset` -- which
        the plugin relies on to keep the drivers' `np.exp(supplemental)` in range -- is no longer an
        upper bound on anything, so the overflow protection silently stops protecting.  Nothing
        downstream can notice: the arithmetic is finite and the array shapes are right.  A fitter
        that has not converged, or one that fitted a boundary-railed direction badly, produces
        exactly this, so it is a realistic input rather than a hypothetical one.

        SINGULAR IS ALSO REJECTED, explicitly.  `cov()` -- hence `marginal`, `log_mass`,
        `write_gwalk_view` and the truncation machinery -- needs Gamma^-1, and a numerically
        singular Gamma inverts to garbage rather than to an error.  A genuinely unconstrained
        direction is a real thing, but it is not representable as a normalizable likelihood without
        bounds; it belongs in the artifact's `unconstrained_dirs` metadata, with the direction
        removed from the chart.

        The threshold is RELATIVE to the largest eigenvalue, so it is invariant under an overall
        rescaling of lnL and does not depend on the units of the chart.  It is set far above the
        ~1e-16 relative error of a symmetric eigendecomposition and far below the conditioning of
        any usable fit, so it separates "zero" from "small" without rejecting the honestly
        ill-conditioned mass/spin blocks these fits produce.
        """
        if not np.all(np.isfinite(self.gamma)):
            raise ValueError("nal_io: gamma contains non-finite entries -- an unconverged or "
                             "failed fit, not a likelihood")
        if not np.all(np.isfinite(self.mu)):
            raise ValueError("nal_io: mu contains non-finite entries")
        w = np.linalg.eigvalsh(self.gamma)
        scale = float(np.max(np.abs(w))) if len(w) else 0.0
        if scale <= 0.0:
            raise ValueError("nal_io: gamma is identically zero -- no likelihood is defined")
        if w[0] < -_PD_RTOL * scale:
            raise ValueError(
                "nal_io: gamma is not positive definite (eigenvalues %s): lnL would INCREASE away "
                "from mu along the negative direction, so this is a saddle and lnL_peak is not the "
                "peak. The plugin's overflow guard subtracts a bound built from lnL_peak, which "
                "such an artifact does not respect. Refit, or drop the unconverged direction."
                % np.array2string(w, precision=4))
        if w[0] <= _PD_RTOL * scale:
            raise ValueError(
                "nal_io: gamma is numerically singular (eigenvalues %s, smallest is %.3g of the "
                "largest): it has an unconstrained direction, and Gamma^-1 -- needed by cov(), "
                "marginal(), log_mass() and the gwalk view -- would be numerically meaningless "
                "rather than an error. A flat direction is not a normalizable likelihood: remove it "
                "from the chart and record it in the artifact's 'unconstrained_dirs', or bound it "
                "and refit." % (np.array2string(w, precision=4), w[0] / scale))

    @property
    def ndim(self):
        return len(self.mu)

    def cov(self):
        return np.linalg.inv(self.gamma)

    def marginal(self, keep, ignore_truncation=False, shape_only=False):
        """Marginal NAL over a subset of coordinates, by name or index.

        Uses Sigma = Gamma^-1 and takes the SUB-BLOCK -- equivalently the Schur complement
        Gamma_AA - Gamma_AB Gamma_BB^-1 Gamma_BA.  This is the MARGINAL.  Taking `Gamma_AA`
        instead would give the CONDITIONAL (nuisance held fixed), which is systematically too
        narrow; they are easy to confuse and are not the same object.

        ABSOLUTE SCALE.  Marginalising is an INTEGRAL, so it changes the peak as well as the
        shape, and the module promises absolute lnL elsewhere -- so the constant is computed, not
        dropped:

            lnL_peak_marg = lnL_peak + (k/2) ln(2 pi) - 1/2 ln det Gamma_BB  [+ ln P(B in bounds)]

        for the k dropped coordinates, where Gamma_BB is the DROPPED sub-block of Gamma (the
        conditional precision, not a sub-block of Sigma).  Dropping one independent unit-variance
        coordinate therefore raises the peak by 0.5 ln(2 pi) = 0.919, not by nothing: a marginal
        that kept `lnL_peak` would be low by that much per coordinate, and the error compounds --
        it is a factor 1e3 after 15 coordinates, applied to a quantity read as an evidence.
        The bracketed term is the enclosed mass of the dropped block, present only when its bounds
        bite; under the conditions this method allows (below) it is a constant, and it is evaluated
        by the same controlled Monte Carlo as `log_mass`, so it can RAISE if it is too small to
        resolve.  `shape_only=True` restores the projection-with-the-original-peak behaviour, for a
        caller who wants the shape and will supply the normalisation themselves.

        TRUNCATION.  That identity is the UNTRUNCATED marginal.  Integrating out a coordinate that
        is genuinely truncated multiplies the result by the mass of that coordinate's CONDITIONAL
        distribution inside its own bounds -- and the conditional mean slides with the retained
        coordinates whenever the two are correlated, so the factor is a theta-DEPENDENT
        conditional-CDF ratio.  It is not Gaussian and cannot be absorbed into (mu, Gamma), so the
        result would have the wrong SHAPE, not merely the wrong normalisation.  Boundary-railing
        fits -- the case this module exists to represent -- are exactly where it bites, so it is
        REJECTED rather than silently approximated.  Two situations are provably safe and are
        allowed: a dropped coordinate whose bounds lie at least 5 marginal sigma from mu on both
        sides (factor 1 to ~1e-6), or one uncorrelated with every retained coordinate (factor
        constant, so only lnL_peak moves -- which the integration constant above accounts for).
        Pass `ignore_truncation=True` to take the untruncated marginal anyway; the constant is then
        the untruncated one, since that is what was asked for.
        """
        idx = [self.coord_names.index(k) if isinstance(k, str) else int(k) for k in keep]
        drop = [i for i in range(self.ndim) if i not in idx]
        Sigma = self.cov()
        if drop and self.bounds is not None and not ignore_truncation:
            self._reject_theta_dependent_truncation(idx, drop, Sigma)
        S = Sigma[np.ix_(idx, idx)]
        b = None if self.bounds is None else self.bounds[idx]
        peak = self.lnL_peak
        if drop and not shape_only:
            peak = peak + self._marginalization_constant(idx, drop, Sigma, ignore_truncation)
        return NAL(self.mu[idx], np.linalg.inv(S), [self.coord_names[i] for i in idx],
                   lnL_peak=peak, bounds=b, meta=self.meta)

    def _marginalization_constant(self, keep_idx, drop_idx, Sigma, ignore_truncation):
        """ln of the factor picked up by integrating exp(lnL) over the dropped coordinates.

        Completing the square in the joint quadratic gives

            int exp(-1/2 d^T Gamma d) d(theta_B)
                = (2 pi)^(k/2) |Gamma_BB|^(-1/2) exp(-1/2 d_A^T (Gamma/Gamma_BB) d_A)

        -- the Schur complement in the exponent (the shape `marginal` already returns) and a
        constant in front.  Gamma_BB is the sub-block of GAMMA, not of Sigma: it is the precision
        of B CONDITIONED on A, which is what completing the square produces.

        With bounds, the integral over the box multiplies this by P(theta_B in B_B | theta_A).
        `_reject_theta_dependent_truncation` has already established that this probability does not
        depend on theta_A -- every dropped coordinate is either effectively unbounded or
        uncorrelated with everything retained -- so it may be evaluated once, at theta_A = mu_A,
        where theta_B ~ N(mu_B, Gamma_BB^-1).  It is skipped entirely when no dropped bound bites,
        which is both the common case and the one where the Monte Carlo would be a waste (the
        answer is 1 to ~1e-6 per side by the 5-sigma criterion that let it through).
        """
        k = len(drop_idx)
        G_BB = self.gamma[np.ix_(drop_idx, drop_idx)]
        sign, logdet = np.linalg.slogdet(G_BB)
        if sign <= 0:                                     # unreachable for a positive-definite
            raise ValueError("nal_io: dropped block of gamma is not positive definite")
        const = 0.5 * k * np.log(2 * np.pi) - 0.5 * logdet
        if self.bounds is None or ignore_truncation:
            return const
        sd = np.sqrt(np.diag(Sigma))
        bites = [j for j in drop_idx
                 if (self.mu[j] - self.bounds[j][0]) < _UNBOUNDED_SIGMA * sd[j]
                 or (self.bounds[j][1] - self.mu[j]) < _UNBOUNDED_SIGMA * sd[j]]
        if not bites:
            return const
        block = NAL(self.mu[drop_idx], G_BB, [self.coord_names[j] for j in drop_idx],
                    bounds=self.bounds[drop_idx])
        return const + block.log_mass()

    def _reject_theta_dependent_truncation(self, keep_idx, drop_idx, Sigma):
        """Raise unless every dropped coordinate is effectively unbounded or uncorrelated."""
        sd = np.sqrt(np.diag(Sigma))
        bad = []
        for j in drop_idx:
            lo, hi = self.bounds[j]
            if (self.mu[j] - lo) >= _UNBOUNDED_SIGMA * sd[j] and \
                    (hi - self.mu[j]) >= _UNBOUNDED_SIGMA * sd[j]:
                continue                                  # bounds do not bite
            rho = np.abs(Sigma[j, keep_idx]) / (sd[j] * sd[keep_idx])
            if np.all(rho <= _CORR_TOL):
                continue                                  # factor is a constant
            bad.append(self.coord_names[j])
        if bad:
            raise ValueError(
                "nal_io: cannot marginalise over %s: those coordinates are truncated by `bounds` "
                "AND correlated with the ones kept, so the exact marginal carries a "
                "theta-dependent conditional-CDF factor that a Gaussian sub-block cannot "
                "represent -- the marginal would have the wrong shape, most severely for the "
                "boundary-railing fits this module is written for. Keep them, widen their bounds "
                "if the truncation is not physical, or pass ignore_truncation=True if you have "
                "established the factor is harmless." % bad)

    def lnL(self, theta, renormalize=False):
        """ln L at theta, shape (N, ndim) or (ndim,).  Outside `bounds` returns -inf."""
        X = np.atleast_2d(np.asarray(theta, float))
        d = X - self.mu
        out = self.lnL_peak - 0.5 * np.einsum("ij,jk,ik->i", d, self.gamma, d)
        if self.bounds is not None:
            inside = np.all((X >= self.bounds[:, 0]) & (X <= self.bounds[:, 1]), axis=1)
            out = np.where(inside, out, -np.inf)
        if renormalize:
            out = out - self.log_mass()
        return out

    def log_mass(self, rel_tol=0.01, max_draws=4000000, batch=200000, seed=0):
        """log of the Gaussian mass inside `bounds` -- computed ONCE per artifact, then cached.

        Deliberately NOT a product of 1-D marginal masses.  That factorisation ignores correlations
        and is badly biased for a correlated fit -- measured against brute force on a 3-D Gaussian
        with rho = 0.9 pairwise, the factorised value is 42% low.  This is a per-event CONSTANT and
        cancels in any hyper-posterior, which is why `renormalize` defaults to False.

        Monte Carlo, but to a CONTROLLED error: batches are drawn until the relative standard error
        of the hit fraction, sqrt((1-f)/hits), is at or below `rel_tol`, and a mass too small to
        resolve within `max_draws` RAISES.  Flooring the estimate at 1/n instead reports a number
        with no error bar as if it were a measurement: for a 1-D standard normal truncated to
        [6, 7] the true mass is ~1e-9 while a 200k-draw floor returns 5e-6, shifting the
        renormalized lnL by ~8.5 nat.  A known value may be declared as
        meta['log_truncation_mass'] and is then used verbatim.

        Caching is not an optimisation but a requirement: `lnL(..., renormalize=True)` is called
        once per likelihood evaluation, and re-running a 200k-draw Monte Carlo inside it makes the
        opt-in path unusable from any sampler.
        """
        if self.bounds is None:
            return 0.0
        if self.meta.get("log_truncation_mass") is not None:
            return float(self.meta["log_truncation_mass"])
        key = (rel_tol, max_draws, batch, seed)
        if self._log_mass_cache is not None and self._log_mass_cache[0] == key:
            return self._log_mass_cache[1]
        rng = _rng(seed)
        cov = self.cov()
        hits = 0
        drawn = 0
        rse = np.inf
        while drawn < max_draws:
            k = int(min(batch, max_draws - drawn))
            G = rng.multivariate_normal(self.mu, cov, size=k)
            hits += int(np.count_nonzero(
                np.all((G >= self.bounds[:, 0]) & (G <= self.bounds[:, 1]), axis=1)))
            drawn += k
            if hits > 0:
                f = hits / float(drawn)
                rse = float(np.sqrt((1.0 - f) / hits))    # relative standard error of f
                if rse <= rel_tol:
                    break
        if hits == 0 or rse > rel_tol:
            raise ValueError(
                "nal_io: truncation mass unresolved -- %d of %d draws landed inside `bounds`, a "
                "relative error of %s against the requested %g. The normalisation is too small to "
                "estimate here and a floored guess would be wrong by an unknown amount, so it is "
                "not returned. Record the known value as meta['log_truncation_mass'], raise "
                "max_draws, widen `bounds` if the truncation is not physical, or leave "
                "renormalize=False -- the constant cancels in any hyper-posterior."
                % (hits, drawn, ("undefined" if hits == 0 else "%.3g" % rse), rel_tol))
        self._log_mass_cache = (key, float(np.log(hits / float(drawn))))
        return self._log_mass_cache[1]


def _canon(value):
    """Comparable form of a metadata value (dicts compare by content, not by key order)."""
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def check_set_compatible(nals):
    """Refuse to ADD artifacts whose metadata does not establish that they share a chart.

    Equal `coord_names` is NOT enough.  'mc' is a detector-frame chirp mass in one artifact and a
    source-frame one in another; two source-frame artifacts built under different cosmologies, or
    with different distance priors integrated out, are likewise in different charts wearing the
    same labels.  Summing any of those evaluates a joint likelihood at a theta that means
    something different in each term -- silently, because the arithmetic is perfectly well defined
    and only the physics is wrong.

    Fails closed.  An artifact that does not declare its `frame` cannot be SHOWN compatible with
    another, so it is rejected rather than assumed.  `chart` is required of EVERY member of a set
    for the same reason: matching coordinate names in a matching frame still do not establish
    matching coordinate CONVENTIONS -- which spin basis, which mass pairing, which angle
    reference -- so a set in which nobody declares a chart is not a set that has been shown
    compatible, it is one in which the question was never asked.  `cosmology` and `d_prior` must
    agree whenever the frame is 'source' (there they define what the masses mean) or whenever any
    member of the set declares them at all.  A single artifact is never checked: nothing is being
    added to it.
    """
    nals = list(nals)
    if len(nals) < 2:
        return
    frames = [n.meta.get("frame") for n in nals]
    undeclared = [i for i, f in enumerate(frames) if not f]
    if undeclared:
        raise ValueError(
            "nal_io: artifact(s) at position(s) %s declare no 'frame', so they cannot be shown to "
            "be in the same chart as the rest of the set -- detector- and source-frame masses "
            "carry identical coordinate names. Write them with write_nal(), which records the "
            "frame, before summing." % undeclared)
    if len(set(frames)) != 1:
        raise ValueError("nal_io: cannot sum artifacts across frames %s -- the same coordinate "
                         "names denote different physical quantities" % sorted(set(frames)))
    for key in ("chart", "cosmology", "d_prior"):
        vals = [_canon(n.meta.get(key)) for n in nals]
        declared = [v for v in vals if v is not None and v != ""]
        # `chart` is never optional in a set: "nobody declared one" is not evidence of agreement,
        # and write_nal(chart=None) will happily produce a whole catalogue of such artifacts.
        required = key == "chart" or (frames[0] == "source" and key in ("cosmology", "d_prior"))
        if not (declared or required):
            continue                                      # nobody claims it; nothing to reconcile
        if len(declared) != len(vals) or len(set(declared)) != 1:
            raise ValueError(
                "nal_io: artifacts in a set must agree on a non-empty %r before they may be "
                "added; got %d of %d declaring it, %d distinct value(s). Artifacts that differ "
                "in %r are in different charts even when their coordinate names match, and one "
                "that does not declare it cannot be shown to agree. Record it at write time as "
                "write_nal(..., %s=...)."
                % (key, len(declared), len(vals), len(set(declared)), key, key))


def check_sampler_compatible(nals, frame, chart):
    """Refuse to evaluate artifacts against a run whose chart is not DECLARED to match them.

    `check_set_compatible` only compares artifacts with each other, and it is skipped entirely for
    a single artifact -- nothing is being added to it.  Neither fact says anything about the
    SAMPLER: `coords` carries names alone, and 'mc'/'delta_mc' are spelt identically whether the
    run walks in detector-frame or source-frame masses.  A source-frame NAL evaluated at
    detector-frame samples (or the reverse) has the right array count, the right names and no
    error -- only a wrong answer, biased by the redshift of the event.

    Conversion is not an option here: it needs a redshift per sample, which the plugin is never
    handed.  So the run declares its frame and its chart and they are compared, and every artifact
    must state its own.  Fails closed in both directions -- undeclared on either side is a
    mismatch, not a pass.
    """
    nals = list(nals)
    if not frame:
        raise ValueError(
            "nal_io: the run's sampling frame is undeclared, so these artifacts cannot be shown "
            "to be in the chart the sampler is walking in -- detector- and source-frame masses "
            "wear identical coordinate names, and no dimension or name check can tell them "
            "apart. Declare it as [nal] sampler_frame = detector|source in the ini, or "
            "RIFT_NAL_SAMPLER_FRAME.")
    if frame not in ("detector", "source"):
        raise ValueError("nal_io: sampler_frame must be 'detector' or 'source', got %r" % (frame,))
    if not chart:
        raise ValueError(
            "nal_io: the run's sampling chart is undeclared. Matching coordinate names in a "
            "matching frame still do not establish matching coordinate CONVENTIONS -- which spin "
            "basis, which mass pairing, which angle reference. Declare it as [nal] sampler_chart "
            "in the ini, or RIFT_NAL_SAMPLER_CHART, naming the same chart the artifacts do.")
    for key, want in (("frame", frame), ("chart", chart)):
        got = sorted({(n.meta.get(key) or "<undeclared>") for n in nals
                      if n.meta.get(key) != want})
        if got:
            raise ValueError(
                "nal_io: artifact %s %s does not match the run's declared %s %r. The artifacts "
                "would be evaluated at coordinates that mean something else in the chart they "
                "were fitted in, silently: the array count and the coordinate names are "
                "identical either way. Use artifacts built for this run's chart, or correct the "
                "declaration." % (key, got, key, want))


class NALSet(object):
    """A catalogue of NALs, summed.  Each event contributes additively in lnL."""

    def __init__(self, nals, require_compatible=True):
        """`require_compatible=False` skips `check_set_compatible` -- only for a caller who has
        established equivalence of the charts by other means."""
        self.nals = list(nals)
        if not self.nals:
            raise ValueError("nal_io: empty NALSet")
        ch = {tuple(n.coord_names) for n in self.nals}
        if len(ch) != 1:
            raise ValueError("nal_io: NALSet requires one common chart, got %s" % sorted(ch))
        if require_compatible:
            check_set_compatible(self.nals)
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
    out = NAL(d["theta_star"], g, names,
              lnL_peak=float(meta.get("lnL_peak", 0.0)),
              bounds=d["bounds"] if "bounds" in d else None, meta=meta)
    out.source = base
    # Enforce the frame invariant on the CONSUMER side too: most artifacts a run loads were not
    # written by write_nal(), so a check that only runs in the writer never sees them.
    check_artifact_frame_invariant(out, where=base)
    return out


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

      * a distance coordinate present  =>  frame must be 'detector' (distance has not been
                         marginalised yet).  BOTH spellings count: `_derive` makes u_d and dist
                         interchangeable, so a chart carrying `dist` is exactly as
                         distance-carrying as one carrying `u_d`.
      * frame source =>  no distance coordinate at all, AND the cosmology and the distance prior
                         that was integrated must both be recorded, since the source-frame masses
                         are meaningless without them.

    Raises ValueError rather than warning: an artifact that cannot state its own frame honestly
    should not be written at all.  (The shipped O3/O4 NAL catalogue records none of this -- its
    npz carries only names/labels/centers/sigs/covs/ess/method/kl -- so a consumer cannot tell
    which cosmology produced it.)
    """
    dist_coords = [c for c in coord_names if c in _DISTANCE_COORDS]
    if frame not in ("detector", "source"):
        raise ValueError("nal_io: frame must be 'detector' or 'source', got %r" % (frame,))
    if dist_coords and frame != "detector":
        raise ValueError("nal_io: chart carries %s, so masses are detector-frame; got frame=%r"
                         % (dist_coords, frame))
    if frame == "source":
        if dist_coords:
            raise ValueError("nal_io: frame='source' must not also carry a distance coordinate "
                             "(%s)" % dist_coords)
        if not cosmology:
            raise ValueError("nal_io: frame='source' requires a declared cosmology")
        if not d_prior:
            raise ValueError("nal_io: frame='source' requires the distance prior that was "
                             "integrated out (name and range)")


def check_artifact_frame_invariant(nal, where=None, require_frame=False):
    """Apply the frame invariant to an artifact this module did not write.

    `write_nal` enforces `check_frame_invariant` at WRITE time, but nothing a consumer is handed
    has necessarily been through it.  Artifacts arrive from exporters that predate this module --
    the shipped O3/O4 NAL catalogue records no frame at all, only names/labels/centers/sigs/covs --
    and from fitting scripts that assemble the npz and the meta.json themselves.  Such an artifact
    can perfectly well declare `frame='source'` while still carrying `u_d`, or declare it with no
    cosmology and no distance prior: a distance-marginalised quadratic whose mass-redshift
    degeneracy was integrated against a prior nobody recorded, which no consumer can undo and none
    can detect from the numbers.  A check that only runs in the writer is a check the artifacts
    that matter never meet, so it is re-run on LOAD against the artifact's own recorded metadata.

    An artifact declaring no frame at all is left to `check_set_compatible` /
    `check_sampler_compatible`, which reject it with a message naming the comparison that cannot be
    made.  `require_frame=True` makes it an error here instead, so the plugin entry point fails
    closed whatever the order of the checks around it.
    """
    meta = nal.meta or {}
    frame = meta.get("frame")
    where = where or getattr(nal, "source", None) or "<in memory>"
    if not frame:
        if require_frame:
            raise ValueError(
                "nal_io: artifact %s declares no 'frame', so its own consistency cannot be "
                "established: whether its masses are detector- or source-frame decides whether "
                "carrying a distance coordinate is normal or means the distance has already been "
                "integrated out against an unrecorded prior. Rewrite it with write_nal(), which "
                "records the frame and checks the invariant." % where)
        return
    try:
        check_frame_invariant(nal.coord_names, frame, meta.get("cosmology"), meta.get("d_prior"))
    except ValueError as exc:
        raise ValueError(
            "nal_io: artifact %s fails the frame invariant on its own recorded metadata (%s). It "
            "was not written by write_nal(), or was edited afterwards; the run cannot interpret "
            "it and no downstream step can detect the error from the numbers alone." % (where, exc))


def write_nal(base, nal, chart=None, frame="detector", cosmology=None, d_prior=None,
              symmetry=None, unconstrained_dirs=None, validation=None, parents=None,
              run_id=None, lnL_ref="noise_hypothesis_ratio", extra=None):
    """Write <base>.npz + <base>.meta.json.

    `parents` should be the source files the fit consumed (all.net / all_dslice.dat); their
    sha256 is recorded so an artifact can always be traced to the grid that produced it.

    `chart` is optional here -- a lone artifact is self-consistent without one -- but an artifact
    written without it cannot later be ADDED to another: `check_set_compatible` requires every
    member of a set to declare the same non-empty chart.  Name it now if the artifact is destined
    for a catalogue.

    `extra` may only ADD metadata.  It is applied after the frame invariant has been checked, so
    allowing it to overwrite a validated key would let frame='detector' pass the check while
    frame='source' is what gets recorded -- an artifact claiming source-frame masses with no
    cosmology, no distance prior, possibly carrying u_d.  Collisions raise, before any file is
    written.
    """
    coord_names = list(nal.coord_names)
    check_frame_invariant(coord_names, frame, cosmology, d_prior)
    clash = sorted(set(extra or ()) & set(_RESERVED_META_KEYS))
    if clash:
        raise ValueError(
            "nal_io: extra=%s would overwrite metadata this writer validates or derives. Those "
            "keys are checked BEFORE extra is applied, so overwriting them records something "
            "that was never validated -- notably a frame whose cosmology, distance prior and "
            "distance coordinate went unchecked. Pass them as the named arguments instead "
            "(write_nal(..., frame=..., cosmology=...)), or rename the extra key." % clash)
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
_STATE = {"set": None, "coords": None, "renormalize": False, "offset": 0.0}


def _peak_offset(nals, renormalize):
    """Largest value the summed contribution can take: sum of the per-artifact peaks.

    Each term is at most its own peak, so subtracting this makes `nal_lnL` non-positive
    everywhere and `np.exp` of it safe.  With `renormalize` the per-artifact peak is
    lnL_peak - log_mass (log_mass <= 0, so the peak RISES); computing it here also fails early and
    fills the cache rather than surprising the sampler on its first call.
    """
    return float(sum(n.lnL_peak - (n.log_mass() if renormalize else 0.0) for n in nals))


def prepare_nal_lnL(config=None, coords=None):
    """Called once by CIP / EOSPosterior with the parsed ini and the run's SAMPLING coordinates.

    ini section:
        [nal]
        artifacts = /path/to/*.npz      ; glob, or a comma-separated list
        sampler_frame = detector        ; REQUIRED: the frame the RUN samples in
        sampler_chart = NAL:aligned     ; REQUIRED: the chart the RUN samples in
        renormalize = false             ; per-event constant, cancels in a hyper-posterior
        sampler_coords = mc,eta         ; only needed when the driver cannot pass coords=
    Falls back to the environment variables RIFT_NAL_ARTIFACTS, RIFT_NAL_SAMPLER_FRAME,
    RIFT_NAL_SAMPLER_CHART and RIFT_NAL_SAMPLER_COORDS when no ini is supplied, so the plugin also
    works on RIFT versions predating the prepare-hook fix.  `coords` from the driver always wins:
    it is the authoritative sampling basis.

    The frame and chart of the RUN cannot be read off the artifacts or the coordinate names, and
    the driver does not pass them, so they must be declared and are then checked against every
    artifact -- including a single one, which no set check ever examines.
    """
    pat = os.environ.get("RIFT_NAL_ARTIFACTS")
    declared = os.environ.get("RIFT_NAL_SAMPLER_COORDS")
    frame = os.environ.get("RIFT_NAL_SAMPLER_FRAME")
    chart = os.environ.get("RIFT_NAL_SAMPLER_CHART")
    if config is not None and config.has_section("nal"):
        if config.has_option("nal", "artifacts"):
            pat = config.get("nal", "artifacts")
        if config.has_option("nal", "renormalize"):
            _STATE["renormalize"] = config.get("nal", "renormalize").strip().lower() \
                in ("1", "true", "yes")
        if config.has_option("nal", "sampler_coords"):
            declared = config.get("nal", "sampler_coords")
        if config.has_option("nal", "sampler_frame"):
            frame = config.get("nal", "sampler_frame")
        if config.has_option("nal", "sampler_chart"):
            chart = config.get("nal", "sampler_chart")
    if not pat:
        raise ValueError("nal_io: no artifacts configured -- set [nal] artifacts in the ini or "
                         "the RIFT_NAL_ARTIFACTS environment variable")
    nals = []
    for part in pat.split(","):
        part = part.strip()
        nals += load_nal_dir(part) if any(c in part for c in "*?[") else [load_nal(part)]
    check_sampler_compatible(nals, (frame or "").strip(), (chart or "").strip())
    # Every artifact must ALSO be self-consistent, not merely consistent with the run: agreeing
    # with a declared frame says nothing about whether the artifact's own chart and metadata are
    # compatible with that frame.  load_nal() already enforces this for a declared frame; repeated
    # here with require_frame=True so the entry point fails closed regardless of check order, and
    # for NALs assembled in memory rather than loaded from disk.
    for n in nals:
        check_artifact_frame_invariant(n, require_frame=True)
    _STATE["set"] = NALSet(nals)
    if coords is not None:
        _STATE["coords"] = list(coords)
    elif declared:
        _STATE["coords"] = [s.strip() for s in declared.split(",") if s.strip()]
    else:
        _STATE["coords"] = None
    _STATE["offset"] = _peak_offset(nals, _STATE["renormalize"])
    print("nal_io: loaded %d NAL artifact(s), chart %s (run frame %s, chart %s); sampler coords "
          "%s; contribution centred by %.6g"
          % (len(nals), _STATE["set"].coord_names, frame, chart, _STATE["coords"],
             _STATE["offset"]))


def nal_lnL_offset():
    """The constant `nal_lnL` subtracts (sum of the artifacts' peak lnL, less the truncation mass
    when `renormalize` is on -- i.e. exactly the constant that was removed, whichever mode is in
    force).

    A fixed multiplicative factor on the likelihood.  It cancels in any posterior, but NOT in an
    absolute lnL or an evidence: reporting the centred value makes `integral_result.dat` low by
    this amount and any odds ratio against a run without the factor wrong by exp(offset).  Both
    drivers query `<supplementary-likelihood-factor-function>_offset`, which is this function for
    the `nal_lnL` entry point, and add it back to the absolute quantities they write.  Zero before
    `prepare_nal_lnL` has run, so a driver may call it unconditionally.
    """
    return _STATE["offset"]


def nal_lnL(*x):
    """Additive lnL contribution.  `x` is one array per sampling coordinate, in coord_names order.

    Matches the calling convention in util_ConstructIntrinsicPosterior_GenericCoordinates.py:2952
    and util_ConstructEOSPosterior.py:946 (`log_likelihood_function(*x) + supplemental(*x)`).

    CENTRED: the artifacts' summed peak (`nal_lnL_offset()`) is subtracted, so the return value is
    never positive.  The drivers' DEFAULT path is not the log one above but
    `likelihood_function(*x) * np.exp(supplemental(*x))`, where float64 overflows past ~709 and a
    perfectly valid loud-event artifact (lnL_peak ~ SNR^2/2) would return inf for every sample;
    the drivers' lnL_shift rescales their own fit, not this separate exponentiation.  The
    subtracted constant multiplies the likelihood by exp(-offset) and so cancels in any posterior.
    """
    if _STATE["set"] is None:
        prepare_nal_lnL(config=None, coords=None)         # legacy: environment-only configuration
    S = _STATE["set"]
    names = _STATE["coords"]
    arrs = [np.atleast_1d(np.asarray(a, float)).ravel() for a in x]
    if names is None:
        # Fail closed.  Assuming the sampler integrates in the artifact's own chart is NOT the
        # safe default: a sampler in (mc, eta) against an artifact in (mc, delta_mc) has the right
        # number of arrays, so nothing would raise -- eta would simply be evaluated as delta_mc.
        raise ValueError(
            "nal_io: the sampling basis is unknown, so the arrays handed to nal_lnL cannot be "
            "named. The driver should call prepare_nal_lnL(config=..., coords=<sampling coords>); "
            "on a RIFT that never reaches the prepare hook, declare the basis explicitly as "
            "RIFT_NAL_SAMPLER_COORDS='name1,name2,...' (or [nal] sampler_coords in the ini). "
            "Refusing to assume the sampler integrates in the artifact chart %s." % S.coord_names)
    if len(arrs) != len(names):
        raise ValueError("nal_io: called with %d coordinate array(s) but the declared sampling "
                         "basis %s has %d -- the arrays would be mislabelled"
                         % (len(arrs), names, len(names)))
    have = dict(zip(names, arrs))
    theta = np.stack([_derive(k, have) for k in S.coord_names], 1)
    return S.lnL(theta, renormalize=_STATE["renormalize"]) - _STATE["offset"]
