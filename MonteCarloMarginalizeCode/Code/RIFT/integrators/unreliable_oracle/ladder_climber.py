"""
ladder_climber.py -- CMA-ES climbs, free and along a ladder of pinned values.

``LadderClimbingOracle`` maximizes an arbitrary ln L over a box with a small
CMA-ES (rank-one + rank-mu covariance update).  One climb is FREE; each further
climb pins one parameter (the ladder parameter, e.g. chi1_perp) at a rung value
and climbs the rest.  Rungs run outward from the free optimum, each warm-started
from the best point of its neighbour, so a narrow ridge found once is followed
rather than re-found.  This is the shape needed to seed a grid with near-peak
points across a parameter the posterior is broad in (transverse spin) while the
others (chirp mass, mass ratio, aligned spin) are narrow and correlated.

As for every unreliable_oracle, the output is only a PROPOSAL: the points must be
re-evaluated with the production likelihood before they carry weight.

Bounds REFLECT (never truncate) and parameters named in ``periodic`` wrap.
Interface follows MCSamplerGeneric: add_parameter(), setup(lnL=...),
update_sampling_prior(ln_weights, n_history, external_rvs=...), draw_simplified().
``lnL`` takes one array per parameter in params_ordered order and returns an
array (the CIP convention); a scalar-only objective can be wrapped with
``np.vectorize`` or a loop.
"""
import numpy as np

from RIFT.integrators.mcsampler_generic import MCSamplerGeneric


def _reflect_into(x, lo, hi):
    """Reflect x into [lo, hi] (repeatedly, for large excursions)."""
    w = hi - lo
    y = np.mod(x - lo, 2 * w)
    return lo + np.where(y > w, 2 * w - y, y)


class LadderClimbingOracle(MCSamplerGeneric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lnL = None
        self.periodic = set()
        self.history = []          # list of (climb_label, x (ndim,), lnL)
        self.climbs = {}           # label -> dict(mean, cov, best_x, best_lnL)
        self.reference_samples = None
        self.rng = np.random.default_rng()

    def add_parameter(self, params, pdf=None, **kwargs):
        super().add_parameter(params, pdf, **kwargs)

    def setup(self, lnL=None, periodic=(), ladder_param=None, ladder_values=(),
              popsize=10, n_generations=25, n_generations_ladder=None, sigma0=0.3, seed=None,
              verbose=False, callback=None, **kwargs):
        """
        lnL : callable(*cols) -> array.  ladder_param : name pinned on rungs.
        sigma0 : initial step, in units of the scale passed to update (or the
        box width when no scale is known).  callback(label, gen, best_lnL) is
        called once per generation (for logging / pre-emption).
        """
        super().setup(**kwargs)
        self.lnL = lnL
        self.periodic = set(periodic)
        self.ladder_param = ladder_param
        self.ladder_values = list(ladder_values)
        self.popsize = int(popsize)
        self.n_generations = int(n_generations)
        # rungs warm-start from a neighbour, so they usually need fewer generations
        self.n_generations_ladder = int(n_generations_ladder or n_generations)
        self.sigma0 = float(sigma0)
        self.verbose = verbose
        self.callback = callback
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._lo = np.array([self.llim[p] for p in self.params_ordered], dtype=float)
        self._hi = np.array([self.rlim[p] for p in self.params_ordered], dtype=float)

    # -- box handling ---------------------------------------------------------
    def _fold(self, X):
        X = np.array(X, dtype=float, copy=True)
        for k, p in enumerate(self.params_ordered):
            if p in self.periodic:
                X[..., k] = self._lo[k] + np.mod(X[..., k] - self._lo[k], self._hi[k] - self._lo[k])
            else:
                X[..., k] = _reflect_into(X[..., k], self._lo[k], self._hi[k])
        return X

    def _eval(self, X, label):
        cols = [X[:, k] for k in range(X.shape[1])]
        vals = np.asarray(self.lnL(*cols), dtype=float).reshape(-1)
        vals = np.where(np.isfinite(vals), vals, -np.inf)
        for x, v in zip(X, vals):
            self.history.append((label, x.copy(), float(v)))
        return vals

    # -- one CMA-ES climb -----------------------------------------------------
    def climb(self, x0, scale, label="free", pinned=None, n_generations=None):
        """Climb from x0 with per-axis step `scale`; `pinned` = {index: value}."""
        pinned = pinned or {}
        free = np.array([k for k in range(len(x0)) if k not in pinned])
        periodic_free = np.array([self.params_ordered[k] in self.periodic for k in free])
        n = len(free)
        lam = self.popsize
        mu = lam // 2
        wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        wts /= wts.sum()
        mueff = 1.0 / np.sum(wts ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x_full = np.array(x0, dtype=float)
        for k, v in pinned.items():
            x_full[k] = v
        S = np.asarray(scale, dtype=float)[free]          # work in units of scale
        m = np.zeros(n)
        sigma = self.sigma0
        C = np.eye(n)
        pc = np.zeros(n); ps = np.zeros(n)
        best_x, best_v = self._fold(x_full[None, :])[0], -np.inf
        n_gen = n_generations or self.n_generations
        for g in range(n_gen):
            evals, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(evals, 1e-20))
            Z = self.rng.normal(size=(lam, n))
            Y = Z @ np.diag(D) @ B.T                     # ~ N(0, C)
            Xw = m + sigma * Y
            X = np.tile(x_full, (lam, 1))
            X[:, free] = x_full[free] + Xw * S
            X = self._fold(X)
            vals = self._eval(X, label)
            # Reflect bounded coordinates back into the working frame. Periodic
            # coordinates keep their unwrapped optimizer values: a wrap is only
            # an equivalent evaluation point, not a long step across the box.
            Xw[:, ~periodic_free] = ((X[:, free] - x_full[free]) / S)[:, ~periodic_free]
            order = np.argsort(-vals)
            if vals[order[0]] > best_v:
                best_v, best_x = vals[order[0]], X[order[0]].copy()
            sel = Xw[order[:mu]]
            m_old = m
            m = wts @ sel
            y = (m - m_old) / sigma
            Cinvsqrt = B @ np.diag(1 / D) @ B.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinvsqrt @ y)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chiN < 1.4 + 2 / (n + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y
            ysel = (sel - m_old) / sigma
            C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                 + cmu * (ysel.T * wts) @ ysel)
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            if self.callback:
                self.callback(label, g, best_v)
            if self.verbose:
                print(" [ladder] %-14s gen %2d  best %.3f  gen-max %.3f  sigma %.3g"
                      % (label, g, best_v, vals[order[0]], sigma))
        # final Gaussian in physical units, for draws
        mean_phys = x_full.copy()
        mean_phys[free] = x_full[free] + m * S
        cov_phys = np.zeros((len(x0), len(x0)))
        cov_phys[np.ix_(free, free)] = (sigma ** 2) * (S[:, None] * C * S[None, :])
        rec = dict(mean=self._fold(mean_phys[None, :])[0], cov=cov_phys,
                   best_x=best_x, best_lnL=float(best_v), pinned=dict(pinned))
        self.climbs[label] = rec
        return rec

    # -- oracle interface -----------------------------------------------------
    def update_sampling_prior(self, ln_weights, n_history, external_rvs=None,
                              lnw_cut=-10, **kwargs):
        """Start from the weighted history: best point, weighted spread as scale."""
        rvs = external_rvs if external_rvs is not None else self._rvs
        if not rvs:
            raise Exception(" ladder oracle: update_sampling_prior requires history")
        X = np.column_stack([np.asarray(rvs[p])[-n_history:] for p in self.params_ordered])
        lnw = np.asarray(ln_weights)[-len(X):]
        keep = lnw > np.max(lnw) + lnw_cut
        if np.sum(keep) < 3:
            keep = np.argsort(-lnw)[:max(3, min(len(lnw), 10))]
        Xk, wk = X[keep], np.exp(lnw[keep] - np.max(lnw[keep]))
        wk /= wk.sum()
        x0 = X[np.argmax(lnw)]
        sd = np.sqrt(np.sum(wk[:, None] * (Xk - wk @ Xk) ** 2, axis=0))
        width = self._hi - self._lo
        scale = np.where(sd > 1e-3 * width, sd, 0.05 * width)
        for k, p in enumerate(self.params_ordered):   # angles / ladder: box scale
            if p in self.periodic or p == self.ladder_param:
                scale[k] = 0.25 * width[k]
        self.run(x0, scale)

    def run(self, x0, scale, between_climbs=None):
        """Free climb(s), then the ladder outward from the best free optimum.

        x0 is one start (ndim,) or several (n_start, ndim); each gets a free
        climb.  between_climbs(x_start) is called before each climb (e.g. to
        re-adapt an auxiliary proposal at the start point).
        """
        starts = np.atleast_2d(np.asarray(x0, dtype=float))
        best = None
        for i, xs in enumerate(starts):
            if between_climbs:
                fixed = between_climbs(xs)
                if fixed is not None:
                    xs = np.asarray(fixed, dtype=float)
            rec = self.climb(xs, scale, label="free" if len(starts) == 1 else "free%d" % i)
            if best is None or rec["best_lnL"] > best["best_lnL"]:
                best = rec
        if self.ladder_param is None or not self.ladder_values:
            self._finish(); return
        k = self.params_ordered.index(self.ladder_param)
        v0 = best["best_x"][k]
        rungs = sorted(self.ladder_values, key=lambda v: abs(v - v0))
        done = {v0: best["best_x"]}
        for v in rungs:
            nearest = min(done, key=lambda u: abs(u - v))
            start = done[nearest].copy()
            start[k] = v
            if between_climbs:
                # may return a corrected start (e.g. projected inside a constraint
                # the pinned value has just violated)
                fixed = between_climbs(start)
                if fixed is not None:
                    start = np.asarray(fixed, dtype=float)
            rec = self.climb(start, scale, label="%s=%.3g" % (self.ladder_param, v),
                             pinned={k: v}, n_generations=self.n_generations_ladder)
            done[v] = rec["best_x"]
        self._finish()

    def _finish(self):
        self.reference_samples = np.array([h[1] for h in self.history])
        self.reference_lnL = np.array([h[2] for h in self.history])

    def near_peak(self, delta_lnL=10.0):
        """History points within delta_lnL of the overall best: (X, lnL, labels)."""
        lnl = np.array([h[2] for h in self.history])
        keep = lnl > np.max(lnl) - delta_lnL
        X = np.array([h[1] for h in self.history])[keep]
        lab = [h[0] for h, k in zip(self.history, keep) if k]
        return X, lnl[keep], lab

    def place(self, n_samples, delta_lnL=10.0, inflate=1.5, pinned_sd=None):
        """Covariance placement: equal draws per good climb from N(best_x, inflate^2 cov).

        A rung's pinned coordinate has zero variance in its CMA covariance; give it
        sd ``pinned_sd`` (e.g. half the rung spacing) so seeds fill between rungs.
        """
        best = max(c["best_lnL"] for c in self.climbs.values())
        good = [c for c in self.climbs.values() if c["best_lnL"] > best - delta_lnL]
        per = int(np.ceil(n_samples / max(1, len(good))))
        out = []
        for c in good:
            cov = (inflate ** 2) * np.array(c["cov"], dtype=float)
            for k in c["pinned"]:
                if pinned_sd:
                    cov[k, k] = pinned_sd ** 2
            out.append(self.rng.multivariate_normal(c["best_x"], cov, size=per, method="eigh"))
        return self._fold(np.concatenate(out))[:n_samples]

    def draw_simplified(self, n_samples, *args, delta_lnL=10.0, **kwargs):
        """Draw from the final Gaussians of climbs whose best is within delta_lnL."""
        best = max(c["best_lnL"] for c in self.climbs.values())
        good = [c for c in self.climbs.values() if c["best_lnL"] > best - delta_lnL]
        idx = self.rng.integers(0, len(good), n_samples)
        rv = np.empty((n_samples, len(self.params_ordered)))
        for j, c in enumerate(good):
            sel = idx == j
            if np.any(sel):
                rv[sel] = self.rng.multivariate_normal(c["mean"], c["cov"], size=int(sel.sum()),
                                                       method="eigh")
        return None, None, self._fold(rv)
