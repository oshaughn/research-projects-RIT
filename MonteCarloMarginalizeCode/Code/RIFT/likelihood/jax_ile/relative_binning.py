"""Relative binning for the Q/U/V reduction of the seed-climb objective.

With a fiducial mode set h0_lm(f) on RIFT's two-sided grid, a nearby template is
written h_lm(f) = r_lm(f) h0_lm(f), with r interpolated linearly in |f| between
bin edges.  Then, per detector,

    Q_lm(t) = sum_edges conj(r_lm,e) C_lm,e(t)                      (+ and - sides)
    U_lm,l'm' = sum_bins sum_{a,c} conj(r_lm,b+a) r_l'm',b+c T_b,ac,lm,l'm'
    V_lm,l'm' = sum_bins sum_{a,c} r_lm(-f),b+a r_l'm'(f),b+c W_b,ac,lm,l'm'

with summary data C, T, W built once from h0, the data and the weights.  Only the
edge frequencies are then needed per template (a few hundred instead of ~1e4).
Conventions follow gpu_precompute.compound_precompute_arrays (A = 1):
Q[j] = df * sum_f 2 conj(h) d w exp(2 pi i f t_j), t_j = (j + n_shift) dT.
Bins on the negative side mirror the positive side, because V couples f and -f.
Accuracy falls with distance from the fiducial; refresh it per climb.
"""
import numpy as np
import jax.numpy as jnp


class UnsupportedFiducial(ValueError):
    """The multiplicative ratio cannot represent a fiducial mode's support.

    Callers with a full-grid waveform path should use it until a representable
    fiducial is available. A zero fiducial mode does not imply that the same
    mode is zero at the candidate intrinsic point.
    """


def phase_bins(fmin, fmax, eps=0.3, gammas=(-5. / 3, -2. / 3, 1., 5. / 3, 7. / 3)):
    """Edges with the Zackay et al. phase bound changing by <= eps per bin."""
    f = np.linspace(fmin, fmax, 200000)
    dpsi = 2 * np.pi * sum(np.sign(g) * (f / (fmin if g < 0 else fmax)) ** g for g in gammas)
    n = int(np.ceil((dpsi[-1] - dpsi[0]) / eps))
    targets = np.linspace(dpsi[0], dpsi[-1], n + 1)
    return np.interp(targets, dpsi, f)


class RelativeBinning:
    """Summary data for one fiducial; ``qvu(hp_e, hn_e)`` gives per-detector Q/U/V."""

    def __init__(self, freqs, dets, n_window, delta_f, delta_t, edges_hz, reg=1e-8):
        """freqs: two-sided grid (N,).  dets: {det: dict(d=(N,), w=(N,), n_shift=int)}.
        edges_hz: approximate positive edges; snapped to the grid."""
        freqs = np.asarray(freqs)
        self.df, self.dt, self.n_window, self.reg = float(delta_f), float(delta_t), int(n_window), reg
        k0 = int(round(-freqs[0] / delta_f))                   # index of f = 0
        kp = np.unique(np.clip(np.rint(np.asarray(edges_hz) / delta_f).astype(int), 1, len(freqs) - k0 - 1))
        self.k_pos = k0 + kp                                    # grid indices of +edges
        self.k_neg = k0 - kp                                    # mirrored -edges
        self.f_edges = kp * delta_f
        self.freqs, self.dets = freqs, dets

    def set_fiducial(self, h0_full):
        """h0_full: (K, N) fiducial modes on the full grid (numpy or jax)."""
        h0 = np.asarray(h0_full)
        K = h0.shape[0]
        # Snapping/clipping may collapse a narrow band to a single edge.
        # Reject during activation so rb_refresh can choose the full grid.
        if len(self.k_pos) < 2:
            raise UnsupportedFiducial(
                "relative binning needs at least two distinct edges; require full-grid evaluation")
        if not np.isfinite(h0).all():
            raise ValueError("relative binning requires finite fiducial modes")
        amplitude = np.max(np.abs(h0), axis=1)
        scale = amplitude ** 2
        missing = np.flatnonzero(amplitude == 0)
        if len(missing):
            raise UnsupportedFiducial(
                "zero fiducial modes %s require full-grid evaluation" % missing.tolist())
        # A zero reference edge cannot recover the candidate ratio wherever
        # that edge's interpolation coefficient is used. Even at the fiducial,
        # replacing its undefined ratio by zero loses weighted interior power.
        active = np.logical_or.reduce([np.asarray(v["w"]) != 0 for v in self.dets.values()])
        # V couples h(f) to h(-f), including for asymmetric detector weights.
        active = active | active[(len(self.freqs)-np.arange(len(self.freqs))) % len(self.freqs)]
        covered = np.zeros(len(self.freqs), dtype=bool)
        idx = np.arange(self.k_pos[0], self.k_pos[-1] + 1)
        covered[idx] = True
        covered[(-idx) % len(self.freqs)] = True
        if np.any(active & ~covered):
            # h0=0 outside the span cannot certify that future candidates stay
            # zero there. Callers may omit only weights known structurally zero.
            raise UnsupportedFiducial(
                "weighted frequencies outside bin edges require full-grid evaluation")
        for side, edges in (("positive", self.k_pos), ("negative", self.k_neg)):
            for b in range(len(edges)-1):
                idx = np.arange(self.k_pos[b], self.k_pos[b+1] + (b == len(edges)-2))
                if side == "negative":
                    idx = (len(self.freqs)-idx) % len(self.freqs)
                # Even finite edge ratios cannot restore candidate power at an
                # interior frequency where the reference is exactly zero.
                missing = np.flatnonzero(np.any((h0[:, idx] == 0) & active[idx], axis=1))
                if len(missing):
                    raise UnsupportedFiducial(
                        "zero fiducial support for modes %s in weighted %s bin %d "
                        "requires full-grid evaluation" % (missing.tolist(), side, b))
                u = (np.abs(self.freqs[idx])-self.f_edges[b]) / (self.f_edges[b+1]-self.f_edges[b])
                left_used = np.any(active[idx] & (u != 1))
                right_used = np.any(active[idx] & (u != 0))
                missing = np.flatnonzero(((h0[:, edges[b]] == 0) & left_used)
                                         | ((h0[:, edges[b+1]] == 0) & right_used))
                if len(missing):
                    raise UnsupportedFiducial(
                        "zero fiducial edges for modes %s in weighted %s bin %d "
                        "require full-grid evaluation" % (missing.tolist(), side, b))
        self.h0p, self.h0n = h0[:, self.k_pos], h0[:, self.k_neg]
        ne = len(self.k_pos)
        tj = None
        self.summary = {}
        for det, v in self.dets.items():
            d, w = np.asarray(v["d"]), np.asarray(v["w"])
            tj = (np.arange(self.n_window) + v["n_shift"]) * self.dt
            out = {}
            for side, ke in (("p", self.k_pos), ("n", self.k_neg)):
                C = np.zeros((K, ne, self.n_window), complex)
                T = np.zeros((ne - 1, 2, 2, K, K), complex)
                W = np.zeros((ne - 1, 2, 2, K, K), complex)
                Ntot = len(self.freqs)
                for b in range(ne - 1):
                    # positive-side bin [e_b, e_b+1) (last bin closed); the negative
                    # side uses the mirrored index set, so coverage is identical
                    idx = np.arange(self.k_pos[b], self.k_pos[b + 1] + (1 if b == ne - 2 else 0))
                    if side == "n":
                        idx = (Ntot - idx) % Ntot
                    if len(idx) == 0:
                        continue
                    f = self.freqs[idx]
                    u = (np.abs(f) - self.f_edges[b]) / (self.f_edges[b + 1] - self.f_edges[b])
                    phi = (1 - u, u)
                    g = 2 * self.df * np.conj(h0[:, idx]) * d[idx] * w[idx]          # (K, nf)
                    E = np.exp(2j * np.pi * f[:, None] * tj[None, :])                 # (nf, nt)
                    C[:, b] += (g * phi[0]) @ E
                    C[:, b + 1] += (g * phi[1]) @ E
                    mirror = 2 * (len(self.freqs) // 2) - idx                         # index of -f
                    mirror = np.where(mirror >= len(self.freqs), mirror - len(self.freqs), mirror)
                    for a in (0, 1):
                        for c in (0, 1):
                            ww = 2 * self.df * w[idx] * phi[a] * phi[c]
                            T[b, a, c] = (np.conj(h0[:, idx]) * ww) @ h0[:, idx].T
                            W[b, a, c] = (h0[:, mirror] * ww) @ h0[:, idx].T
                out[side] = (jnp.asarray(C), jnp.asarray(T), jnp.asarray(W))
            self.summary[det] = out
        self._h0p, self._h0n = jnp.asarray(self.h0p), jnp.asarray(self.h0n)
        self._scale = jnp.asarray(scale)
        return self

    def state(self):
        """Pytree for qvu; scale is retained for cached-state compatibility."""
        return dict(summary=self.summary, h0p=self._h0p, h0n=self._h0n, scale=self._scale)

    def qvu(self, hp, hn, state=None):
        return qvu_state(state or self.state(), hp, hn, self.reg)


def qvu_state(st, hp, hn, reg=1e-8):
    """hp, hn: (K, n_edges) template modes at +edges and -edges.

    ``st`` must come from a validated fiducial. ``reg`` is retained for API
    compatibility: damping nonzero edge ratios changes the fiducial itself.
    """
    def ratio(h, h0):
        # Complex division avoids squaring tiny denominators. Zero edges are
        # allowed only where their interpolation basis has no weighted support.
        supported = h0 != 0
        denominator = jnp.where(supported, h0, jnp.ones_like(h0))
        return jnp.where(supported, h / denominator, 0.)
    rp, rn = ratio(hp, st["h0p"]), ratio(hn, st["h0n"])
    res = {}
    for det, s in st["summary"].items():
        Cp, Tp, Wp = s["p"]
        Cn, Tn, Wn = s["n"]
        q = (jnp.einsum("ke,kej->jk", jnp.conj(rp), Cp)
             + jnp.einsum("ke,kej->jk", jnp.conj(rn), Cn))
        u = 0.0
        v = 0.0
        for r, rm, T, W in ((rp, rn, Tp, Wp), (rn, rp, Tn, Wn)):
            for a in (0, 1):
                for c in (0, 1):
                    ra = r[:, a:r.shape[1] - 1 + a]                 # (K, n_bins) at edge b+a
                    rc = r[:, c:r.shape[1] - 1 + c]
                    rma = rm[:, a:rm.shape[1] - 1 + a]
                    u = u + jnp.einsum("kb,lb,bkl->kl", jnp.conj(ra), rc, T[:, a, c])
                    v = v + jnp.einsum("kb,lb,bkl->kl", rma, rc, W[:, a, c])
        res[det] = (q, u, v)
    return res
