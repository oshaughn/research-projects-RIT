"""Extrinsic-marginalized likelihood as a function of intrinsic parameters.

``IntrinsicMarginalObjective(P)`` returns ln L marginalized over time, distance,
orbital phase and polarization (the JAX ILE fused kernel), and averaged over sky
position and inclination with a FIXED importance-sampled batch.  The batch is held
fixed between calls (common random numbers), so the objective is a deterministic,
smooth function of the intrinsic parameters, suitable for a hillclimber.  It is
not an evidence estimate: the sky/inclination average uses a few thousand points.

The waveform is whatever ``factored_likelihood.PrecomputeLikelihoodTerms`` builds
for ``P.approx`` (lalsimulation, CPU).  The marginalization runs on the JAX
backend (GPU if present).  ``JAXLikelihoodData`` is not a pytree, so the stock
wrappers close over one data object and recompile for each intrinsic point.  Here
the per-point arrays (Q, U, V) are jit ARGUMENTS substituted into a fixed
template, so the kernel compiles once per (modes, buffer length) shape.
"""
import copy
import os
import time

import numpy as np

import lal
import lalsimulation as lalsim

import jax
jax.config.update("jax_enable_x64", True)   # the kernels assume float64
import jax.numpy as jnp
import jax.dlpack
from jax.scipy.special import logsumexp

from RIFT import lalsimutils
from RIFT.likelihood import factored_likelihood
from . import core
from .wrapper import build_data_from_precompute

_PC = lal.PC_SI


@jax.jit
def _qvu_all_finite(arrays):
    """One device scalar; do not copy the Q/U/V arrays back to the host."""
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(arrays)]))


def load_frame_data(cache_file, channel_names, psd_files, data_start_time,
                    data_end_time, srate, window_shape=0.0, psd_window_shape=0.0,
                    fmin_ifo=None):
    """Data and PSDs, read as in ``integrate_likelihood_extrinsic_jax.load_frames``.

    channel_names, psd_files: lists of 'IFO=value'.  fmin_ifo: dict IFO -> Hz.
    """
    fmin_ifo = fmin_ifo or {}
    deltaT = 1.0 / srate
    data_dict, psd_dict = {}, {}
    for inst, chan in (c.split("=") for c in channel_names):
        data_dict[inst] = lalsimutils.frame_data_to_non_herm_hoff(
            cache_file, inst + ":" + chan, start=data_start_time,
            stop=data_end_time, window_shape=window_shape, deltaT=deltaT)
    for inst, psdf in (c.split("=") for c in psd_files):
        psd = lalsimutils.get_psd_series_from_xmldoc(psdf, inst)
        psd = lalsimutils.resample_psd_series(psd, data_dict[inst].deltaF)
        if psd_window_shape > 0 or window_shape > 0:
            psd.data.data *= (
                lalsimutils.psd_windowing_factor(window_shape, len(data_dict[inst].data.data))
                / lalsimutils.psd_windowing_factor(psd_window_shape, len(psd.data.data)))
        if inst in fmin_ifo:
            freqs = psd.f0 + psd.deltaF * np.arange(psd.data.length)
            psd.data.data[freqs < fmin_ifo[inst]] = 0
        psd_dict[inst] = psd
    return data_dict, psd_dict


# ---------------------------------------------------------------------------
# Sky / inclination batch: prior and a reflected-kernel importance proposal
# ---------------------------------------------------------------------------
def _prior_draw(rng, n):
    ra = rng.uniform(0, 2 * np.pi, n)
    dec = np.arcsin(rng.uniform(-1, 1, n))
    incl = np.arccos(rng.uniform(-1, 1, n))
    return ra, dec, incl


def _to_unit(ra, dec, incl):
    """(ra, sin dec, cos incl): the prior is uniform on [0,2pi) x [-1,1]^2."""
    return np.stack([ra, np.sin(dec), np.cos(incl)], axis=-1)


class SkyBatch:
    """Fixed batch of (ra, dec, incl) with log importance weights ln(p/q).

    q = f_prior * prior + (1 - f_prior) * GMM, in u = (ra, sin dec, cos incl),
    where the prior is uniform.  The mixture (component count by BIC) is fitted to
    weight-resampled pilot points, so the separate sky modes of a two-detector
    network (and the incl <-> pi - incl mirror) each get their own component; a
    single kernel bandwidth would span them.  Draws wrap in ra and reflect at
    +-1; the density sums the 27 periodic/mirror images, so the weights are exact.
    """

    _LOG_PRIOR = -np.log(8 * np.pi)   # 1/(2pi * 2 * 2)

    def __init__(self, n, rng, adapt_incl=False):
        self.n = int(n)
        self.rng = rng
        # adapt_incl False: the mixture covers (ra, sin dec) only and cos incl is
        # drawn from its prior.  Inclination is frame-dependent (J vs L frame, and
        # the L frame moves with the spins); a batch concentrated in incl biases
        # the comparison between intrinsic points.  Sky position is not.
        self.adapt_incl = bool(adapt_incl)
        self.gmm = None          # (weights, means, covs)
        self.f_prior = 1.0
        self.ra, self.dec, self.incl = _prior_draw(rng, self.n)
        self.log_w = np.zeros(self.n)

    def _log_gmm(self, u):
        wts, mus, covs = self.gmm
        d = mus.shape[1]
        v0 = u[:, :d]
        shifts = [np.array([a] + [0.0] * (d - 1)) for a in (0.0, 2 * np.pi, -2 * np.pi)]
        mirrors = [(0, 0), (1, 0), (2, 0)] if d == 2 else [(a, b) for a in (0, 1, 2) for b in (0, 1, 2)]
        out = []
        for wk, mu, cov in zip(wts, mus, covs):
            L = np.linalg.cholesky(cov)
            logdet = 2 * np.sum(np.log(np.diag(L)))
            terms = []
            for sh in shifts:
                for m1, m2 in mirrors:
                    v = v0 + sh
                    if m1: v = v.copy(); v[:, 1] = (2 if m1 == 1 else -2) - v[:, 1]
                    if m2: v = v.copy(); v[:, 2] = (2 if m2 == 1 else -2) - v[:, 2]
                    z = np.linalg.solve(L, (v - mu).T)
                    terms.append(-0.5 * np.sum(z ** 2, axis=0))
            out.append(np.log(wk) - 0.5 * logdet - 0.5 * d * np.log(2 * np.pi)
                       + logsumexp(np.array(terms), axis=0))
        lg = logsumexp(np.array(out), axis=0)
        return lg if d == 3 else lg - np.log(2.0)      # times the uniform cos-incl density

    def log_q(self, u):
        lp = self._LOG_PRIOR * np.ones(len(u))
        if self.gmm is None:
            return lp
        return np.logaddexp(np.log(self.f_prior) + lp,
                            np.log1p(-self.f_prior) + self._log_gmm(u))

    def adapt(self, ra, dec, incl, log_weight, n_resample=3000, max_components=4,
              f_prior=0.1, cov_inflate=1.5, **kw):
        """Refit q from pilot samples with log weights ln(L p / q_pilot)."""
        from sklearn.mixture import GaussianMixture
        u = _to_unit(ra, dec, incl)
        lw = np.asarray(log_weight) - np.max(log_weight)
        w = np.exp(lw); w /= w.sum()
        neff = 1.0 / np.sum(w ** 2)
        d = 3 if self.adapt_incl else 2
        x = u[self.rng.choice(len(w), size=n_resample, p=w)][:, :d]
        # cut the ra circle at its widest empty gap so no mode straddles 0/2pi
        r = np.sort(np.mod(x[:, 0], 2 * np.pi))
        gaps = np.diff(np.concatenate([r, [r[0] + 2 * np.pi]]))
        cut = r[np.argmax(gaps)] + 0.5 * np.max(gaps)
        x[:, 0] = np.mod(x[:, 0] - cut, 2 * np.pi) + cut
        jitter = self.rng.normal(scale=[0.01, 0.005, 0.01][:d], size=x.shape)   # break exact duplicates
        x = x + jitter
        best, best_bic = None, np.inf
        for k in range(1, max_components + 1):
            g = GaussianMixture(k, covariance_type="full", reg_covar=1e-4,
                                random_state=int(self.rng.integers(1 << 31))).fit(x)
            b = g.bic(x)
            if b < best_bic:
                best, best_bic = g, b
        covs = best.covariances_ * cov_inflate ** 2
        self.gmm = (best.weights_, best.means_, covs)
        self.f_prior = float(f_prior)
        self.redraw()
        return dict(neff_pilot=float(neff), n_components=int(best.n_components))

    def redraw(self):
        n_prior = self.rng.binomial(self.n, self.f_prior) if self.gmm is not None else self.n
        u = _to_unit(*_prior_draw(self.rng, n_prior))
        if self.gmm is not None and n_prior < self.n:
            wts, mus, covs = self.gmm
            m = self.n - n_prior
            comp = self.rng.choice(len(wts), size=m, p=wts / wts.sum())
            d = mus.shape[1]
            u_k = np.empty((m, 3))
            u_k[:, 2] = self.rng.uniform(-1, 1, m)          # cos incl from the prior unless adapted
            for c in range(len(wts)):
                sel = comp == c
                if np.any(sel):
                    u_k[sel, :d] = self.rng.multivariate_normal(mus[c], covs[c], size=int(sel.sum()))
            u_k[:, 0] = np.mod(u_k[:, 0], 2 * np.pi)
            for k in range(1, d):
                y = np.mod(u_k[:, k] + 1, 4)                      # reflect into [-1, 1]
                u_k[:, k] = np.where(y > 2, 4 - y, y) - 1
            u = np.concatenate([u, u_k])
        u[:, 0] = np.mod(u[:, 0], 2 * np.pi)
        self.ra = u[:, 0]
        self.dec = np.arcsin(np.clip(u[:, 1], -1, 1))
        self.incl = np.arccos(np.clip(u[:, 2], -1, 1))
        self.log_w = self._LOG_PRIOR - self.log_q(u)


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------
class IntrinsicMarginalObjective:
    """ln L(intrinsic), marginalized over time/distance/phase/psi, sky-averaged.

    Parameters
    ----------
    data_dict, psd_dict : as for the production precompute.
    P_base : ChooseWaveformParams holding approximant, fmin, fref, deltaT, deltaF.
    reduce : 'marginal' (log-mean over the sky batch, with importance weights) or
        'profile' (max over the batch).
    """

    def __init__(self, data_dict, psd_dict, P_base, event_time, d_min, d_max,
                 l_max=2, fmax=None, n_sky=1024, nphi=16, npsi=8, n_dist=128,
                 d_prior="euclidean", storage_window_half=0.15,
                 integration_window_half=0.075, interp=None, reduce="marginal",
                 waveform_kwargs=None, seed=0, verbose=False, analyticPSD_Q=False,
                 device_precompute=False, device_check_rtol=1e-8,
                 mode_backend="lalsim", adapt_incl=False, angle_scheme="grid",
                 rb_eps=None, batch_size=10, export_dir=None):
        self.data_dict, self.psd_dict = data_dict, psd_dict
        self.P_base = P_base.copy()
        self.event_time = float(event_time)
        self.l_max, self.fmax = int(l_max), fmax
        self.swh, self.iwh = float(storage_window_half), float(integration_window_half)
        self.interp = interp or core.JAX_INTERP_DEFAULT
        self.reduce = reduce
        self.waveform_kwargs = dict(waveform_kwargs or {})
        self.verbose = verbose
        self.analyticPSD_Q = analyticPSD_Q
        # device_precompute: lalsim modes on CPU, Q/U/V reduced on the device by
        # gpu_precompute.compound_precompute_arrays (A=1).  True needs cupy; 'numpy'
        # runs the same array code on the host (tests).  The first point is also
        # built by the CPU precompute and the two are compared (device_check_rtol).
        self.device_precompute = device_precompute
        self.device_check_rtol = float(device_check_rtol)
        self._device = None
        # mode_backend 'ripple': XPHM modes from ripple (ripple_modes.py), reduced
        # to Q/U/V in JAX; the whole objective runs on the accelerator.  J-frame
        # modes, t = 0 at the LAL FD reference: valid for this orientation-
        # marginalized objective only.  P.approx is ignored on this path.
        self.mode_backend = mode_backend
        # angle_scheme 'exact': anglemarg's exact-coefficient (phi, psi) marginalization,
        # whose dense lattice is sized from a data-derived amplitude bound (static
        # under jit).  The bound is refreshed at each adapt_sky; the kernel is
        # rebuilt only when it grows by more than 25%.
        self.angle_scheme = angle_scheme
        self._amp = None
        self._ripple = None
        # rb_eps: relative binning (relative_binning.py) for the ripple backend, with
        # phase-bound eps per bin; the fiducial is refreshed at each adapt_sky.
        self.rb_eps = rb_eps
        # ripple + RB: single points run through the batched program (padded to
        # batch_size), so ripple is traced for one batched shape only; with
        # export_dir the traced programs are cached on disk (export_cache.py).
        self.batch_size = int(batch_size)
        self.export_dir = export_dir or os.environ.get("JAX_EXPORT_CACHE_DIR")
        self.t_waveform = 0.0
        self.x_grid, self.log_w_grid = core.make_distance_grid(
            d_min, d_max, n_grid=n_dist, d_prior=d_prior)
        self.phi_grid = core.phi_ref_grid(nphi)
        self.psi_grid = core.psi_grid(npsi)
        self.rng = np.random.default_rng(seed)
        self.sky = SkyBatch(n_sky, self.rng, adapt_incl=adapt_incl)
        self._template = None
        self._key = None
        self._kernel = None
        self.n_calls = 0
        self.n_compiles = 0
        self.n_nonfinite = 0
        self.n_exceptions = 0
        self._consecutive_failures = 0
        self.t_precompute = 0.0
        self.t_kernel = 0.0

    # -- per-point precompute -------------------------------------------------
    def _prepare(self, P):
        P = P.copy()
        for attr in ("approx", "fmin", "fref", "deltaT", "deltaF"):
            setattr(P, attr, getattr(self.P_base, attr))
        P.fmax = 0.0
        P.radec = True
        P.tref = self.event_time
        P.dist = core.DIST_MPC_REF * 1e6 * _PC
        P.phiref = P.psi = P.incl = 0.0
        P.theta = P.phi = 0.0          # sky enters n_shift/epoch; keep it fixed
        return P

    def precompute(self, P):
        if self.mode_backend == "ripple":
            t0 = time.time()
            data = self._ripple_precompute(P)
            self.t_precompute += time.time() - t0
            return data, {}
        if self.device_precompute and self._template is not None:
            t0 = time.time()
            data = self._device_precompute(P)
            self.t_precompute += time.time() - t0
            return data, {}
        t0 = time.time()
        data, extras = build_data_from_precompute(
            self._prepare(P), self.data_dict, self.psd_dict, self.event_time,
            self.swh, self.iwh, self.l_max, self.fmax, verbose=False,
            analyticPSD_Q=self.analyticPSD_Q,
            **self.waveform_kwargs)
        self.t_precompute += time.time() - t0
        if self.device_precompute and self._template is None:
            self._check_device_against(P, data)
        return data, extras

    # -- ripple backend ---------------------------------------------------------
    def _ripple_setup(self):
        from .ripple_modes import RippleXPHMModes, reduce_qvu
        P = self._prepare(self.P_base)
        d0 = self.data_dict[list(self.data_dict)[0]]
        freqs = d0.f0 + d0.deltaF * np.arange(d0.data.length)
        fNyq = 0.5 / P.deltaT
        modes = RippleXPHMModes(freqs, P.fmin, self.fmax or fNyq, P.fref, l_max=self.l_max,
                                dist_mpc=core.DIST_MPC_REF)
        dets = {}
        for det, dfd in self.data_dict.items():
            IP = lalsimutils.ComplexOverlap(P.fmin, self.fmax, fNyq, P.deltaF, self.psd_dict[det],
                                            self.analyticPSD_Q, False, 0.0, full_output=True)
            from RIFT.likelihood import factored_likelihood as fl
            t_det = fl.ComputeArrivalTimeAtDetector(det, 0.0, 0.0, self.event_time)
            rho_epoch = float(dfd.epoch)          # mode epoch 0
            n_shift = int((float(t_det) - self.swh - rho_epoch) / P.deltaT + 0.5)
            dets[det] = dict(w=jnp.asarray(np.asarray(IP.weights2side, dtype=np.float64)),
                             d=jnp.asarray(np.asarray(dfd.data.data)), n_shift=n_shift,
                             epoch=rho_epoch + n_shift * P.deltaT)
        n_window = int(2 * self.swh / P.deltaT)
        df = float(d0.deltaF)

        def _all(m1, m2, c1, c2):
            h = modes._modes(m1, m2, c1, c2)
            return {det: reduce_qvu(h, v["d"], v["w"], df, v["n_shift"], n_window)
                    for det, v in dets.items()}
        self._ripple = dict(modes=modes, dets=dets, fn=jax.jit(_all), deltaT=P.deltaT)
        if self.rb_eps:
            from .relative_binning import RelativeBinning, phase_bins, qvu_state
            fw = min(float(np.min(np.abs(freqs[np.asarray(v["w"]) > 0]))) for v in dets.values())
            fhi = float(self.fmax or fNyq)
            rb = RelativeBinning(freqs, dets, n_window, df, P.deltaT,
                                 phase_bins(max(fw, P.fmin), fhi, self.rb_eps))
            f_e = jnp.asarray(rb.f_edges)

            def _rb(m1, m2, c1, c2, state):
                hp, hn = modes.modes_at(f_e, m1, m2, c1, c2)
                return qvu_state(state, hp, hn)
            self._ripple.update(rb=rb, rb_fn=jax.jit(_rb), rb_fn_raw=_rb, rb_state=None)

    def _ripple_precompute(self, P):
        if self._ripple is None:
            self._ripple_setup()
        r = self._ripple
        t0 = time.time()
        args = (jnp.asarray(P.m1 / lal.MSUN_SI), jnp.asarray(P.m2 / lal.MSUN_SI),
                jnp.asarray([P.s1x, P.s1y, P.s1z]), jnp.asarray([P.s2x, P.s2y, P.s2z]))
        if "rb" in r and r["rb_state"] is None and not r.get("rb_fallback"):
            self.rb_refresh(P)
        if "rb" in r and not r.get("rb_fallback"):
            B = self.batch_size
            rep = lambda a: jnp.broadcast_to(a, (B,) + jnp.shape(a))
            out = self._vfn_rb()(*map(rep, args), r["rb_state"])
            out = jax.tree.map(lambda a: a[0], out)
            if not bool(_qvu_all_finite(out)):
                self._decline_rb("non-finite relative-binned Q/U/V")
                out = r["fn"](*args)
        else:
            out = r["fn"](*args)
        jax.block_until_ready(out)        # so t_waveform is not billed to the kernel
        self.t_waveform += time.time() - t0
        lms = r["modes"].lms
        if self._template is None:
            packed = {det: dict(lms=np.array(lms), rholmArray=np.asarray(q).T, U=np.asarray(u),
                                V=np.asarray(v), epoch=r["dets"][det]["epoch"])
                      for det, (q, u, v) in out.items()}
            tvals = factored_likelihood.marginalization_time_grid(self.iwh, r["deltaT"], xpy=np)
            return core.build_likelihood_data(packed, r["deltaT"], self.event_time, tvals)
        d = copy.copy(self._template)
        d.detectors = {det: dict(lms=[tuple(x) for x in lms], npts_full=int(q.shape[0]),
                                 epoch=float(self._template.detectors[det]["epoch"]),
                                 Q=q, U=u, V=v) for det, (q, u, v) in out.items()}
        return d

    def _ripple_sources(self):
        import ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM as X
        from . import ripple_modes, relative_binning
        return [X.__file__, ripple_modes.__file__, relative_binning.__file__, __file__]

    def _vfn_rb(self):
        r = self._ripple
        if "rb_vfn" not in r:
            raw = jax.vmap(r["rb_fn_raw"], in_axes=(0, 0, 0, 0, None))
            B = self.batch_size
            ex = (np.zeros(B), np.zeros(B), np.zeros((B, 3)), np.zeros((B, 3)), r["rb_state"])
            if self.export_dir:
                from .export_cache import exported_call
                m = r["modes"]
                r["rb_vfn"] = exported_call(raw, ex, self.export_dir,
                                         key_parts=("rb", m.lms, m.fref, m.dist_mpc, B),
                                         key_arrays=(r["rb"].f_edges,),
                                         source_files=self._ripple_sources())
            else:
                r["rb_vfn"] = jax.jit(raw)
        return r["rb_vfn"]

    def _full_modes(self, P):
        r = self._ripple
        m = r["modes"]
        if "full_fn" not in r:
            if self.export_dir:
                from .export_cache import exported_call
                r["full_fn"] = exported_call(m._modes, (0.0, 0.0, np.zeros(3), np.zeros(3)), self.export_dir,
                                             key_parts=("full", m.lms, m.fref, m.dist_mpc, m.N),
                                             key_arrays=(np.asarray(m.fband), np.asarray(m._pos),
                                                         np.asarray(m._neg), np.asarray(m._negmap)),
                                             source_files=self._ripple_sources())
            else:
                r["full_fn"] = m._fn
        return r["full_fn"](jnp.asarray(P.m1 / lal.MSUN_SI), jnp.asarray(P.m2 / lal.MSUN_SI),
                            jnp.asarray([P.s1x, P.s1y, P.s1z]), jnp.asarray([P.s2x, P.s2y, P.s2z]))

    def _vfn_full(self):
        r = self._ripple
        if "full_vfn" not in r:
            r["full_vfn"] = jax.jit(jax.vmap(r["fn"]))
        return r["full_vfn"]

    def _decline_rb(self, reason):
        self._ripple["rb_state"] = None
        self._ripple["rb_fallback"] = reason
        print(" [intrinsic_objective] relative binning fallback:", reason)

    def rb_refresh(self, P):
        """Make P the relative-binning fiducial (full-grid ripple evaluation once)."""
        from .relative_binning import UnsupportedFiducial
        r = self._ripple
        h0 = self._full_modes(P)
        try:
            r["rb"].set_fiducial(h0)
        except UnsupportedFiducial as exc:
            # Edge ratios cannot restore a mode absent from the fiducial. Keep
            # the entire candidate waveform on the exact grid, including modes
            # that become nonzero, until a later refresh restores RB support.
            self._decline_rb(str(exc))
            return
        r["rb_state"] = r["rb"].state()
        r.pop("rb_fallback", None)

    # -- device precompute ----------------------------------------------------
    def _device_setup(self):
        from RIFT.likelihood import gpu_precompute as gp
        if self.device_precompute == "numpy":
            xp = np
        else:
            import cupy as xp          # ImportError here is the honest failure
        P = self._prepare(self.P_base)
        fNyq = 0.5 / P.deltaT
        dev = dict(xp=xp, gp=gp, weights={}, data={}, fNyq=fNyq)
        for det in self.data_dict:
            IP = lalsimutils.ComplexOverlap(P.fmin, self.fmax, fNyq, P.deltaF,
                                            self.psd_dict[det], self.analyticPSD_Q,
                                            False, 0.0, full_output=True)
            dev["weights"][det] = xp.asarray(np.asarray(IP.weights2side, dtype=np.float64))
            dev["data"][det] = xp.asarray(np.asarray(self.data_dict[det].data.data))
        self._device = dev

    def _device_arrays(self, P):
        """Return (lms, {det: (Q (npts,K), U, V, epoch)}) built on the device."""
        from RIFT.likelihood import factored_likelihood as fl
        if self._device is None:
            self._device_setup()
        dev = self._device
        xp, gp = dev["xp"], dev["gp"]
        P = self._prepare(P)
        P.dist = core.DIST_MPC_REF * 1e6 * _PC
        P.deltaF = self.data_dict[list(self.data_dict)[0]].deltaF
        wk = {k: v for k, v in self.waveform_kwargs.items()
              if k in ("extra_waveform_kwargs", "use_gwsignal", "use_gwsignal_approx", "no_memory")}
        t0 = time.time()
        hlms, hlms_conj = fl.internal_hlm_generator(P, self.l_max, verbose=False, quiet=True,
                                                    skip_interpolation=True, **wk)
        self.t_waveform += time.time() - t0
        lms = list(hlms.keys())
        basis = xp.asarray(np.stack([hlms[m].data.data for m in lms]))[None]
        basis_c = xp.asarray(np.stack([hlms_conj[m].data.data for m in lms]))[None]
        h0 = hlms[lms[0]]
        out = {}
        for det, dfd in self.data_dict.items():
            t_det = fl.ComputeArrivalTimeAtDetector(det, P.phi, P.theta, self.event_time)
            rho_epoch = dfd.epoch - h0.epoch
            n_shift = int(float(float(t_det) - self.swh - float(rho_epoch)) / P.deltaT + 0.5)
            n_window = int(2 * self.swh / P.deltaT)
            q, u, v = gp.compound_precompute_arrays(
                basis, basis_c, dev["data"][det], dev["weights"][det], P.deltaF, P.deltaT,
                n_shift, n_window, backend=xp, return_device=True)
            out[det] = (q[0].T, u[0, 0], v[0, 0], float(rho_epoch) + n_shift * P.deltaT)
        return lms, out

    def _check_device_against(self, P, data):
        """Compare the device Q/U/V for P with the CPU precompute; raise on mismatch."""
        lms, arr = self._device_arrays(P)
        for det in data.detector_names:
            d = data.detectors[det]
            if [tuple(x) for x in d["lms"]] != [tuple(x) for x in lms]:
                raise RuntimeError("device precompute: mode order %s != CPU %s" % (lms, d["lms"]))
            Q, U, V, ep = arr[det]
            for name, a, b in (("Q", Q, d["Q"]), ("U", U, d["U"]), ("V", V, d["V"])):
                a = np.asarray(self._device["xp"].asnumpy(a) if hasattr(self._device["xp"], "asnumpy") else a)
                b = np.asarray(b)
                err = np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)
                if not err <= self.device_check_rtol:
                    raise RuntimeError("device precompute: %s %s differs from CPU by %.3g (rel)" % (det, name, err))
            if abs(ep - float(d["epoch"])) > 1e-6 * float(data.deltaT):
                raise RuntimeError("device precompute: %s epoch %.9f != CPU %.9f" % (det, ep, d["epoch"]))
        if self.verbose:
            print(" [intrinsic_objective] device precompute matches CPU on the first point")

    def _device_precompute(self, P):
        lms, arr = self._device_arrays(P)
        tmpl = self._template
        dets = {}
        for det in tmpl.detector_names:
            Q, U, V, ep = arr[det]
            td = tmpl.detectors[det]
            if abs(ep - float(td["epoch"])) > 1e-6 * float(tmpl.deltaT):
                raise RuntimeError("device precompute: epoch moved; refusing")
            to_j = (lambda a: jnp.asarray(a)) if self._device["xp"] is np else \
                   (lambda a: jax.dlpack.from_dlpack(a))
            dets[det] = dict(lms=[tuple(x) for x in lms], npts_full=int(Q.shape[0]),
                             epoch=float(td["epoch"]), Q=to_j(Q), U=to_j(U), V=to_j(V))
        d = copy.copy(tmpl)
        d.detectors = dets
        return d

    def _shape_key(self, data):
        return tuple((det, tuple(map(tuple, data.detectors[det]["lms"])),
                      int(data.detectors[det]["npts_full"]),
                      float(data.detectors[det]["epoch"]))
                     for det in data.detector_names)

    def _build_kernel(self, data):
        template = data
        dets = list(data.detector_names)
        xg, lwg, pg, sg = self.x_grid, self.log_w_grid, self.phi_grid, self.psi_grid
        interp = self.interp
        if self.angle_scheme == "exact":
            from . import anglemarg
            if self._amp is None:
                self._amp = self._amp_bound(data)
            amp = self._amp

            def _kernel(arrays, ra, dec, incl):
                d2 = copy.copy(template)
                d2.detectors = {det: dict(template.detectors[det], Q=arrays[det][0],
                                          U=arrays[det][1], V=arrays[det][2])
                                for det in dets}
                return anglemarg.fused_log_likelihood_distphipsimarg_exact(
                    d2, ra, dec, incl, xg, lwg, interp=interp, amp_sizing=amp)
        else:
            def _kernel(arrays, ra, dec, incl):
                d2 = copy.copy(template)
                d2.detectors = {det: dict(template.detectors[det], Q=arrays[det][0],
                                          U=arrays[det][1], V=arrays[det][2])
                                for det in dets}
                return core.fused_log_likelihood_distphipsimarg(
                    d2, ra, dec, incl, xg, lwg, pg, sg, interp=interp)
        self._kernel = jax.jit(_kernel)
        self._template = template
        self.n_compiles += 1

    def _amp_bound(self, data):
        from . import anglemarg
        a = float(anglemarg.estimate_angle_amplitude(data, self.x_grid, interp=self.interp))
        return max(1.5 * a, anglemarg.ANGLE_MARG_CROSSOVER_AMPLITUDE)

    def _refresh_amp(self, data):
        if self.angle_scheme != "exact" or self._template is None:
            return
        a = self._amp_bound(data)
        if a > 1.25 * self._amp:
            if self.verbose:
                print(" [intrinsic_objective] angle amplitude bound %.1f -> %.1f; rebuilding kernel" % (self._amp, a))
            self._amp = a
            self._build_kernel(self._template)

    def lnL_sky(self, P, ra=None, dec=None, incl=None, data=None):
        """Per-sky-point lnL (marginalized over t, d, phi, psi) and the data object."""
        if data is None:
            data, _ = self.precompute(P)
        key = self._shape_key(data)
        if key != self._key:
            if self._key is not None and self.verbose:
                print(" [intrinsic_objective] data shape changed; recompiling")
            self._build_kernel(data)
            self._key = key
        arrays = {det: (data.detectors[det]["Q"], data.detectors[det]["U"],
                        data.detectors[det]["V"]) for det in data.detector_names}
        if ra is None:
            ra, dec, incl = self.sky.ra, self.sky.dec, self.sky.incl
        t0 = time.time()
        out = np.asarray(self._kernel(arrays, jnp.asarray(ra), jnp.asarray(dec),
                                      jnp.asarray(incl)))
        self.t_kernel += time.time() - t0
        return out, data

    def __call__(self, P):
        self.n_calls += 1
        try:
            lnL, _ = self.lnL_sky(P)
            self._consecutive_failures = 0
        except Exception as exc:   # waveform failure / out-of-domain point: score as -inf,
            self.n_exceptions += 1   # but never silently, and not when everything fails
            self._consecutive_failures += 1
            if self.n_exceptions <= 5:
                print(" [intrinsic_objective] evaluation failed (%d so far): %r" % (self.n_exceptions, exc))
            if self.n_calls == 1 or self._consecutive_failures >= 20:
                raise
            return -np.inf
        bad = ~np.isfinite(lnL)
        if np.any(bad):
            # non-finite rows (seen from hlmoft_IMRPv2_dict before #489; see
            # DESIGN_transverse_seed_climb.md) score as zero likelihood
            self.n_nonfinite += int(bad.sum())
            lnL = np.where(bad, -np.inf, lnL)
        if self.reduce == "profile":
            return float(np.max(lnL))
        return float(logsumexp(lnL + self.sky.log_w) - np.log(len(lnL)))

    def batch(self, P_list):
        """Objective for a list of points in one vmapped call (ripple backend only).

        The list length should be constant across calls (e.g. the CMA population)
        so the batched program compiles once.  Falls back to a loop otherwise.
        """
        if self.mode_backend != "ripple":
            return np.array([self(P) for P in P_list])
        if self._template is None:
            self(P_list[0])                       # builds template and kernel
        r = self._ripple
        self.n_calls += len(P_list)
        use_rb = "rb" in r and not r.get("rb_fallback")
        if use_rb:
            vfn = self._vfn_rb()
        else:
            vfn = self._vfn_full()
        # An exact-angle amplitude refresh replaces the kernel without changing
        # the Q/U/V shape key. Keep the batched wrapper tied to that kernel,
        # including when its angular lattice has been rebuilt at the same shape.
        if r.get("vkern_source") is not self._kernel:
            r["vkern"] = jax.jit(jax.vmap(self._kernel, in_axes=(0, None, None, None)))
            r["vkern_source"] = self._kernel
        m1 = jnp.asarray([P.m1 / lal.MSUN_SI for P in P_list]); m2 = jnp.asarray([P.m2 / lal.MSUN_SI for P in P_list])
        c1 = jnp.asarray([[P.s1x, P.s1y, P.s1z] for P in P_list]); c2 = jnp.asarray([[P.s2x, P.s2y, P.s2z] for P in P_list])
        t0 = time.time()
        out = vfn(m1, m2, c1, c2, r["rb_state"]) if use_rb else vfn(m1, m2, c1, c2)
        if use_rb and not bool(_qvu_all_finite(out)):
            self._decline_rb("non-finite relative-binned Q/U/V")
            out = self._vfn_full()(m1, m2, c1, c2)
        jax.block_until_ready(out)
        self.t_waveform += time.time() - t0
        t0 = time.time()
        lnL = np.asarray(r["vkern"](out, jnp.asarray(self.sky.ra), jnp.asarray(self.sky.dec),
                                    jnp.asarray(self.sky.incl)))
        self.t_kernel += time.time() - t0
        bad = ~np.isfinite(lnL)
        if np.any(bad):
            self.n_nonfinite += int(bad.sum())
            lnL = np.where(bad, -np.inf, lnL)
        if self.reduce == "profile":
            return lnL.max(axis=1)
        return logsumexp(lnL + self.sky.log_w[None, :], axis=1) - np.log(lnL.shape[1])

    def adapt_sky(self, P, n_pilot_batches=4, n_rounds=6, neff_target=None,
                  n_prior_batches=16, **kw):
        """Refit the sky/inclination proposal at P.

        Round 0 pilots from the prior (n_prior_batches batches); later rounds
        pilot from the current proposal, with weights ln L + ln(p/q).  Stops early once the batch n_eff
        reaches neff_target (default: 5% of the batch).
        """
        if self._ripple is not None and "rb" in self._ripple:
            self.rb_refresh(P)
        data, _ = self.precompute(P)
        self._refresh_amp(data)
        neff_target = neff_target or 0.05 * self.sky.n
        info = {}
        for rnd in range(n_rounds):
            ras, decs, incls, lws = [], [], [], []
            for _ in range(n_prior_batches if rnd == 0 else n_pilot_batches):
                if rnd == 0:
                    ra, dec, incl = _prior_draw(self.rng, self.sky.n)
                    lw0 = np.zeros(self.sky.n)
                else:
                    self.sky.redraw()
                    ra, dec, incl, lw0 = self.sky.ra, self.sky.dec, self.sky.incl, self.sky.log_w
                lnL, _ = self.lnL_sky(P, ra, dec, incl, data=data)
                ras.append(ra); decs.append(dec); incls.append(incl); lws.append(lnL + lw0)
            ra, dec, incl, lw = map(np.concatenate, (ras, decs, incls, lws))
            info = self.sky.adapt(ra, dec, incl, lw, **kw)
            lnL_new, _ = self.lnL_sky(P, data=data)
            w = lnL_new + self.sky.log_w
            w = np.exp(w - w.max())
            info.update(round=rnd, neff_batch=float(w.sum() ** 2 / np.sum(w ** 2)),
                        lnL_max=float(np.max(lnL_new)))
            if self.verbose:
                print(" [intrinsic_objective] sky adapt", info)
            if info["neff_batch"] >= neff_target:
                break
        return info

    def stats(self):
        return dict(n_calls=self.n_calls, n_compiles=self.n_compiles, n_nonfinite=self.n_nonfinite,
                    n_exceptions=self.n_exceptions,
                    t_waveform=self.t_waveform, device_precompute=str(self.device_precompute),
                    t_precompute=self.t_precompute, t_kernel=self.t_kernel)
