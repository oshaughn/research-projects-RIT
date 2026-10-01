"""Cosmological luminosity-distance priors for the JAX ILE.

``--d-prior cosmo`` and ``--d-prior cosmo_sourceframe`` mean, as in
``integrate_likelihood_extrinsic_batchmode`` and ``util_InitMargTable``,

    p(d_L) dd_L  ∝  dV_c/dz (1+z)^-s dz,     s = 0 (cosmo), 1 (cosmo_sourceframe),

with the Planck15 cosmology from ``priors_utils.get_astropy_cosmology``.  The
batchmode ILE builds this with astropy at setup and evaluates a scipy/cupy
interpolant.  Neither can be traced by JAX, and ``z(d_L)`` has no closed form.

This module tabulates the smooth ratio

    r(d_L) = ln p(d_L) - 2 ln d_L

once, on nodes uniform in ln z, using the closed form for a flat cosmology,

    dV_c/dz  = D_H D_c^2 / E(z)           (per steradian)
    dd_L/dz  = D_c + (1+z) D_H / E(z),

and evaluates ``ln p`` by linear interpolation of ``r`` in ``ln d_L``.  The
same function serves NumPy (setup-time quadrature weights, AV densities, prior
draws) and ``jax.numpy`` (traced inside ``jit``, differentiable in ``d_L``), so
every consumer sees one density.  At low redshift ``r -> 0`` (``p -> d_L^2``),
which is the value used below the first node.
"""

import functools

import numpy as np

COSMO_DISTANCE_PRIORS = ("cosmo", "cosmo_sourceframe")
_EUCLIDEAN_ALIASES = ("euclidean", "volumetric")

# Table: z in [_Z_LO, _Z_HI], uniform in ln z.  _Z_HI = 200 is d_L ~ 2.3e6 Mpc
# for Planck15, above any distance RIFT is run with.
_Z_LO = 1.0e-8
_Z_HI = 200.0
_N_TABLE = 8192
# Dense ln d grid for normalization and inverse-CDF draws.
_N_QUAD = 1 << 16


def canonical_distance_prior(name):
    """Lower-case ``--d-prior`` value; None/'' and the d^2 aliases -> 'euclidean'."""
    key = str(name or "euclidean").strip().lower()
    return "euclidean" if key in _EUCLIDEAN_ALIASES else key


def is_cosmo_distance_prior(name):
    return canonical_distance_prior(name) in COSMO_DISTANCE_PRIORS


class CosmoDistancePrior(object):
    """Tabulated ``p(d_L)`` for one of :data:`COSMO_DISTANCE_PRIORS`.

    Use :func:`cosmo_distance_prior`, which caches instances.  ``d`` is in Mpc.
    """

    def __init__(self, kind, cosmology="Planck15"):
        kind = canonical_distance_prior(kind)
        if kind not in COSMO_DISTANCE_PRIORS:
            raise ValueError("not a cosmological distance prior: %r" % (kind,))
        from RIFT.likelihood import priors_utils
        cosmo = priors_utils.get_astropy_cosmology(cosmology)
        if float(cosmo.Ok0) != 0.0:
            raise ValueError("%s is not flat; dd_L/dz below assumes Ok0 = 0"
                             % (cosmology,))
        self.kind = kind
        self.cosmology = cosmology
        s = 1.0 if kind == "cosmo_sourceframe" else 0.0
        z = np.geomspace(_Z_LO, _Z_HI, _N_TABLE)
        d_h = float(cosmo.hubble_distance.to("Mpc").value)
        d_c = np.asarray(cosmo.comoving_distance(z).to("Mpc").value, dtype=float)
        inv_e = np.asarray(cosmo.inv_efunc(z), dtype=float)
        d_l = (1.0 + z) * d_c
        ln_p = (np.log(d_h) + 2.0 * np.log(d_c) + np.log(inv_e)
                - s * np.log1p(z) - np.log(d_c + (1.0 + z) * d_h * inv_e))
        self.ln_d_nodes = np.log(d_l)
        self.r_nodes = ln_p - 2.0 * self.ln_d_nodes
        self.d_table_max = float(d_l[-1])
        self._r_low = 0.0

    def check_support(self, d_max):
        if not (float(d_max) <= self.d_table_max):
            raise ValueError("--d-prior %s is tabulated to d_L = %.4g Mpc; "
                             "d_max = %.4g Mpc is beyond it"
                             % (self.kind, self.d_table_max, float(d_max)))

    def log_density_unnormalized(self, d, xp=np):
        """``ln p(d_L)`` up to a constant; ``xp`` is ``numpy`` or ``jax.numpy``."""
        d = xp.asarray(d)
        ln_d = xp.log(d)
        r = xp.interp(ln_d, xp.asarray(self.ln_d_nodes), xp.asarray(self.r_nodes),
                      left=self._r_low)
        return r + 2.0 * ln_d

    def density_unnormalized(self, d):
        return np.exp(self.log_density_unnormalized(np.asarray(d, dtype=float)))

    @functools.lru_cache(maxsize=32)
    def _cdf_table(self, lo, hi):
        self.check_support(hi)
        if not (0.0 < lo < hi):
            raise ValueError("need 0 < lo < hi, got (%r, %r)" % (lo, hi))
        u = np.linspace(np.log(lo), np.log(hi), _N_QUAD)
        d = np.exp(u)
        g = np.exp(self.log_density_unnormalized(d)) * d     # p dd = p d du
        cdf = np.concatenate(([0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(u))))
        return d, cdf

    def log_mass(self, lo, hi):
        """``ln ∫_lo^hi p_unnorm(d) dd``."""
        _, cdf = self._cdf_table(float(lo), float(hi))
        return float(np.log(cdf[-1]))

    def log_density(self, d, lo, hi, xp=np):
        """``ln p(d)`` normalized over ``[lo, hi]`` (support is not masked)."""
        return self.log_density_unnormalized(d, xp=xp) - self.log_mass(lo, hi)

    def sample(self, n, rng, lo, hi):
        """``n`` draws from ``p`` restricted to ``[lo, hi]`` (inverse CDF)."""
        d, cdf = self._cdf_table(float(lo), float(hi))
        return np.interp(rng.uniform(0.0, cdf[-1], int(n)), cdf, d)


@functools.lru_cache(maxsize=8)
def cosmo_distance_prior(kind, cosmology="Planck15"):
    return CosmoDistancePrior(canonical_distance_prior(kind), cosmology)
