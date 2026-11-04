"""SMC-MALA followed by a short birth-death rejuvenation (engine version)."""
from .smc_mala import iterate as _smc_mala
from .birth_death import iterate as _bd


def iterate(particles, *, surrogate, surrogate_prev=None,
            prior_box, rng, state=None, **kw):
    X1, info1 = _smc_mala(particles, surrogate=surrogate, surrogate_prev=surrogate_prev,
                          prior_box=prior_box, rng=rng, state=state, **kw)
    X2, info2 = _bd(X1, surrogate=surrogate, prior_box=prior_box, rng=rng,
                    state=info1.get("state"),
                    n_langevin_steps=5, n_bd_passes=2,
                    birth_death_rate=kw.get("birth_death_rate", 1.0))
    info = {**info1, **info2, "state": info2.get("state", info1.get("state"))}
    return X2, info
