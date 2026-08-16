"""
seeding.py:  central RNG seeding for the RIFT drivers.

WHY THIS EXISTS
---------------
Historically ``--seed`` was implemented in the ILE drivers as a bare
``numpy.random.seed(opts.seed)``.  That only covers the CPU code path.

Every sampler in RIFT/integrators draws its variates through the *array
backend* it was configured with -- ``self.xpy`` on an instance, or the
module-level ``xpy_default`` -- and that backend is ``cupy`` whenever the job
runs on a GPU.  The draws that decide the answer are therefore cupy draws:

  * ``self.xpy.random.uniform``  in mcsamplerGPU.draw_simplified (inverse-CDF
    sampling: this is the main integrand proposal)
  * ``xpy_default.random.uniform`` in mcsamplerAdaptiveVolume.sample_from_bins
  * ``self.xpy.random.uniform``  in MonteCarloEnsemble
  * ``self.xpy.random.choice``   in the fair-draw / extrinsic-resample paths of
    mcsamplerGPU, mcsamplerAV, mcsamplerPortfolio, mcsamplerEnsemble, and in
    gaussian_mixture_model's k-means++ initialization

cupy keeps its own global generator, per device, which ``numpy.random.seed``
does not touch.  So a GPU run was irreproducible even when the user asked for a
seed: two byte-identical invocations of the ILE demo with ``--seed 101``
returned lnL = 73.807 (n_eff 5.9) and lnL = 73.520 (n_eff 10.8).  That silently
invalidates any paired / replicate-seed comparison design on GPU, because the
"same seed" arms are not in fact paired.

``seed_everything`` seeds every backend a RIFT sampler can reach, so that the
meaning of ``--seed`` does not depend on which device the job landed on.

CAVEAT (cupy is per-device)
---------------------------
``cupy.random.seed`` seeds the generator of the *current* device only; cupy
holds a separate generator per device, created lazily.  A single-device job --
which is what an ILE process is -- is fully covered.  If more than one device
is visible we say so, rather than implying a guarantee we are not making.
"""

import numpy


__all__ = ['seed_everything', 'get_seed']


# The seed the process was started with, or None if the run was never seeded.
# Exposed via get_seed() so that code needing its own independent stream (e.g.
# a bootstrap diagnostic) can derive one deterministically instead of pulling
# fresh entropy from the OS.
_seed_used = None


def get_seed():
    """Return the seed passed to seed_everything, or None if never seeded."""
    return _seed_used


def seed_everything(seed, verbose=True):
    """Seed every RNG backend a RIFT sampler can draw from.

    Parameters
    ----------
    seed : int
        The seed.  Applied to all backends, so that switching a run between CPU
        and GPU changes which backend is used, not whether the run is seeded.
    verbose : bool
        Print a one-line report of what was actually seeded.  Worth leaving on:
        the failure mode this function exists to fix was invisible.

    Returns
    -------
    dict
        backend name -> status string, one of 'seeded', 'absent' (library not
        installed) or 'failed: <reason>'.  Backends that are absent are not an
        error: a CPU-only install has no cupy, and only mcsamplerNFlow needs
        torch.
    """
    global _seed_used

    seed = int(seed)
    _seed_used = seed
    status = {}

    # Python's stdlib RNG.  Not used by the samplers today, but it is used
    # incidentally elsewhere (and by some dependencies), and it is free.
    import random as _pyrandom
    _pyrandom.seed(seed)
    status['python'] = 'seeded'

    # numpy: the CPU sampler path, and everything that reaches numpy's legacy
    # global RandomState -- which includes scikit-learn estimators constructed
    # with random_state=None, e.g. the KMeans init in weighted_gmm.
    numpy.random.seed(seed)
    status['numpy'] = 'seeded'

    # cupy: the GPU sampler path.  Importing cupy on a machine with no working
    # CUDA install raises, and seeding can raise even when the import succeeds
    # (no device, or a device this cupy build cannot drive), so both steps are
    # guarded -- an unseedable GPU backend must not take down a CPU run.
    n_dev = 0
    try:
        import cupy
    except Exception as e:
        status['cupy'] = 'absent'
    else:
        try:
            n_dev = cupy.cuda.runtime.getDeviceCount()
            cupy.random.seed(seed)
            status['cupy'] = 'seeded'
        except Exception as e:
            status['cupy'] = 'failed: {}'.format(e)

    # torch: only mcsamplerNFlow needs it.
    try:
        import torch
    except Exception:
        status['torch'] = 'absent'
    else:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            status['torch'] = 'seeded'
        except Exception as e:
            status['torch'] = 'failed: {}'.format(e)

    # Seeding the RNGs is necessary but not sufficient on GPU.  The adapted
    # sampling histogram is built with a weighted cupy.bincount, which sums
    # through float atomicAdd; the ordering is set by thread scheduling, so the
    # adapted CDF -- and hence every draw taken through it -- varies at the ULP
    # level between otherwise identical runs.  Switch that one reduction to a
    # scheduler-independent summation order, so that "same seed" really does
    # mean "same answer".  Pushed from here rather than pulled from there so
    # that RIFT.likelihood keeps no dependency on the integrators.
    try:
        from RIFT.likelihood import vectorized_general_tools as _vgt
        _vgt.DETERMINISTIC_REDUCTIONS = True
        status['gpu_reductions'] = 'deterministic'
    except Exception as e:
        status['gpu_reductions'] = 'failed: {}'.format(e)

    if verbose:
        print(" Seeding RNGs with {}: {}".format(
            seed, ", ".join("{}={}".format(k, status[k]) for k in sorted(status))))
        if status.get('cupy') == 'seeded' and n_dev > 1:
            print("   NOTE: cupy generators are per-device; seeded the current"
                  " device only ({} visible).  Pin one device (CUDA_VISIBLE_DEVICES)"
                  " for a fully reproducible GPU run.".format(n_dev))

    return status
