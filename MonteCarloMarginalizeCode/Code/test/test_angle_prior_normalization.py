"""The ILE psi and phi_orb priors are derived from their sampling ranges, so they integrate to 1.

The defect: the drivers sampled psi on (0, 2 pi) (so that --internal-rotate-phase can use
(phi+psi, phi-psi) coordinates) but passed the fixed constant mcsampler.uniform_samp_psi =
1/pi, written for [0, pi), as its prior.  A prior defined independently of the range it is
integrated over has no reason to be normalized: this one had mass 2, mass 4 under
--internal-rotate-phase (range (0, 4 pi)), where phi_orb's fixed 1/(2 pi) also doubled.
Every reported lnZ carried +ln 2 (+ln 8 rotated), invisible in posteriors (the likelihood
is pi-periodic in psi) and equal to the size of AV-vs-JAX or RIFT-vs-bilby agreements people
quote.  Found by the 2026-09-08 marginalization audit (paper repo,
analyses/marginalization_audit_20260908, MATRIX X4).

Two checks, cheapest first:

1. test_range_derived_prior_integrates_to_one: the constructor the drivers now use,
   ret_uniform_samp_vector_alt(lo, hi), integrates to 1 over (lo, hi) on both the numpy and
   the GPU-module backends, and the constant it replaced does not.
2. test_driver_priors_are_built_from_their_ranges: source check on all three drivers that
   the prior objects are built from psi_prior_range / phi_orb_prior_range, captured AFTER the
   rotate-phase widening.
(The rift_O4d line adds an end-to-end --zero-likelihood run, where lnZ is exactly
ln(total prior mass): 0 now, ln 2 before, ln 8 before under --internal-rotate-phase.)
"""

from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent

CODE = HERE.parent
DRIVERS = {
    "batchmode": CODE / "bin" / "integrate_likelihood_extrinsic_batchmode",
    "legacy": CODE / "bin" / "integrate_likelihood_extrinsic",
}


###
### 1. The constructor is normalized over its own range; the constant it replaced is not
###

@pytest.mark.parametrize("lo,hi", [(0.0, 2*np.pi), (0.0, 4*np.pi), (0.3, 1.1)])
def test_range_derived_prior_integrates_to_one(lo, hi):
    import RIFT.integrators.mcsampler as mcsampler
    x = np.linspace(lo, hi, 2001)[1:-1]
    dens = np.asarray(mcsampler.ret_uniform_samp_vector_alt(lo, hi)(x), dtype=float)
    assert np.allclose(dens*(hi - lo), 1.0)
    # the fixed constant assumed [0, pi): over the driver's (0, 2 pi) it is mass 2
    old = np.asarray(mcsampler.uniform_samp_psi(x), dtype=float)
    assert np.allclose(old*2*np.pi, 2.0)

    # the GPU module's constructor, which the AV and GPU samplers bind as mcsampler
    try:
        import RIFT.integrators.mcsamplerGPU as G
    except Exception as exc:              # pragma: no cover  (module imports without cupy)
        pytest.skip("mcsamplerGPU unavailable: %r" % exc)
    val = G.ret_uniform_samp_vector_alt(lo, hi)(x)
    conv = getattr(G, "identity_convert", None)
    if conv is not None:
        val = conv(val)
    assert np.allclose(np.asarray(val, dtype=float)*(hi - lo), 1.0)


###
### 2. Wiring: every driver builds both angle priors from the captured ranges
###

@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_driver_priors_are_built_from_their_ranges(name):
    text = DRIVERS[name].read_text()
    assert "prior_pdf = mcsampler.uniform_samp_psi" not in text, \
        "%s still passes the fixed 1/pi constant as the psi prior" % name
    assert 'psi_prior_range = (param_limits["psi"][0], param_limits["psi"][1])' in text
    assert 'phi_orb_prior_range = (param_limits["phi_orb"][0], param_limits["phi_orb"][1])' in text
    assert "mcsampler.ret_uniform_samp_vector_alt(psi_prior_range[0], psi_prior_range[1])" in text
    assert "mcsampler.ret_uniform_samp_vector_alt(phi_orb_prior_range[0], phi_orb_prior_range[1])" in text
    # phi_orb's prior is the range-derived one, not the fixed phase constant
    i_phi = text.index('sampler.add_parameter("phi_orb"')
    phi_block = text[i_phi:text.index("adaptive_sampling", i_phi) if "adaptive_sampling" in text[i_phi:i_phi+800] else i_phi+800]
    assert "prior_pdf = mcsampler.ret_uniform_samp_vector_alt(phi_orb_prior_range[0], phi_orb_prior_range[1])" in phi_block
    assert "uniform_samp_phase" not in phi_block

    if name == "batchmode":
        assert "prior_pdf = psi_prior_pdf," in text
        # captured after --internal-rotate-phase widens the ranges, so the rotated box is
        # what the prior normalizes over
        i_rot = text.index("param_limits['psi'] = (0, 4*numpy.pi)")
        i_cap = text.index("psi_prior_range = (")
        assert i_rot < i_cap, "%s captures the prior range before the rotate-phase widening" % name
