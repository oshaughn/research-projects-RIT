# Network sky coordinates for JAX AV

A narrow two-detector time-delay ring is curved in right ascension/declination.
JAX AV and AV/GMM portfolio can instead adapt `cos_theta_n` and `phi_n` about
the first two detector locations in the likelihood's detector order:

```
integrate_likelihood_extrinsic_jax ... --sampler-method AV --sky-coordinates network
```

`--internal-sky-network-coordinates` is an AV/portfolio compatibility flag.
Like conventional ILE, this alias excludes V1/K1 when choosing the baseline
(usually H1-L1), while retaining every detector in the physical likelihood.
Explicit `--sky-coordinates network` uses the first two detectors without those
exclusions. If fewer than two eligible detectors remain (HV, LV or single-IFO
events), the alias falls back to equatorial sampling with a printed note, as
classic ILE does. The explicit request fails instead. Adding
`--internal-sky-network-coordinates-raw` keeps V1/K1 and the given detector order,
as in classic ILE; alone it does nothing. If both the alias and
`--sky-coordinates network` are given, the alias rules apply.

The existing ECEF network-frame rotation is reused, including the likelihood's
GMST convention. The physical likelihood still receives RA/DEC and physical
polarization/orientation parameters. An isotropic sky has uniform density in
`cos_theta_n` on [-1,1] and `phi_n` on [0,2 pi), so no extra angular Jacobian is
needed during integration. On conversion back to the exported RA/DEC cloud,
both the joint prior and proposal densities acquire `cos(DEC)`. Their ratio,
importance weights, evidence and ESS are unchanged by that conversion. Distance
and marginalized angle coordinates are untouched. Returned sample arrays and
fair-draw/XML consumers continue to use the physical likelihood parameter order.
The sampler's internal records, sampling bounds and bootstrap `seed_cloud` use
network coordinates; Fisher seed mode locations remain physical.

Caller `initial_samples` and Fisher `seed_initial_points` remain in physical
coordinates. Fisher initialization runs in that physical frame and its cloud is
rotated before AV/GMM bootstrap. This change neither changes the AV coverage
budget nor alters its stopping rule, ESS target or fixed likelihood evaluation
chunk.

AV network sampling requires two distinct detector locations and both sampled
sky axes. Single-detector/degenerate-baseline requests fail clearly. Restricted
RA/DEC sampling windows also fail: a physical rectangular window cannot be
represented by merely relabeling network rectangular bounds. Non-sky limits
remain supported. Equatorial sampling is still the default. Multistart NUTS's
existing network implementation and other samplers are unchanged.

Validation includes isotropic sky moments, coordinate roundtrips, fixed-batch
likelihood parity, actual AV integration against an analytic narrow-ring evidence,
and exported prior/proposal density consistency. These synthetic CPU tests do
not demonstrate a production GPU speedup or establish convergence for a real
sky posterior.
