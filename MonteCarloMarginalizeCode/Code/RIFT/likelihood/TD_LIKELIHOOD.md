# Experimental time-domain likelihood operators

The default `fd-diagonal` likelihood is unchanged. The opt-in `td-toeplitz`
factory supplies PSD-weighted inner products, template-frame temporal cuts,
fixed data gaps and low-rank Gaussian glitch-coefficient marginalization to the
existing Precompute interface. No detector data or learned basis is distributed.

## Installation and dependency boundary

The operator and rank-augmentation modules use the existing NumPy/SciPy/LAL stack.
Only physical-covariance PCG solves require the separately distributed public
`tdanalysis.forms.pcg.PCGQuadraticForm` interface:

```sh
python -m pip install '.[td-pcg]'  # from this checkout
```

This extra requires Python >=3.11 and tdanalysis 1.x. RIFT does not vendor that
package or its private predecessor. tdanalysis has its own noncommercial license
and additional terms; see https://pypi.org/project/tdanalysis/ . Its license does
not change the license of these independently maintained RIFT adapters. PCG
preparation currently allocates a dense covariance, so its memory remains O(N^2).

## Operators and boundaries

The full-segment path applies the same band-limited inverse-PSD operator as the
FD implementation, without inverse-spectrum truncation. TD/FD agreement tests
that shared operator. A principal submatrix of a long-stream precision describes
conditioning on omitted data; it is generally not the inverse of the marginal
covariance of retained data. The cropped-precision and hard-cut-template modes
are experimental analysis conventions, not a general finite-data marginal model.
Data tapering, filtering and padding must be applied consistently with the PSD,
waveform and intended boundary model. Selecting TD does not repair mismatched
conditioning. Band projection and covariance normalization remain model choices.

## ILE and pipeline options

Use the maintained NoLoop route for CPU as well as GPU execution:

```sh
integrate_likelihood_extrinsic_batchmode \
  --time-marginalization --vectorized --gpu --force-xpy \
  --internal-ile-inner-product td-toeplitz [ordinary data and waveform options]
```

`--td-solver {auto,dense,pcg}`, `--td-acf-duration` and `--td-analysis-start/end`
configure the experimental operator. Temporal cut offsets are seconds relative to
the template peak. `--td-gap` is refused at startup: its gap marginal depends on the
K_GG pseudo-inverse cutoff when fMax < fNyq. To excise data use td-gs with disjoint
`--td-piece` intervals. A single dominant-mode peak sets
the template-cut reference for all modes and detectors at each intrinsic point.
`util_RIFT_pseudo_pipe.py` forwards the corresponding `--internal-ile-td-*` flags.
The separate LISA driver is not extended; its drift ledger records the TDI-specific
physics validation needed before porting these flags. Slow rotation, frequency
response and FD-only noise-evidence exports are refused.
Calibration's per-realization FD self term is refused; global-norm calibration
uses the selected factory consistently on the full stationary operator. Calibration
with cuts, gaps or glitch projection is refused because data-side frequency
calibration does not commute with those operators.

## Gaussian glitch marginalization

`--glitch-basis IFO:t1,t2,f1,f2[:r]` constructs a Slepian basis. Alternatively,
`--glitch-basis-file IFO:path.npz` loads user-provided arrays: `basis` is an
`n_snippet x rank` real matrix, `lambda0` is its dimensionless shape variance,
`t0` is the first sample's geocentric offset in seconds from the event, and `fs`
is the sample rate. The pipeline transfers basis files as auxiliary inputs.
Independent Gaussian coefficients can be integrated analytically with Woodbury
identities. Glitch bases require the full segment; combining them with TD cuts,
gaps or shortened ACF duration is refused. Duplicate basis specifications and
requests for detectors without data are also refused. Multiple disturbance blocks retain cross-Gram terms within a detector.
The default scale is empirically profiled, not integrated under a hyperprior.

The current Precompute self-overlaps are scalar values at the nominal geocenter
event, with template epochs translated into the detector data frame. A localized
basis can make them arrival-time dependent; endpoint diagnostics do not bound
that effect. Posterior coverage and precision evidences require validation of this
approximation or time-indexed self terms. A visually accurate reconstruction does
not establish signal preservation. This PR exposes the zero-mean Gaussian CLI;
population-prior helpers are tested at library level, while two-pass sign-mixture
sampling and its output-bookkeeping integration are deferred.

## Tests

```sh
python -m pytest MonteCarloMarginalizeCode/Code/test/test_td*.py -q
```

Tests cover default-path parity, complex Hermitian conventions, covariance solves
against dense references, Woodbury inverses/determinants, mixture identities,
temporal cuts/gaps, NoLoop parity, driver normalization/export accounting and
incompatible configuration guards. The optional PCG check
requires the public extra. The modern dependency-compatibility CI lane installs
the public wheel and runs this case with an explicit import check. Locally, run it with `pytest -k dense_vs_iterative -rs` to make
its execution or skip visible. Full likelihood and GPU regressions follow the
existing `.travis/PRECOMMIT.md` requirements before production merge.
