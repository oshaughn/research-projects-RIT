# EOS interface contract across RIFT, LALSimulation, and NuclearMatter-Backend

RIFT keeps the historical fixed-EOS consumer surface:

```python
eos.lambda_from_m(m)
eos.lambda_from_m_vector(masses)
eos.mMaxMsun
```

Mass arguments may be in solar masses or SI kg, as before.

## LALSimulation families

Released LALSimulation has one stable family and needs no user change.  The
reviewed multipart TOV API can return several stable branches.  RIFT exposes
`branches_for_m(m)` and accepts `branch_id` on direct `radius_from_m` and
`lambda_from_m` calls.  An ambiguous twin-star mass raises instead of silently
choosing a solution.

Legacy scalar consumers select a branch once with `eos.for_branch(branch_id)`.
The fixed-EOS CIP driver exposes the same operation as:

```text
--using-eos lal_<name> --using-eos-branch <integer>
```

A selected branch is bounded at both ends, so the view publishes `mMinMsun`
alongside `mMaxMsun`; a twin-star branch can begin well above the family
minimum mass. Consumers that mask draws should mask against both bounds.
Because the historical consumers mask only on `mMaxMsun`, `lambda_from_m` (and
therefore `lambda_from_m_vector`) returns `-inf` for a mass with no stable star
on the selected branch rather than raising, so a single out-of-branch draw
flags itself instead of aborting the batch it arrived in.

The flag is only useful if the consumer acts on it, and the consumer cannot
read it off the converted coordinates.
`lalsimutils.convert_waveform_coordinates_with_eos` returns only `coord_names`,
so with a mass-only fit basis such as `mc,eta` the lambda carrying the flag is
computed and then discarded, and the row reaches the fit fully finite. On the
vectorized path an above-`mMaxMsun` mass is worse than invisible: it arrives as
`lambda = 0`, which reads as a black hole rather than as "no such star".

The support test is therefore made against the sampled MASSES, by
`RIFT.physics.lalsim_eos_compat.mass_in_eos_support`, which tests each object
against whichever of `mMinMsun` and `mMaxMsun` the EOS publishes and exempts any
object declared a black hole by `--no-matter1` / `--no-matter2` (or
`--assume-eos-but-primary-bh` on export). A bound the EOS does not publish is
read as unbounded, which leaves EOS classes exposing neither unchanged. With a
fixed EOS, CIP applies that test unconditionally -- not only under
`--protect-coordinate-conversions` -- stamping out-of-support rows `-inf` before
the likelihood fit, so they carry zero probability while the rest of the batch
is fit normally in one call. The export path applies the SAME test before asking
the EOS for lambda at all, so the fit and the export cannot disagree about which
draws exist.

Reviewed two- or nine-column tables use:

```text
--using-eos lalsim_file:<path> [--using-eos-dirty-phase-transitions]
    [--using-eos-extended-family] [--using-eos-branch <integer>]
```

By default, `lalsim_file:` preserves released
`SimNeutronStarEOSFromFile` behavior even on a reviewed build. Requesting a
branch, the compatibility dirty-phase-transition option, or the extended
family opts into `SimNeutronStarEOSFromFilePhaseTransition` followed by
`CreateSimNeutronStarFamilyPT(eos, min_fam)`. The historical
`--using-eos-dirty-phase-transitions` spelling is retained as a compatibility
request, but the reviewed SWIG interface has one phase-transition reader; it
does not expose a separate dirty-table boolean. On released LALSimulation,
that flag fails explicitly rather than pretending the legacy reader repaired a
phase transition.

The Python API also accepts `family_log_pressure_min=<value>` on
`EOSLALSimulationFromFile`. This opts into
`CreateSimNeutronStarFamilyPTWithPcmin(eos, min_fam, logPcmin)`; the value must
be finite and is the natural logarithm `ln(Pc / Pa)`. Mass-radius helpers use
the canonical `multipart=` keyword and retain `reviewed_multibranch=` as a
deprecated O4c compatibility alias; conflicting explicit values raise.

Branch-specific causality checks dispatch to the multipart accessors
`SimNeutronStarEOSMultiPartsPseudoEnthalpyOfPressure` and
`SimNeutronStarEOSMultiPartsSpeedOfSoundOfPseudoEnthalpy`; they never pass a
`SimEOSMultiParts` object to a legacy single-EOS accessor. The multipart
sound-speed accessor returns SI metres per second, so RIFT
divides by `lal.C_SI` before applying its historical dimensionless `v/c < 1.1`
test. Full-table causality uses
`SimNeutronStarEOSMultiPartsMaxPseudoEnthalpy`; it fails closed if that reviewed
symbol is unavailable.

For pseudo-pipe workflows, forward the flag with
`--manual-extra-cip-args`.  On O4d, Hydra hyperpipe configurations can put it
in the post driver's `extra-args` when that driver is the fixed-EOS CIP
executable.

## NuclearMatter-Backend sequences

The `nmbackend.nss/1` (`tabular_hc/1`) and `nmbackend.pca/1` (`pca_hc/1`)
producer tags, field names, and `EOSSequenceFromFile` dispatch remain unchanged.
The nmb-papers exact-evidence command remains:

```text
--using-eos nmbseq:<sequence.h5>:<index>
```

That v1 consumer contract intentionally projects each EOS onto its **primary
stable mass-rising branch**.  RIFT now splits central-enthalpy-ordered data at
unstable or decreasing-mass intervals before doing M-to-Lambda interpolation;
it never mixes disconnected branches.  `stable_branch_counts[index]` records
when a native sequence contains more than one stable run.

`--using-eos-branch` is deliberately rejected for `nmbseq:`.  Full NMB
multi-branch inference is not a scalar `M -> Lambda` problem: the required path
is a central-enthalpy likelihood/integration coordinate (one per star), using
the branch-explicit native sequence rather than the legacy Landry projection.
Until that inference path lands, generate or consume the primary branch for
paper-production parity and treat disconnected branches as a separate model.

## Downstream compatibility

- Existing Kedia-style spectral and piecewise-polytrope EOS inference keeps the
  same constructor and scalar lookup APIs.  Ordinary single-branch models need
  no new option.
- Existing nmb-papers and hyperpipe fixed-sequence runs keep the same HDF5 tags
  and `nmbseq:` syntax.
- Twin-star analyses using reviewed LALSimulation must select a branch for
  legacy scalar workflows, or move to the central-enthalpy inference path when
  marginalization over branch identity is scientifically required.

`--using-eos-branch` is rejected with `--using-eos-for-prior`. The current
EOS-hyperprior plugin protocol returns an EOS but does not return branch
identity, so accepting this combination would silently analyze the wrong
branch. A future multibranch hyperprior must extend that plugin contract before
this restriction can be relaxed.

## Reviewed-LALSimulation integration gate

The fake-backed compatibility tests check RIFT's dispatch logic, but do not
certify the reviewed SWIG interface. Fixture-free acceptance uses trusted
`SimNeutronStarEOSMultiPartsByName("SLY")`, checks coexistence with the legacy
named-EOS API, and repeatedly constructs and destroys families. To run it,
build the exact reviewed LALSuite commit, activate that Python environment, and set
`RIFT_REVIEWED_LALSIM_MANIFEST` to a JSON file with this shape:

```json
{
  "lalsuite_ref": "974c0ef468b76e8298e67fd8baf71ed259cc5fee",
  "fixtures": {
    "two_column": {"path": "two-column.dat", "sha256": "..."},
    "nine_column": {"path": "nine-column.dat", "sha256": "..."},
    "twin_star": {"path": "twin-star.dat", "sha256": "...", "columns": 9}
  },
  "known_upstream_crash": {
    "path": "eos_2pt.dat",
    "sha256": "..."
  }
}
```

The `fixtures` object is optional, and `twin_star` is optional within it.
Without fixtures, builtin acceptance runs and external-table coverage is an
explicit skip. File-loader fixtures must be explicit two- or nine-column
numeric tables. Four-column wiki arrays require a separate, provenance-recorded
unit conversion and are not passed to the native file reader. Each external
table is loaded in a subprocess with a 60-second timeout, output sent to
`DEVNULL`, and only a compact JSON status file returned to pytest.

The optional `known_upstream_crash` entry is converted through
`SimNeutronStarEOSFromArraysPhaseTransition` using the reviewed geometrized
factor `1.602176634e32 * G / c^4`. The authoritative two-transition
four-column array fixture currently exits
with signal 11 (shell return code 139) inside the reviewed native implementation.
That is an upstream acceptance blocker, not a passing RIFT fixture. It must be
rerun through a separately provenance-recorded array conversion after the
native crash is fixed; this gate does not silently reinterpret it as a file
reader input.

Run
`pytest MonteCarloMarginalizeCode/Code/test/test_lalsim_eos_reviewed_integration.py`.
Paths are relative to the manifest. The ref must be the full 40-character
commit actually built by the job: the gate requires it to equal
`lalsimulation.SimulationVCSInfo.vcsId` and requires a clean VCS build. Every
supplied fixture hash is mandatory. Once the manifest enables the gate, missing modern
symbols, malformed or mismatched build provenance, missing fixtures, wrong
column counts, or absence of distinct overlapping twin-star solutions are
failures. The gate exercises the real phase-transition reader on two- and
nine-column inputs, minimal and extended family construction, and
branch-indexed radius, Love
number, central pressure, and tidal deformability. Ordinary CI skips this
private-build gate explicitly.

The drift-sentinel registry should eventually declare an EOS contract group
with LALSuite and NuclearMatter-Backend as producers and RIFT/nmb-papers as
consumers.  The current registry only covers RIFT/hyperpipe operational archive
and queue boundaries, so it cannot detect EOS schema or callable drift yet.
