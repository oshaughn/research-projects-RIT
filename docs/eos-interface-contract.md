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

Reviewed two- or nine-column tables use:

```text
--using-eos lalsim_file:<path> [--using-eos-dirty-phase-transitions]
    [--using-eos-extended-family] [--using-eos-branch <integer>]
```

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
certify the reviewed SWIG interface. To run the real-build gate, build the
exact reviewed LALSuite commit, activate that Python environment, and set
`RIFT_REVIEWED_LALSIM_MANIFEST` to a JSON file with this shape:

```json
{
  "lalsuite_ref": "0123456789abcdef0123456789abcdef01234567",
  "fixtures": {
    "two_column": {"path": "two-column.dat", "sha256": "..."},
    "nine_column": {"path": "nine-column.dat", "sha256": "..."},
    "twin_star": {"path": "twin-star.dat", "sha256": "...", "dirty_phase_transitions": true}
  }
}
```

Run
`pytest MonteCarloMarginalizeCode/Code/test/test_lalsim_eos_reviewed_integration.py`.
Paths are relative to the manifest. The ref must be the full 40-character
commit actually built by the job: the gate requires it to equal
`lalsimulation.SimulationVCSInfo.vcsId` and requires a clean VCS build. Every
fixture hash is mandatory. Once the manifest enables the gate, missing modern
symbols, malformed or mismatched build provenance, missing fixtures, wrong
column counts, or absence of distinct overlapping twin-star solutions are
failures. The gate exercises clean and phase-transition-correcting readers,
minimal and extended family construction, and branch-indexed radius, Love
number, central pressure, and tidal deformability. Ordinary CI skips this
private-build gate explicitly.

The drift-sentinel registry should eventually declare an EOS contract group
with LALSuite and NuclearMatter-Backend as producers and RIFT/nmb-papers as
consumers.  The current registry only covers RIFT/hyperpipe operational archive
and queue boundaries, so it cannot detect EOS schema or callable drift yet.
