# EOS interface contract across RIFT, LALSimulation, and NuclearMatter-Backend

RIFT keeps the historical fixed-EOS consumer surface:

```python
eos.lambda_from_m(m)
eos.lambda_from_m_vector(masses)
eos.mMaxMsun
```

Mass arguments may be in solar masses or SI kg, as before.

Multibranch EOS objects add:

```python
eos.branches_for_m(m)                # stable branch ids containing m
eos.lambda_from_m(m, branch_id=b)
eos.lambda_from_m_vector(m, branch_id=b)
eos.for_branch(b)                    # scalar view pinned to one branch
eos.mMaxMsun_all                     # NMB only; lalsim mMaxMsun is already all-branch
```

## Sample evaluation failure policy

Waveform and EOS evaluation failures reject the affected sample and continue
the analysis. This is a design principle for both waveform and EOS sampling.

Exceptions from the waveform or EOS implementation, and nonfinite results,
must follow the existing invalid-sample convention: zero posterior weight or
`-inf` log likelihood. Sampling boundaries translate these failures into sample
rejection. They must not terminate the sampler, worker, batch, or workflow.
Batched evaluation must preserve valid neighboring samples when one evaluation
fails.

A rejected proposed EOS emits log evidence `-inf` and relative error `0`.
A proposed EOS whose actual conversions reject every draw records zero evidence
after the sampler exhausts its existing retry budget. The first invalid batch
does not end those retries; later valid draws remain usable. Unrelated sampler
errors retain their existing behavior.

Repeated estimates combined by an average or sum retain successful replicas;
all failed replicas retain zero evidence. A product over events retains every
event, so a zero-evidence event makes the product zero.

Compatibility changes must preserve this behavior. Regression tests must mix
valid and failing samples, exercise both raised exceptions and nonfinite
outputs, and verify that failed samples are rejected while valid samples remain
usable and sampling continues.

## LALSimulation families

Legacy LALSimulation exposes one stable family. Released multibranch builds
and the reviewed multipart TOV API can return several stable branches.  RIFT exposes
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

By default, `lalsim_file:` preserves released `SimNeutronStarEOSFromFile`
behavior even on a reviewed build. Requesting a branch, the compatibility
dirty-phase-transition option, or the extended family opts into
`SimNeutronStarEOSFromFilePhaseTransition` and `CreateSimNeutronStarFamilyPT`.
Ordinary named, spectral, piecewise-polytrope and file EOS objects use the
one-argument legacy family constructor or the released two-argument constructor,
selected by capability checks. The reviewed multipart object path stays explicit.
The reviewed PT reader always enables its dirty-phase-transition handling, so
`--using-eos-dirty-phase-transitions` remains accepted for command-line
compatibility but does not toggle a second reader mode.

The optional Python API `family_log_pressure_min` selects
`CreateSimNeutronStarFamilyPTWithPcmin`; its value is the finite natural
logarithm of central pressure in pascals, `ln(Pc / Pa)`.

The mass-radius-Lambda helpers use the same canonical keyword as O4d:
`make_mr_lambda_lal(..., multipart=True)` and
`make_mr_lambda_lal_branches(..., multipart=True)`. The older O4c keyword
`reviewed_multibranch` remains a deprecated alias. Supplying both keywords with
different values raises an error rather than selecting an object family
ambiguously.

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

By default this command projects each EOS onto its **primary stable
mass-rising branch**: `lambda_from_m(m)` without a branch id and `mMaxMsun`
both refer to the primary branch.  RIFT splits central-enthalpy-ordered data at
unstable or decreasing-mass intervals before M-to-Lambda interpolation, so it
never mixes disconnected branches.  `stable_branch_counts[index]` records when a
native sequence contains more than one stable run.

`nmbseq:` objects now carry the branch surface above, with `mMaxMsun_all` as the
maximum over every branch.  Two choices are available beyond the primary
branch:

- `--using-eos-branch <k>` analyzes one chosen branch, as for LALSimulation.
- `--eos-affine-parameter central` includes every stable branch (next section).

Unlike LALSimulation, an NMB sequence returns the primary branch for an
ambiguous twin mass when no branch id is given; LALSimulation raises.

## Including every stable branch: `--eos-affine-parameter central`

```text
--using-eos <spec> --eos-affine-parameter central
```

The default, `--eos-affine-parameter mass`, is the historical one-branch
`M -> Lambda` map and runs the same code as before.  `central` is equivalent to
a central-coordinate affine parameter with a dM-per-branch prior.  It is
implemented as mass plus a branch label, so every stable branch contributes.
Each component may be a star on any stable branch.

Late `--fixed-parameter` overrides of mass and tidal coordinates are
incompatible with central mode. Such an override would change the exported
binary after the sampler chose its EOS branches.

The prior is dM per stable branch.  A mass with n(m) stable branches counts
n(m) times, so the joint mass prior gains a factor n(m1) n(m2).  For a
single-branch EOS, n = 1 and the prior is the usual mass prior.  Twin regions
are overcounted by design.

CIP keeps its sampling coordinates and the trained likelihood fit.  It adds two
sampled variates, `eos_branch_u1` and `eos_branch_u2`, uniform on [0, 1).  They
are not fit coordinates.  For component i:

| Step | Definition |
|---|---|
| branches | B_i = `eos.branches_for_m(m_i)`, n_i = len(B_i) |
| branch | b_i = B_i[floor(u_i n_i)] |
| tide | Lambda_i = `eos.lambda_from_m(m_i, branch_id=b_i)` |
| weight | integrand times n_1 n_2 |

Integrating over u_i then sums the integrand over branches with the
dM-per-branch prior.  The factor n_1 n_2 enters the integrand beside the
existing prior-scale correction.  It therefore multiplies the evidence and the
posterior weights for every sampler.  `adaptive_cartesian`, `GMM` (with or
without `--internal-use-lnL`) and `AV` have been checked against exact branch
sums.  With `GMM`, each variate gets its own one-dimensional mixture group.

A mass on no stable branch, or at or above the all-branch maximum
`getattr(eos, 'mMaxMsun_all', eos.mMaxMsun)`, rejects the sample.  The mass path
applies the same rule at `mMaxMsun`.  `--no-matter1` and `--no-matter2` give
that component Lambda = 0 and weight 1.  `--assume-eos-but-primary-bh` acts as
`--no-matter1` in central mode, in the integrand and in the export.

For a single-branch EOS, central and mass mode agree inside the branch.  They
can differ at its edges:

| Mass | mass mode | central mode |
|---|---|---|
| below the branch minimum | EOS-specific (nmbseq clamps Lambda) | rejected |
| in [0.999 mMax, mMax) | Lambda = 1e-8 | branch value of Lambda |

Central mode uses `branches_for_m` membership and the branch's own Lambda.
It does not depend on the 1e-8 cutoff.

Exported samples keep the `sim_inspiral` XML format, with Lambda from the
sampled branch.  The XML has no branch column, so CIP writes
`<fname-output-samples>_eos_branch.dat` with columns `b1 b2 n1 n2`, row for row
with the XML.  The ids follow component swaps; `-1` marks a component without
matter.  The sidecar belongs to one CIP process.  Pipeline joins
(`write_joingrids_sub`, `igwn_ligolw_add`) merge only the XML and do not carry
it.  The low-`n_eff` quadpuff exit path writes no sidecar.

Central mode fails at startup with a clear error for these combinations:

- no `--using-eos`, or an EOS without `branches_for_m` or without a `branch_id`
  keyword on `lambda_from_m`;
- an EOS whose `branches_for_m` fails at the startup probe masses, such as an
  `nmbseq:` Landry-format file (a proposed `file:` grid EOS is rejected
  instead, with evidence `-inf`);
- `--using-eos-branch`, which selects one branch;
- `--tabular-eos-file`: the fit over the ordering coordinate cannot represent
  branch labels;
- eccentric phase coordinates or `--eccentric-phase-sampling`;
- `--sampler-oracle` or `--oracle-reference-sample-file`: reference samples have
  no branch variates.

A supplementary likelihood factor receives the native sampling coordinates
only, as in mass mode.  The sample evaluation failure policy above applies
unchanged.  Exceptions or nonfinite values from `branches_for_m` or
`lambda_from_m` reject that sample and keep its neighbors.

`util_RIFT_pseudo_pipe.py --assume-matter-eos <spec>` already adds
`--input-tides --using-eos <spec>` to CIP, and `--assume-matter-but-primary-bh`
adds `--assume-eos-but-primary-bh`.  Add the new option with
`--manual-extra-cip-args "--eos-affine-parameter central"`.

## Downstream compatibility

- Existing Kedia-style spectral and piecewise-polytrope EOS inference keeps the
  same constructor and scalar lookup APIs.  Ordinary single-branch models need
  no new option.
- Existing nmb-papers and hyperpipe fixed-sequence runs keep the same HDF5 tags
  and `nmbseq:` syntax.
- Twin-star analyses select one branch with `--using-eos-branch`, or include
  every branch with `--eos-affine-parameter central`.

`--using-eos-branch` is rejected with `--using-eos-for-prior`. The current
EOS-hyperprior plugin protocol returns an EOS but does not return branch
identity, so accepting this combination would silently analyze the wrong
branch. A future multibranch hyperprior must extend that plugin contract before
this restriction can be relaxed.

## Reviewed-LALSimulation integration gate

The fake-backed compatibility tests check RIFT's dispatch logic, but do not
certify the reviewed SWIG interface. Fixture-free acceptance uses trusted
`SimNeutronStarEOSMultiPartsByName("SLY")`, checks coexistence with the legacy
named-EOS API, extended-family fields when exported, and repeated family
destruction. To run it, build the exact reviewed LALSuite commit, activate that
Python environment, and set
`RIFT_REVIEWED_LALSIM_MANIFEST` to a JSON file with this shape:

```json
{
  "lalsuite_ref": "974c0ef468b76e8298e67fd8baf71ed259cc5fee",
  "fixtures": {
    "two_column": {"path": "two-column.dat", "sha256": "..."},
    "nine_column": {"path": "nine-column.dat", "sha256": "..."},
    "twin_star": {
      "path": "twin_star.dat",
      "sha256": "1bc0fa5f92788cfb10cea3a7f04d5e6587a342043c94f1d6e7fbef2f7294ac16",
      "columns": 2,
      "raw_source_sha256": "3905405889b7da968e07a7ec97d37ca92c710a15af9407e014d6b30869b9608e",
      "transform": "pressure=raw col4*1.602176634e32*G/c^4; energy=raw col3*1.602176634e32*G/c^4"
    }
  },
  "known_upstream_crash": {
    "path": "upstream_2pt_raw.dat",
    "sha256": "e8439b356e0815cfd710c3e4f0ccb7330a9ee7e42050ae6c5963c9d5890f8db7",
    "expected_returncodes": [-11, 139]
  }
}
```

Run
`pytest MonteCarloMarginalizeCode/Code/test/test_lalsim_eos_reviewed_integration.py`.
The `fixtures` object is optional, and `twin_star` is optional within it.
Without fixtures, builtin acceptance runs and external-table coverage is an
explicit skip. File-loader fixtures must be exact two- or nine-column numeric
tables; four-column wiki arrays require a separate provenance-recorded
transformation. Each table is loaded in a subprocess with a 60-second timeout,
stdout and stderr sent to `DEVNULL`, and only a status file capped at 64 KiB
returned to pytest. Paths are relative to the manifest. The ref must be the
full 40-character
commit actually built by the job: the gate requires it to equal
`lalsimulation.SimulationVCSInfo.vcsId` and requires a clean VCS build. Every
supplied fixture hash is mandatory. Once the manifest enables the gate, missing modern
symbols, malformed or mismatched build provenance, missing fixtures, wrong
column counts, or absence of distinct overlapping twin-star solutions are
failures. Ordinary CI skips this private-build gate explicitly.

The recorded one-transition twin table is a two-column file derived using
`pressure = raw column 4 * 1.602176634e32 * G / c^4` and
`energy = raw column 3 * 1.602176634e32 * G / c^4`; both its derived and raw
SHA-256 values are retained above. The optional `known_upstream_crash` record
uses `SimNeutronStarEOSFromArraysPhaseTransition` with the same geometrized
factor. At reviewed commit `974c0ef`, the authoritative two-transition RC139
array reaches `CreateSimNeutronStarFamilyPT` and exits with SIGSEGV (`-11`, or
shell `139`). The gate records that as an expected upstream `xfail`; if it
starts passing, the test fails with instructions to promote it to a required
passing fixture rather than silently accepting changed behavior.

The drift-sentinel registry should eventually declare an EOS contract group
with LALSuite and NuclearMatter-Backend as producers and RIFT/nmb-papers as
consumers.  The current registry only covers RIFT/hyperpipe operational archive
and queue boundaries, so it cannot detect EOS schema or callable drift yet.
