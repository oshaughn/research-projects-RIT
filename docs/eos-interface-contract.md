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

For pseudo-pipe workflows, forward the flag with
`--manual-extra-cip-args`.  Hydra hyperpipe configurations can put it in the
post driver's `extra-args` when that driver is the fixed-EOS CIP executable.

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

The drift-sentinel registry should eventually declare an EOS contract group
with LALSuite and NuclearMatter-Backend as producers and RIFT/nmb-papers as
consumers.  The current registry only covers RIFT/hyperpipe operational archive
and queue boundaries, so it cannot detect EOS schema or callable drift yet.
