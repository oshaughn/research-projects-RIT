# Bounded multipeak JAX marginalization

`--angle-marg-scheme multipeak-jax` integrates time, distance and both
angles with fixed-cap device discovery and fixed-order local quadrature.
There is no reserve, including on a decline. Discrete planning is stopped
from the AD graph; derivatives describe the retained fixed-plan integral.
This scheme is explicit only and is not selected by `auto`.

## Configuration and accuracy

`--multipeak-jax-<field>` exposes every `BoundedMultipeakConfig` field, with
underscores replaced by hyphens. These are static values chosen before JIT
compilation. Important controls include `time-guard`, `base-max-starts`,
`max-time-nodes`, `max-modes`, `enriched-max-modes`, the four quadrature
orders, `convergence-tol-nats` and `total-value-error-budget-nats`.
The separate `--direct-marginalization-*` flags still configure the policy
with a reserve; they do not configure this scheme.

The default resource envelope is compact: **guard 16, starts 32, time nodes
64, base/enriched modes 8/8**. This is a deliberate cost choice, not an
assumption that the policy's wider operating point is unnecessary everywhere.
With corrected quadrature it accepts the analytic reference at one tenth of
the wider envelope's warm A100 cost. Quadrature orders are **11/13/13/15**, convergence
**0.01 nat**, total empirical error budget **0.03 nat**. These are practical
starting settings, not a universal scientific accuracy requirement. An
application can lower orders, relax tolerances or change capacities to meet
its accuracy and runtime needs. The complete configuration is recorded in both output headers.

To try the wider policy capacity profile on important declines, add:

```text
--multipeak-jax-time-guard 128 --multipeak-jax-base-max-starts 128
--multipeak-jax-max-time-nodes 256 --multipeak-jax-max-modes 16
--multipeak-jax-enriched-max-modes 16
```

The policy's recorded full-sky acceptance improvement at these wider caps
motivates exposing this profile, but acceptance fraction is not missed mass.
The compact default does not establish that its declined production rows are
negligible. Use their diagnostics and contribution comparisons to decide.

## Declines and contribution

The low-level bounded kernel returns `nan` plus its acceptance ledger for
unusable rows. The wrapper offers two actions:

* `--multipeak-jax-decline-action drop` (default) gives declined proposals
  finite log-zero (`-1e30`) in sampler calls. Their numerical importance
  weight is zero, but they remain in the proposal count. Simply removing
  nonfinite log weights would renormalize over survivors and change the
  evidence. Declined rows are excluded from the exported cloud.
* `--multipeak-jax-decline-action refuse` stops on any declined host batch.
  The refusal is latched, so an optional sampler stage catching an exception
  cannot subsequently publish survivors.

Dropping is useful when those rows contribute negligibly. A decline by itself
is not evidence of negligible contribution. Outputs in drop mode explicitly
identify an accepted-region target with unbounded omitted mass. The host audit
reports evaluated/declined counts (evaluations, not unique samples), decline
reasons, the maximum accepted log likelihood, the maximum finite *diagnostic*
value among declines, and the count with no finite diagnostic. Diagnostic
values failed the warrant: they are neither accepted likelihoods nor upper
bounds. A large declined diagnostic, missing modes, or unknown diagnostic
calls for a configuration change and an acceptance/value comparison, not a
reserve. A small diagnostic alone does not prove absent missed modes.

For qualification, widen the relevant capacities or relax the appropriate
accuracy check, rerun the same proposals, and compare the contribution and
posterior/evidence stability at the accuracy needed by the application.
No output cloud consisting entirely of declined rows is published.
The audit and refusal latch cover only batched `log_likelihood` calls
(pilot/reweight/output evaluations). Scalar evaluations, including MAP,
Fisher and internal MALA training, are excluded. Under `refuse`, a scalar
decline returns NaN without raising or latching; a scalar-only decline can
therefore go unreported and does not prevent later publication of accepted
batches. Zero audited declines is not certification of scalar acceptance.
Both output headers record this scope and limitation explicitly.

## Regression coverage

The bounded suite exercises real discovery, refinement, quadrature and AD
on the analytic guarded-table fixture with an independent fine-time
reference. It also exercises low-order decline, invalid inputs/envelopes,
CLI-to-wrapper configuration, mixed-cloud publication, proposal-count
preservation, and refusal after a swallowed decline. Synthetic inputs are
constructed in tests; no captured scientific products are needed.

## Operating-point comparison (2026-09-12)

Two analytic `_guarded_problem` rows at scales 1 and 1.01, 33 native time
samples, guard 128, starts/time capacities 128/256 and 16 retained modes:
orders 11/13/13/15 accepted both rows even at the policy's tighter 0.001-nat
convergence setting and 0.01-nat total budget. Relaxing these tolerances to
0.01/0.03 nat gave no acceptance or accuracy benefit on this fixture; the
relaxation is a configurable starting accuracy choice, not a measured speedup.
The A100 (JAX 0.9.2, float64) took 3.70 s for the warm
two-row call and reported 2.57 MB of XLA temporary workspace (not total GPU
memory); CPU JAX 0.7.1 took 9.07 s. Returned log likelihoods were
36.3868065844 and 37.4751416172. The independent fine-time reference and AD
comparison are exercised by the regression test.

Orders 7/9/9/11 still declined both rows with convergence loosened to 0.03 nat
and total budget 0.1 nat, and took 9.09 s on CPU / 3.66 s on A100. Lowering
orders alone did not reduce this fixture's dominant discovery/refinement cost. These are
synthetic checks, not a production acceptance-rate or throughput claim.

The compact caps (guard 16, starts/time nodes 32/64, modes 8/8), with
11/13/13/15 orders, also accepted both rows at the tighter 0.001-nat setting:
log likelihoods 36.3868061719 and 37.4751411797. Warm A100 cost was **0.355 s
per two-row call** and temporary workspace 2.20 MB. That measured cost,
together with explicit drop diagnostics and configurable caps, is why this
profile is the default instead of automatically inheriting the wider policy.
