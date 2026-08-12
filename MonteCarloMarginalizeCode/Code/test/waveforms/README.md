# Waveform-level symmetry tests

Checks on the `U` and `V` mode-cross-term matrices built by
`RIFT.likelihood.factored_likelihood` (`ComputeModeCrossTermIP`):

```
U[(A,B)] = < h_A   | h_B >    (crossTerms)
V[(A,B)] = < h_A^* | h_B >    (crossTermsV)
```

with `A=(l,m)`, `B=(l',m')`.

## `test_uv_symmetry.py`

Verifies, over semi-random parameters looped across an active waveform list:

1. **`U` Hermitian** — `U[(A,B)] = conj(U[(B,A)])` (definitional)
2. **`U` diagonal real & positive** — `U[(A,A)] > 0`, real (definitional)
3. **`V` complex-symmetric** — `V[(A,B)] = V[(B,A)]` (definitional)
4. **Reflection / parity (aligned-spin only)** —
   `V[((l,m),B)] = (-1)^l U[((l,-m),B)]`, from
   `h_{l,-m} = (-1)^l conj(h_{l,m})`.

Every matrix element is recomputed independently (`same_waveform_Q=False`), so
the code's internal symmetrization shortcut is bypassed and the checks are real.

Checks 1–3 are exercised on both aligned-spin (`ACTIVE_WAVEFORMS`) and
precessing (`PRECESSING_WAVEFORMS`) models; check 4 only on aligned-spin models,
since precessing approximants do not obey the simple reflection relation even at
zero in-plane spin. Models the local `lalsuite` build cannot generate are
skipped with a reason, not failed.

```bash
pytest -v test_uv_symmetry.py
python  test_uv_symmetry.py --approximant IMRPhenomXHM --Lmax 3 --seed 42
python  test_uv_symmetry.py --list
```

### Placeholder (expected-fail) check

`test_full_nonlinear_reflection_symmetry_left_as_exercise` is a placeholder for
the full non-linear reflection algebra that is not yet implemented. It is marked
`@pytest.mark.xfail(strict=True)` so it stays visible (reported `XFAIL`) without
reddening the suite; if the algebra is ever implemented and it starts passing,
strict xfail turns the `XPASS` into a failure so the placeholder gets removed.
Deselect with `-k 'not left_as_exercise'`, or run the script with
`--skip-ludicrous`.

## Parity (orbital-plane reflection) diagnostics — NOT collected by pytest

Two script-only diagnostics implement the physics the placeholder above points
at: the exact parity identity of GR,

```
h_lm[(s_x,s_y,s_z) -> (-s_x,-s_y,s_z)](t) = (-1)^l conj( h_{l,-m}(t) ),
```

with no time- or phase-shift freedom.  They are deliberately named so pytest
does NOT collect them: run against currently released precessing models they
FAIL, correctly (NRSur7dq4 at the percent level generically and tens of
percent in superkick subdominant-mode amplitudes; SEOBNRv5PHM with
antisymmetric modes at few x 1e-4; SEOBNRv4PHM at 0.3-1% with a frame origin).
Their role is rapid assessment of upstream "developer leakage" when adopting a
model version or interface — not CI gating.

- `parity_check_hlm.py [models...]` — mode-level check of the identity above
  over superkick-like / generic-precessing / nonprecessing configurations.
  Use *perturbed* superkicks: the exact degenerate point is a convention
  branch point for several models.
- `uv_parity_diagnostics.py [models...]` — the same physics on the U/V
  cross-term matrices ILE builds:
  D1 (any waveform; failure = code bug): U = U^dagger, V = V^T;
  D2 (single generation, nonprecessing points, including nonprecessing limits
  of precessing models): V_{(l,m),B} = (-1)^l U_{(l,-m),B};
  D3 (two generations, any configuration): reflected-pair relations
  U'_{(lm),(l'm')} = (-1)^{l+l'} U_{(l',-m'),(l,-m)},
  V'_{(lm),(l'm')} = (-1)^{l+l'} conj(V_{(l,-m),(l',-m')}).
  Clean models: <= 1e-10 relative Frobenius residual; violating models:
  >= 1e-4.  Suggested tolerance: 1e-8.  (This extends check (4) of
  `test_uv_symmetry.py` to precessing models, where the known violations
  live; note that check (4)'s 3e-2 tolerance would pass NRSur7dq4's 4e-4
  aligned-spin violation.)

NRSur7dq4 needs `LAL_DATA_PATH` pointing at a directory containing
`NRSur7dq4_v1.0.h5`.  A marginal-likelihood impact demonstration (at what SNR
a failed check biases PE) lives in the RIFT_roboto_paper repository under
`demos/waveform_symmetry/`.

### X-family (ChooseFDModes) models: two caveats

1. **Frame convention**: raw `ChooseFDModes` output for IMRPhenomXPHM /
   IMRPhenomXPNR satisfies the parity identity only after a global rotation
   by exactly pi about z (their mode frame rotates under reflection of the
   in-plane spins).  Degenerate with phi_ref — cancels in marginalized PE —
   but naive complex mode-level residuals report O(1) "violations".  The
   amplitude residual is the convention-robust column.  After removing this
   rotation: XPNR raw modes are parity-clean to 1e-5 (amplitudes 1e-14);
   XPHM likewise, apart from a small genuine 2e-3 (2,+-1) equatorial
   asymmetry in its aligned-spin limit.
2. **RIFT interface artifact**: raw IMRPhenomXHM satisfies the equatorial
   identity exactly (residual 0.0), but through `RIFT.lalsimutils.hlmoft`'s
   ChooseFDModes->TD conditioning acquires ~1% (2,+-2) spurious amplitude
   asymmetry.  This affects every ChooseFDModes-consumed model and dominates
   the through-RIFT parity residuals for the X family (D2 for XHM sits at
   ~1e-2 instead of <=1e-10 until the conditioning is fixed).
