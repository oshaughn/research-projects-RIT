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
