# demo/rift/slowrot — slow-response likelihood value demos

Verify-anywhere (no condor, no GPU) quick-looks that RIFT's **time-/frequency-dependent
detector-response** likelihoods *add value*, not merely run. Both generalize the maintained
marginalized ILE likelihood so a long or long-armed signal is modeled with the correct
detector response instead of a single constant antenna pattern.

```
make demo              # both demos below -> outputs/*.{txt,png}
make demo-rotation     # Path A/B  (Earth rotation)
make demo-finite-size  # Path D    (finite-size / frequency-dependent response)
```

Each demo builds data that carries the true response, then compares, **at the true
parameters on the same data and time grid**, the recovered time-maximized log-likelihood of
the standard analysis vs the response-aware one. The difference `gain = lnL_aware − lnL_standard`
is the SNR the standard analysis throws away. Both use the maintained vectorized (NoLoop)
path with cubic sub-bin time interpolation, so the peak-resolution floor cancels in the gain.

---

## Path A/B — Earth rotation (`demo_rotation.py`)

Over a long signal the antenna pattern `F(t)` drifts as the Earth rotates; the static
(Earth-fixed-response) likelihood loses match. RIFT injections already carry the true
time-varying response (`hoft → SimDetectorStrainREAL8TimeSeries`), so a standard injection is
"rotating". The gain grows with the rotation phase `Ω⊕·T` over the signal; a short signal is a
null control. Single H1, SNR≈30 held fixed so the gain tracks `Ω⊕·T`, not loudness:

| config | seglen | Ω⊕·T | gain `lnL_rot − lnL_static` |
|---|---|---|---|
| null_bbh (30+25) | 2 s  | 1.5e-4 | +0.004 (null control) |
| bbh_8_8          | 16 s | 1.2e-3 | +0.043 |
| bbh_4_4          | 64 s | 4.7e-3 | +0.187 |

Grows ~50× from the short null to a minute-scale signal; extrapolated to a 90-minute XG BNS
(`Ω⊕·T`≈0.4) it is orders of magnitude larger — the 3G headline. Figure:
`outputs/rotation_gain_vs_duration.png`. ILE: `--rotation-slow` (Path A),
`--rotation-slow --rotation-p-max 2` (Path B, adds propagation-delay drift).

## Path D — finite-size / frequency-dependent response (`demo_finite_size.py`)

On a multi-km arm the light-travel time across the arm is not negligible vs the GW period, so
the response is per-frequency: `h_k(f) = F₊(f;sky) h₊(f) + Fₓ(f;sky) hₓ(f)`. We inject an exact
finite-size signal (`antenna_response_fd`) and compare the standard long-wavelength (LWL,
constant-response) likelihood to the finite-size one (`--freqresponse`, sky-harmonic route (b),
sky stays extrinsic). The gain grows with the arm length — i.e. with `fL/c`, the in-band
light-crossing phase. 15+13 M☉, fmax=2000 Hz, loud (SNR≈320), Qmax=6:

| detector | arm L | fL/c @ fmax | gain `lnL_finite − lnL_LWL` |
|---|---|---|---|
| LIGO | 4 km  | 0.027 | +0.25 (null control) |
| ET   | 10 km | 0.067 | +3.5 |
| CE   | 20 km | 0.133 | +12.4 |
| CE   | 40 km | 0.267 | +39.6 |

Null at the 4-km LIGO arm; tens of nats at a 40-km Cosmic Explorer. Only the
**direction-dependent** part of the response contributes — the common `e^{−i2πfL/c}`
light-crossing delay is degenerate with arrival time and is absorbed by both likelihoods'
time maximization, so it does not inflate the gain. Figure:
`outputs/finite_size_gain_vs_arm.png`. ILE: `--freqresponse`
`--freqresponse-arm-length 40000` `--freqresponse-qmax 6`.

Higher `fL/c` (heavier system / higher fmax / longer arm) needs higher `--freqresponse-qmax`
(the response is a power series in `fL/c · (â·n̂)`); Qmax=6 converges through CE-40 km at this
config.

---

## Full injection–recovery PE (cluster)

These quick-looks are the likelihood-level core. The headline parameter-bias / single-network
sky-localization figures are DAG PE runs (structurally the standard RIFT pipeline with the
extra ILE flag appended: `--rotation-slow` / `--freqresponse`). The three analyses per event
differ *only* in that appended option. See `RIFT/likelihood/SLOWROT_HANDOFF.md` and
`~/RIFT_roboto_paper/analyses/slowrot_demo/` for the pipeline scaffolding.

## Validation (the demos show value; these prove correctness)

`RIFT/likelihood/test_slowrot_*.py` (run under the venv with this Code dir on `PYTHONPATH`):
- `test_slowrot_noloop.py` — rotation NoLoop == baseline at f_sidereal=0 (3.6e-12).
- `test_slowrot_headtohead.py` — matched-sample rotation vs baseline; Cauchy-Schwarz bound.
- `test_slowrot_freqresponse_likelihood.py` — finite-size: L→0 reduces to baseline (3e-9),
  bound respected, and a **V4 positive control** asserting finite-size beats LWL by +38.9 nats
  in an in-band-effect config.
