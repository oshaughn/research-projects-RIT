# Prototype low-mass RF transverse-spin fitting

This is an opt-in change to RF fitting features, motivated by controlled existing-grid
comparisons. It adds the tested cone squared, phase deficit and torque squared
features to full two-spin Cartesian fitting stages, retaining every original fitting
coordinate (including mu1/mu2 and all four transverse spin components). The top-level option selects the tested native mu1/mu2 phase-fit schedule
when applicable. In a standalone helper invocation with no explicit fit-method
choice it also selects RF; an explicit non-RF choice is preserved (auto skips,
explicit physics3 cannot activate and fails). This can change CIP stage
parameterization and iteration counts to the existing phase-coordinate schedule;
selection happens after initial-grid construction, so no new initial-grid rule
is introduced. The normal Asimov template already selects RF and the phase basis.
It does not
replace four components with three scalars, change the L frame, modify priors or
sampling coordinates, or alter ILE, integrators, worker counts or stopping rules.
The compact four-dimensional chart is not adopted as a universal replacement.

Pipeline CLI:

```
util_RIFT_pseudo_pipe.py ... --rf-transverse-spin-coordinates physics3
helper_LDG_Events.py ... --rf-transverse-spin-coordinates physics3
```

`off` explicitly disables the option; omitted is unchanged pipeline behavior.
`auto` enables only for a finite positive detector-frame **event chirp-mass
estimate below 20 solar masses**, precessing BBH, with a full two-spin RF stage.
It does not use a prior lower bound, source-frame mass, spin amplitude, placeholder
mass from an event-time-only invocation, or a failed GraceDB fallback. The exact
20-solar-mass boundary is excluded. Aligned, tides/EOS, eccentric and high-q
single-spin schedules are outside this prototype; explicit `physics3` rejects
inapplicable analyses. Reduced-spin, quadratic, covariance and GP stages remain
unchanged. No GP dependency or automatic interpolator substitution is introduced.

The bundled Asimov template selects `auto`. Override it with:

```yaml
sampler:
  cip:
    transverse spin coordinates: "off"  # or physics3 / auto
    fit method: rf
```

YAML boolean `false` also means `off`, and boolean `true` means explicit
`physics3`; quote enum strings to avoid YAML 1.1 coercion. An explicit false
value is never replaced by the automatic default.

The helper resolves the event estimate using the existing event ingestion path and
emits the opt-in flag and actual ILE spin reference frequency in applicable CIP
stage arguments. Generated stage files therefore record the choice. `fref` is
`engine.fref` for INI workflows, or the helper's existing ILE reference-frequency
choice without an INI. It is never inferred from an independent GP model.
Direct CIP also accepts `--rf-transverse-spin-coordinates physics3 --fref FREQ`
and requires a fresh RF fit with delta_mc, mu1, mu2, chiMinus and all four
transverse Cartesian fit coordinates;
cached GP loading and incompatible models fail explicitly. Training, mirrored
rows, predictions and diagnostic output use the same scalar implementation.

For component masses in solar masses, x_i=m_i/M, eta=x_1*x_2,
v=(pi*M*4.9254909476412675e-6*fref)^(1/3), define L=eta/v,
D=L+x_1^2*s1z+x_2^2*s2z, T=x_1^2*s1perp+x_2^2*s2perp,
J=sqrt(D^2+|T|^2), G=(2+1.5*q)*x_1^2*s1perp+(2+1.5/q)*x_2^2*s2perp.
The appended features are |T|^2/[D^2+(0.1L)^2], (J-D)/L and
|G|^2/(eta*v^2)^2. For D>=0, (J-D) is evaluated as |T|^2/(J+D).
These are fitting summaries, not exact SEOBNRv5PHM Fisher coordinates.

Controlled tests substantially improved both spin marginals in S250628am,
S250205bk and S250328ae. S250830bp remains a partial improvement, and
sky/prior/multiple-cause cases are held out. Finite-bin joint JS is descriptive,
not by itself an interpolation-failure criterion. Calibration-treated references
and provisional intrinsic CIP are not identical statistical targets. This is a
prototype, not a universal default or independently calibrated population result.
