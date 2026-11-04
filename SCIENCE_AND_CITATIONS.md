# Science and citations

[Back to the RIFT README](README.md)

> [!WARNING]
> **READ BEFORE USING RIFT FOR A SCIENTIFIC PUBLICATION OR PRODUCTION ANALYSIS.**
> Contact the developers before starting a production analysis to confirm the
> branch, configuration, validation requirements, citations, and collaboration
> expectations for the features you will use. Development features evolve rapidly;
> installing RIFT or completing a workflow does not establish scientific validity.
> Read the [collaboration and authorship policy](#collaboration-and-authorship-opt-in-model) below.

## Choose the analysis branch deliberately

The stable `0.0.17.xx` series is maintained on **`rift_O4c`**. It is a separate
maintenance line, will not be a descendant of `master`, and is not intended to
merge back into it. Use `rift_O4c` when you want the stable branch, rather than
assuming `main` or `master` is the stable release.

New scientific development and ongoing short-author research analyses use
**`rift_O4d`**, the `0.0.18.xx` development line. See [CHANGES.rst](CHANGES.rst)
for the branch policy and numerical changes. Record the exact commit,
dependencies, waveform models, priors, and configuration used in your analysis.

## Core RIFT methods

Cite the foundational rapid_pe/RIFT methods and the algorithm updates relevant
to your analysis. The years below identify the linked preprints; journal
publication years can differ.

| Paper | Contribution |
| --- | --- |
| Pankow et al. (2015), [A novel scheme for rapid parallel parameter estimation of gravitational waves from compact binary coalescences][pankow] | Original parallel likelihood and extrinsic marginalization architecture. |
| Lange, O'Shaughnessy & Rizzo (2018), [Rapid and accurate parameter inference for coalescing, precessing compact binaries][lange] | Iterative RIFT inference. |
| Wysocki et al. (2019), [Accelerating parameter inference with graphics processing units][wysocki] | GPU acceleration. |
| Wofford et al. (2022), [Expanding RIFT: Improving performance for GW parameter inference][wofford] | Updated inference, integration, coordinates, and performance. Published in *Physical Review D* **107**, 024040 (2023). |
| Wagner et al. (2025), [Narrowing RIFT: Focused simulation-based-inference for interpreting exceptional GW sources][wagner] | Focused inference, samplers, waveform interfaces, and updated operating choices. |

## Cite the features you use

### Automation, hyperparameter inference, and simulation management

| Feature or workflow | Relevant references and guidance |
| --- | --- |
| RIFT / Asimov integration and reproducible reanalysis | [Fernando, O'Shaughnessy & Williams (2024)][fernando] and [Manning & O'Shaughnessy (2026)][manning]. Also cite [Williams et al., Asimov (2023)][asimov] for the orchestration framework itself. |
| Hyperpipeline: EOS and population inference | [Kedia et al. (2024)][kedia] and [Ebiri & O'Shaughnessy (2026)][ebiri]. Ebiri extends Hyperpipe to flexible user-provided priors and joint population/EOS inference. |
| `simulation_manager`, simulation archives, and adaptive simulation campaigns | Cite the scientific method and application supported by your campaign: e.g. [Ristic et al. (2021)][ristic] for adaptive learning with detailed kilonova simulations, and [Kedia et al.][kedia] / [Ebiri & O'Shaughnessy][ebiri] for EOS and population campaigns. Choose references according to the methods and simulations actually used. See the [archive design](https://oshaughnessy-junior.github.io/research-projects-RIT/api_reference/simulation_manager/design_overview.html) and [backend documentation](https://oshaughnessy-junior.github.io/research-projects-RIT/api_reference/simulation_manager/backends/index.html). |

### Likelihood, integration, and coordinates

| Feature | Relevant references and contributors |
| --- | --- |
| Surrogate-basis likelihood | [O'Shaughnessy, Blackman & Field (2017)][surrogate]. Also cite the waveform model used. |
| GPU Monte Carlo integration | [Wysocki et al.][wysocki] and [Wofford et al.][wofford]. |
| GMM / MonteCarloEnsemble integration | [Ristic et al.][ristic]; integration implementation by E. Champion ([original repository](https://git.ligo.org/benjamin.champion/Monte-Carlo-Integrator)). |
| Adaptive-volume (AV) integration | [Wagner et al.][wagner] and the underlying [VARAHA method, Tiwari et al. (2023)][varaha]. |
| Portfolio sampling, oracles, and normalizing flows | [Wagner et al.][wagner]. |
| Distance marginalization | [Wofford et al.][wofford]; contributions by S. Morisaki and D. Wysocki. |
| Rotated inspiral-phase coordinates | [Lee, Morisaki & Tagoshi (2022)][lee] and [Wofford et al.][wofford]. |
| Pseudo-cylindrical coordinates | [Wofford et al.][wofford]. |
| Calibration marginalization | [Payne et al. (2020)][payne] and the RIFT updates in [Wagner et al.][wagner]; RIFT integration by J. Lange and D. Williams. |

### Waveform interfaces and detector extensions

Cite both the RIFT interface/method and the original waveform or surrogate
papers appropriate to the model actually used.

| Interface or extension | Relevant RIFT references |
| --- | --- |
| `gwsurrogate` and NRSur interfaces | [O'Shaughnessy, Blackman & Field][surrogate] and [Lange et al.][lange]. Contact the developers about additional interface contributions, including work by F. Shaik and collaborators. |
| Original TEOBResumS interface | [Lange et al.][lange]. |
| Direct numerical-relativity comparisons | [Lange et al. (2017), Physical Review D **96**, 104041][nr]. |
| `gwsignal` interface | [Wagner et al.][wagner]. |
| Eccentric TEOBResumS interface | [Iglesias et al. (2022)][iglesias]. |
| Hyperbolic TEOBResumS interface | [Henshaw et al. (2025)][henshaw], replacing the earlier in-preparation reference. |
| LISA-RIFT | [Jan et al. (2024)][jan] and [O'Shaughnessy et al. (2026), Scaling RIFT 1][scaling3g]. |
| Third-generation detectors, slow rotation, and finite-size response | [O'Shaughnessy et al. (2026), Scaling RIFT 1: Extending RIFT to third-generation gravitational-wave analyses][scaling3g]. |

## Astrophysical applications

Selected examples show how these methods have been used. Cite the applications
relevant to your work in addition to the software and methods references.

| Science case | Examples |
| --- | --- |
| Catalog reanalysis and waveform systematics | [Fernando et al., Efficient reanalysis of events from GWTC-3 with RIFT and asimov][fernando]; [Manning & O'Shaughnessy, Analysis of GWTC-3 with multiple quasicircular models][manning]. |
| Neutron-star EOS and population inference | [Kedia et al., Astrophysical assumptions and equation of state framework have larger impact on equation of state inference than individual neutron star observations][kedia]; [Ebiri & O'Shaughnessy, Improved Population and EOS Joint Inference for Binary Neutron Star Systems][ebiri]. |
| Eccentric binaries | [Iglesias et al., Eccentricity estimation for five binary black hole mergers with higher-order gravitational wave modes][iglesias]. |
| Hyperbolic encounters | [Henshaw et al., Parameter estimation of gravitational waves from hyperbolic black hole encounters][henshaw]. |
| Massive black hole binaries with LISA | [Jan et al., Adapting a novel framework for rapid inference of massive black hole binaries for LISA][jan]; see also the detector-response extensions in [Scaling RIFT 1][scaling3g]. |
| Third-generation gravitational-wave analyses | [O'Shaughnessy et al., Scaling RIFT 1: Extending RIFT to third-generation gravitational-wave analyses][scaling3g]. |
| Kilonova simulation and inference | [Ristic et al., Interpolating Detailed Simulations of Kilonovae: Adaptive Learning and Parameter Inference Applications][ristic]. |

## Collaboration and authorship: opt-in model

Several parts of RIFT are under active development. We encourage close
collaboration with the lead developers, O'Shaughnessy and Lange, particularly
when using recent features or evolving interfaces.

### How the model applies

The age of a code commit is measured on the date an analysis **commences**.

| Code used in the analysis | Collaboration expectation |
| --- | --- |
| Commits older than three years | Free to use for any scientific work under this collaboration policy. |
| More recent commits | Offer opt-in authorship to the lead developers and to developers who contributed significantly to the features used in the adopted version. |

The newer the features you use, the earlier you should contact the relevant
developers so they can engage with the scientific work and final publication.
Citations and opt-in authorship are separate considerations; discuss both
before starting the analysis.

### Scope and review

This policy applies to commits in this repository. External resources and
software, including numerical-relativity surrogates, have their own policies.

Release-based distribution remains the intended model for the final developed
code. The opt-in model is scheduled for review four times per year:
**January 15, April 15, July 15, and October 15**.

### Who to contact

| Area | Developers |
| --- | --- |
| Iterative pipeline, fitting, posterior generation, and external EOB/surrogate interfaces | O'Shaughnessy and Lange |
| Surrogate-basis method | Field, O'Shaughnessy, and Blackman |
| GPU optimizations | Wysocki, O'Shaughnessy, Fang, and Lange |

For other recent features, ask the lead developers which contributors should
be included in the discussion.

## Software license

The software is distributed under the [MIT license](LICENSE.md). The
collaboration and scientific attribution expectations above are retained from
the previous README; the license text is unchanged.

[pankow]: https://arxiv.org/abs/1502.04370
[lange]: https://arxiv.org/abs/1805.10457
[wysocki]: https://arxiv.org/abs/1902.04934
[wofford]: https://arxiv.org/abs/2210.07912
[wagner]: https://arxiv.org/abs/2505.11655
[surrogate]: https://arxiv.org/abs/1701.01137
[fernando]: https://arxiv.org/abs/2412.02999
[manning]: https://arxiv.org/abs/2609.15827
[asimov]: https://doi.org/10.21105/joss.04170
[kedia]: https://arxiv.org/abs/2405.17326
[ebiri]: https://arxiv.org/abs/2609.39057
[ristic]: https://arxiv.org/abs/2105.07013
[varaha]: https://arxiv.org/abs/2303.01463
[lee]: https://doi.org/10.1103/PhysRevD.105.124057
[payne]: https://doi.org/10.1103/PhysRevD.102.122004
[nr]: https://doi.org/10.1103/PhysRevD.96.104041
[iglesias]: https://arxiv.org/abs/2208.01766
[henshaw]: https://arxiv.org/abs/2507.01156
[jan]: https://arxiv.org/abs/2410.15542

[scaling3g]: https://arxiv.org/abs/2609.37393
