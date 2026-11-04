# RIFT

**Rapid Iterative FiTting for gravitational-wave parameter inference**

RIFT estimates the properties of gravitational-wave sources by combining
parallel likelihood evaluations, marginalization over extrinsic parameters,
and iterative fitting of the intrinsic likelihood. It supports compact-binary
analyses, multiple waveform models, GPU acceleration, and extensions for
population and equation-of-state inference.

[Documentation](https://oshaughnessy-junior.github.io/research-projects-RIT/) ·
[Installation](https://oshaughnessy-junior.github.io/research-projects-RIT/installation.html) ·
[Tutorials](https://github.com/kjtwagner/rift-tutorials) ·
[Science & citations](SCIENCE_AND_CITATIONS.md) ·
[Release notes](CHANGES.rst)

> [!WARNING]
> **Planning a publication or production analysis?** Read
> [Science and citations](SCIENCE_AND_CITATIONS.md) before starting, including
> the production-use guidance and collaboration/authorship policy.

## Choose a branch

| Your goal | Branch | Release series |
| --- | --- | --- |
| Stable analyses and maintenance updates | **`rift_O4c`** | `0.0.17.xx` |
| New features and ongoing short-author research | **`rift_O4d`** | `0.0.18.xx` development |

**Use `rift_O4c` for the stable line.** `main` and `master` are not the stable
branch recommendation. O4c is maintained separately, will not be a descendant
of `master`, and has no planned merge back into it. See the
[branch maintenance policy](CHANGES.rst) for details.

## Get started

Install LALSuite and the dependencies described in [INSTALL.md](https://oshaughnessy-junior.github.io/research-projects-RIT/installation.html),
then install from the branch appropriate to your analysis. For the stable line:

```bash
git clone --branch rift_O4c https://github.com/oshaughn/research-projects-RIT.git
cd research-projects-RIT
python -m pip install -e .
export GW_SURROGATE=''
```

For development, select `rift_O4d` when cloning. GPU execution requires a
compatible CuPy/CUDA environment; cluster jobs also require the appropriate
accounting configuration. See [optional dependencies](https://oshaughnessy-junior.github.io/research-projects-RIT/using-pipeline.html#strongly-recommended-dependencies).

Start with the [tutorials](https://github.com/kjtwagner/rift-tutorials) and
[pipeline documentation](https://oshaughnessy-junior.github.io/research-projects-RIT/).
The [worked examples](https://oshaughnessy-junior.github.io/research-projects-RIT/demos.html) cover pipeline,
hyperparameter, population, and simulation workflows.

## Workflows and examples

| Task | Start here |
| --- | --- |
| Analyze an event | `util_RIFT_pseudo_pipe.py` — [pipeline examples](https://oshaughnessy-junior.github.io/research-projects-RIT/using-pipeline.html) |
| Coordinate analyses with Asimov | [RIFT / Asimov integration](https://oshaughnessy-junior.github.io/research-projects-RIT/api_reference/asimov/index.html) |
| Infer EOS or population hyperparameters | [Hyperpipe examples](https://oshaughnessy-junior.github.io/research-projects-RIT/hyperpipe.html) |
| Manage simulation campaigns and archives | [Simulation-manager design](https://oshaughnessy-junior.github.io/research-projects-RIT/api_reference/simulation_manager/design_overview.html) and [backends](https://oshaughnessy-junior.github.io/research-projects-RIT/api_reference/simulation_manager/backends/index.html) |
| Explore applications and choose citations | [Science and citations](SCIENCE_AND_CITATIONS.md) |

Features and examples vary between release branches; consult the documentation
in your selected checkout.

## Development

The `rift_O4d` root [pixi.toml](pixi.toml) provides local development environments:

```bash
pixi install
pixi run install-rift
pixi run import-check
```

The default environment pins SWIG below 4.4.0. Compatibility environments are
available as `swig-pre44` and `swig-post44`; run `swig-version` and `import-check`
in each when testing binding compatibility.

Read the [architecture guide](https://oshaughnessy-junior.github.io/research-projects-RIT/architecture.html) and
[API reference](https://oshaughnessy-junior.github.io/research-projects-RIT/api_reference/index.html) for the source layout
and interfaces. Repository contributors should also follow
[AGENTS.md](AGENTS.md) and [.travis/PRECOMMIT.md](.travis/PRECOMMIT.md)
for module design and validation requirements.
For help, see [troubleshooting](https://oshaughnessy-junior.github.io/research-projects-RIT/troubleshooting.html) or
[open an issue](https://github.com/oshaughn/research-projects-RIT/issues).

## Project and license

RIFT and rapid_pe share their origin in the parallel inference method of
[Pankow et al. (2015)](https://arxiv.org/abs/1502.04370).
The RIFT project is also hosted at [rapidpe-rift/rift](https://git.ligo.org/rapidpe-rift/rift).

Distributed under the [MIT license](LICENSE.md). Scientific references and the
collaboration policy are collected in [SCIENCE_AND_CITATIONS.md](SCIENCE_AND_CITATIONS.md).
