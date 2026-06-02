"""
RIFT.hyperpipe.config
=====================

Schema + defaults for the Hydra/OmegaConf configuration consumed by
``util_RIFT_hyperpipe.py``. The top-level keys correspond to the
sections sketched in the original (pre-rewrite) ``util_RIFT_hyperpipe.py``
header comment:

    arch       : iteration / chunking / batch architecture
    post       : posterior-construction stage (CIP-style executable)
    marg-list  : list of per-event/per-driver marg jobs (heterogeneous OK)
    puff       : puffball randomization stage
    test       : convergence-test stage
    init       : initial-grid sourcing (file or generation)
    general    : retries, resources, condor / OSG / singularity knobs

A default config is provided as :data:`DEFAULT_CONFIG_YAML` so a user can
write a minimal override and get a working pipeline.

Validation is intentionally light here --- the heavy lifting is delegated
to :mod:`RIFT.hyperpipe.coords` and :mod:`RIFT.hyperpipe.marg_list`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Default config (Hydra writes this verbatim if no override is given)
# --------------------------------------------------------------------------

DEFAULT_CONFIG_YAML = """\
# RIFT hyperpipe default configuration.
#
# Override on the CLI with Hydra syntax, e.g.:
#     util_RIFT_hyperpipe.py arch.n-iterations=10 general.use-osg=true
# or by writing your own hyperpipe_conf.yaml and pointing Hydra at it.

arch:
  # High-level iterative architecture knobs.
  method: default               # reserved for future hyperpipe strategies
  n-iterations: 5
  n-samples-per-job: 1000
  explode-marg-jobs: 5          # -> --eos-post-explode-jobs
  explode-marg-jobs-last: null  # -> --eos-post-explode-jobs-last (default = same)
  start-iteration: 0
  # Parsimonious-placement workflow (set tracer-only-marg: true to enable).
  # When true, only the final ``tracer-final-marg-iterations`` iterations
  # spawn MARG_* (posterior) nodes; intermediate iterations rely on
  # MARG_PUFF (placement) jobs alone, recovering most of a CIP iteration's
  # cost. Requires puff.exe to be a tracer-aware updater.
  tracer-only-marg: false       # -> --tracer-only-marg
  tracer-final-marg-iterations: 1  # -> --tracer-final-marg-iterations

post:
  # Final posterior-construction stage (CIP-family executable).
  exe: null                     # default = `which util_ConstructEOSPosterior.py`
  coord-module: null            # importable module name, e.g. "rift_default"
  coords-fit: ""                # "x y z"
  coords-sample: ""             # "x:[-8,8] y:[-8,8] z:[-8,8]"
  coords-implied: ""            # "R1.4 Mmax"
  coords-nofit: ""              # "delta_mc s1z s2z"
  likelihood-factor-module: null
  likelihood-factor-function: null
  likelihood-factor-ini: null
  extra-args: ""                # appended verbatim to args_eos_post.txt
  settings:
    n-max: null                 # -> --n-max
    n-step: null                # -> --n-step
    n-eff: null                 # -> --n-eff
    sampler-method: null        # -> --sampler-method
    fit-method: null            # -> --fit-method
    sigma-cut: null             # -> --sigma-cut

marg-list:
  # One entry per (likelihood driver, event) pair. Heterogeneous OK ---
  # different drivers may have different batch sizes (n-chunk) and
  # different per-driver coord modules.
  #
  # Example (Gaussian toy):
  #   - name: gaussian
  #     exe: example_gaussian.py
  #     args: "--outdir Gaussian_example --conforming-output-name"
  #     event-file: null      # null -> 'empty_event_file' sentinel
  #     n-chunk: 100
  #     coord-module: null    # per-driver coord override (rarely needed)
  #     extra-args: ""
  - name: example
    exe: example_gaussian.py
    args: "--outdir example_output --conforming-output-name"
    event-file: null
    n-chunk: 100
    coord-module: null
    extra-args: ""

puff:
  exe: null                     # default = `which util_HyperparameterPuffball.py`
  puff-factor: 0.5
  force-away: 0.03
  extra-args: ""
  # Tracer-placement sampler hyperparameters. These are only consumed when
  # exe points at a tracer-aware updater (e.g. util_HyperparameterTracerUpdate.py
  # or util_ParameterTracerUpdate.py); the legacy puffball binaries ignore them.
  # Null values fall through to the updater's built-in defaults.
  settings:
    update-method: null         # smc-mala-bd | smc-mala | birth-death | puffball
    tracer-fit-method: null     # rf | rbf | polynomial | quadratic
    n-mala-steps: null          # -> --n-mala-steps
    target-ess-frac: null       # -> --target-ess-frac
    birth-death-rate: null      # -> --birth-death-rate
    inj-file-prev: null         # -> --inj-file-prev (SMC bridging input)
    no-union-refit: false       # -> --no-union-refit
    regularize: false           # -> --regularize (passes through to puffball-compat code)
    rng-seed: null              # -> --rng-seed (deterministic when set)
    state-in: null              # -> --state-in
    state-out: null             # -> --state-out

test:
  # The on-disk script is named convergence_test_samples.py (setup.py ships
  # bin/* verbatim).  Using the bare name here would make the test-exe
  # `which` resolution miss and leave test.sub with a non-absolute
  # `executable =` line.
  exe: convergence_test_samples.py
  method: JS
  threshold: 0.05
  extra-args: ""

init:
  # Initial parameter-grid sourcing. Provide *either* ``file`` (path to an
  # existing grid file) *or* a ``generation`` block (auto-generate via
  # util_HyperparameterGrid.py-style helper). If both are set, ``file``
  # wins.
  file: null
  generation:
    placement-method: null      # e.g. "uniform"
    params-and-ranges: null     # e.g. "x:[-8,8] y:[-8,8] z:[-8,8]"
    npts: null
    external-code: null
    external-args: null

general:
  rundir: null                  # optional; created if missing, cd into it
  retries: 0                    # -> --general-retries
  request-disk: 10M             # -> --general-request-disk
  request-memory: 16384         # -> --request-memory-marg  (MB)
  use-osg: false
  use-singularity: false
  use-singularity-local: false
  condor-local-nonworker: false
  condor-local-nonworker-igwn-prefix: false
  condor-nogrid-nonworker: false
  use-full-submit-paths: true
  transfer-files: []            # extra files to ship beyond auto-detected exes
"""


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _get(node, key, default=None):
    """Forgiving get that works on OmegaConf DictConfig *and* plain dicts."""
    if node is None:
        return default
    try:
        return node.get(key, default)  # type: ignore[union-attr]
    except AttributeError:
        return node[key] if key in node else default


def validate_config(cfg) -> None:
    """Raise informatively if the config has structural problems we can detect early.

    This does *not* validate coord modules or marg-list executables ---
    those are deferred to coords.HyperCoordSpec.validate() and
    marg_list.assemble_marg_list() so the messages can be more specific.
    """
    if cfg is None:
        raise ValueError("hyperpipe config is empty.")

    # Required top-level sections
    for section in ("arch", "post", "marg-list", "puff", "init", "general"):
        if _get(cfg, section) is None:
            raise ValueError(
                f"hyperpipe config is missing required section: {section!r}"
            )

    # marg-list must be a non-empty sequence
    marg_list = _get(cfg, "marg-list")
    try:
        n = len(marg_list)
    except TypeError as exc:
        raise ValueError("'marg-list' must be a list of marg-driver entries.") from exc
    if n == 0:
        raise ValueError("'marg-list' must contain at least one entry.")

    # arch
    arch = _get(cfg, "arch")
    if not isinstance(_get(arch, "n-iterations"), int) or _get(arch, "n-iterations") <= 0:
        raise ValueError("arch.n-iterations must be a positive integer.")
    if not isinstance(_get(arch, "n-samples-per-job"), int) or _get(arch, "n-samples-per-job") <= 0:
        raise ValueError("arch.n-samples-per-job must be a positive integer.")

    # post: must have at least one fit dim AND at least one MC sampling dim.
    # Fit basis = coords-fit + coords-implied; MC basis = coords-fit + coords-nofit.
    # Pre-decoupling this only required coords-fit, because the fit basis was
    # forced to equal the MC basis -- now an EOS-style "fit in a transformed
    # basis" config can legally have empty coords-fit (everything routed via
    # coords-implied + coords-nofit through the coordinate plugin).
    post = _get(cfg, "post")
    has_fit  = bool(_get(post, "coords-fit")) or bool(_get(post, "coords-implied"))
    has_samp = bool(_get(post, "coords-fit")) or bool(_get(post, "coords-nofit"))
    if not has_fit:
        raise ValueError(
            "post: must list at least one fit dimension "
            "(coords-fit or coords-implied; e.g. 'x y z' or 'u v w')."
        )
    if not has_samp:
        raise ValueError(
            "post: must list at least one MC sampling dimension "
            "(coords-fit or coords-nofit; e.g. 'x y z')."
        )

    # init: must have either file or generation set
    init = _get(cfg, "init")
    has_file = bool(_get(init, "file"))
    gen = _get(init, "generation") or {}
    has_gen = bool(_get(gen, "placement-method") or _get(gen, "external-code"))
    if not (has_file or has_gen):
        raise ValueError(
            "init: must provide either init.file (existing grid) or "
            "init.generation.placement-method / init.generation.external-code."
        )


def expand_path(p: str, base_dir: str) -> str:
    """Expand a config path: leave absolutes alone; resolve relatives against *base_dir*."""
    if not p:
        return p
    p = os.path.expanduser(p)
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def truthy(val: Any) -> bool:
    """Robustly coerce config values to bool. Tolerates 'true'/'True'/'1'/etc."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        return val.strip().lower() in {"true", "yes", "1", "on"}
    return False
