#! /usr/bin/env python
"""
util_RIFT_hyperpipe.py
======================

Top-level driver for the RIFT hyperpipeline --- the iterative
marginalize + fit + puff loop used to infer hyperparameters (e.g. EOS
parameters, population-model parameters, or any user-defined high-level
quantity) from one or more underlying likelihood evaluators ("marg
drivers").

This is the hyperpipe analog of :file:`util_RIFT_pseudo_pipe.py`. Where
``pseudo_pipe`` builds a single-event GW PE pipeline from argparse
flags and a ``.ini`` file, this script builds a hyperpipe DAG from an
ini-style **Hydra** configuration with three first-class features:

  * **ini-based configuration** (Hydra/OmegaConf) --- see the default
    template :file:`hyperpipe_conf.yaml` next to this script and the
    schema documented in :mod:`RIFT.hyperpipe.config`.

  * **Flexible multi-event input** via the ``marg-list:`` section ---
    one entry per (likelihood-driver, event) pair, with per-entry
    executables, args, event files, ``n-chunk`` (heterogeneous batch
    sizes), and an optional per-entry coord module.

  * **Coordinate-transformation framework** --- the ``post`` stage
    consumes a CIP-mirror coord module via
    ``--supplementary-coordinate-code``, the same convention
    ``util_ConstructEOSPosterior.py`` already understands; the
    parameters / integration ranges are emitted automatically from
    ``post.coords-fit`` / ``post.coords-sample`` (and optionally
    ``coords-implied`` / ``coords-nofit``).

Examples
--------
Reproduce a minimal Gaussian-toy hyperpipe analysis::

    util_RIFT_hyperpipe.py \\
        general.rundir=./my_run \\
        init.file=./blind_gaussian_plus_minus.dat

Override a few defaults from the CLI (Hydra syntax)::

    util_RIFT_hyperpipe.py \\
        arch.n-iterations=15 \\
        general.use-osg=true general.use-singularity=true \\
        general.condor-local-nonworker=true

Point at a config file living anywhere on disk (no install / no
``--config-dir`` plumbing needed)::

    util_RIFT_hyperpipe.py --config ./my_tracer.yaml
    util_RIFT_hyperpipe.py -c /full/path/to/run_config.yaml \\
        puff.settings.update-method=ucb

The ``--config PATH`` shim accepts an absolute or relative path and
splits it into the ``--config-dir`` / ``--config-name`` pair Hydra
expects. Trailing ``.yaml`` / ``.yml`` extensions are stripped
automatically (so ``--config-name foo.yaml`` works too). A bare name
falls through to Hydra's normal search path (the installed
``hyperpipe_conf.yaml`` next to the script).

Implementation notes
--------------------
The real work lives in :mod:`RIFT.hyperpipe`:

  * :func:`RIFT.hyperpipe.config.validate_config`     -- structural validation
  * :func:`RIFT.hyperpipe.coords.coord_spec_from_config_section` -- coord-spec
  * :func:`RIFT.hyperpipe.marg_list.assemble_marg_list` -- multi-event assembly

This script is intentionally thin: it parses Hydra config, materializes
the args / exe / event / nchunk files that
``create_eos_posterior_pipeline`` consumes, builds the matching
command line, and either runs it or (with ``general.dry-run=true``)
prints it.
"""

from __future__ import annotations

import logging
import os
import shutil
import shlex
import sys
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from RIFT.hyperpipe import config as hyper_config
from RIFT.hyperpipe.coords import coord_spec_from_config_section
from RIFT.hyperpipe.marg_list import assemble_marg_list

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _which_or_self(name: str) -> str:
    """Resolve an exe via PATH (or RIFT's which); fall back to the bare name."""
    found = shutil.which(name)
    if found:
        return found
    try:
        from RIFT.misc.dag_utils_generic import which as rift_which

        rw = rift_which(name)
        if rw:
            return rw
    except Exception:
        pass
    return name


def _cfg_get(node, key, default=None):
    """Forgiving get() that works on DictConfig or plain dict."""
    if node is None:
        return default
    try:
        return node.get(key, default)
    except AttributeError:
        return node[key] if key in node else default


def _maybe_generate_initial_grid(
    init_cfg, base_dir: str, run_dir: str
) -> str:
    """Materialize an initial_grid.dat in run_dir.

    Honors (in priority order):
      1. ``init.file``: copy verbatim
      2. ``init.generation.placement-method == 'uniform'``: call out to
         ``util_HyperparameterGrid.py`` with random-parameter/range/npts
         pulled from the generation block.
      3. ``init.generation.external-code``: invoke an arbitrary user
         command (``external-code external-args``) expected to drop
         ``initial_grid.dat`` in cwd.
    """
    initial_grid = os.path.join(run_dir, "initial_grid.dat")
    init_file = _cfg_get(init_cfg, "file")
    if init_file:
        src = hyper_config.expand_path(init_file, base_dir)
        if not os.path.exists(src):
            raise FileNotFoundError(f"init.file does not exist: {src!r}")
        shutil.copyfile(src, initial_grid)
        return initial_grid

    gen = _cfg_get(init_cfg, "generation") or {}
    placement = _cfg_get(gen, "placement-method")
    external_code = _cfg_get(gen, "external-code")

    if placement == "uniform":
        params_and_ranges = (_cfg_get(gen, "params-and-ranges") or "").strip()
        npts = _cfg_get(gen, "npts")
        if not params_and_ranges or not npts:
            raise ValueError(
                "init.generation.placement-method='uniform' requires "
                "params-and-ranges and npts."
            )
        # Parse params-and-ranges: "x:[a,b] y:[c,d] ..."
        # and emit --random-parameter / --random-parameter-range pairs.
        from RIFT.hyperpipe.coords import parse_range_string

        ranges = parse_range_string(params_and_ranges)
        bits = [_which_or_self("util_HyperparameterGrid.py")]
        for name, (lo, hi) in ranges.items():
            bits.append(f"--random-parameter {name}")
            bits.append(f"--random-parameter-range [{lo},{hi}]")
        bits.append(f"--npts {int(npts)}")
        bits.append(f"--fname-out {initial_grid}")
        cmd = " ".join(bits)
        logger.info("Generating initial grid: %s", cmd)
        rc = os.system(cmd)
        if rc != 0:
            raise SystemExit(f"util_HyperparameterGrid.py exited with code {rc}.")
        return initial_grid

    if external_code:
        ext_args = _cfg_get(gen, "external-args") or ""
        cmd = f"{external_code} {ext_args}".strip()
        logger.info("Generating initial grid via external code: %s", cmd)
        rc = os.system(f"cd {shlex.quote(run_dir)} && {cmd}")
        if rc != 0:
            raise SystemExit(f"external-code exited with code {rc}.")
        if not os.path.exists(initial_grid):
            raise FileNotFoundError(
                f"external-code did not produce {initial_grid!r}; "
                "it must write to that exact path."
            )
        return initial_grid

    raise ValueError(
        "init: no usable initialization. Set either init.file, "
        "init.generation.placement-method='uniform' (with params-and-ranges + npts), "
        "or init.generation.external-code."
    )


def _build_post_args(cfg, coord_spec) -> str:
    """Compose the args_eos_post.txt body from the coord spec + post.settings + extras."""
    post = cfg["post"]
    args = coord_spec.to_post_args()
    settings = _cfg_get(post, "settings") or {}
    setting_flags = [
        ("n-max", "--n-max"),
        ("n-step", "--n-step"),
        ("n-eff", "--n-eff"),
        ("sampler-method", "--sampler-method"),
        ("fit-method", "--fit-method"),
        ("sigma-cut", "--sigma-cut"),
    ]
    for key, flag in setting_flags:
        val = _cfg_get(settings, key)
        if val is not None and val != "":
            args += f" {flag} {val}"
    extra = (_cfg_get(post, "extra-args") or "").strip()
    if extra:
        args += " " + extra
    return args


def _build_puff_args(cfg, coord_spec) -> str:
    """Compose args_puff.txt from the coord spec + puff.settings + extras.

    The legacy puffball (util_HyperparameterPuffball.py) accepts only the
    coord-spec flags + force-away/puff-factor. The tracer drop-ins
    (util_HyperparameterTracerUpdate.py / util_ParameterTracerUpdate.py)
    accept those plus a handful of sampler hyperparameters; we expose those
    declaratively here so users can stay in Hydra rather than escaping into
    extra-args strings.
    """
    puff = _cfg_get(cfg, "puff") or {}
    args = coord_spec.to_puff_args(
        force_away=_cfg_get(puff, "force-away", 0.03),
        puff_factor=_cfg_get(puff, "puff-factor", 0.5),
    )
    settings = _cfg_get(puff, "settings") or {}
    # Value-bearing flags. Each pair is (yaml key, CLI flag). Empty / null is skipped.
    setting_flags = [
        ("update-method", "--update-method"),
        ("tracer-fit-method", "--tracer-fit-method"),
        ("ucb-kappa", "--ucb-kappa"),
        ("ucb-n-candidates", "--ucb-n-candidates"),
        ("n-mala-steps", "--n-mala-steps"),
        ("target-ess-frac", "--target-ess-frac"),
        ("birth-death-rate", "--birth-death-rate"),
        ("inj-file-prev", "--inj-file-prev"),
        ("rng-seed", "--rng-seed"),
        ("state-in", "--state-in"),
        ("state-out", "--state-out"),
    ]
    for key, flag in setting_flags:
        val = _cfg_get(settings, key)
        if val is not None and val != "":
            args += f" {flag} {val}"
    # Bool flags. Each pair is (yaml key, CLI flag).
    setting_bool_flags = [
        ("no-union-refit", "--no-union-refit"),
        ("regularize", "--regularize"),
    ]
    for key, flag in setting_bool_flags:
        if hyper_config.truthy(_cfg_get(settings, key, False)):
            args += f" {flag}"
    extra = (_cfg_get(puff, "extra-args") or "").strip()
    if extra:
        args += " " + extra
    return args


def _build_test_args(cfg, coord_spec) -> str:
    test = _cfg_get(cfg, "test") or {}
    args = coord_spec.to_test_args(
        method=_cfg_get(test, "method", "JS"),
        threshold=_cfg_get(test, "threshold", 0.05),
    )
    # test.settings: forwards convergence_test_samples.py-specific flags. All
    # nullable -- omitted keys produce no flag, mirroring post.settings.
    settings = _cfg_get(test, "settings") or {}
    setting_flags = [
        ("iteration-threshold",    "--iteration-threshold"),
        ("write-file-on-success",  "--write-file-on-success"),
        ("test-output",            "--test-output"),
    ]
    for key, flag in setting_flags:
        val = _cfg_get(settings, key)
        if val is not None and val != "":
            args += f" {flag} {val}"
    bool_flags = [
        ("always-succeed", "--always-succeed"),
        ("verbose",        "--verbose"),
    ]
    for key, flag in bool_flags:
        if hyper_config.truthy(_cfg_get(settings, key, False)):
            args += f" {flag}"
    extra = (_cfg_get(test, "extra-args") or "").strip()
    if extra:
        args += " " + extra
    return args


# --------------------------------------------------------------------------
# Hydra entry point
# --------------------------------------------------------------------------


@hydra.main(version_base=None, config_path=".", config_name="hyperpipe_conf")
def my_app(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    base_dir = hydra.utils.get_original_cwd()

    logger.info("---- INPUT CONFIG ----")
    print(OmegaConf.to_yaml(cfg))

    hyper_config.validate_config(cfg)

    # ----- run directory ---------------------------------------------------
    rundir = _cfg_get(cfg["general"], "rundir")
    if rundir:
        rundir = hyper_config.expand_path(rundir, base_dir)
        os.makedirs(rundir, exist_ok=True)
        os.chdir(rundir)
    run_dir = os.getcwd()
    logger.info("Working directory: %s", run_dir)

    # ----- coord spec ------------------------------------------------------
    coord_spec = coord_spec_from_config_section(cfg["post"])
    coord_spec.validate(strict_import=False)

    # ----- marg-list assembly ---------------------------------------------
    marg = assemble_marg_list(cfg, base_dir=base_dir, run_dir=run_dir)
    logger.info(
        "marg-list: %d entries (%s)",
        len(marg.names),
        ", ".join(marg.names),
    )

    # ----- emit per-stage files -------------------------------------------
    args_marg_path = os.path.join(run_dir, "args_marg_eos.txt")
    args_marg_exe_path = os.path.join(run_dir, "args_marg_eos_exe.txt")
    args_post_path = os.path.join(run_dir, "args_eos_post.txt")
    args_puff_path = os.path.join(run_dir, "args_puff.txt")
    args_test_path = os.path.join(run_dir, "args_test.txt")
    event_nchunk_path = os.path.join(run_dir, "event_nchunk.txt")
    transfer_list_path = os.path.join(run_dir, "transfer_file_list.txt")

    marg.write_args_file(args_marg_path)
    marg.write_exe_file(args_marg_exe_path)
    marg.write_nchunk_file(event_nchunk_path)

    # create_eos_posterior_pipeline reads these via --eos-post-args /
    # --puff-args / --test-args and discards the first whitespace-separated
    # token (see split(' ')[1:] in create_eos_posterior_pipeline). We prepend
    # a dummy "X" so the real first flag survives. The marg-list args file is
    # read line-by-line and does NOT get this treatment.
    _PIPELINE_DUMMY = "X"
    with open(args_post_path, "w") as f:
        f.write(_PIPELINE_DUMMY + " " + _build_post_args(cfg, coord_spec) + "\n")
    with open(args_puff_path, "w") as f:
        f.write(_PIPELINE_DUMMY + " " + _build_puff_args(cfg, coord_spec) + "\n")
    with open(args_test_path, "w") as f:
        f.write(_PIPELINE_DUMMY + " " + _build_test_args(cfg, coord_spec) + "\n")

    # ----- initial grid ---------------------------------------------------
    initial_grid = _maybe_generate_initial_grid(cfg["init"], base_dir, run_dir)

    # ----- transfer-file list ---------------------------------------------
    general = cfg["general"]
    extra_transfer: List[str] = list(_cfg_get(general, "transfer-files") or [])
    use_osg = hyper_config.truthy(_cfg_get(general, "use-osg", False))
    use_singularity = hyper_config.truthy(_cfg_get(general, "use-singularity", False))
    use_transfer = use_osg or use_singularity or bool(extra_transfer)
    if use_transfer:
        marg.write_transfer_file_list(transfer_list_path, extra=extra_transfer)

    # ----- command assembly -----------------------------------------------
    arch = cfg["arch"]
    post_exe = _cfg_get(cfg["post"], "exe") or _which_or_self("util_ConstructEOSPosterior.py")
    puff_exe = (
        _cfg_get(_cfg_get(cfg, "puff"), "exe")
        or _which_or_self("util_HyperparameterPuffball.py")
    )
    test_exe = _cfg_get(_cfg_get(cfg, "test"), "exe", "convergence_test_samples")

    cmd_parts: List[str] = [
        _which_or_self("create_eos_posterior_pipeline"),
        f"--n-samples-per-job {int(arch['n-samples-per-job'])}",
        f"--n-iterations {int(arch['n-iterations'])}",
        f"--working-directory {run_dir}",
        f"--input-grid {initial_grid}",
        f"--marg-event-exe-list-file {args_marg_exe_path}",
        f"--marg-event-args-list-file {args_marg_path}",
        f"--marg-event-nchunk-list-file {event_nchunk_path}",
        f"--eos-post-args {args_post_path}",
        f"--eos-post-exe {post_exe}",
        f"--puff-exe {puff_exe}",
        f"--puff-args {args_puff_path}",
        f"--test-args {args_test_path}",
        f"--test-exe {test_exe}",
    ]
    # one --event-file per marg entry (critical for multi-event!)
    for ev in marg.event_files:
        cmd_parts.append(f"--event-file {ev}")

    # arch tunables
    explode = _cfg_get(arch, "explode-marg-jobs")
    if explode is not None:
        cmd_parts.append(f"--eos-post-explode-jobs {int(explode)}")
    explode_last = _cfg_get(arch, "explode-marg-jobs-last")
    if explode_last is not None:
        cmd_parts.append(f"--eos-post-explode-jobs-last {int(explode_last)}")
    start_iter = int(_cfg_get(arch, "start-iteration", 0) or 0)
    if start_iter:
        cmd_parts.append(f"--start-iteration {start_iter}")
    # tracer-aware puff input source (posterior|marg_net). Default posterior
    # keeps the legacy puffball path byte-identical.
    puff_input_source = _cfg_get(_cfg_get(cfg, "puff"), "input-source")
    if puff_input_source:
        cmd_parts.append(f"--puff-input-source {puff_input_source}")

    # general
    bool_flag_pairs = [
        ("use-osg", "--use-osg"),
        ("use-singularity", "--use-singularity"),
        ("use-singularity-local", "--use-singularity-local"),
        ("condor-local-nonworker", "--condor-local-nonworker"),
        ("condor-local-nonworker-igwn-prefix", "--condor-local-nonworker-igwn-prefix"),
        ("condor-nogrid-nonworker", "--condor-nogrid-nonworker"),
        ("use-full-submit-paths", "--use-full-submit-paths"),
    ]
    for key, flag in bool_flag_pairs:
        if hyper_config.truthy(_cfg_get(general, key, False)):
            cmd_parts.append(flag)

    retries = int(_cfg_get(general, "retries", 0) or 0)
    if retries:
        cmd_parts.append(f"--general-retries {retries}")
    req_disk = _cfg_get(general, "request-disk")
    if req_disk:
        cmd_parts.append(f"--general-request-disk {req_disk}")
    req_mem = _cfg_get(general, "request-memory")
    if req_mem is not None:
        cmd_parts.append(f"--request-memory-marg {int(req_mem)}")
    if use_transfer:
        cmd_parts.append(f"--transfer-file-list {transfer_list_path}")

    cmd = " ".join(cmd_parts)
    print(cmd)

    if hyper_config.truthy(_cfg_get(general, "dry-run", False)):
        logger.info("dry-run set; not invoking create_eos_posterior_pipeline.")
        return

    rc = os.system(cmd)
    if rc != 0:
        raise SystemExit(
            f"create_eos_posterior_pipeline exited with code {rc} (cmd: {cmd!r})."
        )


def _preprocess_config_args(argv):
    """User-friendly --config shim translated into Hydra's --config-dir / --config-name.

    Hydra's native CLI requires two flags (``--config-dir DIR --config-name NAME``)
    and the config-name must be bare (no ``.yaml`` extension). Users who pass a full
    path to a config file expect the script to "just use it". This shim accepts:

      --config /full/path/to/myconf.yaml
      --config myconf.yaml                          (resolved against CWD)
      --config myconf                               (Hydra default search path)
      -c /full/path/to/myconf.yaml                  (short form)

    and rewrites argv so Hydra sees the equivalent of::

      --config-dir /full/path/to    --config-name myconf

    Also strips a trailing ``.yaml`` / ``.yml`` from ``--config-name=...`` /
    ``--config-name foo.yaml`` (the most common user-mistake from Hydra's docs).

    Fails fast with a clear error if the resolved file does not exist.
    """
    out = []
    i = 0
    n = len(argv)

    def _strip_ext(name):
        for ext in (".yaml", ".yml"):
            if name.endswith(ext):
                return name[: -len(ext)]
        return name

    def _split_path_or_name(raw):
        """Given what the user passed, return (dir_or_None, bare_name)."""
        if os.sep in raw or raw.startswith(".") or raw.endswith(".yaml") or raw.endswith(".yml"):
            # path-shaped or extension-bearing
            abs_path = os.path.abspath(raw)
            d = os.path.dirname(abs_path)
            base = os.path.basename(abs_path)
            return d, _strip_ext(base)
        # bare name -> leave dir alone (use built-in search path)
        return None, _strip_ext(raw)

    def _emit(dirpart, namepart, raw):
        # Validate existence when we have a directory commitment
        if dirpart is not None:
            full = os.path.join(dirpart, namepart + ".yaml")
            full_yml = os.path.join(dirpart, namepart + ".yml")
            if not (os.path.isfile(full) or os.path.isfile(full_yml)):
                sys.stderr.write(
                    f"util_RIFT_hyperpipe.py: config file not found from --config {raw!r}\n"
                    f"  looked for: {full}\n"
                    f"          or: {full_yml}\n"
                )
                sys.exit(2)
            out.append(f"--config-dir={dirpart}")
        out.append(f"--config-name={namepart}")

    while i < n:
        a = argv[i]
        # Long form: --config X  or  --config=X
        if a == "--config" or a == "-c":
            if i + 1 >= n:
                sys.stderr.write("util_RIFT_hyperpipe.py: --config requires an argument\n")
                sys.exit(2)
            raw = argv[i + 1]
            d, name = _split_path_or_name(raw)
            _emit(d, name, raw)
            i += 2
            continue
        if a.startswith("--config=") or a.startswith("-c="):
            raw = a.split("=", 1)[1]
            d, name = _split_path_or_name(raw)
            _emit(d, name, raw)
            i += 1
            continue
        # Belt-and-braces: if the user passed --config-name foo.yaml, strip the ext.
        if a == "--config-name" and i + 1 < n:
            out.append(a)
            out.append(_strip_ext(argv[i + 1]))
            i += 2
            continue
        if a.startswith("--config-name="):
            out.append("--config-name=" + _strip_ext(a.split("=", 1)[1]))
            i += 1
            continue
        out.append(a)
        i += 1
    return out


if __name__ == "__main__":
    # Translate any user-friendly --config PATH into Hydra's native flags
    # before @hydra.main parses sys.argv. Idempotent: passing Hydra's
    # native --config-dir / --config-name through unchanged still works.
    sys.argv = [sys.argv[0]] + _preprocess_config_args(sys.argv[1:])
    my_app()
