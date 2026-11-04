"""Worked example: GW PE synthetic-targeted backend.

Runs in two modes depending on `--factory`:

    --factory stub          (default; no lalsuite/RIFT/condor needed)
        Builds an archive, registers 3 events, submits in embed mode
        with the no-op `factory_stub` subdag_factory, simulates the
        per-(sim, level) sub-DAGs completing inline, refreshes
        statuses, and demonstrates dedup + refinement.

    --factory pseudo_pipe   (the real path; needs lalsuite + RIFT)
        Builds an archive, registers 3 events, invokes the real
        `factory_pseudo_pipe` subdag_factory which renders frames /
        coinc.xml / PSD / ini and shells out to util_RIFT_pseudo_pipe.
        Stops after producing the wrapper DAG; ACTUAL PE completion
        is condor's job, not this demo's. Use `--submit-mode submit`
        to also dispatch the wrapper DAG to your local schedd.

CLI:
    python -m RIFT.simulation_manager.examples.gw_pe_synthetic_hello \\
        [--base PATH] [--keep] \\
        [--factory {stub,pseudo_pipe}] \\
        [--submit-mode {submit,embed}] \\
        [--base-ini-path FILE]            # for pseudo_pipe: pop-example.ini etc.
        [--singularity-image PATH]
        [--accounting-group X --accounting-group-user Y]

The default (`--factory stub --submit-mode embed`) runs end-to-end
with no external deps. Swap `--factory pseudo_pipe` and supply at
least `--base-ini-path` to drive the real pipeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from RIFT.simulation_manager.database import StatusRecord
from RIFT.simulation_manager.backends.gw_pe_synthetic import make_archive
from RIFT.simulation_manager.backends.gw_pe_synthetic import factory_stub


# Default canned events. Adjust here if you want to demo different params;
# the more interesting overrides (factory, base ini, mode) are CLI-driven.
DEFAULT_EVENTS = [
    {"mc": 25.0, "eta": 0.22, "distance": 400.0,
     "ra": 1.234, "dec": 0.567, "geocent_time": 1.2e9,
     "inclination": 0.5, "polarization": 0.3, "s1z": 0.0, "s2z": 0.0},
    {"mc": 31.0, "eta": 0.20, "distance": 500.0,
     "ra": 2.0,   "dec": -0.3,  "geocent_time": 1.2e9 + 100,
     "inclination": 1.0, "polarization": 0.0, "s1z": 0.1, "s2z": -0.1},
    {"mc": 12.5, "eta": 0.245, "distance": 200.0,
     "ra": 0.7,   "dec": 0.1,   "geocent_time": 1.2e9 + 200,
     "inclination": 0.2, "polarization": 1.0, "s1z": 0.0, "s2z": 0.0},
]


def _resolve_factory(name):
    if name == "stub":
        from RIFT.simulation_manager.backends.gw_pe_synthetic import factory_stub
        return factory_stub.subdag_factory
    if name == "pseudo_pipe":
        from RIFT.simulation_manager.backends.gw_pe_synthetic import factory_pseudo_pipe
        return factory_pseudo_pipe.subdag_factory
    raise ValueError("Unknown factory: {!r}".format(name))


def _resolve_ini_localizer(spec):
    """Resolve a 'module:callable' spec into the callable. Used for the
    --ini-localizer CLI knob so users can plug in their own
    domain-specific ini construction without editing this example."""
    import importlib
    if ":" not in spec:
        raise ValueError("--ini-localizer must be 'module:callable'; got {!r}"
                         .format(spec))
    mod_name, _, attr = spec.partition(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError("{}.{} is not callable".format(mod_name, attr))
    return fn


def _augment_events_for_factory(events, args):
    """Some factories need extra params on each event. The stub doesn't
    care; pseudo_pipe wants base_ini_path / noise_source / etc."""
    if args.factory != "pseudo_pipe":
        return events
    augmented = []
    for e in events:
        e = dict(e)
        if args.base_ini_path:
            e["base_ini_path"] = args.base_ini_path
        e.setdefault("noise_source", args.noise_source)
        e.setdefault("psd_source", args.psd_source)
        if args.user_psd_path:
            e["user_psd_path"] = args.user_psd_path
        augmented.append(e)
    return augmented


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--base", default=None,
                    help="Archive directory (default: $TMPDIR/gw_pe_synth_hello)")
    ap.add_argument("--keep", action="store_true",
                    help="Don't delete --base on success")
    ap.add_argument("--factory", choices=("stub", "pseudo_pipe"),
                    default="stub",
                    help="Which subdag_factory to wire into the run queue. "
                         "stub = no-op (no external deps); pseudo_pipe = real "
                         "(requires lalsuite + RIFT)")
    ap.add_argument("--submit-mode", choices=("submit", "embed"),
                    default="embed",
                    help="DualCondorRunQueue submit_mode. embed = write "
                         "wrapper DAG without dispatching (default; "
                         "suitable for hyperpipeline-style use). submit = "
                         "call condor_submit_dag now.")
    ap.add_argument("--target-level", type=int, default=1,
                    help="target level for the registered events; default 1")
    # pseudo_pipe-specific knobs
    ap.add_argument("--base-ini-path", default=None,
                    help="Path to a base RIFT ini. The default ini_localizer "
                         "copies this through with level-scaled iterations/"
                         "n-eff layered on top — that is NOT enough for "
                         "production runs (see --ini-localizer).")
    ap.add_argument("--ini-localizer", default=None,
                    help="Production-style ini localizer, as 'module:callable'. "
                         "Signature: (base_ini_path, params, level, out_path) "
                         "-> Path. Use this to inject per-event mass priors, "
                         "signal duration, fmin, etc. that the default "
                         "localizer leaves untouched.")
    ap.add_argument("--noise-source", choices=("gwpy", "user"), default="gwpy",
                    help="pseudo_pipe noise: gwpy-generated or user frames")
    ap.add_argument("--psd-source", choices=("design_aligo", "user"),
                    default="design_aligo")
    ap.add_argument("--user-psd-path", default=None)
    # Run-pool knobs (passed to DualCondorRunQueue.extra)
    ap.add_argument("--singularity-image", default=None)
    ap.add_argument("--accounting-group", default=None)
    ap.add_argument("--accounting-group-user", default=None)
    ap.add_argument("--no-refinement-demo", action="store_true",
                    help="Skip the dedup + refinement block at the end "
                         "(automatic when --factory pseudo_pipe).")
    args = ap.parse_args()

    base = Path(args.base or (tempfile.gettempdir() + "/gw_pe_synth_hello"))
    if base.exists():
        shutil.rmtree(base)

    # Build run_queue_extra from CLI flags.
    run_queue_extra = {}
    if args.singularity_image:
        run_queue_extra["use_singularity"] = True
        run_queue_extra["singularity_image"] = args.singularity_image
    if args.accounting_group:
        run_queue_extra["accounting_group"] = args.accounting_group
    if args.accounting_group_user:
        run_queue_extra["accounting_group_user"] = args.accounting_group_user

    # Resolve the factory and (optional) ini_localizer.
    factory = _resolve_factory(args.factory)
    ini_localizer = None
    if args.ini_localizer:
        ini_localizer = _resolve_ini_localizer(args.ini_localizer)
        print("  ini_localizer: {}".format(args.ini_localizer))

    archive = make_archive(
        base_location=base,
        subdag_factory=factory,
        submit_mode=args.submit_mode,
        run_queue_extra=run_queue_extra,
        ini_localizer=ini_localizer,
    )
    print("Built archive at: {}".format(archive.base))
    print("  factory      : {}".format(args.factory))
    print("  submit_mode  : {}".format(args.submit_mode))
    if args.singularity_image:
        print("  singularity  : {}".format(args.singularity_image))

    # Register events.
    events = _augment_events_for_factory(DEFAULT_EVENTS, args)
    names = [archive.register(e, target_level=args.target_level) for e in events]
    print(" Registered: {}".format(names))

    # Submit. With submit_mode='embed' the wrapper DAG is written but
    # not dispatched; per-(sim, level) sub-DAGs are written by the
    # selected factory.
    archive.request_queue.submit_pending(archive)
    wrapper = Path(archive.run_queue.last_wrapper_dag_path)
    print("\n Wrapper DAG: {}".format(wrapper))
    print(" Wrapper contents:")
    for line in wrapper.read_text().splitlines():
        print("    {}".format(line))

    # Mode split: stub demos completion + refinement; pseudo_pipe stops
    # here (real completion is condor's job).
    if args.factory != "stub":
        print("\n--- pseudo_pipe path ---")
        print(" The wrapper DAG above is ready for inclusion as a SUBDAG")
        print(" EXTERNAL in a parent workflow (or direct submission via")
        print(" condor_submit_dag if you used --submit-mode submit).")
        print(" Per-(sim, level) DAGs were produced by util_RIFT_pseudo_pipe;")
        print(" each lives under sims/<n>/level_<lvl>/ alongside its")
        print(" rundir's frames, coinc.xml, ini, and PSD.")
        print("\nPASS: GW PE synthetic-targeted backend (pseudo_pipe factory)")
        print("      end-to-end through wrapper-DAG generation.")
        if not args.keep:
            print("(Re-run with --keep to inspect.)")
        else:
            print("(Archive retained at {}.)".format(archive.base))
        return 0

    # ---- stub path: simulate completion + demo refinement -----------------
    for n in names:
        factory_stub.execute_stub_inline(archive, n, level=args.target_level)
    promoted = archive.refresh_status_from_disk()
    print("\n promoted: {}".format(promoted))
    assert all(archive.get_status(n) == "complete" for n in names)

    print("\nindex.jsonl:")
    print(archive.index.path.read_text())

    if args.no_refinement_demo:
        print("PASS (refinement demo skipped per --no-refinement-demo).")
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)
        return 0

    # Dedup: re-register events[0] -> same name, no-op.
    n_again = archive.register(events[0], target_level=args.target_level)
    assert n_again == names[0], "dedup failed"
    # Bump target_level -> refine_ready -> resubmit -> chained sub-DAGs.
    n_bump = archive.register(events[0], target_level=args.target_level + 2)
    assert n_bump == names[0]
    assert archive.get_status(n_bump) == "refine_ready"
    archive.request_queue.submit_pending(archive)
    print("\nAfter refine to lvl {}:".format(args.target_level + 2))
    print(" wrapper: {}".format(archive.run_queue.last_wrapper_dag_path))
    print(" wrapper contents:")
    print(Path(archive.run_queue.last_wrapper_dag_path).read_text())

    for lvl in range(args.target_level + 1, args.target_level + 3):
        factory_stub.execute_stub_inline(archive, n_bump, level=lvl)
    archive.refresh_status_from_disk()
    rec = StatusRecord.read(archive.sim_dir(n_bump))
    assert rec.data["status"] == "complete"
    assert rec.data["current_level"] == args.target_level + 2
    print("\nFinal status of refined sim: {} at level {}".format(
        rec.data["status"], rec.data["current_level"]))

    print("\nPASS: GW PE synthetic-targeted backend (stub factory) end-to-end.")
    if not args.keep:
        shutil.rmtree(base, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
