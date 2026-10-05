#!/usr/bin/env python3
"""Audit an untouched RIFT DAG build through calibration marginalization."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from .dag_structure import external_dags, parse_dag, validate_dag
except ImportError:  # Direct execution from .travis/test-build.sh.
    from dag_structure import external_dags, parse_dag, validate_dag


DAG_NAME = "marginalize_intrinsic_parameters_BasicIterationWorkflow.dag"
REQUIRED_ROLES = {
    "ILE.sub",
    "ILE_extr.sub",
    "convert_extr.sub",
    "Bilby_pickle.sub",
    "Calib_reweight.sub",
    "CAL_REWEIGHT_COMBINE.sub",
}


def _submit_path(dag, node):
    path = Path(dag.nodes[node]["submit"])
    if not path.is_absolute():
        path = dag.path.parent / path
    return path


def _submit_has(dag, node, value):
    path = _submit_path(dag, node)
    return path.is_file() and value in path.read_text(encoding="utf-8", errors="replace")


def audit_pipeline(root):
    root = Path(root).resolve()
    failures = []
    checks = {}
    dag_path = root / DAG_NAME
    if not dag_path.is_file():
        return {"pass": False, "root": str(root), "failures": ["missing {}".format(dag_path)]}

    dag = parse_dag(dag_path)
    graph_errors = validate_dag(dag)
    roles = defaultdict(set)
    for node, data in dag.nodes.items():
        roles[Path(data["submit"]).name].add(node)

    for role in sorted(REQUIRED_ROLES):
        if not roles[role]:
            failures.append("DAG has no job using {}".format(role))
    cip_nodes = {
        node
        for role, nodes in roles.items()
        if role.startswith("CIP")
        for node in nodes
    }
    if not cip_nodes:
        failures.append("DAG has no CIP or exploded CIP-worker job")

    extrinsic = roles["ILE_extr.sub"]
    converts = roles["convert_extr.sub"]
    pickles = roles["Bilby_pickle.sub"]
    calibration = roles["Calib_reweight.sub"]
    combines = roles["CAL_REWEIGHT_COMBINE.sub"]

    missing_extrinsic = set()
    if len(converts) != 1:
        failures.append("expected one extrinsic conversion, found {}".format(len(converts)))
    else:
        convert = next(iter(converts))
        missing_extrinsic = extrinsic - dag.ancestors(convert)
        if missing_extrinsic:
            failures.append(
                "{} extrinsic jobs do not feed conversion".format(len(missing_extrinsic))
            )

    if len(pickles) != 1:
        failures.append("expected one Bilby pickle, found {}".format(len(pickles)))
    elif len(converts) == 1:
        if next(iter(converts)) not in dag.ancestors(next(iter(pickles))):
            failures.append("extrinsic conversion is not upstream of Bilby pickle")

    for node in calibration:
        if not (dag.ancestors(node) & pickles):
            failures.append("calibration job {} is not downstream of Bilby pickle".format(node))
    if len(combines) != 1:
        failures.append("expected one calibration combine, found {}".format(len(combines)))
        product = None
    else:
        product = next(iter(combines))
        missing_batches = calibration - dag.ancestors(product)
        if missing_batches:
            failures.append(
                "calibration combine misses {} batches".format(len(missing_batches))
            )
        if product not in dag.sinks():
            failures.append("calibration combine is not the terminal product")

    critical_sinks = [
        node
        for node in dag.sinks()
        if Path(dag.nodes[node]["submit"]).name
        in {"ILE_extr.sub", "convert_extr.sub", "Bilby_pickle.sub", "Calib_reweight.sub"}
    ]
    if critical_sinks:
        failures.append("critical pre-combine terminal sinks: " + ", ".join(critical_sinks))

    unsafe_abort_nodes = []
    abort_nodes_without_product = []
    if product:
        for node in dag.abort:
            if product not in dag.descendants(node):
                abort_nodes_without_product.append(node)
            if not _submit_has(dag, node, "--always-succeed"):
                unsafe_abort_nodes.append(node)
    if abort_nodes_without_product:
        failures.append(
            "top-level abort nodes lack terminal-product descendants: "
            + ", ".join(abort_nodes_without_product)
        )
    if unsafe_abort_nodes:
        failures.append(
            "top-level abort nodes can terminate before handoff: " + ", ".join(unsafe_abort_nodes)
        )

    external_reports = []
    parsed = {}
    # Each queued boundary carries the DAG that declares it, whether that
    # declaring node is connected to the top-level terminal product, and the
    # chain of boundaries it was reached through, so a second-level external
    # DAG is evaluated in its own containing graph: two SUBDAG nodes may share
    # one container file yet differ in how they connect to the terminal
    # product.
    pending = [
        (dag, node, path, bool(product) and product in dag.descendants(node), ())
        for node, path in external_dags(dag)
    ]
    seen = set()
    analysed = set()
    while pending:
        container, external_node, path, reaches_product, ancestry = pending.pop()
        invocation = ancestry + ((container.path, external_node, path),)
        # Key on what decides the verdict, not on the whole chain.  A boundary's
        # three findings are fixed by the container it is declared in, the node
        # declaring it, the file it points at, and its connection to the
        # terminal product; two chains agreeing on those agree on the result.
        # Keying on the chain itself is exact but exponential -- a file
        # referenced twice per level costs 2**depth invocations, 4.3 s and
        # 155 MB at depth 16 -- and this gate runs on production DAGs.
        key = (container.path, external_node, path, reaches_product)
        if key in seen:
            continue
        seen.add(key)
        if any(step[0] == path for step in invocation):
            graph_errors.append(
                "external DAG {} re-enters its own container: {}".format(external_node, path)
            )
            continue
        if not path.is_file():
            graph_errors.append("external DAG {} is missing: {}".format(external_node, path))
            continue
        if path not in parsed:
            parsed[path] = parse_dag(path)
            graph_errors.extend(
                "{}: {}".format(path, item) for item in validate_dag(parsed[path])
            )
        nested = parsed[path]
        active_abort_nodes = [
            node
            for node in nested.abort
            if not _submit_has(nested, node, "--always-succeed")
        ]
        if path not in analysed:
            analysed.add(path)
            abort_without_grid = []
            for node in nested.abort:
                prior_roles = {
                    Path(nested.nodes[ancestor]["submit"]).name
                    for ancestor in nested.ancestors(node)
                    if ancestor in nested.nodes
                }
                if "convert.sub" not in prior_roles:
                    abort_without_grid.append(node)
            if abort_without_grid:
                failures.append(
                    "external DAG aborts before a grid conversion: "
                    + ", ".join(abort_without_grid)
                )
        child_roles = sorted(
            {
                Path(container.nodes[child]["submit"]).name
                for child in container.children.get(external_node, set())
                if child in container.nodes
            }
        )
        has_fetch = any(role.startswith("FETCH_") for role in child_roles)
        invocation_nodes = [step[1] for step in invocation]
        boundary = "{} (in {})".format(external_node, container.path)
        if ancestry:
            boundary += " via " + " -> ".join(invocation_nodes[:-1])
        if active_abort_nodes and not has_fetch:
            failures.append("active external abort lacks immediate FETCH child: " + boundary)
        if active_abort_nodes and not reaches_product:
            failures.append("active external abort does not feed terminal product: " + boundary)
        external_reports.append(
            {
                "node": external_node,
                "container": str(container.path),
                "invocation": invocation_nodes,
                "path": str(path),
                "nodes": len(nested.nodes),
                "active_abort_nodes": len(active_abort_nodes),
                "has_fetch_child": has_fetch,
                "feeds_terminal_product": reaches_product,
            }
        )
        # Every node of ``nested`` must complete before this SUBDAG node does,
        # so a boundary declared inside it inherits this node's connection to
        # the top-level product.
        pending.extend(
            (nested, child_node, child_path, reaches_product, invocation)
            for child_node, child_path in external_dags(nested)
        )

    if graph_errors:
        failures.extend("DAG graph integrity: " + item for item in sorted(set(graph_errors)))

    checks.update(
        {
            "nodes": len(dag.nodes),
            "edges": sum(len(items) for items in dag.parents.values()),
            "graph_errors": sorted(set(graph_errors)),
            "role_counts": {role: len(nodes) for role, nodes in sorted(roles.items())},
            "cip_nodes": len(cip_nodes),
            "extrinsic_nodes_missing_from_convert": len(missing_extrinsic),
            "top_level_abort_nodes": len(dag.abort),
            "unsafe_top_level_abort_nodes": unsafe_abort_nodes,
            "external_dags": external_reports,
            "critical_sinks": critical_sinks,
        }
    )
    return {
        "pass": not failures,
        "root": str(root),
        "failures": failures,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory")
    parser.add_argument("--report")
    args = parser.parse_args()
    report = audit_pipeline(args.run_directory)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        Path(args.report).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
