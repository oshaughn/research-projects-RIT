"""Dependency-free HTCondor DAG parser used by build-only CI tests.

Only control-flow directives are parsed. Job-specific ``VARS`` content is
deliberately ignored: this module answers whether the builder connected the
workflow correctly, without importing HTCondor or executing any science job.
"""

import shlex
from collections import defaultdict, deque
from pathlib import Path


NODE_COMMANDS = {"JOB", "FINAL", "SERVICE", "PROVISIONER"}


class Dag:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.nodes = {}
        self.parents = defaultdict(set)
        self.children = defaultdict(set)
        self.abort = {}
        self.retries = {}
        self.scripts = defaultdict(list)
        self.parse_errors = []

    def ancestors(self, node):
        return self._walk(node, self.parents)

    def descendants(self, node):
        return self._walk(node, self.children)

    @staticmethod
    def _walk(node, adjacency):
        seen = set()
        queue = deque(adjacency.get(node, set()))
        while queue:
            item = queue.popleft()
            if item in seen:
                continue
            seen.add(item)
            queue.extend(adjacency.get(item, set()))
        return seen

    def roots(self):
        return sorted(name for name in self.nodes if not self.parents.get(name))

    def sinks(self):
        return sorted(name for name in self.nodes if not self.children.get(name))

    def nodes_for_submit(self, submit_basename):
        return {
            name
            for name, data in self.nodes.items()
            if data.get("submit") and Path(data["submit"]).name == submit_basename
        }


def _tokens(raw, line_number, dag):
    try:
        return shlex.split(raw, comments=True, posix=True)
    except ValueError as exc:
        dag.parse_errors.append("line {}: {}".format(line_number, exc))
        return []


def _declare(dag, name, kind, submit, line_number):
    if name in dag.nodes:
        dag.parse_errors.append("line {}: duplicate node {}".format(line_number, name))
        return
    dag.nodes[name] = {"kind": kind, "submit": submit, "line": line_number}


def parse_dag(path):
    dag = Dag(path)
    for line_number, raw in enumerate(dag.path.read_text(encoding="utf-8").splitlines(), 1):
        fields = _tokens(raw, line_number, dag)
        if not fields:
            continue
        command = fields[0].upper()
        if command in NODE_COMMANDS:
            if len(fields) < 3:
                dag.parse_errors.append("line {}: malformed {}".format(line_number, command))
            else:
                _declare(dag, fields[1], command, fields[2], line_number)
        elif command == "SUBDAG":
            if len(fields) < 4 or fields[1].upper() != "EXTERNAL":
                dag.parse_errors.append("line {}: malformed SUBDAG".format(line_number))
            else:
                _declare(dag, fields[2], "SUBDAG", fields[3], line_number)
        elif command == "SPLICE":
            if len(fields) < 3:
                dag.parse_errors.append("line {}: malformed SPLICE".format(line_number))
            else:
                _declare(dag, fields[1], "SPLICE", fields[2], line_number)
        elif command == "PARENT":
            upper = [field.upper() for field in fields]
            if "CHILD" not in upper:
                dag.parse_errors.append("line {}: PARENT lacks CHILD".format(line_number))
                continue
            split = upper.index("CHILD")
            if split < 2 or split == len(fields) - 1:
                dag.parse_errors.append("line {}: malformed PARENT/CHILD".format(line_number))
                continue
            for parent in fields[1:split]:
                for child in fields[split + 1 :]:
                    dag.parents[child].add(parent)
                    dag.children[parent].add(child)
        elif command == "ABORT-DAG-ON":
            if len(fields) < 3:
                dag.parse_errors.append("line {}: malformed ABORT-DAG-ON".format(line_number))
            else:
                dag.abort[fields[1]] = {
                    "exit": fields[2],
                    "return": fields[4]
                    if len(fields) >= 5 and fields[3].upper() == "RETURN"
                    else None,
                    "line": line_number,
                }
        elif command == "RETRY":
            if len(fields) < 3:
                dag.parse_errors.append("line {}: malformed RETRY".format(line_number))
            else:
                dag.retries[fields[1]] = fields[2:]
        elif command == "SCRIPT":
            # SCRIPT [DEFER status time] PRE|POST|HOLD node executable ...
            upper = [field.upper() for field in fields]
            positions = [i for i, field in enumerate(upper) if field in ("PRE", "POST", "HOLD")]
            if not positions or positions[0] + 1 >= len(fields):
                dag.parse_errors.append("line {}: malformed SCRIPT".format(line_number))
            else:
                position = positions[0]
                dag.scripts[fields[position + 1]].append(
                    {"kind": upper[position], "line": line_number}
                )
    return dag


def validate_dag(dag):
    errors = list(dag.parse_errors)
    declared = set(dag.nodes)
    for child, parents in dag.parents.items():
        if child not in declared:
            errors.append("edge references undefined child {}".format(child))
        for parent in parents:
            if parent not in declared:
                errors.append("edge references undefined parent {}".format(parent))
            if parent == child:
                errors.append("self edge on {}".format(parent))
    for directive, names in (
        ("ABORT-DAG-ON", dag.abort),
        ("RETRY", dag.retries),
        ("SCRIPT", dag.scripts),
    ):
        for name in names:
            if name not in declared:
                errors.append("{} references undefined node {}".format(directive, name))

    indegree = {name: len(dag.parents.get(name, set())) for name in declared}
    queue = deque(name for name, count in indegree.items() if count == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in dag.children.get(node, set()):
            if child not in indegree:
                continue
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(declared):
        errors.append("cycle detected involving {} nodes".format(len(declared) - visited))
    return sorted(set(errors))


def external_dags(dag):
    """Return SUBDAG/SPLICE paths resolved relative to the containing DAG."""
    result = []
    for name, data in dag.nodes.items():
        if data["kind"] not in ("SUBDAG", "SPLICE"):
            continue
        path = Path(data["submit"])
        if not path.is_absolute():
            path = dag.path.parent / path
        result.append((name, path.resolve()))
    return result
