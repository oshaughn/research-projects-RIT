"""pseudo_pipe forwards each --internal-*-request-disk to its own job class."""
import ast
from pathlib import Path
from types import SimpleNamespace

PSEUDO = Path(__file__).resolve().parents[1] / "bin" / "util_RIFT_pseudo_pipe.py"
DISK_OPTS = ("internal_ile_request_disk", "internal_cip_request_disk",
             "internal_general_request_disk")


def _disk_blocks():
    """The module-level `if opts.internal_*_request_disk:` statements, in order."""
    tree = ast.parse(PSEUDO.read_text())
    blocks = [node for node in tree.body if isinstance(node, ast.If)
              and isinstance(node.test, ast.Attribute)
              and node.test.attr in DISK_OPTS]
    assert sorted(b.test.attr for b in blocks) == sorted(DISK_OPTS)
    return ast.Module(body=blocks, type_ignores=[])


def _emitted(**disks):
    opts = SimpleNamespace(**{name: disks.get(name) for name in DISK_OPTS})
    scope = {"opts": opts, "cmd": ""}
    exec(compile(_disk_blocks(), str(PSEUDO), "exec"), scope)
    return scope["cmd"].split()


def test_cip_disk_is_not_the_ile_disk():
    cmd = _emitted(internal_ile_request_disk="4G", internal_cip_request_disk="9G")
    assert cmd[cmd.index("--ile-request-disk") + 1] == "4G"
    assert cmd[cmd.index("--cip-request-disk") + 1] == "9G"


def test_cip_disk_alone_does_not_emit_none():
    cmd = _emitted(internal_cip_request_disk="9G")
    assert "--ile-request-disk" not in cmd
    assert cmd[cmd.index("--cip-request-disk") + 1] == "9G"
