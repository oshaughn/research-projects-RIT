"""Verify pinned dependencies and installed candidate bytes, never claim PyPI qualification."""
import base64
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
import RIFT
expected = set(Path(sys.argv[1]).read_text().splitlines())
actual = set(subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True).splitlines())
rift = {line for line in actual if line.lower().startswith("rift @ ")}
assert len(rift) == 1 and next(iter(rift)).startswith("RIFT @ file:"), rift
assert actual - rift == expected, dict(extra=sorted(actual - rift - expected), missing=sorted(expected - actual))
assert sys.version_info[:3] == (3, 11, 14) and md.version("asimov") == "0.8.0"
assert Path(RIFT.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
dist = md.distribution("RIFT")
verified = 0
executables = {}
for item in dist.files:
    if item.hash:
        digest = base64.urlsafe_b64encode(hashlib.sha256(dist.locate_file(item).read_bytes()).digest()).decode().rstrip("=")
        assert item.hash.mode == "sha256" and item.hash.value == digest
        verified += 1
        if Path(str(item)).name in ("util_RIFT_pseudo_pipe.py", "helper_LDG_Events.py",
                                    "create_event_parameter_pipeline_BasicIteration"):
            executables[Path(str(item)).name] = hashlib.sha256(dist.locate_file(item).read_bytes()).hexdigest()
assert verified >= 102 and len(executables) == 3, executables
print(json.dumps(dict(passed=True, development_only=True, frozen_rc12_qualification=False,
    python=sys.version.split()[0], rift_version_label=md.version("RIFT"), asimov=md.version("asimov"),
    dependency_lock_sha256=hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
    candidate_direct_url=json.loads(dist.read_text("direct_url.json")), record_files_verified=verified, installed_executable_sha256=executables), indent=2))
