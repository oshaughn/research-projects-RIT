"""Shared helpers for container survey_scan tooling."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ContainerEntry:
    label: str
    image: str
    cuda_capability_min: float | None
    cuda_capability_max: float | None
    note: str = ""


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    if value in ("", "null", "None", "~"):
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_simple_yaml_manifest(text: str) -> dict[str, Any]:
    """Parse the simple RIFT container-family YAML schema without dependencies.

    This is not a general YAML parser. It handles the schema emitted by
    containers/build_family.sh and the CIT build kit: top-level scalars plus a
    `containers:` list of scalar mappings.
    """

    result: dict[str, Any] = {"containers": []}
    in_containers = False
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.strip() == "containers:":
            in_containers = True
            continue
        if not in_containers:
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = _coerce_scalar(value)
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            if current is not None:
                result["containers"].append(current)
            current = {}
            stripped = stripped[2:]
            if stripped and ":" in stripped:
                key, value = stripped.split(":", 1)
                current[key.strip()] = _coerce_scalar(value)
        elif current is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key.strip()] = _coerce_scalar(value)
    if current is not None:
        result["containers"].append(current)
    return result


def load_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    return _parse_simple_yaml_manifest(text)


def manifest_entries(path: Path) -> list[ContainerEntry]:
    manifest = load_manifest(path)
    entries = []
    for item in manifest.get("containers", []):
        entries.append(
            ContainerEntry(
                label=str(item.get("label", "")),
                image=str(item.get("image", "")),
                cuda_capability_min=_as_float_or_none(
                    item.get("cuda_capability_min")
                ),
                cuda_capability_max=_as_float_or_none(
                    item.get("cuda_capability_max")
                ),
                note=str(item.get("note", "") or ""),
            )
        )
    return entries


def _as_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_name(value: str) -> str:
    value = value.strip() or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def rel_or_abs(path: Path) -> str:
    try:
        return os.path.relpath(path, Path.cwd())
    except ValueError:
        return str(path)
