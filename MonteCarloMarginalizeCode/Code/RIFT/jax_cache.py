"""Persistent, transferable JAX compilation-cache support for RIFT ILE.

JAX includes compiler options and argument shapes in its cache keys.  RIFT adds
an outer compatibility namespace so cache bundles are never mixed across the
JAX/JAXLIB/backend/device combinations that matter on heterogeneous GPU pools.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


MANIFEST_NAME = "rift-jax-cache-manifest.json"
IMPORT_MANIFEST_NAME = "rift-jax-cache-import.json"
IMPORT_MANIFEST_PREFIX = "rift-jax-cache-import-"
FORMAT_VERSION = 1
MAX_BUNDLE_FILES = 100_000
MAX_BUNDLE_MEMBER_BYTES = 4 * 1024**3
MAX_BUNDLE_TOTAL_BYTES = 16 * 1024**3
MAX_BUNDLE_COMPRESSION_RATIO = 10_000
MAX_MANIFEST_BYTES = 16 * 1024**2
_ACCELERATOR_PLUGIN_PACKAGES = (
    "jax-cuda13-plugin", "jax-cuda12-plugin", "jax-cuda11-plugin",
    "jax-rocm7-plugin", "jax-rocm60-plugin", "jax-rocm-plugin", "jax-metal",
)


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_compatibility(jax_module=None):
    """Return the conservative runtime identity used for cache transfer."""
    if jax_module is None:
        import jax as jax_module
    backend = jax_module.default_backend()
    devices = list(jax_module.devices(backend))
    device = devices[0] if devices else None
    client = getattr(device, "client", None)
    capability = getattr(device, "compute_capability", None)
    if callable(capability):
        capability = capability()
    if isinstance(capability, (tuple, list)):
        capability = ".".join(str(part) for part in capability)
    accelerator_plugins = {
        name: version for name in _ACCELERATOR_PLUGIN_PACKAGES
        if (version := _package_version(name)) is not None
    }
    return {
        "python": platform.python_version(),
        "jax": getattr(jax_module, "__version__", _package_version("jax")),
        "jaxlib": _package_version("jaxlib"),
        "accelerator_plugins": accelerator_plugins,
        "backend": backend,
        "platform_version": str(getattr(client, "platform_version", None)),
        "device_kind": str(getattr(device, "device_kind", None)),
        "compute_capability": capability,
    }


def compatibility_key(compatibility):
    raw = json.dumps(compatibility, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _argv_cache_controls(argv):
    root = None
    disabled = False
    argv = list(argv or ())
    for i, token in enumerate(argv):
        if token == "--no-jax-persistent-cache":
            disabled = True
        elif token.startswith("--jax-cache-dir="):
            root = token.split("=", 1)[1]
        elif token == "--jax-cache-dir" and i + 1 < len(argv):
            root = argv[i + 1]
    return root, disabled


def argv_option(argv, name):
    """Return the last ``--name value``/``--name=value`` occurrence."""
    value = None
    argv = list(argv or ())
    for i, token in enumerate(argv):
        if token.startswith(name + "="):
            value = token.split("=", 1)[1]
        elif token == name and i + 1 < len(argv):
            value = argv[i + 1]
    return value


def default_cache_root():
    explicit = os.environ.get("RIFT_JAX_CACHE_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    scratch = os.environ.get("_CONDOR_SCRATCH_DIR")
    if scratch:
        return Path(scratch) / ".rift_cache" / "jax"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "rift" / "jax"


def _write_manifest(directory, compatibility, extra=None):
    manifest = {
        "format_version": FORMAT_VERSION,
        "compatibility": compatibility,
        "compatibility_key": compatibility_key(compatibility),
    }
    if extra:
        manifest.update(extra)
    return _write_json_atomic(directory / MANIFEST_NAME, manifest)


def _write_import_manifest(directory, compatibility, bundle_manifest):
    """Persist one immutable record per contributing bundle manifest."""
    manifest_raw = json.dumps(
        bundle_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest_digest = hashlib.sha256(manifest_raw).hexdigest()
    record = {
        "format_version": FORMAT_VERSION,
        "compatibility": compatibility,
        "compatibility_key": compatibility_key(compatibility),
        "bundle_manifest_sha256": manifest_digest,
        "imported_profile": bundle_manifest.get("profile"),
        "static_shapes": bundle_manifest.get("static_shapes", {}),
    }
    target = directory / (IMPORT_MANIFEST_PREFIX + manifest_digest + ".json")
    return _write_json_atomic(target, record)


def _is_provenance_file(path):
    """Return whether *path* is runtime/import metadata, not compiler data."""
    return (path.name in (MANIFEST_NAME, IMPORT_MANIFEST_NAME)
            or (path.name.startswith(IMPORT_MANIFEST_PREFIX)
                and path.name.endswith(".json")))


def _write_json_atomic(target, value):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".%s." % target.name, suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return value


def _publish_file_atomic(source, target):
    """Copy one validated entry without exposing a partial target to readers."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".%s." % target.name, suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    try:
        shutil.copy2(source, temporary_name)
        sync_fd = os.open(temporary_name, os.O_RDONLY)
        try:
            os.fsync(sync_fd)
        finally:
            os.close(sync_fd)
        os.replace(temporary_name, target)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def configure_persistent_cache(jax_module, argv=None):
    """Enable RIFT's cache before any ILE JIT is constructed.

    ``--jax-cache-dir`` and ``RIFT_JAX_CACHE_ROOT`` name a cache *root*.  A
    compatibility-keyed child is selected automatically.  The standard
    ``JAX_COMPILATION_CACHE_DIR`` remains supported as an exact expert override.
    """
    cli_root, disabled = _argv_cache_controls(argv)
    if disabled or os.environ.get("RIFT_DISABLE_JAX_CACHE") == "1":
        jax_module.config.update("jax_enable_compilation_cache", False)
        return None

    # runtime_compatibility() PROBES THE DEVICE, and that probe can fail on a
    # perfectly ordinary execute node: jax.default_backend()/jax.devices() force
    # backend init, so unloadable CUDA libraries or a card that is busy for every
    # tenant raise here.  This function runs at driver IMPORT, before the option
    # parser exists, so an escaping exception turns "no usable accelerator" into
    # a driver that cannot even print --help -- a performance optimization
    # failing a scientific run, which is the thing the OSError handler below
    # exists to prevent.  Catch it in the same place and for the same reason.
    try:
        compatibility = runtime_compatibility(jax_module)
    except Exception as exc:
        print("WARNING: disabling JAX persistent cache, cannot identify the "
              "JAX runtime/device: %s" % exc, file=sys.stderr)
        try:
            jax_module.config.update("jax_enable_compilation_cache", False)
        except Exception:
            pass
        return None
    exact = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if exact and not cli_root:
        directory = Path(exact).expanduser()
    else:
        root = Path(cli_root).expanduser() if cli_root else default_cache_root()
        directory = root / compatibility_key(compatibility)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        _write_manifest(directory, compatibility)
    except OSError as exc:
        # A read-only/missing home must not turn a performance optimization into
        # a failed scientific run. Condor normally avoids this via its scratch
        # fallback above; unusual sites can still opt in explicitly.
        print("WARNING: disabling JAX persistent cache: %s" % exc, file=sys.stderr)
        jax_module.config.update("jax_enable_compilation_cache", False)
        return None
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(directory.resolve())
    # JAX 0.9.2's auxiliary per-fusion GPU autotune cache embeds its absolute
    # directory in CompileOptions, but does not exclude that field from the
    # persistent-executable cache key.  A bundle imported under a different
    # absolute root would therefore miss every executable it contains.  Keep
    # the portable executable cache enabled, but disable only that auxiliary
    # path-valued cache.  Deliberately let config.update fail loudly if a future
    # JAX stops accepting a setting it advertises: silently restoring
    # non-portable keys would make a successful-looking transfer useless.
    # Older supported JAX (for example 0.4.24) predates this auxiliary cache and
    # does not advertise the option, so there is nothing path-valued to disable.
    if hasattr(jax_module.config, "jax_persistent_cache_enable_xla_caches"):
        jax_module.config.update("jax_persistent_cache_enable_xla_caches", "")
    jax_module.config.update("jax_enable_compilation_cache", True)
    jax_module.config.update("jax_compilation_cache_dir", str(directory.resolve()))
    return directory.resolve()


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_bundle(cache_dir, output, compatibility, profile=None, static_shapes=None):
    """Create a self-describing zip bundle from an already-warmed cache."""
    cache_dir = Path(cache_dir)
    output = Path(output)
    if not cache_dir.is_dir():
        raise ValueError("cache directory does not exist: %s" % cache_dir)
    files = {}
    total_size = 0
    for path in sorted(cache_dir.rglob("*")):
        is_manifest_temp = (path.name.startswith(".rift-jax-cache-")
                            and path.name.endswith(".tmp"))
        is_provenance = _is_provenance_file(path)
        if path.is_file() and not is_provenance and not is_manifest_temp:
            rel = path.relative_to(cache_dir).as_posix()
            size = path.stat().st_size
            if size > MAX_BUNDLE_MEMBER_BYTES:
                raise ValueError("cache member exceeds the bundle size limit: %s" % rel)
            total_size += size
            if total_size > MAX_BUNDLE_TOTAL_BYTES:
                raise ValueError("cache exceeds the total bundle size limit")
            files[rel] = _file_hash(path)
    if len(files) > MAX_BUNDLE_FILES:
        raise ValueError("cache has too many files to bundle safely")
    manifest = {
        "format_version": FORMAT_VERSION,
        "compatibility": compatibility,
        "compatibility_key": compatibility_key(compatibility),
        "profile": profile,
        "static_shapes": static_shapes or {},
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for rel in files:
            archive.write(cache_dir / rel, "cache/" + rel)
    return manifest


def _validate_member(info, *, manifest=False):
    limit = MAX_MANIFEST_BYTES if manifest else MAX_BUNDLE_MEMBER_BYTES
    if info.file_size > limit:
        raise ValueError("cache bundle member exceeds the size limit: %s" % info.filename)
    if info.file_size and not info.compress_size:
        raise ValueError("cache bundle member has an invalid compressed size: %s" % info.filename)
    if info.compress_size and info.file_size / info.compress_size > MAX_BUNDLE_COMPRESSION_RATIO:
        raise ValueError("cache bundle member exceeds the compression-ratio limit: %s" % info.filename)


def _read_limited(archive, info, limit):
    with archive.open(info, "r") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("cache bundle member exceeds the size limit: %s" % info.filename)
    return data


def import_bundle(bundle, cache_root, compatibility, expected_profile=None,
                  destination=None):
    """Validate and merge a bundle into this runtime's cache namespace."""
    bundle = Path(bundle)
    with zipfile.ZipFile(bundle, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("cache bundle contains duplicate archive members")
        if len(names) > MAX_BUNDLE_FILES + 1:
            raise ValueError("cache bundle contains too many archive members")
        if MANIFEST_NAME not in names:
            raise ValueError("bundle has no %s" % MANIFEST_NAME)
        info_by_name = {info.filename: info for info in infos}
        _validate_member(info_by_name[MANIFEST_NAME], manifest=True)
        manifest = json.loads(_read_limited(
            archive, info_by_name[MANIFEST_NAME], MAX_MANIFEST_BYTES))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported cache bundle format")
        if manifest.get("compatibility") != compatibility:
            raise ValueError("cache bundle is incompatible with this JAX runtime/device")
        if expected_profile is not None and manifest.get("profile") != expected_profile:
            raise ValueError("cache bundle warmup profile does not match --expect-profile")
        declared = manifest.get("files", {})
        expected_names = {"cache/" + rel for rel in declared}
        actual_names = {name for name in names if name.startswith("cache/") and not name.endswith("/")}
        if actual_names != expected_names:
            raise ValueError("cache bundle contents do not match its manifest")
        if set(names) != expected_names | {MANIFEST_NAME}:
            raise ValueError("cache bundle contains unexpected archive members")
        total_size = 0
        for name in expected_names:
            info = info_by_name[name]
            _validate_member(info)
            total_size += info.file_size
            if total_size > MAX_BUNDLE_TOTAL_BYTES:
                raise ValueError("cache bundle exceeds the total size limit")
        with tempfile.TemporaryDirectory(prefix="rift-jax-cache-") as temp:
            temp_root = Path(temp)
            for rel, expected_hash in declared.items():
                pure = PurePosixPath(rel)
                if pure.is_absolute() or ".." in pure.parts:
                    raise ValueError("unsafe cache bundle path: %s" % rel)
                target = temp_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                written = 0
                with archive.open(info_by_name["cache/" + rel], "r") as source, target.open("wb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        written += len(block)
                        if written > MAX_BUNDLE_MEMBER_BYTES:
                            raise ValueError("cache bundle member exceeds the size limit: %s" % rel)
                        digest.update(block)
                        output.write(block)
                if digest.hexdigest() != expected_hash:
                    raise ValueError("cache bundle checksum mismatch: %s" % rel)
            destination = (Path(destination).expanduser() if destination is not None
                           else Path(cache_root).expanduser() / compatibility_key(compatibility))
            destination.mkdir(parents=True, exist_ok=True)
            for source in temp_root.rglob("*"):
                if source.is_file():
                    target = destination / source.relative_to(temp_root)
                    _publish_file_atomic(source, target)
            _write_import_manifest(destination, compatibility, manifest)
    return destination
