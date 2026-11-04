import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import threading
import zipfile
from pathlib import Path

import pytest

from RIFT import jax_cache as cache


COMPAT = {
    "python": "3.11.9", "jax": "0.4.35", "jaxlib": "0.4.35",
    "accelerator_plugins": {"jax-cuda12-plugin": "0.4.35"},
    "backend": "gpu", "platform_version": "CUDA 12.4",
    "device_kind": "NVIDIA A30", "compute_capability": "8.0",
}


class _Config:
    jax_persistent_cache_enable_xla_caches = None

    def __init__(self):
        self.updates = []

    def update(self, name, value):
        self.updates.append((name, value))


class _Jax:
    config = _Config()


def test_configure_uses_compatibility_namespace(tmp_path, monkeypatch):
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setattr(cache, "runtime_compatibility", lambda unused: COMPAT)
    fake = _Jax()
    selected = cache.configure_persistent_cache(fake, ["--jax-cache-dir", str(tmp_path)])
    assert selected == (tmp_path / cache.compatibility_key(COMPAT)).resolve()
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == str(selected)
    manifest = json.loads((selected / cache.MANIFEST_NAME).read_text())
    assert manifest["compatibility"] == COMPAT
    assert ("jax_persistent_cache_enable_xla_caches", "") in fake.config.updates
    assert ("jax_enable_compilation_cache", True) in fake.config.updates


def test_configure_supports_jax_before_auxiliary_xla_caches(tmp_path, monkeypatch):
    """JAX 0.4.24 has executable caching but not the path-valued XLA option."""
    class LegacyConfig:
        def __init__(self):
            self.updates = []

        def update(self, name, value):
            if name == "jax_persistent_cache_enable_xla_caches":
                raise AttributeError("Unrecognized config option")
            self.updates.append((name, value))

    class LegacyJax:
        config = LegacyConfig()

    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setattr(cache, "runtime_compatibility", lambda unused: COMPAT)
    selected = cache.configure_persistent_cache(
        LegacyJax(), ["--jax-cache-dir", str(tmp_path)])
    assert selected == (tmp_path / cache.compatibility_key(COMPAT)).resolve()
    assert ("jax_enable_compilation_cache", True) in LegacyJax.config.updates


@pytest.mark.parametrize("argv,env", [
    (["--no-jax-persistent-cache"], None),
    ([], "RIFT_DISABLE_JAX_CACHE"),
])
def test_disable_does_not_create_cache(tmp_path, monkeypatch, argv, env):
    """Each opt-out separately, against a fake that COULD have succeeded.

    A mutation sweep replaced the whole disable condition with ``False`` and
    this test still passed.  The reason is that the bare ``_Jax()`` fake has no
    default_backend/devices, so with the early return gone the run instead hit
    the device-probe fail-open handler -- which ALSO returns None, ALSO records
    jax_enable_compilation_cache=False, and ALSO leaves tmp_path empty.  Every
    observable the test checked was reproduced by a different code path, so it
    was pinning nothing.  Stubbing runtime_compatibility gives the fake a
    working probe, and naming a cache root means a non-disabled run must create
    a directory.  Both halves of the condition get their own case, because the
    sweep flipped them together.
    """
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setattr(cache, "runtime_compatibility", lambda unused: COMPAT)
    if env:
        monkeypatch.setenv(env, "1")
    fake = _Jax()
    assert cache.configure_persistent_cache(
        fake, argv + ["--jax-cache-dir", str(tmp_path)]) is None
    assert ("jax_enable_compilation_cache", False) in fake.config.updates
    assert not list(tmp_path.iterdir()), (
        "a disabled cache must not create its namespace directory")


@pytest.mark.parametrize("spelling", ["separate", "equals"])
def test_cache_root_is_read_from_either_cli_spelling(tmp_path, monkeypatch,
                                                     spelling):
    """optparse accepts --jax-cache-dir X and --jax-cache-dir=X; so must this.

    configure_persistent_cache scans sys.argv itself, before the option parser
    exists, so the two spellings are two separate branches.  Only the separate
    form was covered: a mutation sweep deleted the "=" branch and the whole
    file still passed.  With it gone, `--jax-cache-dir=/shared/cache` silently
    selects the DEFAULT root instead -- no error, and no reuse of the shared
    cache the operator asked for.
    """
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("RIFT_JAX_CACHE_ROOT", raising=False)
    monkeypatch.setattr(cache, "runtime_compatibility", lambda unused: COMPAT)
    # a default root that is NOT tmp_path, so falling back is visible
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "default"))
    argv = ([str(tmp_path / "asked")] if spelling == "separate" else [])
    argv = (["--jax-cache-dir"] + argv if spelling == "separate"
            else ["--jax-cache-dir=" + str(tmp_path / "asked")])
    selected = cache.configure_persistent_cache(_Jax(), argv)
    assert selected == (
        tmp_path / "asked" / cache.compatibility_key(COMPAT)).resolve(), selected
    assert not (tmp_path / "default").exists(), (
        "the requested root was ignored and the default was used")


def test_condor_scratch_is_the_default_root(tmp_path, monkeypatch):
    monkeypatch.delenv("RIFT_JAX_CACHE_ROOT", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("_CONDOR_SCRATCH_DIR", str(tmp_path))
    assert cache.default_cache_root() == tmp_path / ".rift_cache" / "jax"


def test_bundle_option_scan_uses_last_cli_value():
    assert cache.argv_option(["--jax-cache-bundle", "old.zip",
                              "--jax-cache-bundle=new.zip"],
                             "--jax-cache-bundle") == "new.zip"


def test_cache_cli_help_does_not_require_optional_jax():
    script = Path(__file__).resolve().parents[2] / "bin" / "rift_jax_cache"
    code = textwrap.dedent("""
        import runpy
        import sys
        sys.modules["jax"] = None
        sys.argv = ["rift_jax_cache", "--help"]
        runpy.run_path(%r, run_name="__main__")
    """ % str(script))
    completed = subprocess.run([sys.executable, "-c", code], check=False,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert "Inspect, export, and safely import" in completed.stdout


def test_unwritable_cache_disables_without_failing(monkeypatch, capsys):
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setattr(cache, "runtime_compatibility", lambda unused: COMPAT)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read only")))
    fake = _Jax()
    assert cache.configure_persistent_cache(fake, ["--jax-cache-dir", "/unwritable"]) is None
    assert "disabling JAX persistent cache" in capsys.readouterr().err
    assert ("jax_enable_compilation_cache", False) in fake.config.updates


def test_an_unusable_backend_disables_the_cache_instead_of_raising(tmp_path,
                                                                   monkeypatch,
                                                                   capsys):
    """A device probe that fails must not kill the driver.

    runtime_compatibility() calls default_backend()/devices(), which force
    backend init; unloadable CUDA libraries, or a card busy for every tenant,
    raise there.  configure_persistent_cache runs at driver IMPORT, before the
    option parser exists, so an escaping exception makes the ILE unable even to
    print --help.  Verified against the real jaxlib: with JAX_PLATFORMS=cuda on
    a CPU-only host, jax.devices() raised and this function propagated it.
    """
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)

    def _explode(unused):
        raise RuntimeError("cuInit(0) failed: Unknown CUDA error 303")

    monkeypatch.setattr(cache, "runtime_compatibility", _explode)
    fake = _Jax()
    assert cache.configure_persistent_cache(
        fake, ["--jax-cache-dir", str(tmp_path)]) is None
    assert "disabling JAX persistent cache" in capsys.readouterr().err
    assert ("jax_enable_compilation_cache", False) in fake.config.updates
    assert not list(tmp_path.iterdir()), "a failed probe must not create a cache"


def test_manifest_updates_use_unique_atomic_temporary_files(tmp_path, monkeypatch):
    sources = []
    lock = threading.Lock()
    real_replace = os.replace

    def recording_replace(source, target):
        with lock:
            sources.append(str(source))
        real_replace(source, target)

    monkeypatch.setattr(cache.os, "replace", recording_replace)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: cache._write_manifest(tmp_path, COMPAT, {"writer": i}),
                      range(24)))
    assert len(sources) == 24
    assert len(set(sources)) == 24
    assert json.loads((tmp_path / cache.MANIFEST_NAME).read_text())["writer"] in range(24)


def test_runtime_fingerprint_records_current_accelerator_plugins(monkeypatch):
    class Client:
        platform_version = "PJRT CUDA 13"

    class Device:
        client = Client()
        device_kind = "Future GPU"
        compute_capability = (13, 0)

    class Jax:
        __version__ = "1.0"

        @staticmethod
        def default_backend():
            return "gpu"

        @staticmethod
        def devices(backend):
            assert backend == "gpu"
            return [Device()]

    versions = {"jaxlib": "1.0", "jax-cuda13-plugin": "1.0"}
    monkeypatch.setattr(cache, "_package_version", versions.get)
    identity = cache.runtime_compatibility(Jax)
    assert identity["accelerator_plugins"] == {"jax-cuda13-plugin": "1.0"}
    assert identity["compute_capability"] == "13.0"


def test_bundle_round_trip_and_profile_guard(tmp_path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "compiled-entry").write_bytes(b"compiled")
    bundle = tmp_path / "warm.zip"
    cache.export_bundle(source, bundle, COMPAT, "o4-laplace", {"n_chunk": 8000})
    destination = cache.import_bundle(bundle, tmp_path / "target", COMPAT, "o4-laplace")
    assert (destination / "nested" / "compiled-entry").read_bytes() == b"compiled"
    records = sorted(destination.glob(cache.IMPORT_MANIFEST_PREFIX + "*.json"))
    assert len(records) == 1
    manifest = json.loads(records[0].read_text())
    assert manifest["static_shapes"] == {"n_chunk": 8000}
    cache._write_manifest(destination, COMPAT)
    assert json.loads(records[0].read_text()) == manifest

    # Compatible bundles merge compiler entries, so provenance must retain
    # every contributor rather than silently replacing the previous profile.
    (source / "nested" / "second-entry").write_bytes(b"second")
    second_bundle = tmp_path / "second.zip"
    cache.export_bundle(source, second_bundle, COMPAT, "o4-exact",
                        {"n_chunk": 1000})
    cache.import_bundle(second_bundle, tmp_path / "target", COMPAT,
                        "o4-exact")
    records = sorted(destination.glob(cache.IMPORT_MANIFEST_PREFIX + "*.json"))
    assert len(records) == 2
    imported_profiles = {
        json.loads(path.read_text())["imported_profile"] for path in records}
    assert imported_profiles == {"o4-laplace", "o4-exact"}
    cache.import_bundle(bundle, tmp_path / "target", COMPAT, "o4-laplace")
    assert len(list(destination.glob(
        cache.IMPORT_MANIFEST_PREFIX + "*.json"))) == 2

    # A cache warmed by an older PR may still contain the former singular
    # import record. Neither legacy nor current provenance is compiler data.
    (destination / cache.IMPORT_MANIFEST_NAME).write_text("{}\n")
    reexport = tmp_path / "reexport.zip"
    reexport_manifest = cache.export_bundle(destination, reexport, COMPAT)
    exported_names = {Path(rel).name for rel in reexport_manifest["files"]}
    assert cache.MANIFEST_NAME not in exported_names
    assert cache.IMPORT_MANIFEST_NAME not in exported_names
    assert not any(name.startswith(cache.IMPORT_MANIFEST_PREFIX)
                   for name in exported_names)
    with pytest.raises(ValueError, match="profile"):
        cache.import_bundle(bundle, tmp_path / "wrong-profile", COMPAT, "other")

    exact = tmp_path / "standard-jax-exact-dir"
    imported = cache.import_bundle(bundle, tmp_path / "ignored-root", COMPAT,
                                   destination=exact)
    assert imported == exact
    assert (exact / "nested" / "compiled-entry").read_bytes() == b"compiled"


def test_import_publishes_cache_entries_atomically(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "entry").write_bytes(b"compiled")
    bundle = tmp_path / "warm.zip"
    cache.export_bundle(source, bundle, COMPAT)
    replacements = []
    real_replace = os.replace

    def recording_replace(temporary, target):
        replacements.append((Path(temporary), Path(target)))
        real_replace(temporary, target)

    monkeypatch.setattr(cache.os, "replace", recording_replace)
    destination = cache.import_bundle(bundle, tmp_path / "target", COMPAT)
    entry_publications = [(temporary, target) for temporary, target in replacements
                          if target == destination / "entry"]
    assert len(entry_publications) == 1
    temporary, target = entry_publications[0]
    assert temporary.parent == target.parent
    assert temporary != target


def test_bundle_rejects_runtime_mismatch_and_tampering(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "entry").write_bytes(b"one")
    bundle = tmp_path / "warm.zip"
    cache.export_bundle(source, bundle, COMPAT)
    mismatch = dict(COMPAT, jaxlib="0.5.0")
    with pytest.raises(ValueError, match="incompatible"):
        cache.import_bundle(bundle, tmp_path / "mismatch", mismatch)

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(bundle) as old, zipfile.ZipFile(tampered, "w") as new:
        for name in old.namelist():
            new.writestr(name, b"two" if name == "cache/entry" else old.read(name))
    with pytest.raises(ValueError, match="checksum"):
        cache.import_bundle(tampered, tmp_path / "tampered", COMPAT)

    unexpected = tmp_path / "unexpected.zip"
    with zipfile.ZipFile(bundle) as old, zipfile.ZipFile(unexpected, "w") as new:
        for name in old.namelist():
            new.writestr(name, old.read(name))
        new.writestr("unrelated", b"surprise")
    with pytest.raises(ValueError, match="unexpected"):
        cache.import_bundle(unexpected, tmp_path / "unexpected", COMPAT)


def test_bundle_rejects_oversized_or_overcompressed_members(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "entry").write_bytes(b"0" * 10_000)
    bundle = tmp_path / "warm.zip"
    cache.export_bundle(source, bundle, COMPAT)

    monkeypatch.setattr(cache, "MAX_BUNDLE_MEMBER_BYTES", 100)
    with pytest.raises(ValueError, match="size limit"):
        cache.import_bundle(bundle, tmp_path / "oversized", COMPAT)
    monkeypatch.setattr(cache, "MAX_BUNDLE_MEMBER_BYTES", 20_000)
    monkeypatch.setattr(cache, "MAX_BUNDLE_COMPRESSION_RATIO", 2)
    with pytest.raises(ValueError, match="compression-ratio"):
        cache.import_bundle(bundle, tmp_path / "overcompressed", COMPAT)


def test_real_jax_cache_reused_across_fresh_processes(tmp_path):
    pytest.importorskip("jax")
    code = textwrap.dedent("""
        import jax
        import jax.numpy as jnp
        from RIFT.jax_cache import configure_persistent_cache
        configure_persistent_cache(jax, ["--jax-cache-dir", r"%s"])
        @jax.jit
        def work(x):
            for _ in range(8):
                x = jnp.sin(x @ x + 0.01)
            return x.sum()
        print(float(work(jnp.eye(64)).block_until_ready()))
    """ % tmp_path)
    env = os.environ.copy()
    env.update({
        "JAX_PLATFORMS": "cpu",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1",
    })

    def run():
        subprocess.run([sys.executable, "-c", code], env=env, check=True,
                       capture_output=True, text=True, timeout=120)

    def entries():
        return {
            str(path.relative_to(tmp_path)): (hashlib.sha256(path.read_bytes()).hexdigest(),
                                              path.stat().st_mtime_ns)
            for path in tmp_path.rglob("*")
            if path.is_file() and path.name != cache.MANIFEST_NAME
            and not path.name.endswith(".tmp")
        }

    run()
    first = entries()
    assert first, "the first fresh process did not populate JAX's persistent cache"
    run()
    assert entries() == first, "the second fresh process recompiled or rewrote cache entries"


def test_real_jax_cache_bundle_reused_from_different_absolute_root(tmp_path):
    """A transferred executable must not be keyed by its original cache path."""
    jax = pytest.importorskip("jax")
    source_root = tmp_path / "producer" / "cache"
    target_root = tmp_path / "consumer-at-a-different-path" / "cache"
    code = textwrap.dedent("""
        import jax
        import jax.numpy as jnp
        from RIFT.jax_cache import configure_persistent_cache
        configure_persistent_cache(jax, ["--jax-cache-dir", r"%s"])
        @jax.jit
        def transferred_work(x):
            for _ in range(8):
                x = jnp.sin(x @ x + 0.01)
            return x.sum()
        print(float(transferred_work(jnp.eye(64)).block_until_ready()))
    """)
    env = os.environ.copy()
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env.update({
        "JAX_PLATFORMS": "cpu",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
        "JAX_DEBUG_LOG_MODULES": "jax._src.compiler,jax._src.compilation_cache",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1",
    })

    producer = subprocess.run(
        [sys.executable, "-c", code % source_root], env=env, check=True,
        capture_output=True, text=True, timeout=120)
    compatibility = cache.runtime_compatibility(jax)
    source = source_root / cache.compatibility_key(compatibility)
    bundle = tmp_path / "portable.zip"
    cache.export_bundle(source, bundle, compatibility, "different-root-test")
    target = cache.import_bundle(bundle, target_root, compatibility,
                                 "different-root-test")

    def entries():
        return {
            str(path.relative_to(target)): (hashlib.sha256(path.read_bytes()).hexdigest(),
                                            path.stat().st_mtime_ns)
            for path in target.rglob("*")
            if path.is_file() and not cache._is_provenance_file(path)
            and not path.name.endswith(".tmp")
        }

    imported = entries()
    assert imported, "producer did not create any persistent executable entry"
    consumer = subprocess.run(
        [sys.executable, "-c", code % target_root], env=env, check=True,
        capture_output=True, text=True, timeout=120)
    assert consumer.stdout == producer.stdout
    # JAX publishes an executable cache entry atomically after compilation.  A
    # fresh compile would therefore replace it and change its mtime; preserving
    # every imported byte and mtime pins an actual persistent-cache load without
    # depending on JAX's version-specific debug-log formatting.
    assert entries() == imported, "consumer recompiled after cache-root transfer"


@pytest.mark.parametrize("scheme", ["exact", "laplace"])
def test_angle_batched_kernel_persists_without_host_effects(tmp_path, scheme):
    """Pin both real anglemarg graphs, not a toy matmul cache entry.

    JAX refuses to persist any graph containing debug callbacks.  This test
    executes the shipped exact coefficient/reconstruction/scan kernel in two
    fresh processes and requires its named cache entry to survive unchanged;
    reintroducing the former amplitude callback therefore fails behaviorally.
    """
    pytest.importorskip("jax")
    test_dir = Path(__file__).resolve().parent
    code = textwrap.dedent("""
        import sys
        sys.path.insert(0, r"%s")
        import jax
        import jax.numpy as jnp
        from RIFT.jax_cache import configure_persistent_cache
        configure_persistent_cache(jax, ["--jax-cache-dir", r"%s"])
        from test_angle_marg_exact import make_synth, _dist_grid, RA, DEC, INCL, INTERP
        from RIFT.likelihood.jax_ile import anglemarg as AM
        data = make_synth(npts=16)
        xg, lwg = _dist_grid(data, n=16)
        if %r == "exact":
            @jax.jit
            def persisted_work(ra, dec, incl):
                return AM.fused_log_likelihood_distphipsimarg_exact(
                    data, ra, dec, incl, xg, lwg, interp=INTERP,
                    amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE,
                    dense_chunk=8, grid_block=8, return_amp=True)
        else:
            @jax.jit
            def persisted_work(ra, dec, incl):
                return AM.fused_log_likelihood_distphipsimarg_laplace(
                    data, ra, dec, incl, xg, lwg, interp=INTERP,
                    amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE,
                    phi_chunk=8, dist_block=8, return_amp=True)
        value, amp = persisted_work(jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL))
        print(float(value.block_until_ready()[0]), float(amp.block_until_ready()))
    """ % (test_dir, tmp_path, scheme))
    env = os.environ.copy()
    env.update({
        "JAX_PLATFORMS": "cpu",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
        "JAX_DEBUG_LOG_MODULES": "jax._src.compiler,jax._src.compilation_cache",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1",
    })

    def run():
        return subprocess.run([sys.executable, "-c", code], env=env, check=True,
                              capture_output=True, text=True, timeout=180)

    def persisted_entries():
        return {
            str(path.relative_to(tmp_path)): (hashlib.sha256(path.read_bytes()).hexdigest(),
                                              path.stat().st_mtime_ns)
            for path in tmp_path.rglob("*")
            if path.is_file() and "jit_persisted_work-" in path.name
        }

    first_run = run()
    assert "because it uses host callbacks" not in first_run.stderr
    first = persisted_entries()
    assert first, "the shipped %s-angle batch graph was not persisted" % scheme
    second_run = run()
    assert "because it uses host callbacks" not in second_run.stderr
    assert persisted_entries() == first, (
        "fresh-process %s kernel cache entry changed" % scheme)


# ---------------------------------------------------------------------------
# Bundle-validation guards.  Every test below was added because the guard it
# covers SURVIVED a mutation sweep: its condition could be replaced with
# ``False`` and the whole file still passed.  A limit nothing reaches is not a
# limit, and this module's whole job is refusing a bundle it should not trust.
# ---------------------------------------------------------------------------


def _one_entry_bundle(tmp_path, name="warm.zip", payload=b"compiled"):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    (source / "entry").write_bytes(payload)
    bundle = tmp_path / name
    cache.export_bundle(source, bundle, COMPAT)
    return source, bundle


def _rebuild(bundle, target, *, manifest=None, members=None, drop=()):
    """Write a modified copy of *bundle*: patched manifest, extra/dropped members."""
    with zipfile.ZipFile(bundle) as old:
        original = json.loads(old.read(cache.MANIFEST_NAME))
        data = {n: old.read(n) for n in old.namelist()}
    if manifest is not None:
        original = manifest(original)
    with zipfile.ZipFile(target, "w") as new:
        new.writestr(cache.MANIFEST_NAME,
                     json.dumps(original, indent=2, sort_keys=True) + "\n")
        for name, blob in data.items():
            if name == cache.MANIFEST_NAME or name in drop:
                continue
            new.writestr(name, blob)
        for name, blob in (members or {}).items():
            new.writestr(name, blob)
    return target


def test_import_refuses_a_member_path_escaping_the_cache(tmp_path):
    """A '..' member must be refused, not written outside the destination.

    The manifest is what names the files, so a hostile bundle controls those
    strings.  Without the traversal guard ``temp_root / rel`` resolves above
    the extraction directory and the write lands wherever the relative path
    points -- while the publication walk, which only rglobs INSIDE temp_root,
    never sees the file and reports nothing.
    """
    _, bundle = _one_entry_bundle(tmp_path)
    escaped = tmp_path / "escape.zip"
    with zipfile.ZipFile(bundle) as old:
        manifest = json.loads(old.read(cache.MANIFEST_NAME))
        blob = old.read("cache/entry")
    digest = manifest["files"].pop("entry")
    manifest["files"]["../../escaped-entry"] = digest
    with zipfile.ZipFile(escaped, "w") as new:
        new.writestr(cache.MANIFEST_NAME,
                     json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        new.writestr("cache/../../escaped-entry", blob)
    with pytest.raises(ValueError, match="unsafe"):
        cache.import_bundle(escaped, tmp_path / "target", COMPAT)
    assert not (tmp_path.parent / "escaped-entry").exists()

    absolute = tmp_path / "absolute.zip"
    manifest["files"] = {"/etc/escaped": digest}
    with zipfile.ZipFile(absolute, "w") as new:
        new.writestr(cache.MANIFEST_NAME,
                     json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        new.writestr("cache//etc/escaped", blob)
    with pytest.raises(ValueError, match="unsafe"):
        cache.import_bundle(absolute, tmp_path / "target-abs", COMPAT)


def test_a_member_declaring_zero_compressed_size_is_refused(tmp_path):
    """A member claiming N uncompressed bytes in 0 compressed bytes.

    Reachability, stated because it decides how much this guard is worth: the
    ratio check below it short-circuits on compress_size == 0, so removing this
    one raises no ZeroDivisionError, and such a member then fails the checksum
    during extraction instead.  It is defence in depth -- refuse at the header,
    with a message naming the reason, rather than reporting a checksum failure
    for an archive whose real defect is a lying header.  A mutation sweep found
    it survived: nothing reached it at all.

    Unit-level on purpose.  zipfile writes consistent sizes, so producing this
    member end to end means forging central-directory fields; the guard is a
    pure predicate on ZipInfo, so drive it directly and say so.
    """
    info = zipfile.ZipInfo("cache/entry")
    info.file_size = 4096
    info.compress_size = 0
    with pytest.raises(ValueError, match="invalid compressed size"):
        cache._validate_member(info)

    # and the honest member with the same declared size passes
    info.compress_size = 4096
    cache._validate_member(info)


def test_import_refuses_an_unknown_format_version(tmp_path):
    """A future bundle layout must fail closed, not be read with today's rules."""
    _, bundle = _one_entry_bundle(tmp_path)
    future = _rebuild(bundle, tmp_path / "future.zip",
                      manifest=lambda m: dict(m, format_version=
                                              cache.FORMAT_VERSION + 1))
    with pytest.raises(ValueError, match="unsupported cache bundle format"):
        cache.import_bundle(future, tmp_path / "target", COMPAT)


def test_import_refuses_a_bundle_with_no_manifest(tmp_path):
    """Without the manifest there is nothing to check compatibility against."""
    _, bundle = _one_entry_bundle(tmp_path)
    headless = tmp_path / "headless.zip"
    with zipfile.ZipFile(bundle) as old, zipfile.ZipFile(headless, "w") as new:
        for name in old.namelist():
            if name != cache.MANIFEST_NAME:
                new.writestr(name, old.read(name))
    with pytest.raises(ValueError, match=cache.MANIFEST_NAME):
        cache.import_bundle(headless, tmp_path / "target", COMPAT)


def test_import_refuses_duplicate_archive_members(tmp_path):
    """Two members with one name: readers disagree about which is the content.

    zipfile resolves a duplicate name to the LAST entry, while the checksum
    walk and the name-set comparison both see the name only once, so a
    duplicate is exactly how a validated bundle and an extracted bundle come
    apart.
    """
    _, bundle = _one_entry_bundle(tmp_path)
    duplicated = tmp_path / "duplicated.zip"
    with zipfile.ZipFile(bundle) as old:
        data = {n: old.read(n) for n in old.namelist()}
    with zipfile.ZipFile(duplicated, "w") as new:
        for name, blob in data.items():
            new.writestr(name, blob)
        new.writestr("cache/entry", b"second copy")
    with pytest.raises(ValueError, match="duplicate"):
        cache.import_bundle(duplicated, tmp_path / "target", COMPAT)


def test_import_refuses_a_manifest_naming_absent_members(tmp_path):
    """Declared-but-missing is the mirror of the extra-member case.

    The suite already covered an UNEXPECTED member; a manifest that promises a
    file the archive does not carry took the same branch's other side, and
    nothing exercised it.
    """
    source = tmp_path / "source"
    source.mkdir()
    (source / "one").write_bytes(b"a")
    (source / "two").write_bytes(b"b")
    bundle = tmp_path / "pair.zip"
    cache.export_bundle(source, bundle, COMPAT)
    truncated = _rebuild(bundle, tmp_path / "truncated.zip",
                         drop=("cache/two",))
    with pytest.raises(ValueError, match="do not match its manifest"):
        cache.import_bundle(truncated, tmp_path / "target", COMPAT)


def test_import_bounds_member_count_and_total_size(tmp_path, monkeypatch):
    """Both import-side aggregate limits, each on its own.

    They are separate guards with separate messages, and neither was reached:
    the file-count ceiling is 100k and the total-size ceiling 16 GiB, so no
    honest fixture gets near either.  Lower the constants instead of building
    a hostile archive.
    """
    source = tmp_path / "source"
    source.mkdir()
    for i in range(4):
        (source / ("entry%d" % i)).write_bytes(b"0" * 64)
    bundle = tmp_path / "many.zip"
    cache.export_bundle(source, bundle, COMPAT)

    monkeypatch.setattr(cache, "MAX_BUNDLE_FILES", 2)
    with pytest.raises(ValueError, match="too many archive members"):
        cache.import_bundle(bundle, tmp_path / "count", COMPAT)

    monkeypatch.setattr(cache, "MAX_BUNDLE_FILES", 100_000)
    monkeypatch.setattr(cache, "MAX_BUNDLE_TOTAL_BYTES", 100)
    with pytest.raises(ValueError, match="total size limit"):
        cache.import_bundle(bundle, tmp_path / "total", COMPAT)


def test_import_member_size_is_checked_before_and_during_extraction(tmp_path,
                                                                    monkeypatch):
    """The declared size and the streamed size are two guards, not one.

    Each masked the other in the sweep: defeating either alone still raised
    "size limit" from its partner, so both read as covered while neither was.
    The header check refuses a member whose DECLARED size is too large; the
    streaming check refuses one that lies about it and keeps producing bytes.
    """
    _, bundle = _one_entry_bundle(tmp_path, payload=b"0" * 4_000)

    # Header guard alone.  Matching "size limit" is NOT enough to pin it: the
    # streaming guard raises the same message, so that assertion passes with
    # this guard deleted.  What only the header guard can do is refuse BEFORE
    # any extraction begins, so make reaching extraction an error.
    monkeypatch.setattr(cache, "MAX_BUNDLE_MEMBER_BYTES", 100)

    class _ExtractionReached(Exception):
        pass

    def _no_extraction(*args, **kwargs):
        raise _ExtractionReached("import began extracting an oversized member")

    monkeypatch.setattr(cache.tempfile, "TemporaryDirectory", _no_extraction)
    with pytest.raises(ValueError, match="size limit"):
        cache.import_bundle(bundle, tmp_path / "declared", COMPAT)
    monkeypatch.undo()

    # Streaming guard alone: headers pass, extraction must still stop.  A
    # ZipInfo reporting a small file_size passes _validate_member, so only the
    # byte counter in the extraction loop can catch the real length.
    monkeypatch.setattr(cache, "MAX_BUNDLE_MEMBER_BYTES", 1_000)
    real_validate = cache._validate_member
    monkeypatch.setattr(cache, "_validate_member",
                        lambda info, **kw: None if not kw else
                        real_validate(info, **kw))
    with pytest.raises(ValueError, match="size limit"):
        cache.import_bundle(bundle, tmp_path / "streamed", COMPAT)


def test_export_refuses_a_cache_too_large_or_too_numerous_to_bundle(tmp_path,
                                                                    monkeypatch):
    """The export-side ceilings, which no fixture came near either.

    Export builds the bundle from a directory this process already trusts, so
    these are resource guards rather than security ones -- but an unbounded
    export is how a 16 GiB cache becomes an OOM on a submit node.
    """
    source = tmp_path / "source"
    source.mkdir()
    for i in range(3):
        (source / ("entry%d" % i)).write_bytes(b"0" * 512)

    monkeypatch.setattr(cache, "MAX_BUNDLE_MEMBER_BYTES", 100)
    with pytest.raises(ValueError, match="member exceeds the bundle size"):
        cache.export_bundle(source, tmp_path / "a.zip", COMPAT)

    monkeypatch.setattr(cache, "MAX_BUNDLE_MEMBER_BYTES", 4 * 1024**3)
    monkeypatch.setattr(cache, "MAX_BUNDLE_TOTAL_BYTES", 600)
    with pytest.raises(ValueError, match="total bundle size limit"):
        cache.export_bundle(source, tmp_path / "b.zip", COMPAT)

    monkeypatch.setattr(cache, "MAX_BUNDLE_TOTAL_BYTES", 16 * 1024**3)
    monkeypatch.setattr(cache, "MAX_BUNDLE_FILES", 2)
    with pytest.raises(ValueError, match="too many files"):
        cache.export_bundle(source, tmp_path / "c.zip", COMPAT)
