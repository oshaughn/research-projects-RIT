"""Compatibility boundary for the optional TEOBResumS Python extension.

``EOBRun_module`` is a native extension with several incompatible interfaces in
active use.  Invalid enum values can terminate Python instead of raising an
exception, so callers must normalize parameters and probe a new interface in a
child process before making the first in-process call.
"""

import hashlib
import json
import math
import os
import subprocess
import sys

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # Python 3.7 compatibility
    try:
        import importlib_metadata
    except ImportError:  # fingerprinting is optional on minimal Python 3.6 installs
        importlib_metadata = None


class TEOBResumSCompatibilityError(RuntimeError):
    pass


_DALI_MARKERS = {
    "eob_dyn_j0_py",
    "eob_ham_s_py",
    "eob_metric_A5PNlog_py",
}

# Profiles intentionally contain semantic values, not C enum ordinals.  The
# default is the common string API verified against legacy-hyperbolic and DALI
# builds.  Future incompatible profiles should be added here rather than
# branching throughout lalsimutils.
_PROFILE_VALUES = {
    "default": {
        "arg_out": "yes",
        "nqc": "no",
        "nqc_coefs_hlm": "none",
        "nqc_coefs_flx": "none",
        "use_geometric_units": "no",
        "interp_uniform_grid": "yes",
        "output_hpc": "no",
    },
    "dali": {},
    "legacy": {},
}

_PROBED_SCHEMAS = set()


# TEOBResumS-DALI decides whether a system is precessing from
# hypot(chi1x, chi1y) + hypot(chi2x, chi2y) > 1e-4.  Keep the value here in
# sync with TEOBResumSPars.c.  It is a native-backend safety boundary, not a
# generic numerical-zero tolerance.
DALI_TRANSVERSE_SPIN_THRESHOLD = 1e-4
DALI_INITIAL_TRANSVERSE_SPIN_RANGE = (1e-3, 3e-3)
LEGACY_INITIAL_TRANSVERSE_SPIN_RANGE = (1e-5, 3e-5)


def is_teobresums_approximant(approximant):
    """Return whether an approximant name selects the TEOBResumS family."""
    return str(approximant or "").lower().startswith("teobresums")


def initial_transverse_spin_range(approximant):
    """Return a non-aligned initial-grid seed appropriate to ``approximant``.

    Other precessing models retain RIFT's long-standing tiny seed.  ResumS
    needs a seed safely above its native 1e-4 aligned/precessing boundary.
    """
    if is_teobresums_approximant(approximant):
        return DALI_INITIAL_TRANSVERSE_SPIN_RANGE
    return LEGACY_INITIAL_TRANSVERSE_SPIN_RANGE


def _dimensionless_float(value):
    return float(value.value if hasattr(value, "value") else value)


def total_transverse_spin(s1x, s1y, s2x, s2y):
    """Return the transverse spin magnitude TEOBResumS-DALI classifies with."""
    return math.hypot(_dimensionless_float(s1x), _dimensionless_float(s1y)) + math.hypot(
        _dimensionless_float(s2x), _dimensionless_float(s2y)
    )


def is_precessing_for_resums(s1x, s1y, s2x, s2y):
    """Return whether TEOBResumS-DALI evolves these spins on its precessing path.

    Callers must ask the same question the backend does before requesting
    inertial-frame modes: a component that is merely nonzero can still leave the
    summed transverse magnitude inside the native aligned interval, and asking
    for inertial modes there disagrees with the dynamics DALI actually runs.
    """
    return total_transverse_spin(s1x, s1y, s2x, s2y) > DALI_TRANSVERSE_SPIN_THRESHOLD


def guard_gwsignal_transverse_spins(parameters, approximant):
    """Return GWSignal parameters safe at the ResumS alignment boundary.

    TEOBResumS-DALI treats total transverse spin at or below 1e-4 as aligned,
    but its GWSignal wrapper requests inertial modes for *any* exactly nonzero
    transverse component.  Some native builds segfault when those two choices
    disagree.  Match the backend's own classification by zeroing only that
    interval; do not mutate the caller's dictionary.
    """
    if not is_teobresums_approximant(approximant):
        return parameters

    keys = ("spin1x", "spin1y", "spin2x", "spin2y")
    transverse_spin = total_transverse_spin(*(parameters[key] for key in keys))
    if not 0.0 < transverse_spin <= DALI_TRANSVERSE_SPIN_THRESHOLD:
        return parameters

    safe_parameters = dict(parameters)
    for key in keys:
        value = parameters[key]
        safe_parameters[key] = 0.0 * value.unit if hasattr(value, "unit") else 0.0
    return safe_parameters


def _json_compatible(value):
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def detect_profile(module, requested=None):
    """Return the configured or best-effort TEOBResumS interface profile.

    ``auto`` is the default.  Unknown extensions deliberately fall back to the
    conservative ``default`` profile; an explicit unknown profile is rejected
    so misspellings cannot silently change waveform settings.
    """
    requested = requested or os.environ.get("RIFT_TEOBRESUMS_PROFILE", "auto")
    requested = requested.lower()
    if requested != "auto":
        if requested not in _PROFILE_VALUES:
            raise TEOBResumSCompatibilityError(
                "Unknown RIFT_TEOBRESUMS_PROFILE={!r}; expected auto, default, dali, or legacy".format(
                    requested
                )
            )
        return requested
    if _DALI_MARKERS.issubset(set(dir(module))):
        return "dali"
    return "default"


def _profile_values(profile):
    values = dict(_PROFILE_VALUES["default"])
    values.update(_PROFILE_VALUES.get(profile, {}))
    return values


def normalize_parameters(parameters, profile="default"):
    """Return a copy using the selected profile's stable enum representation."""
    if profile not in _PROFILE_VALUES:
        raise TEOBResumSCompatibilityError(
            "Unknown TEOBResumS profile {!r}".format(profile)
        )
    normalized = dict(parameters)
    values = _profile_values(profile)

    # These integer spellings occur in older RIFT branches.  Their apparent C
    # enum ordinals changed meaning in DALI, so translate their RIFT semantics
    # before applying the profile representation.
    legacy_semantics = {
        "arg_out": {0: "no", 1: "yes"},
        "nqc": {2: "no"},
        "nqc_coefs_hlm": {0: "none"},
        "nqc_coefs_flx": {0: "none"},
        "use_geometric_units": {0: "no", 1: "yes"},
        "interp_uniform_grid": {0: "no", 1: "yes"},
        "output_hpc": {0: "no", 1: "yes"},
    }
    for key, translations in legacy_semantics.items():
        if key not in normalized:
            continue
        value = normalized[key]
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, int) and value not in translations:
            raise TEOBResumSCompatibilityError(
                "Unsupported numeric TEOBResumS value {}={!r}; use a semantic string".format(
                    key, value
                )
            )
        semantic_value = translations.get(value, value)
        if semantic_value == values.get(key):
            normalized[key] = values[key]
        elif isinstance(semantic_value, str):
            normalized[key] = semantic_value
    return normalized


def runtime_fingerprint(module, profile=None):
    """Return reproducibility metadata without assuming package version is unique."""
    module_path = os.path.realpath(getattr(module, "__file__", ""))
    package_version = None
    if importlib_metadata is not None:
        try:
            package_version = importlib_metadata.version("teobresums")
        except importlib_metadata.PackageNotFoundError:
            pass
    digest = None
    if module_path and os.path.isfile(module_path):
        hasher = hashlib.sha256()
        with open(module_path, "rb") as module_file:
            for block in iter(lambda: module_file.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
    return {
        "profile": profile or detect_profile(module),
        "module_path": module_path or None,
        "module_sha256": digest,
        "package_version": package_version,
        "exported_symbols": sorted(
            name for name in _DALI_MARKERS if hasattr(module, name)
        ),
    }


def _probe_schema(module, parameters, profile, purpose, timeout):
    module_path = os.path.realpath(getattr(module, "__file__", ""))
    schema = (profile, purpose, module_path, tuple(sorted(parameters)))
    if schema in _PROBED_SCHEMAS:
        return
    if os.environ.get("RIFT_TEOBRESUMS_SKIP_PROBE", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return

    probe_code = r"""
import json
import importlib.util
import os
import sys

expected_path = os.path.realpath(sys.argv[1])
if expected_path:
    spec = importlib.util.spec_from_file_location("EOBRun_module", expected_path)
    EOBRun_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(EOBRun_module)
else:
    import EOBRun_module
parameters = json.loads(sys.argv[2])
result = EOBRun_module.EOBRunPy(parameters)
if not isinstance(result, tuple) or len(result) < 4:
    raise RuntimeError("EOBRunPy returned an unexpected result")
"""
    probe_parameters = _json_compatible(parameters)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_code, module_path, json.dumps(probe_parameters)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TEOBResumSCompatibilityError(
            "TEOBResumS {} compatibility probe exceeded {} seconds".format(
                purpose, timeout
            )
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-1000:]
        raise TEOBResumSCompatibilityError(
            "TEOBResumS {} compatibility probe failed with return code {}. {}".format(
                purpose, completed.returncode, detail
            )
        )
    _PROBED_SCHEMAS.add(schema)


def run(module, parameters, purpose="waveform", profile=None, probe=True, timeout=None):
    """Normalize, safely preflight, and call ``EOBRunPy``.

    The first call for each parameter schema is duplicated in a child process.
    That cost is intentional: an incompatible native extension may segfault.
    Set ``RIFT_TEOBRESUMS_SKIP_PROBE=1`` only for a separately validated,
    pinned runtime.
    """
    selected_profile = profile or detect_profile(module)
    normalized = normalize_parameters(parameters, selected_profile)
    if probe:
        if timeout is None:
            timeout = float(os.environ.get("RIFT_TEOBRESUMS_PROBE_TIMEOUT", "60"))
        _probe_schema(module, normalized, selected_profile, purpose, timeout)
    return module.EOBRunPy(normalized)
