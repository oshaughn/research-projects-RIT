"""
container_manifest
==================

Support for "container family" manifests used by the RIFT pipeline.

Historically ``SINGULARITY_RIFT_IMAGE`` names a single ``.sif`` image (a local
path or an ``osdf://`` URL), and the ILE/CIP Condor jobs hard-code

    MY.SingularityImage = "<that image>"

A *manifest* lets us instead advertise a *family* of images, each targeting a
different GPU compute capability, and let HTCondor pick the right one per matched
machine.  When ``SINGULARITY_RIFT_IMAGE`` points at a ``.yaml``/``.yml`` file,
the job-submission code turns it into:

  * an expression-valued ``MY.SingularityImage`` -- a nested ``ifThenElse`` over
    the matched machine's GPU capability attribute (default ``GPUs_Capability``)
    that selects the highest-capability image the machine can run; and
  * a ``require_gpus`` capability floor (the lowest capability any image in the
    family supports), composed (``&&``) with any user-supplied
    ``RIFT_REQUIRE_GPUS``; and
  * for ``osdf://`` images, a *selective* ``transfer_input_files`` entry using
    HTCondor ``$$()`` match-time substitution, so only the *matched* image is
    transferred (CVMFS/local images are referenced in place and never
    transferred).

Single-``.sif`` behavior is completely unchanged: only ``.yaml``/``.yml`` values
exercise any of this.

YAML schema
-----------

    version: 1
    capability_attr: GPUs_Capability   # machine ClassAd attr the ifThenElse tests
    fallback: ancient                  # label used as the innermost else-branch
    containers:
      - label: ancient
        image: /cvmfs/.../rift_ancient_cuda11.sif   # in-place (CVMFS/local)
        cuda_capability_min: 3.0       # inclusive
        cuda_capability_max: 7.0       # exclusive; null/omitted => open-ended
        note: "cupy-cuda11x, ancient base"
      - label: modern
        image: osdf:///igwn/.../rift_modern_cuda12.sif   # selectively transferred
        cuda_capability_min: 7.0
        cuda_capability_max: null
        note: "cupy-cuda12x, newer base"

Per-host ILE profiles (optional)
--------------------------------

A container entry may also carry

    select_requirements: TARGET.GPUs_GlobalMemoryMb >= 40000  # ANDed into the branch test
    ile_exe: integrate_likelihood_extrinsic_jax              # bare name or in-container path
    request_memory: 16000                                     # MB

If any entry sets ``ile_exe`` or ``request_memory``, a GPU ILE job selects its
executable and memory request from the same branch that selects its image.
Entries are tested highest ``cuda_capability_min`` first; at equal minimum an
entry with ``select_requirements`` is tested first.  All three choices are
ClassAd expressions over the matched machine, re-evaluated on every match.
Entries without the fields use the job defaults.
"""

import os

__all__ = [
    "ContainerManifestError",
    "is_container_manifest",
    "load_container_manifest",
    "build_singularity_image_expr",
    "build_transfer_input_expr",
    "build_require_gpus_floor",
    "build_container_image_select",
    "build_capability_defined_requirement",
    "build_fallback_single_image",
    "build_runtime_selection_wrapper",
    "has_ile_profiles",
    "build_request_memory_expr",
    "build_ile_exe_expr",
    "build_profile_label_expr",
]

# Default machine ClassAd attribute advertising GPU compute capability.  The
# user's pools advertise this via e.g.
#   condor_status -constraint 'TotalGPUs > 0' -autoformat GPUs_DeviceName GPUs_Capability
DEFAULT_CAPABILITY_ATTR = "GPUs_Capability"


class ContainerManifestError(Exception):
    """Raised for a missing/malformed container family manifest."""


def is_container_manifest(value):
    """Return True iff ``value`` (the ``SINGULARITY_RIFT_IMAGE`` string) names a
    multi-container manifest rather than a single ``.sif``/``osdf://`` image.

    Pure string check (no filesystem access) so single-image callers pay zero
    cost and their behavior is unchanged.
    """
    if not value or not isinstance(value, str):
        return False
    return value.lower().endswith((".yaml", ".yml"))


def _image_needs_transfer(image):
    """True iff ``image`` is a URL that must be fetched via Condor file transfer
    (e.g. ``osdf://``).  CVMFS/local paths (``/cvmfs/...``, ``./foo.sif``) are
    resolved in place and return False.
    """
    return "://" in image


def _image_runtime_path(image):
    """The string used *inside* ``MY.SingularityImage`` for this image.

    Transferred (URL) images land in the job scratch dir under their basename,
    so the pilot must reference ``./<basename>`` -- matching the existing
    single-image osdf rewrite convention.  In-place (CVMFS/local) images are
    referenced verbatim.
    """
    if _image_needs_transfer(image):
        return "./{}".format(image.rstrip("/").split("/")[-1])
    return image


def _image_basename(image):
    """The bare file name an image has once transferred into the job scratch dir.

    Used by the container-universe selector, which MUST NOT contain a ``/``.
    """
    return image.rstrip("/").split("/")[-1]


def _fmt_cap(value):
    """Format a capability number for a ClassAd expression (e.g. 7.0 -> '7.0')."""
    return repr(float(value))


def load_container_manifest(path):
    """Parse and validate a YAML container family manifest.

    Returns a dict ``{capability_attr, fallback, containers}`` where
    ``containers`` is sorted by ``cuda_capability_min`` *descending* (containers
    with no min sort last).

    Raises ``ContainerManifestError`` on a missing pyyaml, an unreadable or
    malformed file, an empty container list, or an unknown ``fallback`` label.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ContainerManifestError(
            "PyYAML is required to read a container family manifest ({}); "
            "install pyyaml or point SINGULARITY_RIFT_IMAGE at a single .sif".format(path)
        ) from exc

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except (IOError, OSError) as exc:
        raise ContainerManifestError("Cannot read container manifest {}: {}".format(path, exc))
    except yaml.YAMLError as exc:
        raise ContainerManifestError("Malformed container manifest {}: {}".format(path, exc))

    if not isinstance(data, dict):
        raise ContainerManifestError("Container manifest {} is not a mapping".format(path))

    raw_containers = data.get("containers")
    if not raw_containers or not isinstance(raw_containers, list):
        raise ContainerManifestError(
            "Container manifest {} must define a non-empty 'containers' list".format(path)
        )

    containers = []
    for idx, entry in enumerate(raw_containers):
        if not isinstance(entry, dict):
            raise ContainerManifestError(
                "Container manifest {} entry #{} is not a mapping".format(path, idx)
            )
        image = entry.get("image")
        label = entry.get("label")
        if not image:
            raise ContainerManifestError(
                "Container manifest {} entry #{} is missing 'image'".format(path, idx)
            )
        if not label:
            raise ContainerManifestError(
                "Container manifest {} entry #{} is missing 'label'".format(path, idx)
            )
        cap_min = entry.get("cuda_capability_min")
        cap_max = entry.get("cuda_capability_max")
        try:
            cap_min = None if cap_min is None else float(cap_min)
            cap_max = None if cap_max is None else float(cap_max)
        except (TypeError, ValueError):
            raise ContainerManifestError(
                "Container manifest {} entry '{}' has non-numeric capability bounds".format(
                    path, label
                )
            )
        select_req = entry.get("select_requirements")
        if select_req is not None:
            select_req = str(select_req).strip()
            # The selector is also embedded in a comma-split transfer_input_files
            # entry and in a container_image value that may not contain '/'.
            if any(ch in select_req for ch in ",/\n"):
                raise ContainerManifestError(
                    "Container manifest {} entry '{}': select_requirements may not contain "
                    "',', '/' or a newline".format(path, label)
                )
        req_mem = entry.get("request_memory")
        if req_mem is not None:
            try:
                req_mem = int(req_mem)
            except (TypeError, ValueError):
                req_mem = 0
            if req_mem <= 0:
                raise ContainerManifestError(
                    "Container manifest {} entry '{}': request_memory must be a positive "
                    "integer (MB)".format(path, label)
                )
        ile_exe = entry.get("ile_exe")
        if ile_exe is not None and (not isinstance(ile_exe, str) or any(ch in ile_exe for ch in ' ,"')):
            raise ContainerManifestError(
                "Container manifest {} entry '{}': ile_exe must be a path without spaces, "
                "commas or quotes".format(path, label)
            )
        containers.append(
            {
                "label": label,
                "image": image,
                "cuda_capability_min": cap_min,
                "cuda_capability_max": cap_max,
                "note": entry.get("note"),
                "select_requirements": select_req,
                "ile_exe": ile_exe,
                "request_memory": req_mem,
            }
        )

    # Sort by min capability descending; None mins (open-ended-low catch-alls)
    # sort last.  float('-inf') keeps them at the bottom.
    containers.sort(
        key=lambda c: (c["cuda_capability_min"] if c["cuda_capability_min"] is not None else float("-inf")),
        reverse=True,
    )

    labels = {c["label"] for c in containers}
    fallback = data.get("fallback")
    if fallback is None:
        # Default fallback = the most-compatible (lowest-min) container, i.e. the
        # last one after the descending sort.  This is the CPU-safe catch-all.
        fallback = containers[-1]["label"]
    elif fallback not in labels:
        raise ContainerManifestError(
            "Container manifest {} fallback '{}' is not one of {}".format(
                path, fallback, sorted(labels)
            )
        )

    if {c["label"]: c for c in containers}[fallback]["select_requirements"]:
        raise ContainerManifestError(
            "Container manifest {} fallback '{}' may not set select_requirements: it is "
            "the unconditional else-branch".format(path, fallback)
        )

    capability_attr = data.get("capability_attr") or DEFAULT_CAPABILITY_ATTR

    return {
        "capability_attr": capability_attr,
        "fallback": fallback,
        "containers": containers,
    }


def _capability_attr(manifest):
    """Resolve the machine attribute used by the selection ifThenElse.

    Precedence: ``RIFT_GPU_CAPABILITY_ATTR`` env override > manifest
    ``capability_attr`` > module default.
    """
    return os.environ.get("RIFT_GPU_CAPABILITY_ATTR") or manifest["capability_attr"]


def _build_selector(manifest, value_fn, ternary=False):
    """Build a nested capability selector over the family.

    ``value_fn(container)`` returns the ClassAd literal for a container branch
    (already quoted as appropriate).  The highest-min container is the outermost
    test; the ``fallback`` container is the innermost else (catch-all, used when
    the capability is below every threshold).

    With ``ternary=False`` the selector uses ``ifThenElse(cond, a, b)`` (commas).
    With ``ternary=True`` it uses the comma-free ClassAd ternary ``cond ? a : b``
    -- required when the result is embedded as one element of a comma-separated
    ``transfer_input_files`` list, where internal commas would be mis-split.

    NOTE: the selector is intentionally NOT undefined-guarded.  A GPU job must add
    a ``Requirements`` clause excluding slots that do not advertise the capability
    attribute (:func:`build_capability_defined_requirement`); guessing an image
    for an undefined slot is unsafe (it could be a Blackwell that hard-fails on the
    older fallback image), so the correct action is to NOT match such a slot.  A
    non-GPU job must not use this selector at all -- it has no capability to read.
    """
    attr = _capability_attr(manifest)
    containers = manifest["containers"]  # sorted desc by min
    by_label = {c["label"]: c for c in containers}
    fb = by_label[manifest["fallback"]]

    # Containers that contribute a capability threshold test (exclude the
    # fallback so it is not duplicated as both a branch and the else).
    thresholds = [
        c
        for c in containers
        if (c["cuda_capability_min"] is not None or c.get("select_requirements"))
        and c["label"] != fb["label"]
    ]
    # Fold ascending so the highest min ends up outermost; at equal min, an entry
    # with select_requirements is folded last, i.e. tested first.
    thresholds.sort(key=lambda c: (
        c["cuda_capability_min"] if c["cuda_capability_min"] is not None else float("-inf"),
        bool(c.get("select_requirements")),
    ))

    expr = value_fn(fb)
    for c in thresholds:
        terms = []
        if c["cuda_capability_min"] is not None:
            terms.append("TARGET.{attr} >= {mn}".format(attr=attr, mn=_fmt_cap(c["cuda_capability_min"])))
        if c.get("select_requirements"):
            terms.append("({})".format(c["select_requirements"]))
        cond = " && ".join(terms)
        if ternary:
            expr = "({cond} ? {val} : {inner})".format(cond=cond, val=value_fn(c), inner=expr)
        else:
            expr = "ifThenElse({cond}, {val}, {inner})".format(
                cond=cond, val=value_fn(c), inner=expr
            )
    return expr


def build_singularity_image_expr(manifest):
    """Return the unquoted ClassAd expression for ``MY.SingularityImage``.

    Each branch literal is the container's *runtime* path (CVMFS/local verbatim,
    ``./<basename>`` for transferred images).

    GPU jobs that emit this MUST also add
    :func:`build_capability_defined_requirement` so they never match a slot that
    does not advertise the capability attribute (where this expression would be
    ``undefined``).
    """
    return _build_selector(
        manifest, lambda c: '"{}"'.format(_image_runtime_path(c["image"]))
    )


def build_transfer_input_expr(manifest):
    """Return a single ``$$([ ... ])`` token for ``transfer_input_files`` that
    fetches *only the matched* image, or ``None`` if no container in the family
    needs transfer.

    Transfer branches yield the URL verbatim; in-place (CVMFS/local) branches
    yield ``""`` (no transfer on those machines).  Uses the comma-free ternary
    form so the token survives comma-splitting of ``transfer_input_files``.
    """
    if not any(_image_needs_transfer(c["image"]) for c in manifest["containers"]):
        return None

    def value_fn(c):
        return '"{}"'.format(c["image"]) if _image_needs_transfer(c["image"]) else '""'

    # GPU jobs that emit this MUST also add build_capability_defined_requirement so
    # they never match a slot where TARGET.<attr> is undefined (this $$ token would
    # then "cannot expand" and HOLD the job).
    return "$$([ {} ])".format(_build_selector(manifest, value_fn, ternary=True))


def build_container_image_select(manifest, request_gpu=True):
    """Return the value for the HTCondor *container universe* ``container_image``
    submit command for this family.

    GPU jobs (``request_gpu=True``, the default) get a per-machine selection: an
    unquoted ``$$([ ... ])`` token.  ``$$()`` is HTCondor's *match-time machine-ad
    substitution* -- the schedd evaluates the bracketed expression against the
    matched machine ad and substitutes a literal string before the job reaches the
    execution point.  Unlike :func:`build_singularity_image_expr` (an execute-side
    ClassAd expression that OSPool glidein pilots read as a literal string and hold
    on), the pilot only ever sees a literal image name.  ``container_image`` is a
    single submit command (not a comma list), so the comma-bearing ``ifThenElse``
    form is fine here.

    **The branch values are BASENAMES, not full URLs.**  ``condor_submit`` parses
    ``container_image`` *before* any ``$$`` expansion and derives the job ad's
    ``ContainerImage`` -- the name the image will have in the job scratch dir -- as
    the text after the **last** ``/``.  A selector containing full paths therefore
    gets cut in half, and what survives is not even a valid image name.  This is not
    theoretical: submitting the full-URL form to the IGWN pool holds the job at the
    execute point with::

        PREPARE_JOB (prepare-hook) failed (reported status 001):
        Unable to download or build singularity image cutest_busybox_...sif") ])

    With no ``/`` in the value, that derivation is a no-op, the whole ``$$`` token
    survives into ``ContainerImage``, and the schedd expands it at match time
    (``MATCH_EXP_ContainerImage = "rift_container_modern.sif"``) -- verified end to
    end on an OSPool glidein.

    Because the selector now names only basenames, the caller MUST also deliver the
    matched image itself: add :func:`build_transfer_input_expr` (the comma-free
    ternary over the full URLs) to ``transfer_input_files`` **and** emit it as
    ``MY.TransferInput`` so it overrides the entry ``condor_submit`` would otherwise
    derive from ``container_image``.  All images in the family must therefore be
    transferable URLs; a family that references an image in place (CVMFS/local path)
    cannot be selected this way and raises :class:`ContainerManifestError`.

    **Non-GPU jobs (``request_gpu=False``) collapse to a SINGLE fixed container**:
    the plain ``fallback`` image (a literal ``container_image``, no ``$$()``).
    A CPU-only job (e.g. CIP) matches a slot that advertises **no** GPU capability
    attribute, so a ``$$()`` capability expression has nothing to resolve against
    -- it fails to expand and HTCondor *holds the job*.  There is also nothing to
    select between, so the CPU-safe fallback image is the right (and only) choice.

    The GPU-path ``$$()`` is NOT undefined-guarded: the GPU job that emits it MUST
    also add :func:`build_capability_defined_requirement` so it never matches a
    slot where the capability attr is undefined (guessing an image for such a slot
    is unsafe -- it could be a Blackwell that hard-fails on the older fallback).
    The non-GPU case never reaches the ``$$()`` (it returns the literal fallback).
    """
    by_label = {c["label"]: c for c in manifest["containers"]}
    fb_image = by_label[manifest["fallback"]]["image"]
    if not request_gpu:
        # Single fixed container: no capability, no $$() -- a plain literal.  This is
        # the ordinary single-image path condor_submit handles correctly (it derives
        # ContainerImage as the basename, which is exactly right).
        return fb_image
    in_place = [c["label"] for c in manifest["containers"] if not _image_needs_transfer(c["image"])]
    if in_place:
        raise ContainerManifestError(
            "container universe per-machine selection requires every image in the family "
            "to be a transferable URL (e.g. osdf://), because the selector may not contain "
            "a '/' -- condor_submit would truncate it.  In-place image(s): {}.  Either "
            "stage those images at a URL, or use RIFT_CONTAINER_RUNTIME_SELECT=1 instead "
            "of RIFT_CONTAINER_UNIVERSE=1.".format(", ".join(sorted(in_place)))
        )
    selector = _build_selector(manifest, lambda c: '"{}"'.format(_image_basename(c["image"])))
    return "$$([ {} ])".format(selector)


def build_capability_defined_requirement(manifest):
    """Return a ``Requirements`` clause that excludes machines which do not
    advertise the capability attribute the family selection reads.

    A GPU family job MUST add this.  Measured on the CIT pool (2026-06-12), ~45%
    of GPU slots satisfy the per-GPU ``require_gpus`` floor (which matches the
    per-GPU ``Capability`` inside ``AvailableGPUs``) yet do NOT advertise the
    machine-level rollup attribute (default ``GPUs_Capability``) that the
    ``$$()``/``ifThenElse`` selection reads.  On such a slot the selection cannot
    expand and the job HOLDS ("Cannot expand $$ expression").  Excluding these
    slots is the safe fix: an undefined-capability slot could be a Blackwell that
    hard-fails on the older fallback image, so we must NOT match it (rather than
    guess its image).  The defined set still includes the high-capability nodes,
    so the family's purpose is preserved.

    Generic on ``capability_attr``; a no-op on pools where every GPU slot
    advertises it, hence merge-safe.
    """
    return "TARGET.{attr} =!= undefined".format(attr=_capability_attr(manifest))


def build_fallback_single_image(manifest):
    """For jobs that must use a SINGLE fixed container (no capability selection) --
    e.g. CPU-only CIP, which requests no GPU and so cannot resolve a
    ``$$()``/``ifThenElse`` capability selection (its matched slot advertises no
    capability attribute -> the selection holds the job).

    Returns ``(runtime_path, transfer_url)`` for the manifest ``fallback`` (the
    CPU-safe image):

      * ``runtime_path`` -- what ``MY.SingularityImage`` / ``container_image``
        references: ``./<basename>`` for a transferred ``osdf://`` image, the path
        verbatim for a CVMFS/local image.  (``MY.SingularityImage`` callers must
        quote it; ``container_image`` takes it unquoted.)
      * ``transfer_url``  -- the ``osdf://`` URL to add to ``transfer_input_files``,
        or ``None`` if the image is referenced in place (CVMFS/local).
    """
    fb_image = {c["label"]: c for c in manifest["containers"]}[manifest["fallback"]]["image"]
    runtime_path = _image_runtime_path(fb_image)
    transfer_url = fb_image if _image_needs_transfer(fb_image) else None
    return runtime_path, transfer_url


# Body of the OSG-safe runtime-selection wrapper. @@TOKENS@@ are substituted by
# build_runtime_selection_wrapper (str.replace, not .format; the script is full
# of ${...} bash expansions that would collide with format()).
_RUNTIME_WRAPPER_BODY = r'''#!/bin/bash
# AUTO-GENERATED by RIFT.misc.container_manifest.build_runtime_selection_wrapper.
# OSG-safe runtime container selection. Runs as the Condor executable on the
# bare execute node, with no +SingularityImage. At job start it detects the real
# GPU compute capability, selects the matching family image, acquires only that
# image, and execs the real command inside it via nested apptainer.
set -euo pipefail
LABELS=( @@LABELS@@ )
CAP_MIN=( @@MINS@@ )
CAP_MAX=( @@MAXS@@ )
RTPATH=( @@RTPATHS@@ )
FETCH=( @@FETCHES@@ )
FALLBACK_LABEL="@@FALLBACK@@"
INNER_COMMAND="@@INNER@@"

log() { echo "[rift_container_select] $*" >&2; }

cap="${RIFT_CONTAINER_FORCE_CAP:-}"
if [ -z "$cap" ] && command -v nvidia-smi >/dev/null 2>&1; then
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')" || cap=""
fi
log "detected compute capability: ${cap:-<none>}"

sel=-1
if [ -n "$cap" ]; then
    best_min=-1
    for i in "${!LABELS[@]}"; do
        lo="${CAP_MIN[$i]}"; hi="${CAP_MAX[$i]}"
        if awk -v c="$cap" -v lo="$lo" -v hi="$hi" 'BEGIN{exit !(c+0>=lo+0 && c+0<=hi+0)}'; then
            if awk -v lo="$lo" -v b="$best_min" 'BEGIN{exit !(lo+0>b+0)}'; then
                best_min="$lo"; sel="$i"
            fi
        fi
    done
fi
if [ "$sel" -lt 0 ]; then
    log "no GPU match for cap='${cap:-<none>}' -> fallback ${FALLBACK_LABEL}"
    for i in "${!LABELS[@]}"; do
        if [ "${LABELS[$i]}" = "$FALLBACK_LABEL" ]; then sel="$i"; fi
    done
fi
[ "$sel" -lt 0 ] && { log "FATAL: fallback '${FALLBACK_LABEL}' not in table"; exit 3; }
log "selected: ${LABELS[$sel]} (${RTPATH[$sel]}) [cap band ${CAP_MIN[$sel]}-${CAP_MAX[$sel]}]"

rt="${RTPATH[$sel]}"; fetch="${FETCH[$sel]}"; SIF=""
if [ -e "$rt" ]; then
    SIF="$rt"; log "using in-place/local image: $SIF"
elif [ -n "$fetch" ]; then
    log "fetching single image: $fetch -> $rt"
    if   command -v stashcp >/dev/null 2>&1; then stashcp "$fetch" "$rt"
    elif command -v pelican  >/dev/null 2>&1; then pelican object get "$fetch" "$rt"
    else log "FATAL: no local image and no stashcp/pelican to fetch $fetch"; exit 4; fi
    SIF="$rt"
else
    log "FATAL: image '$rt' absent and no fetch URL"; exit 4
fi

if [ -n "$INNER_COMMAND" ]; then
    log "exec: apptainer exec --nv ${RIFT_CONTAINER_APPTAINER_FLAGS:-} $SIF $INNER_COMMAND $*"
    exec apptainer exec --nv ${RIFT_CONTAINER_APPTAINER_FLAGS:-} "$SIF" $INNER_COMMAND "$@"
else
    log "exec: apptainer exec --nv ${RIFT_CONTAINER_APPTAINER_FLAGS:-} $SIF $*"
    exec apptainer exec --nv ${RIFT_CONTAINER_APPTAINER_FLAGS:-} "$SIF" "$@"
fi
'''


def _runtime_image_fields(container):
    """Return (runtime_path, fetch_url, cap_min, cap_max) for one container."""
    image = container["image"]
    runtime_path = _image_runtime_path(image)
    fetch_url = image if _image_needs_transfer(image) else ""
    return runtime_path, fetch_url, container["cuda_capability_min"], container["cuda_capability_max"]


def build_runtime_selection_wrapper(manifest, inner_command=None):
    """Return an OSG-safe runtime container-selection wrapper script.

    The wrapper is intended to run as the Condor executable on the bare execute
    node. It chooses a container at job start from the same manifest used by the
    ClassAd/container-universe selectors, then runs ``inner_command`` or the
    wrapper arguments inside the selected image with apptainer.
    """
    labels, mins, maxs, rtpaths, fetches = [], [], [], [], []
    for c in manifest["containers"]:
        runtime_path, fetch_url, cap_min, cap_max = _runtime_image_fields(c)
        labels.append(c["label"])
        mins.append("-1" if cap_min is None else repr(float(cap_min)))
        maxs.append("9999" if cap_max is None else repr(float(cap_max)))
        rtpaths.append(runtime_path)
        fetches.append(fetch_url)

    def _arr(values):
        return " ".join('"{}"'.format(v) for v in values)

    return (
        _RUNTIME_WRAPPER_BODY
        .replace("@@LABELS@@", _arr(labels))
        .replace("@@MINS@@", _arr(mins))
        .replace("@@MAXS@@", _arr(maxs))
        .replace("@@RTPATHS@@", _arr(rtpaths))
        .replace("@@FETCHES@@", _arr(fetches))
        .replace("@@FALLBACK@@", manifest["fallback"])
        .replace("@@INNER@@", "" if not inner_command else str(inner_command))
    )


def build_require_gpus_floor(manifest):
    """Return a ``require_gpus`` capability floor expression for the family, or
    ``None``.

    The floor is the lowest ``cuda_capability_min`` across the family -- i.e. do
    not match a GPU less capable than anything we ship.  Uses the require_gpus
    sub-ad attribute ``Capability`` (unprefixed -- *not* ``TARGET.`` and *not*
    ``GPUs_Capability``).

    If any container has no min (open-ended-low catch-all), there is effectively
    no lower bound and ``None`` is returned.
    """
    mins = [c["cuda_capability_min"] for c in manifest["containers"]]
    if any(m is None for m in mins) or not mins:
        return None
    return "Capability >= {}".format(_fmt_cap(min(mins)))


def has_ile_profiles(manifest):
    """True iff any container sets ``ile_exe`` or ``request_memory``."""
    return any(c.get("ile_exe") or c.get("request_memory") for c in manifest["containers"])


def build_request_memory_expr(manifest, default_mb):
    """Return the ``request_memory`` value (MB) for a GPU ILE job of this family.

    A ClassAd selector over ``TARGET`` when entries differ, else a plain integer.
    HTCondor evaluates ``RequestMemory`` against the matched slot, so the dynamic
    slot is carved at the selected size, and the schedd keeps the expression, so a
    rematch after eviction selects again.  Like the image selector it is
    ``undefined`` on a slot without the capability attribute; the job's
    :func:`build_capability_defined_requirement` keeps it off such slots.
    """
    values = {c["label"]: int(c.get("request_memory") or default_mb) for c in manifest["containers"]}
    if len(set(values.values())) == 1:
        return str(values[manifest["fallback"]])
    return _build_selector(manifest, lambda c: str(values[c["label"]]))


def _ile_exe_path(container, default_exe, exe_dir):
    exe = container.get("ile_exe")
    if not exe:
        return default_exe
    if exe.startswith("/") or not exe_dir:
        return exe
    return exe_dir.rstrip("/") + "/" + exe


def build_ile_exe_expr(manifest, default_exe, exe_dir=None):
    """Return a ClassAd selector over quoted in-container ILE executable paths.

    A bare ``ile_exe`` name is prefixed with ``exe_dir`` (the in-container bin
    directory); entries without ``ile_exe`` use ``default_exe``.  The job carries
    this as ``MY.RIFTILEExe`` and passes it to the job as
    ``RIFT_ILE_EXE=$$([MY.RIFTILEExe])``.
    """
    return _build_selector(
        manifest, lambda c: '"{}"'.format(_ile_exe_path(c, default_exe, exe_dir))
    )


def build_profile_label_expr(manifest):
    """Return a ClassAd selector over quoted container labels (for the job record)."""
    return _build_selector(manifest, lambda c: '"{}"'.format(c["label"]))
