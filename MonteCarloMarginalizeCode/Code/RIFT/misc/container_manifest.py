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
        containers.append(
            {
                "label": label,
                "image": image,
                "cuda_capability_min": cap_min,
                "cuda_capability_max": cap_max,
                "note": entry.get("note"),
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
    test; the ``fallback`` container is the innermost else (catch-all, also used
    when the capability attribute is ``undefined``).

    With ``ternary=False`` the selector uses ``ifThenElse(cond, a, b)`` (commas).
    With ``ternary=True`` it uses the comma-free ClassAd ternary ``cond ? a : b``
    -- required when the result is embedded as one element of a comma-separated
    ``transfer_input_files`` list, where internal commas would be mis-split.
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
        if c["cuda_capability_min"] is not None and c["label"] != fb["label"]
    ]
    # Fold ascending so the highest min ends up outermost.
    thresholds.sort(key=lambda c: c["cuda_capability_min"])

    expr = value_fn(fb)
    for c in thresholds:
        cond = "TARGET.{attr} >= {mn}".format(attr=attr, mn=_fmt_cap(c["cuda_capability_min"]))
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

    return "$$([ {} ])".format(_build_selector(manifest, value_fn, ternary=True))


def build_container_image_select(manifest, request_gpu=True):
    """Return the value for the HTCondor *container universe* ``container_image``
    submit command for this family.

    GPU jobs (``request_gpu=True``, the default) get a per-machine selection: an
    unquoted ``$$([ ... ])`` token.  ``$$()`` is HTCondor's *match-time machine-ad
    substitution* -- the schedd evaluates the bracketed expression against the
    matched machine ad and substitutes a literal image string into
    ``container_image`` before the job reaches the execution point.  Unlike
    :func:`build_singularity_image_expr` (an execute-side ClassAd expression that
    OSPool glidein pilots read as a literal string and hold on), the pilot only
    ever sees a literal URL.  ``$$`` in ``container_image`` is HTCondor's
    documented mechanism for per-GPU-capability image selection, and it works on
    both the CIT-local pool and OSPool glideins.  The branch value is the manifest
    image *verbatim* (an ``osdf://`` URL the container-universe file-transfer
    plugin fetches, or a CVMFS/local path used in place) -- NOT a ``./basename``
    rewrite.  ``container_image`` is a single submit command (not a comma list),
    so the comma-bearing ``ifThenElse`` form is fine.

    **Non-GPU jobs (``request_gpu=False``) collapse to a SINGLE fixed container**:
    the plain ``fallback`` image (a literal ``container_image``, no ``$$()``).
    A CPU-only job (e.g. CIP) matches a slot that advertises **no** GPU capability
    attribute, so a ``$$()`` capability expression has nothing to resolve against
    -- it fails to expand and HTCondor *holds the job*.  There is also nothing to
    select between, so the CPU-safe fallback image is the right (and only) choice.

    (The GPU-path expression is also written undefined-safe -- ``=?= undefined``
    yields the fallback -- but a GPU job that requested a GPU will match a slot
    that advertises the capability, so that guard is belt-and-suspenders; the
    non-GPU case must not use ``$$()`` at all.)
    """
    by_label = {c["label"]: c for c in manifest["containers"]}
    fb_image = by_label[manifest["fallback"]]["image"]
    if not request_gpu:
        # Single fixed container: no capability, no $$() -- a plain literal.
        return fb_image
    attr = _capability_attr(manifest)
    selector = _build_selector(manifest, lambda c: '"{}"'.format(c["image"]))
    guarded = 'ifThenElse(TARGET.{attr} =?= undefined, "{fb}", {sel})'.format(
        attr=attr, fb=fb_image, sel=selector
    )
    return "$$([ {} ])".format(guarded)


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
