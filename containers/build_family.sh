#!/bin/bash
# Build a *family* of RIFT containers from containers/rift_container.def.in, one
# per build-matrix entry (different base image + cupy/CUDA variant, targeting
# different GPU compute capabilities).
#
# Usage:
#   containers/build_family.sh [--render-only] [--fakeroot] [--sandbox] \
#                              [other apptainer build flags] [OUTPUT_DIR]
#
#   --render-only   render the per-entry .def files but do NOT run apptainer
#                   (useful on machines without apptainer, or in CI)
#   --fakeroot      pass --fakeroot to `apptainer build` (RECOMMENDED on shared
#                   clusters such as CIT: avoids the unprivileged `proot` engine,
#                   whose mksquashfs step fails. Needs /etc/subuid + /etc/subgid
#                   entries for your user and unprivileged user namespaces.)
#   --sandbox       build a writable directory instead of a .sif. This SKIPS the
#                   mksquashfs step entirely, so it sidesteps the proot squashfs
#                   failure when even --fakeroot is unavailable. Convert to .sif
#                   later on a capable host: apptainer build out.sif sandbox_dir/
#   any other --flag is passed straight through to `apptainer build`.
#   OUTPUT_DIR      where rendered .def and built images land (default: ./container_family)
#
# The DEFAULT (first) matrix entry keeps the current production base image, so
# the family always includes a broadly-compatible image for older machines.
# Add rows to MATRIX to target more architectures.
#
# After building, publish the .sif files to CVMFS or osdf and edit
# containers/rift_container_family.yaml so SINGULARITY_RIFT_IMAGE can point at it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${HERE}/rift_container.def.in"

RENDER_ONLY=0
SANDBOX=0
BUILD_OPTS=()
OUTPUT_DIR="./container_family"
for arg in "$@"; do
    case "$arg" in
        --render-only) RENDER_ONLY=1 ;;
        --sandbox)     SANDBOX=1 ;;
        --fakeroot)    BUILD_OPTS+=(--fakeroot) ;;
        --*)           BUILD_OPTS+=("$arg") ;;   # passthrough to apptainer build
        *)             OUTPUT_DIR="$arg" ;;
    esac
done

# Build matrix: "label|base_image|cupy_pkg|cuda_capability_min|cuda_capability_max"
# - The first entry is the DEFAULT and uses the current production base image.
# - cuda_capability_max may be empty (open-ended); it is informational and is
#   echoed into the manifest stub for convenience.
MATRIX=(
  "default|nvidia/cuda:11.8.0-runtime-ubuntu22.04|cupy-cuda11x|3.5|8.0"
  "modern|nvidia/cuda:12.4.1-runtime-ubuntu22.04|cupy-cuda12x|8.0|"
)

# Workaround for unprivileged apptainer builds that fall back to the `proot`
# engine (no usable user namespaces / setuid apptainer, common on shared
# clusters like CIT): proot's mksquashfs step is blocked by seccomp on ptrace
# ("proot error: ptrace(TRACEME): Operation not permitted"). PROOT_NO_SECCOMP=1
# disables proot's seccomp filtering and lets the build finish. Harmless when
# proot is not used; override by exporting it yourself before running.
# A fully privileged / fakeroot / userns-capable build does not need this.
export PROOT_NO_SECCOMP="${PROOT_NO_SECCOMP:-1}"

mkdir -p "${OUTPUT_DIR}"
MANIFEST_STUB="${OUTPUT_DIR}/rift_container_family.generated.yaml"
{
  echo "# Auto-generated manifest stub from containers/build_family.sh."
  echo "# Edit 'image:' to the published CVMFS path or osdf:// URL of each .sif."
  echo "# IMPORTANT: a single SINGULARITY_BASE_EXE_DIR is applied to the whole"
  echo "# family -- all images must install RIFT executables at the SAME"
  echo "# in-container path. These are built from one template, so they are"
  echo "# consistent; do not hand-swap in images with a different layout."
  echo "version: 1"
  echo "capability_attr: GPUs_Capability"
  echo "fallback: default"
  echo "containers:"
} > "${MANIFEST_STUB}"

for row in "${MATRIX[@]}"; do
    IFS='|' read -r label base cupy cap_min cap_max <<< "$row"
    rendered="${OUTPUT_DIR}/rift_container_${label}.def"
    sif="${OUTPUT_DIR}/rift_container_${label}.sif"

    echo ">>> Rendering ${label}: base=${base} cupy=${cupy}"
    sed -e "s#@@BASE_IMAGE@@#${base}#g" \
        -e "s#@@CUPY_PKG@@#${cupy}#g" \
        -e "s#@@REQFILE@@#${HERE}/requirements-container.txt#g" \
        "${TEMPLATE}" > "${rendered}"

    {
      echo "  - label: ${label}"
      echo "    image: REPLACE_ME/rift_container_${label}.sif   # publish to CVMFS or osdf"
      echo "    cuda_capability_min: ${cap_min}"
      if [ -n "${cap_max}" ]; then
        echo "    cuda_capability_max: ${cap_max}"
      else
        echo "    cuda_capability_max: null"
      fi
      echo "    note: \"base=${base}, ${cupy}\""
    } >> "${MANIFEST_STUB}"

    if [ "${RENDER_ONLY}" -eq 1 ]; then
        echo "    (render-only) wrote ${rendered}"
        continue
    fi
    if ! command -v apptainer >/dev/null 2>&1; then
        echo "    apptainer not found; wrote ${rendered} (build skipped)" >&2
        continue
    fi

    if [ "${SANDBOX}" -eq 1 ]; then
        target="${OUTPUT_DIR}/rift_container_${label}"   # writable directory (no mksquashfs)
        sandbox_opt=(--sandbox)
    else
        target="${sif}"
        sandbox_opt=()
    fi
    echo ">>> Building ${target}${BUILD_OPTS[*]:+ (opts: ${BUILD_OPTS[*]})}"
    # ${arr[@]+"${arr[@]}"} expands safely even when the array is empty under set -u
    apptainer build ${sandbox_opt[@]+"${sandbox_opt[@]}"} ${BUILD_OPTS[@]+"${BUILD_OPTS[@]}"} "${target}" "${rendered}"
done

echo
echo "Done. Rendered defs (and any built .sif) are in ${OUTPUT_DIR}/"
echo "Manifest stub: ${MANIFEST_STUB}"
echo "Next: publish the .sif images, fill in their 'image:' locations, and point"
echo "SINGULARITY_RIFT_IMAGE at the resulting .yaml manifest."
