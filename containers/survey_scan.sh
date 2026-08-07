#!/usr/bin/env bash
# Survey target GPU pools and emit representative RIFT container warmup probes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

usage() {
    cat <<EOF
Usage:
  containers/survey_scan.sh survey [--out DIR] [--constraint EXPR] [--manifest FILE]
  containers/survey_scan.sh emit-jobs --survey DIR --manifest FILE [--out DIR] [--profiles LIST]
  containers/survey_scan.sh collect --survey DIR [--out FILE]

Commands:
  survey      Query condor_status for GPU inventory and write JSON/TSV/Markdown.
  emit-jobs   Generate Condor submit files that run common warmup profiles.
  collect     Summarize JSON outputs from completed warmup jobs.

Examples:
  containers/survey_scan.sh survey \\
    --out survey/cit-20260704 \\
    --manifest container_family/rift_container_family.generated.yaml
  containers/survey_scan.sh emit-jobs \\
    --survey survey/cit-20260704 \\
    --manifest container_family/rift_container_family.generated.yaml
  containers/survey_scan.sh collect --survey survey/cit-20260704
EOF
}

cmd="${1:-}"
if [ -z "${cmd}" ] || [ "${cmd}" = "-h" ] || [ "${cmd}" = "--help" ]; then
    usage
    exit 0
fi
shift

case "${cmd}" in
    survey)
        exec "${PYTHON}" "${HERE}/survey_scan/gpu_inventory.py" "$@"
        ;;
    emit-jobs)
        exec "${PYTHON}" "${HERE}/survey_scan/emit_condor_jobs.py" "$@"
        ;;
    collect)
        exec "${PYTHON}" "${HERE}/survey_scan/collect_results.py" "$@"
        ;;
    *)
        echo "Unknown survey_scan command: ${cmd}" >&2
        usage >&2
        exit 2
        ;;
esac
