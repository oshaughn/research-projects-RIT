#!/bin/bash
# run_e2e_consistency.sh -- GPU<->CPU END-TO-END parity for the slowrot ILE (rotation + freqresponse).
#
# Runs the REAL integrate_likelihood_extrinsic_batchmode on a finite-size ILE case (frames/PSD/grid +
# case.json) four ways -- {rotation,finite} x {cpu,gpu} -- and asserts the marginalized lnL (evidence,
# .dat col $(NF-3)) agrees between CPU and GPU within TOL_SIGMA * combined sampler error.  This is the
# end-to-end complement to test_slowrot_gpu.py / test_slowrot_freqresponse_gpu.py (which prove the
# likelihood is bit-identical; this proves the WHOLE ILE -- precompute, device transfer, GPU
# likelihood_function branch, AV sampler -- runs and agrees to sampler noise).  Run in a cupy container
# on a GPU host (e.g. `make e2e`).  Exit 0 = PASS.
#
# Env: RIFT_CODE (required), PYTHON (python3), RIFT_E2E_CASE (an ILE case dir), TOL_SIGMA (4),
#      E2E_NEFF/E2E_NMAX/E2E_NCHUNK/E2E_SEED (see e2e_mkargs.py).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${RIFT_CODE:?set RIFT_CODE to the Code dir}"
PYTHON="${PYTHON:-python3}"
TOL_SIGMA="${TOL_SIGMA:-4}"
# Default case: the finite-size pipeline's CE-ET SNR30 inputs (frames + PSD xml + grid + case.json,
# with ile_common / ile_finite_extra).  Override RIFT_E2E_CASE to point at any such case dir.
RIFT_E2E_CASE="${RIFT_E2E_CASE:-/home/richard.oshaughnessy/RIFT_roboto_paper/analyses/slowrot_finite-size/3g/run_CE-ET_snr30}"
export PYTHONPATH="$RIFT_CODE:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$(mktemp -d)}"

for f in case.json grid.xml.gz; do
  if [ ! -e "$RIFT_E2E_CASE/$f" ]; then
    echo "ERROR: RIFT_E2E_CASE is missing '$f': $RIFT_E2E_CASE"
    echo "  Point RIFT_E2E_CASE at a finite-size ILE case dir: frames (*.gwf) + PSD (*-psd.xml.gz) +"
    echo "  grid.xml.gz + case.json (carrying ile_common / ile_finite_extra)."
    echo "  Regenerate one with the finite-size pipeline generator:"
    echo "    ~/RIFT_roboto_paper/analyses/slowrot_finite-size/3g/make_inputs.py  (writes run_<net>_snr<N>/)"
    exit 2
  fi
done

WORK="$(mktemp -d)"
cp "$RIFT_E2E_CASE"/*.gwf "$RIFT_E2E_CASE"/*.xml.gz "$RIFT_E2E_CASE"/case.json "$WORK"/ 2>/dev/null
cp "$HERE/e2e_mkargs.py" "$WORK"/
# Rebuild data.cache against the LOCAL frame copies (the submit-host absolute paths in the case's own
# data.cache may not resolve here).
( cd "$WORK" && $PYTHON - <<'PY'
import glob, os
lines = []
for g in sorted(glob.glob("*.gwf")):
    base = g[:-4]; parts = base.split("-")          # OBS-desc-START-DUR.gwf
    obs = parts[0]; dur = parts[-1]; start = parts[-2]; desc = "-".join(parts[1:-2])
    lines.append("%s %s %s %s file://localhost%s\n" % (obs, desc, start, dur, os.path.abspath(g)))
open("data.cache", "w").writelines(lines)
PY
)
cd "$WORK"

echo "=== slowrot GPU<->CPU end-to-end consistency ==="
echo "    case: $RIFT_E2E_CASE"
echo "    work: $WORK   (n-eff=${E2E_NEFF:-200}, tol=${TOL_SIGMA} sigma)"

declare -A LNL ERR
FAIL=0
for mode in rotation_cpu rotation_gpu finite_cpu finite_gpu; do
  args="$($PYTHON e2e_mkargs.py "$mode" "ile_$mode")"
  t0=$SECONDS
  $PYTHON "$RIFT_CODE/bin/integrate_likelihood_extrinsic_batchmode" $args > "run_$mode.log" 2>&1
  ec=$?; dt=$((SECONDS - t0))
  if grep -q "FAILED ANALYSIS" "run_$mode.log" || [ $ec -ne 0 ] || [ ! -s "ile_${mode}_0_.dat" ]; then
    echo "  $mode: FAILED (exit $ec, ${dt}s) -> $(grep -A1 'FAILED ANALYSIS' "run_$mode.log" | tail -1)"
    FAIL=1; continue
  fi
  # .dat trailing columns are always [... lnL sigma_lnL ntotal neff]; lnL=$(NF-3), sigma=$(NF-2).
  read -r lnL err < <(awk 'END{print $(NF-3), $(NF-2)}' "ile_${mode}_0_.dat")
  LNL[$mode]="$lnL"; ERR[$mode]="$err"
  echo "  $mode: OK (${dt}s)  lnL=$lnL  sigma=$err"
done
[ $FAIL -ne 0 ] && { echo "RESULT: FAILED (a mode crashed) -- see $WORK/run_*.log"; exit 1; }

$PYTHON - "$TOL_SIGMA" \
  "${LNL[rotation_cpu]}" "${ERR[rotation_cpu]}" "${LNL[rotation_gpu]}" "${ERR[rotation_gpu]}" \
  "${LNL[finite_cpu]}"   "${ERR[finite_cpu]}"   "${LNL[finite_gpu]}"   "${ERR[finite_gpu]}" <<'PY'
import sys, math
tol = float(sys.argv[1])
rc, rce, rg, rge, fc, fce, fg, fge = map(float, sys.argv[2:10])
ok = True
print("--- parity (marginalized lnL, GPU vs CPU) ---")
for name, (cpu, ecpu, gpu, egpu) in [("rotation", (rc, rce, rg, rge)),
                                     ("freqresponse", (fc, fce, fg, fge))]:
    d = abs(gpu - cpu); sig = math.hypot(ecpu, egpu); lim = tol * sig
    verdict = "PASS" if d <= lim else "FAIL"
    ok = ok and verdict == "PASS"
    print("  %-13s CPU=%+8.4f  GPU=%+8.4f  |d|=%.4f  <= %g*sigma=%.4f ?  %s"
          % (name, cpu, gpu, d, tol, lim, verdict))
print("RESULT:", "PASS -- GPU matches CPU within sampler noise" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
