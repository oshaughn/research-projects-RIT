#!/usr/bin/env bash
#
# make_reconstruct_subfile.sh  ILE.sub  [ILE_extr.sub]
#
# Turn a STANDARD RIFT pipeline ILE submit file into an "extrinsic reconstruction"
# submit file: one that emits fair-draw posterior samples each carrying its own
# coalescence time, ready for reconstruct_strain.py.
#
# Most end users run RIFT under HTCondor via the standard pipeline
# (util_RIFT_pseudo_pipe.py / create_event_parameter_pipeline_BasicIteration),
# which writes an ILE.sub.  Rather than hand-run ILE, copy that submit file and
# inject the two required flags, then submit the copy.  This keeps the cluster's
# accounting group, container (+SingularityImage), GPU request, and data/PSD
# arguments exactly as the pipeline configured them.
#
# Required additions (see ../README.md):
#   --fairdraw-extrinsic-output --resample-time-marginalization
# and we make sure --time-marginalization is present and --maximize-only is absent.
#
# After running this, submit with:   condor_submit ILE_extr.sub
# (or add it as a node in your DAG).  Then extract each output *_0_.xml.gz with
# ../extract_ile_samples.py and feed the .npz to ../reconstruct_strain.py.
set -e
SRC=${1:?usage: make_reconstruct_subfile.sh ILE.sub [ILE_extr.sub]}
DST=${2:-ILE_extr.sub}
ADD="--fairdraw-extrinsic-output --resample-time-marginalization"

cp "$SRC" "$DST"
# The condor 'arguments = ...' line holds the ILE command line.
python3 - "$DST" "$ADD" <<'PY'
import re, sys
path, add = sys.argv[1], sys.argv[2]
s = open(path).read().splitlines(keepends=True)
out = []
for line in s:
    if re.match(r'\s*arguments\s*=', line, re.I):
        core = line.rstrip("\n")
        # drop --maximize-only if present (it corrupts the per-sample lnL scale)
        core = core.replace("--maximize-only", "")
        # ensure time marginalization
        if "--time-marginalization" not in core:
            core += " --time-marginalization"
        # add the fair-draw + resample-in-time flags if not already present
        for flag in add.split():
            if flag not in core:
                core += " " + flag
        line = core + "\n"
    out.append(line)
open(path, "w").writelines(out)
print("wrote", path)
PY
echo "Created $DST from $SRC."
echo "Submit with:  condor_submit $DST     (output: <prefix>_0_.xml.gz)"
echo "Then:         ../extract_ile_samples.py <prefix>_0_.xml.gz samples.npz"
echo "              ../reconstruct_strain.py --samples samples.npz --fair-draw ..."
