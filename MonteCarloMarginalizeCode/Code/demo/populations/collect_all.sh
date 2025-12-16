#!/bin/bash

# Set your main directory where analysis_event_* live
MAIN_DIR="/home/muhammad.zeeshan/projects/research-projects-RIT/MonteCarloMarginalizeCode/Code/demo/populations/ecc_injections"

# Set output directory
OUT_DIR="./collected_dat_files"
mkdir -p "$OUT_DIR"

# Loop over all analysis_event_* directories
for EVENT_DIR in "$MAIN_DIR"/analysis_event_*; do
    # Make sure this is a directory
    [ -d "$EVENT_DIR" ] || continue

    # Extract event number from directory name
    EVENT_NAME=$(basename "$EVENT_DIR")           # example: analysis_event_0
    EVENT_NUM=${EVENT_NAME#analysis_event_}       # example: 0

    # Full path of the file inside the rundir
    SRC_FILE="$EVENT_DIR/rundir/extrinsic_posterior_samples.dat"

    if [[ -f "$SRC_FILE" ]]; then
        DEST_FILE="$OUT_DIR/event_${EVENT_NUM}.dat"
        echo "Copying $SRC_FILE → $DEST_FILE"
        cp "$SRC_FILE" "$DEST_FILE"
    else
        echo "WARNING: File not found in $EVENT_NAME"
    fi
done
