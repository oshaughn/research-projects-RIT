#!/usr/bin/env bash
# Config for the GW190521 NR strain reconstruction (RIT-Five eBBH-1794).
# Sourced by reconstruct.sh / run_extrinsic_fairdraw.sh.

# --- environment ---
ENV=/home/patricia.mcmillin/.conda/envs/myigwn-py311
export PYBIN=$ENV/bin/python
export ILEBIN=$ENV/bin/integrate_likelihood_extrinsic_batchmode
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXTRACT_PY=$HERE/../extract_ile_samples.py

# --- where to work / write ---
export WORKDIR=/home/patricia.mcmillin/rift_nr/gw190521/reconstruction_demo
mkdir -p "$WORKDIR"

# --- event / analysis settings (match the intrinsic analysis) ---
export EVENT_TIME=1242442967.459473
export EVENT_NAME=GW190521
export SIM_ID=RIT-eBBH-1794
export SRATE=512
export FMAX=224.0
export FLOW=20.0
export D_MIN=2000          # Mpc; brackets the distance posterior, excludes <~2 Mpc NaN region
export D_MAX=13000
export N_MAX=2000000
export N_EFF=3000

# --- NR simulation (fixed) ---
export NR_GROUP=RIT-Five
export NR_PARAM=ExtrapStrain_RIT-eBBH-1794-n100.h5
RD=/home/patricia.mcmillin/rift_nr/gw190521/190521_RIT_Five_aligned_only
export SIM_XML=/home/patricia.mcmillin/rift_nr/gw190521/lvk_peak_NR_wf_plot/RIT-Five_aligned_only/eBBH-1794/overlap-grid-extrinsic.xml.gz

# --- data + PSDs for ILE (must be copied into WORKDIR or referenced by abs path) ---
export DATA_ARGS="--cache $RD/local.cache \
  --channel-name H1=DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01 --psd-file H1=$RD/H1-psd.xml.gz --fmin-ifo H1=20 \
  --channel-name L1=DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01 --psd-file L1=$RD/L1-psd.xml.gz --fmin-ifo L1=20 \
  --channel-name V1=Hrec_hoft_16384Hz          --psd-file V1=$RD/V1-psd.xml.gz --fmin-ifo V1=20"

# --- PSDs for the reconstruction plot (whitening; H1/L1 shown) ---
export PLOT_PSD_ARGS="--psd-file H1=$RD/H1-psd.xml.gz --psd-file L1=$RD/L1-psd.xml.gz"

# --- optional: weight waveforms along the NR mass curve (single point here => no-op) ---
export INTRINSIC=/home/patricia.mcmillin/rift_nr/gw190521/lvk_peak_NR_wf_plot/RIT-Five_aligned_only/eBBH-1794/intrinsic_params_RIT-eBBH-1794-n100.dat

# --- reconstruction / accumulation controls ---
export TARGET_SAMPLES=150
export MAX_RUNS=12
export NPROC=6
export TLO=-0.10
export THI=0.06
export OUT_PNG=$WORKDIR/GW190521_reconstruction.png
