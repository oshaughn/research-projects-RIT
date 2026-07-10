#!/usr/bin/env bash
# config.sh -- central configuration for the GW150914 waveform-MODEL strain
# reconstruction demo. Sourced by every script and by the Makefile.
#
# Self-contained: downloads GW150914 O1 open data from GWOSC, runs a full RIFT
# parameter-estimation DAG (IMRPhenomD, aligned spin pinned to zero), then
# reconstructs the whitened strain band from the resulting posterior.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export MODEL_DIR="$HERE"

# ---------------------------------------------------------------------------
# RIFT execution environment (conda deps + RIFT source tree on PATH/PYTHONPATH)
# ---------------------------------------------------------------------------
# The pipeline GENERATOR (util_RIFT_pseudo_pipe.py, convert_psd_ascii2xml,
# util_WriteInjectionFile.py, util_SimInspiralToCoinc.py) runs on THIS submit
# node.  We supply its python dependencies (lal, gwpy, igwn_ligolw, cupy-less
# numpy path) from an igwn conda env, and the RIFT package + bin/ from the
# source checkout so we run exactly this RIFT version.
export CONDA_SH="${CONDA_SH:-/cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh}"
export CONDA_ENV="${CONDA_ENV:-test_junior_o4d}"
export RIFT_SRC="${RIFT_SRC:-/home/richard.oshaughnessy/RIFT_ralph}"
export RIFT_CODE="$RIFT_SRC/MonteCarloMarginalizeCode/Code"

# rift_env(): activate conda + prepend the RIFT source tree. Idempotent.
# conda's activate script trips `set -e`/`set -u` (it touches unbound vars), so we
# disable those options across activation and restore the caller's state after.
rift_env() {
  local _opts=$-
  set +eu
  # shellcheck disable=SC1090
  source "$CONDA_SH" 2>/dev/null || true
  conda activate "$CONDA_ENV" 2>/dev/null || true
  case "$_opts" in *e*) set -e;; esac
  case "$_opts" in *u*) set -u;; esac
  export PYTHONPATH="$RIFT_CODE:${PYTHONPATH:-}"
  export PATH="$RIFT_CODE/bin:$PATH"
  # CIT thread caps (RLIMIT_NPROC on loaded hosts kills numpy import mid-build)
  export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
}
export PYBIN="python"

# ---------------------------------------------------------------------------
# Container (CIT-local: container universe + local .sif file transfer;
# OSDF .sif delivery fails to open on CIT execute nodes -- see submit.sh)
# ---------------------------------------------------------------------------
export CONTAINER_SIF="${CONTAINER_SIF:-/home/richard.oshaughnessy/rift_cit_build_container_family/built_containers_timeinterp_20260707/rift_o4d-calmarg_in_loop_cc60-90_cuda118_20260707.sif}"
export SINGULARITY_RIFT_IMAGE="$CONTAINER_SIF"
export SINGULARITY_BASE_EXE_DIR="/usr/local/bin/"
# durable CIT DAG-build settings (mirror the manual submit.sh fixes)
export RIFT_GETENV='*' RIFT_GETENV_OSG='*'
export RIFT_CONTAINER_UNIVERSE=1
export RIFT_REQUIRE_GPUS='(Capability >= 6.0)'
# GPU capability band the cc60-90 cuda11.8 image supports (floor .. ceiling)
export GPU_CAP_FLOOR=6.0 GPU_CAP_CEIL=9.0

# CIT-local fix tools (read-only, used by submit.sh)
export FIX_TOOLS="${FIX_TOOLS:-/home/richard.oshaughnessy/LVK/O4_era/investigations/rift_transverse_highSNR_study/tools}"

# --- LIGO accounting (REQUIRED at DAG-build time) --------------------------
# The CIT schedd has a submit transform (/etc/condor/config.d/99-transform) that
# derives LigoSearchTag from the job's accounting_group and REJECTS the submit
# ("Invalid value for search tag: None") if it is unset.  RIFT bakes
# accounting_group into every sub only when these env vars are set at build time;
# without them the build warns "LIGO accounting information not available".
export LIGO_USER_NAME="${LIGO_USER_NAME:-richard.oshaughnessy}"
export LIGO_ACCOUNTING="${LIGO_ACCOUNTING:-ligo.dev.o4.cbc.pe.rift}"

# ---------------------------------------------------------------------------
# Event: GW150914 (O1; H1 + L1 only, no Virgo)
# ---------------------------------------------------------------------------
export EVENT_NAME=GW150914
export EVENT_TIME=1126259462.4        # GPS
export IFOS="H1 L1"

# --- GWOSC O1 4 kHz open data --------------------------------------------
# 4096-s open-data block covering GPS 1126259462 (block start 1126256640).
# IMPORTANT: the in-frame channel for O1 open data is <IFO>:LOSC-STRAIN, NOT
# GWOSC-4KHZ_R1_STRAIN (that is the O4 naming).  Verified by reading the frame
# channel table (gwpy iter_channel_names).  ILE/pseudo_pipe prepend '<IFO>:',
# so scripts store it bare ('LOSC-STRAIN'); the ini stores it prefixed.
export FRAME_BLOCK=1126256640
export FRAME_DUR=4096
export CHANNEL_BARE="LOSC-STRAIN"       # used by ILE-style '--channel-name IFO=...'
export FRAME_TAG_H="H1_LOSC_4_V1"
export FRAME_TAG_L="L1_LOSC_4_V1"

# --- analysis settings (IMRPhenomD, aligned spin, spins pinned to zero) ---
export APPROX=IMRPhenomD
export LMAX=2
export FMIN=20
export FREF=20
export FMAX=1024                       # GW150914 is heavy (Mc~28); srate 4096
export SRATE=4096
export SEGLEN=8                         # short segment (heavy system)
export MC_RANGE="[25,35]"              # force chirp-mass window (Mc~28)
export DIST_MAX=1000                    # Mpc

# --- coinc seed (masses are ONLY a time/IFO seed; grid is proposed fresh) --
export SEED_M1=36 SEED_M2=29 SEED_S1Z=0 SEED_S2Z=0
export SEED_DIST=410 SEED_SNR=24

# --- pipeline size knobs --------------------------------------------------
export NIT=6                           # iterations
export ILE_NEFF=1000
export JOBS_PER_WORKER=100
export NSAMP_LAST=20000                # final fair-draw extrinsic samples
export SRATE_TIME=4096                 # --internal-ile-srate-time-resampling
# per-job disk for OSG file transfer (frames + PSDs ride along)
export DISK_ILE=16G DISK_CIP=16G DISK_GEN=16G

# --- PSD estimation (off-source, gwpy) ------------------------------------
# ~400 s of clean data ~1360 s BEFORE the event (offset 1060 s into the block).
# NB: the first ~500 s of the L1 O1 block contains a data-quality gap (NaNs), so
# we deliberately start the PSD window past it.  Event is at block offset 2822 s.
export PSD_SEG_START=$((FRAME_BLOCK + 1060))   # 1126257700
export PSD_SEG_LEN=400
export PSD_FFTLEN=4                     # s (Welch/median FFT length)

# ---------------------------------------------------------------------------
# Paths / filenames
# ---------------------------------------------------------------------------
export DATA_DIR="$MODEL_DIR/data"
export CACHE="$DATA_DIR/event.cache"
export INI="$MODEL_DIR/GW150914_D.ini"
export COINC="$MODEL_DIR/coinc.xml"
export RUNDIR="$MODEL_DIR/rundir_gw150914_D"
export H1_PSD="$MODEL_DIR/H1-psd.xml.gz"
export L1_PSD="$MODEL_DIR/L1-psd.xml.gz"
export PP_DAG="marginalize_intrinsic_parameters_BasicIterationWorkflow.dag"
export POSTERIOR="$RUNDIR/extrinsic_posterior_samples.dat"
export SAMPLES_NPZ="$MODEL_DIR/gw150914_samples.npz"
export OUT_PNG="$MODEL_DIR/GW150914_reconstruction.png"

# shared reconstruction tools (one level up; read-only)
export RECON_PY="$MODEL_DIR/../reconstruct_strain.py"
export DAT2COMPACT_PY="$MODEL_DIR/../dat_to_compact.py"
