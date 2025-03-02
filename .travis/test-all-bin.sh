#!/bin/bash
#
# Travis CI test runner: test --help for all command-line executables
# Author: Duncan Macleod <duncan.macleod@ligo.org> (2016)

# Environment variables
export GW_SURROGATE=''
export EOB_BASE=''
export EOB_ARCHVE=''
export EOB_C_BASE=''
export EOB_C_ARCHIVE=''
export EOB_C_ARCHIVE_NMAX=1
export EOS_TABLES=''
export LALSIMULATION_DATADIR=''

set -e

# loop over all bin/ scripts
for EXE in MonteCarloMarginalizeCode/Code/bin/*; do
   # skip scripts with explicit bilby dependence
   if [[ ${EXE} == *'calibration_reweighting.py' ]]; then
       continue
   fi
   if [[ ${EXE} == *'combine_weights_and_rejection_sample.py' ]]; then
       continue
   fi
   # skip scripts with pesummary dependence
   if [[ ${EXE} == *'convert_output_format_ascii2h5.py' ]]; then
       continue
   fi
   if [[ ${EXE} == *'make_uni_comov_skymap.py' ]]; then
       continue
   fi
   if [[ ${EXE} == *'util_shuffle_file.py' ]]; then
       continue
   fi
#   if [[ ${EXE} == *'resample_uniform_comoving.py' ]]; then
#       continue
#   fi
   # skip non-python scripts
   if  [[ ${EXE} == *".sh" ]]; then
        echo " Not python : " ${EXE}
        continue
    fi
   if  [[ ${EXE} == *"switcheroo" ]]; then
        echo " Not python : " ${EXE}
        continue
    fi
   if  [[ ${EXE} == *"AnalyticFisher"* ]]; then
        echo " Analytic fisher  : " ${EXE}
        continue
    fi
   # skip tests that have no --help
   if  [[ ${EXE} == *"util_CacheFileConvert"* ]]; then
        continue
    fi
   if  [[ ${EXE} == *"util_CleanILE"* ]]; then
        continue
    fi
   # skip tests that require NR
   if  [[ ${EXE} == *"util_"*"NR"* ]]; then
        echo " NR-based code : " ${EXE}
        continue
   fi
   # skip tests that require condor environment
   if [[ ${EXE} == *"check_CIP_complete_work.py" ]]; then
         continue
   fi
   if  [[ ${EXE} == *"util_CompareWaveformsInDetectors.py"* ]]; then
        continue
    fi
    # get file-name as PATH executable
    EXENAME=`basename ${EXE}`
    EXEPATH=`which ${EXENAME}`
    # execute --help with coverage
    echo "Testing $EXEPATH --help..."
    python3 -m coverage run --append --source=RIFT ${EXEPATH} --help 1>/dev/null;
done
