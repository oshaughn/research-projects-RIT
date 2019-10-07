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
export EOS_TABLES=''
export LALSIMULATION_DATADIR=''

# loop over all bin/ scripts
for EXE in MonteCarloMarginalizeCode/Code/bin/*; do
   # skip non-python scripts
   if  [[ ${EXE} == *".sh" ]]; then
        echo " Not python : " ${EXE}
        continue
    fi
   if  [[ ${EXE} == *"switcheroo" ]]; then
        echo " Not python : " ${EXE}
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
   if  [[ ${EXE} == *"util_NR"* ]]; then
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
    python -m coverage run --append --source=RIFT ${EXEPATH} --help 1>/dev/null;
done
