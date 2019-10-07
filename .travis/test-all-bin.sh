#!/bin/bash
#
# Travis CI test runner: test --help for all command-line executables
# Author: Duncan Macleod <duncan.macleod@ligo.org> (2016)

# loop over all bin/ scripts
for EXE in MonteCarloMarginalizeCode/Code/bin/*; do
   # skip tests that have no --help
   if  [[ ${EXE} == *"util_CacheFileConvert"* ]]; then
        continue
    fi
   # skip tests that require NR
   if  [[ ${EXE} == *"util_NR"* ]]; then
        continue
    fi
    # get file-name as PATH executable
    EXENAME=`basename ${EXE}`
    EXEPATH=`which ${EXENAME}`
    # execute --help with coverage
    echo "Testing $EXEPATH --help..."
    python -m coverage run --append --source=RIFT ${EXEPATH} --help 1>/dev/null;
done
