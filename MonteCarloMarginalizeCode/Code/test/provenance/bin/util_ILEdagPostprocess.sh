#! /bin/bash
# util_NRdagPostprocess.sh
#
# GOAL
#   For NR-based DAGs, (a) consolidates the output, (b) runs ILE simplification, then (c) creates an NR-indexed version.
#   The second format uses a *portable* name, which is stable to me changing the underlying relationship between spins and label.


DIR_PROCESS=$1
BASE_OUT=$2
ECC=$3 # Liz (Capstone): this will only be non-blank in the case where my eccentric PE Makefile has inserted "--eccentricity" into join.sub


# clean them (=join duplicate lines)
echo " Input " `find ${DIR_PROCESS} -name 'CME*.dat' -exec cat {} \;`
echo " Output:  "  > $BASE_OUT.composite


