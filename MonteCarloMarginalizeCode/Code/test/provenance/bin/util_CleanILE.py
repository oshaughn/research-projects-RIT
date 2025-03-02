#! /usr/bin/env python
# util_CleanILE.py
#
#  Reads FILE (not stdin). Consolidates ILE entries for the same physical system.
#  Compare to: util_MassGriCoalesce.py


import sys
import os
import RIFT.misc.xmlutils as xmlutils
#from optparse import OptionParser
from igwn_ligolw import lsctables, table, utils

import numpy as np
import RIFT.misc.weight_simulations as weight_simulations

import fileinput
#import StringIO

data_at_intrinsic = {}

my_digits=5  # safety for high-SNR BNS

tides_on = False
distance_on = False  
col_intrinsic = 9

import argparse
parser = argparse.ArgumentParser(usage="util_CleanILE.py fname1.dat fname2.dat ... ")
parser.add_argument("fname",action='append',nargs='+')
parser.add_argument("--eccentricity", action="store_true")
opts = parser.parse_args()


print(" Input: ", ' '.join(opts.fname))
print(" Output: stdout")
