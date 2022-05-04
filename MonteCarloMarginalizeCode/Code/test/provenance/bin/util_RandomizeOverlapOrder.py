#! /usr/bin/env python
#
# util_RandomizeOverlapOrder.py
#
# 
# GOAL
#   - take an overlap.xml.gz (SimInspiralTable), and randomize its order.  
#   - Important when merging files from many workers, to avoid accidentally using only the output from one of them.  Run after ligolw_add
#   - takes equal number from each file by default ... WARNING THIS MEANS WE CAN HAVE PATHOLOGICAL SIZE LIMITS


import sys
from optparse import OptionParser
import numpy as np


optp = OptionParser()
optp.add_option("--fref",default=20,type=float,help="Reference frequency. Depending on approximant and age of implementation, may be ignored")
optp.add_option("--n-min",default=20,type=int,help="Minimum size of file to include")
optp.add_option("--output-file",default='merged_output',type=str,help="Merged output file")
opts, args = optp.parse_args()

print(" Inputs:  ", '\n\t  '.join(args))
print(" Output: ", opts.output_file)

import os
#os.system(" touch {} ".format(opts.output_file))
os.system(' echo {}  > {}'.format(' '.join(args), opts.output_file))
