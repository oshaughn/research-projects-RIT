#! /usr/bin/env python
#
# GOALS
#   - takes top level directory name for BasicIteration DAG workflow
#   - uses ppc-args.txt to perform a set of plots for those iterations
#
# EXAMPLE
#   ./driver_PlotAllIterations.py --working-directory analyze_0_SEOBNRv3/0-SEOBNRv3-fmin20-Lmax2-Iterative-v0_cit --ppc-args ppc-args.txt



import argparse
import sys
import os
import shutil
import numpy as np
import lalsimutils
import lalsimulation as lalsim
import lal
import functools
import itertools

from glue import pipeline # https://github.com/lscsoft/lalsuite-archive/blob/5a47239a877032e93b1ca34445640360d6c3c990/glue/glue/pipeline.py

import dag_utils
from dag_utils import mkdir
from dag_utils import which

parser = argparse.ArgumentParser()
parser.add_argument("--working-directory",default="./")
parser.add_argument("--ppc-args",default=None,help="filename of args_ppc.txt file  which holds PPC arguments.  Should NOT contain --posterior-file , label, or line styles: these will be auto-set by the script")
opts=  parser.parse_args()

with open(opts.ppc_args) as f:
    ppcargs_list = f.readlines()
ppcargs = ' '.join( map( lambda x: x.replace('\\',''),ppcargs_list) )
ppcargs = ' '.join(ppcargs.split(' ')[1:])
# Some argument protection for later
ppcargs = ppcargs.replace('[', ' \'[')
ppcargs = ppcargs.replace(']', ']\'')
ppcargs=ppcargs.rstrip()
print("CIP", ppcargs)


os.chdir(opts.working_directory)

exe = dag_utils.which("plot_posterior_corner.py")
cvt = dag_utils.which("convert_output_format_ile2inference")




cmd = exe + ' ' + ppcargs
cmd += ' --use-legend '
import glob
fnames = glob.glob("overlap-grid-*.xml.gz")
indx_names = map( lambda x: int(x.replace('overlap-grid-','').replace('.xml.gz','')),fnames)
for indx in indx_names:
    fname_xml = "overlap-grid-"+str(indx)+".xml.gz"
    fname_post = "posterior_samples-"+str(indx)+".dat"
    os.system(cvt + " "+fname_xml + " > " + fname_post) 
    new_args = ' --posterior-file ' + fname_post + ' --posterior-label ' + str(indx)
    cmd += new_args

fname_comp= glob.glob("all.net")
if len(fname_comp) >0:
    cmd+= ' --composite-file all.net '

print(cmd)
os.system(cmd)

