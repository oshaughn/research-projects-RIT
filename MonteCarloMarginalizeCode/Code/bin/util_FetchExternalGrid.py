#! /usr/bin/env python
#
#
# INTENT
#    - retrieve external grid(s), deployed in target format, and merged
#    - used to create cross-run dependencies in iterative structure

# SPECIFICATION: argument file

# * json file
#   label
#   method    : file
#     label
#     command : some methods require commands
#     arguments  : command line arguments.  Note this will not include a target name, which must be allowed as the next argument
#   convert   : None, or block
#     label
#     command
#     arguments
#
# Example
#   {
    #   "method" : "native",
    #   "source":  rundir   # will take latest grid directly, verbatim
    #   "n_max": 1000   # cap on number of points taken
    # }
    # }
# target methods
#    - copy (copy existing file)
#    - copy_latest (copy last item matching pattern, using numerical sort)



# **FETCH SPECIFICATION**


# * json file.  Default is ONE JSON PER EXECUTABLE.
#   label
#   method    : file
#     label
#     command : some methods require commands
#     arguments  : command line arguments
#   convert   : None, or block
#     label
#     command
#     arguments
   

from RIFT.misc.samples_utils import add_field,extract_combination_from_LI
import argparse
import sys
import numpy as np
import numpy.lib.recfunctions
import functools
import itertools
import os

import json
import shutil, glob,re


def retrieve_native(sourcedir,outfile,n_max=None,base_pattern="overlap-grid-*.xml.gz",verbose=True):
    """
    retrieve_native(sourcedir,outfile)

    sourcedir : source directory.  Looks for last file of form "output-grid-N.xml.gz
    outfile : target file name, assume it is full path
    n_max :
    """
    if verbose:
        print("Checking ", sourcedir, " for ", base_pattern)
    # Identify the correct source file in the directory
    fnames = glob.glob(sourcedir+"/"+base_pattern)  # give flexibility to naming/reuse of this code
    fnames.sort(key=lambda f:int(re.sub('\D', '', f))) # trick from https://pretagteam.com/question/how-to-sort-files-by-number-in-filename-of-a-directory-duplicate
    # if verbose:
    #     print(fnames)
    fname_to_use = fnames[-1]

    # If n_max is not None, load in the file, truncate its size
    if n_max is None:
        if verbose:
            print(" Transferring ", fname_to_use, " -> ", outfile)
        shutil.copyfile(fname_to_use, outfile)
    elif n_max > 0:
        import random
        import RIFT.lalsimutils as lalsimutils
        # Load in grid
        P_list = lalsimutils.xml_to_ChooseWaveformParams_array(fname_to_use)
        # select points randomly!
        P_list_reduced = random.sample(P_list, int(n_max))
        lalsimutils.ChooseWaveformParams_array_to_xml(P_list, outfile)
    else:
        print(" Invalid fetch size ", n_max)
        import sys; sys.exit(99)
    return None


parser = argparse.ArgumentParser()
parser.add_argument("--input-json",type=str,default="fetch.json",help="input file")
parser.add_argument("--inj-file-out",type=str,default="merged_grid",help="output file")
opts=  parser.parse_args()


rundir = os.getcwd()

config=None
with open(opts.input_json,'r') as f:
    config = json.load(f)

method = config['method']
if method =='native':
    retrieve_native(config['source'],opts.inj_file_out)
