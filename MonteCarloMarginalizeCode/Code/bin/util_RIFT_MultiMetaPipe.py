#! /usr/bin/env python
#
#  a lazy user syntax for constructing *multiple* RIFT pseudo pipe commands, based on a core command.  
#  Designed for a single event analysis, with multiple related (possibly concurrent) analyses feeding one another
#
#   https://htcondor.readthedocs.io/en/latest/apis/python-bindings/tutorials/DAG-Creation-And-Submission.html
#  NOT using pipeline.py (ao more!)

# common_args argstring("...")
# A  argstring(" --arg1 val1 --arg2 --arg3 ... ")
# B  argstring("...")
# C  ...
# D ...
# parent A child B
# parent B child C
# parent B child D
# B will fetch from A, C and D from B using the native fetch syntax

# we can even have a synchronous_child option, for children that we know are slower, so they can start at the same time and not wait

# this then builds a submit DAG, and runs the job building pipeline for all the specified jobs


import argparse
import numpy as np
try:
    import htcondor
except:
    print(" - no htcondor - ")
    exit(0)


parser = argparse.ArgumentParser()
parser.add_argument("--workflow",help="input file, pseudo-dag specification. ")
opts= parser.parse_args()


