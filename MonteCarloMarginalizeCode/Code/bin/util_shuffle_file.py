#! /usr/bin/env python

import random
import numpy as np

import sys

fname = sys.argv[1]
dat = np.genfromtxt(fname,names=True)
p = np.random.permutation(len(dat))
dat_out  = dat[p]
fname_out = sys.argv[2]
np.savetxt(fname_out,dat_out,header='# ' + ' '.join(dat.dtype.names))
