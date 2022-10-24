#! /usr/bin/env python
import numpy as np
import sys

dat = np.loadtxt(sys.argv[1])
print(np.mean(dat),np.std(dat))
