#! /usr/bin/env python

import RIFT.lalsimutils as lalsimutils
import numpy as np

chi_max = 0.5
npts=2000
dat_all = np.empty((npts,13))
for i in np.arange(npts):
    m1  =np.random.uniform(50,60)
    m2  =np.random.uniform(0.8*m1,m1)
    P =lalsimutils.ChooseWaveformParams()
    P.randomize()
    s1x = chi_max * P.s1x
    s1y = chi_max * P.s1y
    s1z = chi_max * P.s1z
    s2x = chi_max * P.s2x
    s2y = chi_max * P.s2y
    s2z = chi_max * P.s2z

    line =[-1 , m1,m2,s1x, s1y,s1z,s2x, s2y,s2z, 100, 0.1, 100, -1]
    dat_all[i,:] = np.array(line)

np.savetxt("fake.composite",dat_all,header=" indx m1 m2 s1x s1y s1z s2x s2y s2z lnL sigmaL neff npts")
