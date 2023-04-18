#! /usr/bin/env python
#
#  Usage
#     Workaround for an analysis being done in J frame unintentionally.
#     Note fref is MANDATORY for such a conversion

import numpy as np

import lalsimulation as lalsim
import RIFT.lalsimutils as lalsimutils
import numpy.lib.recfunctions as rfn
from optparse import OptionParser

optp = OptionParser()
optp.add_option("--extrinsic-posterior-file",default=None,type=str,help="Input file")
optp.add_option("--fname-out",default="rotated-samples.dat",type=str,help="output file")
optp.add_option("--fref",default=20,type=float,help="Reference frequency. Depending on approximant and age of implementation, may be ignored")
opts, args = optp.parse_args()



# 
dat = np.genfromtxt(opts.extrinsic_posterior_file, names=True)
npts = len(dat['m1'])
dat_out =dat.copy()


for indx in np.arange(npts):
    # Should use the vectorized convert tool, but not immplemented for this yet
    P =lalsimutils.ChooseWaveformParams()
    # Intrinsic parameters
    P.assign_param('m1', dat['m1'][indx])
    P.assign_param('m2', dat['m2'][indx])
    P.assign_param('s1x', dat['a1x'][indx])
    P.assign_param('s1y', dat['a1y'][indx])
    P.assign_param('s1z', dat['a1z'][indx])
    P.assign_param('s2x', dat['a2x'][indx])
    P.assign_param('s2y', dat['a2y'][indx])
    P.assign_param('s2z', dat['a2z'][indx])
    P.phiref = dat['phiorb'][indx]   # phase convention for this waveform specifically, its modes, might be needed here
    P.assign_param('fref', opts.fref)
    # stable extrinsic parameters that we *need* to assign: we've done PE with waveforms such that the z axis is 'J', and thus
    P.assign_param('psiJ', dat['psi'][indx])
    P.assign_param('thetaJN', dat['incl'][indx])
    # Should do something to phiJL. This needs to be one of the two points such that the cone specified by thetaJN intersects the cone specified by thetaJL, then converted to polar angle...

    # We could do it via a frame transformation: the z axis (L) maps to the direction specified by 
    P.assign_param('phiJL',0)  # 
    dat_out['psi'][indx] = P.psi
    dat_out['incl'][indx] = P.incl
    dat_out['phiorb'][indx] = np.mod(P.phiref, 2*np.pi)

np.savetxt( opts.fname_out,dat_out,header='# ' + ' '.join(dat_out.dtype.names))
