#
# sky_rotations.py
#   Rotate the sky to a specific detector frame pair (usually HL)
#   Note this currently uses GLOBAL STATE VARIABLES for efficiency
#
#   this is *horribly inefficient nonvectorized code*
#   Might want to call astropy or some related library that does this better.\\
#
# NOTE
#   - in practice, we should be rotating the vectors defining the detetcor response etc instead of rotating all our sky
# coordinates every iteration.

import numpy as np
import lalsimulation as lalsim
import lal
import RIFT.lalsimutils as lalsimutils
import functools

# defaults at start
# these global variables don't work as well as we would like
frm=np.diag(np.ones(3))
frmInverse=np.diag(np.ones(3))
vecZ = np.array([0,0,1])
vecZnew = np.array([0,0,1])


def assign_sky_frame(det0,det1,theEpochFiducial):
    global frm
    global frmInverse
    global vecZ
    global vecZnew
    theDetectorSeparation = lalsim.DetectorPrefixToLALDetector(det0).location - lalsim.DetectorPrefixToLALDetector(det1).location
    vecZ = np.array(theDetectorSeparation/np.sqrt( np.dot(theDetectorSeparation,theDetectorSeparation)))
    time_angle =  np.mod( lal.GreenwichMeanSiderealTime(theEpochFiducial), 2*np.pi)
    vecZnew = np.dot(lalsimutils.rotation_matrix(np.array([0,0,1]), -time_angle), vecZ)
    frm = lalsimutils.VectorToFrame(vecZnew)   # Create an orthonormal frame related to this particular choice of z axis. (Used as 'rotation' object)
    frmInverse= np.asarray(np.matrix(frm).I)                                    # Create an orthonormal frame to undo the transform above


# USE INSTEAD
#  functools(lalsimutils.polar_angles_in_frame_alt,frm)
def rotate_sky_forwards_scalar(theta,phi,frm=frm):   # When theta=0 we are describing the coordinats of the zhat direction in the vecZ frame
#    global frm
    return lalsimutils.polar_angles_in_frame_alt(frm, theta,phi)

# superfluous
# USE INSTEAD
#  functools(lalsimutils.polar_angles_in_frame_alt,frm)
def rotate_sky_forwards(th,ph,frm=frm):
    th_out= np.zeros(len(th))
    ph_out= np.zeros(len(th))
    for indx in np.arange(len(th)):
        th_out[indx],ph_out[indx] = rotate_sky_forwards_scalar(th[indx],ph[indx],frm=frm)
        return th_out, ph_out
    

#  functools(lalsimutils.polar_angles_in_frame_alt,frmInverse)
def rotate_sky_backwards_scalar(theta,phi,frmInverse=frmInverse): # When theta=0, the vector should be along the vecZ direction and the polar angles returned its polar angles
#    global frmInverse
    return lalsimutils.polar_angles_in_frame_alt(frmInverse, theta,phi)

#  functools(lalsimutils.polar_angles_in_frame_alt,frmInverse)
def rotate_sky_backwards(th,ph,frmInverse=frmInverse):
    th_out= np.zeros(len(th))
    ph_out= np.zeros(len(th))
    for indx in np.arange(len(th)):
            th_out[indx],ph_out[indx] = rotate_sky_backwards_scalar(th[indx],ph[indx],frmInverse=frmInverse)
    return th_out, ph_out
