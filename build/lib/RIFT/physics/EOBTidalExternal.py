#
# EOBTidalExternal
# 
#  Provides interface to Bernuzzi EOB model
#
# REQUIRES
#    - matlab
#    - run_goeob_standalone.sh  # created by EOB code interface
#
# PROVIDES
#    - hlmoft
#
# COMPARE TO
#    NRWaveformCatalogManager3   : very similar interface


debug_output = False

import numpy as np
import os
import sys
import shutil
import lalsimulation as lalsim
import lal

from scipy.interpolate import interp1d

import pickle

from .. import lalsimutils

rosUseArchivedWaveforms = True

rosDebug = False
#dirBaseFiles =os.environ["HOME"] + "/unixhome/Projects/LIGO-ILE-Applications/ILE-Tides/MatlabCodePolished"
dirBaseFiles =os.environ["EOB_BASE"]
dirBaseFilesArchive =os.environ["EOB_ARCHIVE"]
dirBaseMatlab=os.environ["MATLAB_BASE"]

#default_interpolation_kind = 'quadratic'  # spline interpolation   # very slow! 
default_interpolation_kind = 'linear'  # spline interpolation   # very slow! 

internal_ModesAvailable = [(2,2), (2,1), (2,-2), (2,-1), (3,3), (3,2), (3,1), (3,-3), (3,-2), (3,-1)]

MsunInSec = lal.MSUN_SI*lal.G_SI/lal.C_SI**3




def myzero(arg):
    return 0
def RangeWrap1d(bound, val,fn):
    return fn   # IDEALLY not necessary with modern interp1d

def ModeToString(pair):
    return str(pair[0])+str(pair[1])   # this is only used for POSITIVE l,m (single digit)



class WaveformModeCatalog:
    """
    Class containing EOB tidal harmonics,  both in dimensionless and dimensional form
    """


    def __init__(self, P,  lmax=2,
                 align_at_peak_l2_m2_emission=True, mode_list_to_load=[],build_fourier_time_window=1000,clean_with_taper=True,use_internal_interpolation_deltaT=None,build_strain_and_conserve_memory=False,reference_phase_at_peak=None,fix_phase_zero_at_coordinate=False):
        self.P  = P
        self.quantity = "h"
        self.fOrbitLower =0.    #  Used to clean results.  Based on the phase of the 22 mode
        self.fMinMode ={}
        # Mode storage convention
        #   - event time in first element
        #     - t=0 MAY be at peak, if opts.align_at_peak_l2m2_emission.  Cannot promise...but raw samples show the valid range
        #     - t in seconds
        #   - h*r/M in second element
        self.waveform_modes = {}
        self.waveform_modes_uniform_in_time={}
        self.waveform_modes_nonuniform_smallest_timestep = {}
        self.waveform_modes_nonuniform_largest_timestep = {}
        self.waveform_modes[(2,2)] = []
        self.waveform_modes_complex = {}
        self.waveform_modes_complex_interpolated = {}
        self.waveform_modes_complex_interpolated_amplitude = {}
        self.waveform_modes_complex_interpolated_phase = {}
        self.waveform_modes_complex_padded = {}
        self.waveform_modes_fourier = {}

        # Require spins zero for now
        if any([P.s1x,P.s1y,P.s1z,P.s2x,P.s2y,P.s2z]):
            print(" FAILURE: Tidal code assumes a nonspinning approximant for now")

        # Run the external script with the necessary parameters
        # Assume environment set correctly
        # Delete files from previous case"
        m1InMsun = P.m1/lal.MSUN_SI
        m2InMsun = P.m2/lal.MSUN_SI
        m1InMsun, m2InMsun = reversed(sorted([m1InMsun, m2InMsun]))   # FORCE m1 > m2

        # Convert lambda's to kappa
        # Note we are passed lambda1, lambda2.  We convert to (l=2) kappaA, kappaB
        #  - Generate lambdatilde
        #  - generate lambda2
        LambdaTilde, DeltaLambdaTilde = lalsimutils.tidal_lambda_tilde(m1InMsun, m2InMsun, P.lambda1,P.lambda2)
        kappaA2, kappaB2 =  lalsimutils.barlamdel_to_kappal( m1InMsun/m2InMsun, LambdaTilde,2)

        fname_base = "stored_"+"_".join([str(m1InMsun),str(m2InMsun),str(kappaA2),str(kappaB2),str(P.fmin)]) +".dir";
        print("  Saving to file (beware collisions!) ", fname_base)

        retrieve_directory = ''
        cwd = os.getcwd(); 
        if rosUseArchivedWaveforms and os.path.exists(dirBaseFilesArchive+"/"+fname_base) and os.path.exists(dirBaseFilesArchive+"/"+fname_base+"/test_h22.dat"):
            retrieve_directory = dirBaseFilesArchive+"/"+fname_base
            print(" Attempting to use archived waveform data  in ", retrieve_directory)
        else:
            retrieve_directory = dirBaseFilesArchive+"/"+fname_base + "/"
            cmd = dirBaseFiles+"/run_goeob_standalone.sh " + dirBaseMatlab + " " +str(m1InMsun) + " "+str(m2InMsun) + " " + str(kappaA2) + " " + str(kappaB2) + " " + str(P.fmin) + " " + retrieve_directory
            # Create directory 
            if not os.path.exists(retrieve_directory):
                print(" Making directory to archive this run ... ", retrieve_directory)
                os.makedirs(retrieve_directory)
            print(" Generating tidal EOB with ", cmd)
            os.chdir(dirBaseFiles); os.system(cmd); 
            # retrieve_directory = dirBaseFiles
            # if rosUseArchivedWaveforms:
            #     retrieve_directory = dirBaseFilesArchive+"/"+fname_base
            #     # Create directory 
            #     if not os.path.exists(retrieve_directory):
            #         print " Making directory to archive this run ... ", retrieve_directory
            #         os.makedirs(retrieve_directory)
            #     # Copy
            #     print " Copying files into storage directory ", retrieve_directory
            #     shutil.copyfile(dirBaseFiles+"/test_h22.dat", retrieve_directory+"/test_h22.dat")
            #     shutil.copyfile(dirBaseFiles+"/test_h21.dat", retrieve_directory+"/test_h21.dat")
            #     shutil.copyfile(dirBaseFiles+"/test_h33.dat", retrieve_directory+"/test_h33.dat")
            #     shutil.copyfile(dirBaseFiles+"/test_h32.dat", retrieve_directory+"/test_h32.dat")
            #     shutil.copyfile(dirBaseFiles+"/test_h31.dat", retrieve_directory+"/test_h31.dat")




        # First loop: Create all the basic mode data
        # This should ALREADY BE IN PHYSICAL TIME UNITS but have UNPHYSICAL distance scales
        nu = lalsimutils.symRatio(P.m1,P.m2)
        delta = (m1InMsun- m2InMsun)/(m1InMsun+m2InMsun)
        for mode in internal_ModesAvailable:
            if mode[0]<= lmax:   
                # Note all modes are REFLECTION-SYMMETRIC
                self.waveform_modes[mode] = np.loadtxt(retrieve_directory+"/test_h"+ModeToString( (mode[0], np.abs(mode[1])))+".dat")
                self.waveform_modes_nonuniform_smallest_timestep[mode] = self.waveform_modes[mode][1,0]-self.waveform_modes[mode][0,0]  # uniform in time
                self.waveform_modes_nonuniform_largest_timestep[mode] = self.waveform_modes[mode][1,0]-self.waveform_modes[mode][0,0]  # uniform in time
                self.waveform_modes_uniform_in_time[mode] =True
                  
                # Add factor of 'nu' that was missing
                if mode[1] %2 ==0 :
                    self.waveform_modes[mode][:,1]*= nu
                    self.waveform_modes[mode][:,2]*= nu
                else:
                    self.waveform_modes[mode][:,1]*= nu*delta
                    self.waveform_modes[mode][:,2]*= nu*delta
                
                # Create, if symmetric
                if mode[1]<0:  # (-1)^l conjugate
                      self.waveform_modes[mode][:,1] *= (-1)**mode[0]
                      self.waveform_modes[mode][:,2] *= -1.*(-1)**mode[0]


                if clean_with_taper:
                    # chop off the first 1 seconds. 
                    # use *physical time* for tapering, because it is nonuniform
                    ncut = np.argmax( - (1-self.waveform_modes[mode][:,0])**2)
                    tcut = float(self.waveform_modes[mode][ncut,0])
                    vectaper = 0.5-0.5*np.cos(np.pi* self.waveform_modes[mode][:ncut,0]/tcut)
                    self.waveform_modes[mode][:ncut,1]*= vectaper
                    self.waveform_modes[mode][:ncut,2]*= vectaper

                self.waveform_modes_complex[mode] = np.array([self.waveform_modes[mode][:,0], self.waveform_modes[mode][:,1]+1.j*self.waveform_modes[mode][:,2]]).T

                if rosDebug:
                    print("  Loaded ",  mode, "; length = ", len(self.waveform_modes[mode]), " processed interval = ", self.waveform_modes[mode][0,0], self.waveform_modes[mode][-1,0], " sampling interval at start ",  self.waveform_modes[mode][1,0]-self.waveform_modes[mode][0,0], " and smallest interval = ", self.waveform_modes_nonuniform_smallest_timestep[mode])

        nmax_orig = np.argmax(np.abs(self.waveform_modes_complex[(2,2)][:,1]))
        tmax_orig = float(self.waveform_modes_complex[(2,2)][nmax_orig,0])

        for mode in internal_ModesAvailable:
           if mode[0]<=lmax:
                # shift all times, if necessary
                if align_at_peak_l2_m2_emission:
                    self.waveform_modes[mode][:,0] += -tmax_orig
                    self.waveform_modes_complex[mode][:,0] += -tmax_orig
                # Either store in a different field OR reformat
                # Create an interpolating function for the complex amplitude and phase
                datAmp = np.abs(self.waveform_modes_complex[mode][:,1])
                datPhase = lalsimutils.unwind_phase(np.angle(self.waveform_modes_complex[mode][:,1]))    #needs to be unwound to be continuous
                datT = self.waveform_modes_complex[mode][:,0]
                # Compute starting frequency for this mode (used for FFT)
                # Index offset corresponding to 10 M
                dt = self.waveform_modes_complex[mode][1,0]-self.waveform_modes_complex[mode][0,0]
                nOffsetForPhase = 5  # ad-hoc offset based on uniform sampling
                nStride = 5
                self.fMinMode[mode] = np.abs((datPhase[nOffsetForPhase+nStride]-datPhase[nOffsetForPhase])/(2*np.pi*(datT[nOffsetForPhase+nStride]-datT[nOffsetForPhase]))) # for FFT. Modified to work with nonuniform sampling
                # WARNING: this is really only reliable for non-junky resolved modes.  We will clean this lower frequency later.
                if mode ==(2,2):
                    self.fOrbitLower  = 0.5*self.fMinMode[mode]
              
                # Interpolate. (Works even for nonuniform time sampling)
                if True: #not build_strain_and_conserve_memory:
                    # this is gravy - not critical. BUT it is critical for nonuniform
                    self.waveform_modes_complex_interpolated_amplitude[mode] = RangeWrap1d([np.min(self.waveform_modes_complex[mode][:,0]), np.max(self.waveform_modes_complex[mode][:,0])], 0, interp1d( self.waveform_modes_complex[mode][:,0].astype(float), datAmp,kind=default_interpolation_kind, fill_value=0.,bounds_error=False))
                    self.waveform_modes_complex_interpolated_phase[mode] = RangeWrap1d([np.min(self.waveform_modes_complex[mode][:,0]), np.max(self.waveform_modes_complex[mode][:,0])], 0,  interp1d(self.waveform_modes_complex[mode][:,0].astype(float), datPhase,kind=default_interpolation_kind, fill_value=0.,bounds_error=False))

        print(" Restoring current working directory... ",cwd)
        os.chdir(cwd);


    def complex_hoft(self,  force_T=False, deltaT=1./16384, time_over_M_zero=0.,sgn=-1):
        hlmT = self.hlmoft( force_T, deltaT,time_over_M_zero)
        npts = hlmT[(2,2)].data.length
        wfmTS = lal.CreateCOMPLEX16TimeSeries("Psi4", lal.LIGOTimeGPS(0.), 0., deltaT, lalsimutils.lsu_DimensionlessUnit, npts)
        wfmTS.data.data[:] = 0   # SHOULD NOT BE NECESARY, but the creation operator doesn't robustly clean memory
        wfmTS.epoch = hlmT[(2,2)].epoch
        for mode in hlmT.keys():
            # PROBLEM: Be careful with interpretation. The incl and phiref terms are NOT tied to L.
            if rosDebug:
                print(mode, np.max(hlmT[mode].data.data), " running max ",  np.max(np.abs(wfmTS.data.data)))
            wfmTS.data.data += np.exp(-2*sgn*1j*self.P.psi)* hlmT[mode].data.data*lal.SpinWeightedSphericalHarmonic(self.P.incl,-self.P.phiref,-2, int(mode[0]),int(mode[1]))
        return wfmTS
    def complex_hoff(self, force_T=False):
        htC  = self.complex_hoft( force_T=force_T,deltaT= self.P.deltaT)
        TDlen = int(1./self.P.deltaF * 1./self.P.deltaT)
        assert TDlen == htC.data.length
        hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)",
                                                htC.epoch, htC.f0, 1./htC.deltaT/htC.data.length, lalsimutils.lsu_HertzUnit, 
                                                htC.data.length)
        fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(htC.data.length,0)
        lal.COMPLEX16TimeFreqFFT(hf, htC, fwdplan)
        return hf
    def real_hoft(self,Fp=None, Fc=None):
        """
        Returns the real-valued h(t) that would be produced in a single instrument.
        Translates epoch as needed.
        Based on 'hoft' in lalsimutils.py
        """
        # Create complex timessereis
        htC = self.complex_hoft(force_T=1./self.P.deltaF, deltaT= self.P.deltaT)  # note P.tref is NOT used in the low-level code
        TDlen  = htC.data.length
        if rosDebug:
            print("Size sanity check ", TDlen, 1/(self.P.deltaF*self.P.deltaT))
            print(" Raw complex magnitude , ", np.max(htC.data.data))
            
        # Create working buffers to extract data from it -- wasteful.
        hp = lal.CreateREAL8TimeSeries("h(t)", htC.epoch, 0.,
            self.P.deltaT, lalsimutils.lsu_DimensionlessUnit, TDlen)
        hc = lal.CreateREAL8TimeSeries("h(t)", htC.epoch, 0.,
            self.P.deltaT, lalsimutils.lsu_DimensionlessUnit, TDlen)
        hT = lal.CreateREAL8TimeSeries("h(t)", htC.epoch, 0.,
            self.P.deltaT, lalsimutils.lsu_DimensionlessUnit, TDlen)
        # Copy data components over
        hp.data.data = np.real(htC.data.data)
        hc.data.data = np.imag(htC.data.data)
        # transform as in lalsimutils.hoft
        if Fp!=None and Fc!=None:
            hp.data.data *= Fp
            hc.data.data *= Fc
            hp = lal.AddREAL8TimeSeries(hp, hc)
            hoft = hp
        elif self.P.radec==False:
            fp = lalsimutils.Fplus(self.P.theta, self.P.phi, self.P.psi)
            fc = lalsimutils.Fcross(self.P.theta, self.P.phi, self.P.psi)
            hp.data.data *= fp
            hc.data.data *= fc
            hp.data.data  = lal.AddREAL8TimeSeries(hp, hc)
            hoft = hp
        else:
            # Note epoch must be applied FIRST, to make sure the correct event time is being used to construct the modulation functions
            hp.epoch = hp.epoch + self.P.tref
            hc.epoch = hc.epoch + self.P.tref
            if rosDebug:
                print(" Real h(t) before detector weighting, ", np.max(hp.data.data), np.max(hc.data.data))
            hoft = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc,    # beware, this MAY alter the series length??
                self.P.phi, self.P.theta, self.P.psi, 
                lalsim.DetectorPrefixToLALDetector(str(self.P.detector)))
            hoft = lal.CutREAL8TimeSeries(hoft, 0, hp.data.length)       # force same length as before??
            if rosDebug:
                print("Size before and after detector weighting " , hp.data.length, hoft.data.length)
        if rosDebug:
            print(" Real h_{IFO}(t) generated, pre-taper : max strain =", np.max(hoft.data.data))
        if self.P.taper != lalsimutils.lsu_TAPER_NONE: # Taper if requested
            lalsim.SimInspiralREAL8WaveTaper(hoft.data, self.P.taper)
        if self.P.deltaF is not None:
            TDlen = int(1./self.P.deltaF * 1./self.P.deltaT)
            print("Size sanity check 2 ", int(1./self.P.deltaF * 1./self.P.deltaT), hoft.data.length)
            assert TDlen >= hoft.data.length
            npts = hoft.data.length
            hoft = lal.ResizeREAL8TimeSeries(hoft, 0, TDlen)
            # Zero out the last few data elements -- NOT always reliable for all architectures; SHOULD NOT BE NECESSARY
            hoft.data.data[npts:TDlen] = 0

        if rosDebug:
            print(" Real h_{IFO}(t) generated : max strain =", np.max(hoft.data.data))
        return hoft

    def non_herm_hoff(self):
        """
        Returns the 2-sided h(f) associated with the real-valued h(t) seen in a real instrument.
        Translates epoch as needed.
        Based on 'non_herm_hoff' in lalsimutils.py
        """
        htR = self.real_hoft() # Generate real-valued TD waveform, including detector response
        if self.P.deltaF == None: # h(t) was not zero-padded, so do it now
            TDlen = nextPow2(htR.data.length)
            htR = lal.ResizeREAL8TimeSeries(htR, 0, TDlen)
        else: # Check zero-padding was done to expected length
            TDlen = int(1./self.P.deltaF * 1./self.P.deltaT)
            assert TDlen == htR.data.length
        fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(htR.data.length,0)
        htC = lal.CreateCOMPLEX16TimeSeries("hoft", htR.epoch, htR.f0,
            htR.deltaT, htR.sampleUnits, htR.data.length)
        # copy h(t) into a COMPLEX16 array which happens to be purely real
        htC.data.data[:htR.data.length] = htR.data.data
#        for i in range(htR.data.length):
#            htC.data.data[i] = htR.data.data[i]
        hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)",
            htR.epoch, htR.f0, 1./htR.deltaT/htR.data.length, lalsimutils.lsu_HertzUnit, 
            htR.data.length)
        lal.COMPLEX16TimeFreqFFT(hf, htC, fwdplan)
        return hf


    def estimateFminHz(self):
        return 2*self.fOrbitLower/(MsunInSec*(self.P.m1+self.P.m2)/lal.MSUN_SI)

    def estimateDurationSec(self):
        """
        estimateDuration uses the ACTUAL UNITS IN THE WAVEFORM, which are already in sec
        """
        return np.real(self.waveform_modes_complex[(2,2)][-1,0]-self.waveform_modes_complex[(2,2)][0,0]) # self.deltaToverM*(self.len(self.waveform_modes[(2,2)])

    def hlmoft(self,  force_T=False, deltaT=1./16384, time_over_M_zero=0.,taper_start_time=True):
        """
        hlmoft uses stored interpolated values for hlm(t) generated via the standard cleaning process, scaling them 
        to physical units for use in injection code.

        If the time window is sufficiently short, the result is NOT tapered (!!) -- no additional tapering is applied

        The code will ALWAYS have zero padding on the end -- half of the buffer is zero padding!
        This can cause loss of frequency content if you are not careful
        """
        hlmT ={}
        # Define units
        m_total_s = MsunInSec*(self.P.m1+self.P.m2)/lal.MSUN_SI
        distance_s = self.P.dist/lal.C_SI  # insures valid units

        # Create a suitable set of time samples.  Zero pad to 2^n samples.
        # Note waveform is stored in s already
        T_estimated = np.real(self.waveform_modes_complex[(2,2)][-1,0] - self.waveform_modes_complex[(2,2)][0,0])
        npts=0
        n_crit = 0
        if not force_T:
            npts_estimated = int(T_estimated/deltaT)
#            print " Estimated length: ",npts_estimated, T_estimated
            npts = lalsimutils.nextPow2(npts_estimated)
        else:
            npts = int(force_T/deltaT)
            print(" Forcing length T=", force_T, " length ", npts)
        # WARNING: Time range may not cover the necessary time elements.
        # Plan on having a few seconds buffer at the end
        T_buffer_required = npts*deltaT
        print(" EOB internal: Estimated time window (sec) ", T_estimated, " versus buffer duration ", T_buffer_required)
        print(" EOB internal: Requested size vs buffer size",   npts, len(self.waveform_modes_complex[(2,2)]))
        # If the waveform is longer than the buffer, we need to avoid wraparound

        # If the buffer requested is SHORTER than the 2*waveform, work backwards
        # If the buffer requested is LONGER than the waveform, work forwards from the start of all data
        if T_buffer_required/2 > T_estimated:
            tvals =  np.arange(npts)*deltaT + float(self.waveform_modes_complex[(2,2)][0,0])   # start at time t=0 and go forwards (zeros automatically padded by interpolation code)
            t_crit = float( -self.waveform_modes_complex[(2,2)][0,0])
            n_crit = int( t_crit/deltaT) # estiamted peak sample location in the t array, working forward

        else:
            print("  EOB internal: Warning LOSSY conversion to insure half of data is zeros ")
            # Create time samples by walking backwards from the last sample of the waveform, a suitable duration
            # ASSUME we are running in a configuration with align_at_peak_l2m2_emission
            # FIXME: Change this
            tvals = T_buffer_required/2  + (-npts + 1+ np.arange(npts))*deltaT + np.real(self.waveform_modes_complex[(2,2)][-1,0])  # last insures we get some ringdown
            t_crit = T_buffer_required/2 - (np.real(self.waveform_modes_complex[(2,2)][-1,0]))
            n_crit = int(t_crit/deltaT)
            
        if rosDebug:
            print(" time range being sampled ", [min(tvals),max(tvals)], " corresponding to dimensionless range", [min(tvals)/m_total_s,max(tvals)/m_total_s])
            print(" estimated peak sample at ", n_crit)

        # Loop over all modes in the system
        for mode in self.waveform_modes.keys():
            amp_vals = m_total_s/distance_s * self.waveform_modes_complex_interpolated_amplitude[mode](tvals)  # vectorized interpolation with piecewise
            phase_vals = self.waveform_modes_complex_interpolated_phase[mode]( tvals)
            phase_vals = lalsimutils.unwind_phase(phase_vals)

            if rosDebug:
                print("  Mode ", mode, " physical strain max, indx,", np.max(amp_vals), np.argmax(amp_vals))
            
            # Copy into a new LIGO time series object
            wfmTS = lal.CreateCOMPLEX16TimeSeries("h", lal.LIGOTimeGPS(0.), 0., deltaT, lalsimutils.lsu_DimensionlessUnit, npts)
            wfmTS.data.data =amp_vals*np.exp(1j*phase_vals)

            # Set the epoch for the time series correctly: should have peak near center of series by construction
            # note all have the same length
            # wfmTS.epoch = -deltaT*wfmTS.data.length/2  # did not work
            #n_crit = np.argmax(wfmTS.data.data)
            #print n_crit*wfmTS.deltaT, wfmTS.epoch   # this should be nearly zero

            # taper the start (1s. Only needed if I do not grab the whole range, because I taper the raw data)
            if taper_start_time:
                tTaper = 1
                nTaper = int(tTaper/deltaT)
                hoft_window = lal.CreateTukeyREAL8Window(nTaper*2, 0.8)
                factorTaper = hoft_window.data.data[0:nTaper]
                wfmTS.data.data[:nTaper]*=factorTaper

            # Store the resulting mode
            hlmT[mode] = wfmTS


        # Set time at peak of 22 mode. This is a hack, but good enough for us
#        n_crit = np.argmax(hlmT[(2,2)].data.data)
        epoch_crit = float(-t_crit) #-deltaT*n_crit 
        print(" EOB internal: zero epoch sample location", n_crit, np.argmax(np.abs(hlmT[(2,2)].data.data)))
        for mode in hlmT:
            hlmT[mode].epoch = epoch_crit

        return hlmT

    def hlmoff(self, force_T=False, deltaT=1./16384, time_over_M_zero=0.):
        """
        hlmoff takes fourier transforms of LAL timeseries generated from hlmoft.
        All modes have physical units, appropriate to a physical signal.
        """
        hlmF ={}
        hlmT = self.hlmoft(force_T=force_T,deltaT=deltaT,time_over_M_zero=time_over_M_zero)
        for mode in self.waveform_modes.keys():
            wfmTS=hlmT[mode]
            # Take the fourier transform
            wfmFD = lalsimutils.DataFourier(wfmTS)  # this creates a new f series for *each* call.
            # Store the resulting mode
            hlmF[mode] = wfmFD
        return hlmF


class WaveformMode:
    """
    Class representing a dimensionless timeseries X(\tau=t/M)
    """

###
### FFT syntatic sugar
###
def DataFourierNumpy(wfComplex):    # assume (n,2) size array of [tvals, g(t)]; return fvals, tilde(g). 
    # FFT
    # Unroll -- it will save me time later if we are continuous
    T = wfComplex[-1,0] - wfComplex[0,0]
    n = len(wfComplex)
    dt = T/n
    gtilde = np.fft.fft(wfComplex[:,1])*dt   
    # Option 1: Raw pairing, not caring about continuity in the frequency array.
#    fvals = 1./T* np.array([ k if  k<=n/2 else k-n for k in np.arange(n)]) 
    # Option 2: Unroll
    gtilde = np.roll(gtilde, n/2)
    fvals = 1./T*(np.arange(n) - n/2+1)     # consistency with DataFourier: \int dt e(i2\pi f t), note REVERSING
    wfComplexF = np.array([fvals,gtilde[::-1]]).T
    return wfComplexF

def DataInverseFourierNumpy(wfComplex):    # assume (n,2) size array of [fvals, gtilde(f)]; return tvals, g
#    print "NAN check ", wfComplex[np.isnan(wfComplex[:,1]),1]
    df = wfComplex[1,0] - wfComplex[0,0]
    n = len(wfComplex)
    T = 1./df
    dt = T/n
    # Undo the roll, then perform the FFT
    datReversed = wfComplex[:,1][::-1]
    g = np.fft.ifft(np.roll(datReversed, -n/2))*n*df # undo the reverse and roll
    # Assign correct time values.  
    #    - Note the zero of time is now centered in the array -- we don't carry a time reference with us.
    tvals = dt*(np.arange(n) -n/2+1)
    wfComplexT = np.array([tvals,g]).T
    return wfComplexT

###
### Mode syntactic sugar
###
def RawGetModePeakTime(wfMode):   # assumed applied to complex data sequence
    nmax = np.argmax(np.abs(wfMode[:,1]))
    return np.real(wfMode[nmax][0])



###
### Plotting syntactic sugar
###

# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot    
mode_line_style= {}
mode_line_style[(2,2)] = 'r'
mode_line_style[(2,1)] = 'b'
mode_line_style[(2,-1)] = 'b'
mode_line_style[(2,-2)] = 'r'
mode_line_style[(3,3)] = 'r-'
mode_line_style[(3,-3)] = 'r-'
mode_line_style[(3,1)] = 'b-'
mode_line_style[(3,-1)] = 'b-'

mode_line_style[(2,0)] = 'g'


for l in np.arange(2,5):
    for m in np.arange(-l,l+1):
        if not ((l,m) in mode_line_style.keys()):
            mode_line_style[(l,m)] = 'k'


