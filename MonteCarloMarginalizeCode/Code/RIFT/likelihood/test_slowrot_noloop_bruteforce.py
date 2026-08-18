"""
test_slowrot_noloop_bruteforce : the definitive rotation-physics validation.

The vectorized rotation NoLoop lnL_t (real sidereal rate) is compared, over the full time
window, to an INDEPENDENT brute-force likelihood that builds the real detector strain
    h(t') = Re[ F(t') * sum_lm Y_lm h_lm(t' - t_arr) ],   F from lal.ComputeDetAMResponse,
explicitly in the time domain at every arrival sample, and takes BOTH inner products of that
one series.  This confirms that the vectorized harmonic contraction is correct AND that the
(large, at high SNR) shift the sidereal rotation induces in the marginalized lnL is genuine
physics, not an artifact.

The reference deliberately shares NO convention with the implementation -- it never pushes the
modulation onto the data and never pins the template's arrival time -- so, unlike the version
this replaced, it cannot agree with a broken likelihood by making the same mistake.  See
test_slowrot_cauchy_schwarz.py and rotation_post_phase() for what that used to hide.

Run: source ~/RIFT_develUWM/bin/activate;
     PYTHONPATH=~/RIFT_slowrot/MonteCarloMarginalizeCode/Code python <this file>
"""
import numpy as np, lal, lalsimulation as lalsim
import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.factored_likelihood_with_rotation as flwr
fmin=30.;fmax=1700.;event_time=1e9;t_window=0.1;Lmax=2;deltaT=1/4096.;deltaF=1/4.;fNyq=1/2./deltaT
HARM=(-2,-1,0,1,2); OM=flwr.OMEGA_EARTH
Psig=lsu.ChooseWaveformParams(fmin=fmin,radec=True,incl=0.3,phiref=0.0,theta=0.2,phi=1.0,psi=0.4,
    m1=30*lal.MSUN_SI,m2=25*lal.MSUN_SI,detector='H1',dist=200e6*lal.PC_SI,deltaT=deltaT,tref=event_time,deltaF=deltaF)
data_dict={}
for det in ("H1","L1","V1"):
    P=Psig.manual_copy();P.detector=det;data_dict[det]=lsu.non_herm_hoff(P)
psd_dict={det:lalsim.SimNoisePSDaLIGOZeroDetHighPower for det in data_dict}
RA,DEC,PSI,INCL,PHIREF,DIST=1.0,0.2,0.5,0.7,0.9,300e6*lal.PC_SI
gmst_ev=float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(event_time)))
def to_td(fs):
    n=fs.data.length;dt=1./(n*fs.deltaF);ts=lal.CreateCOMPLEX16TimeSeries("x",fs.epoch,0.,dt,lal.DimensionlessUnit,n)
    lal.COMPLEX16FreqTimeFFT(ts,fs,lal.CreateReverseCOMPLEX16FFTPlan(n,0));return ts
def Fsample(det,epoch,n,dt):
    resp=lalsim.DetectorPrefixToLALDetector(det).response
    t=float(epoch)+np.arange(n)*dt
    return np.array([ (lambda a,b:a+1j*b)(*lal.ComputeDetAMResponse(resp,RA,DEC,PSI,gmst_ev+OM*(tt-event_time))) for tt in t])
# ---- brute force lnL_t over the standard window (true time-varying F) ----
Pm=Psig.manual_copy();Pm.dist=fl.distMpcRef*1e6*lsu.lsu_PC;Pm.deltaF=deltaF
hlms,hlms_conj=fl.internal_hlm_generator(Pm,Lmax,verbose=False,quiet=True)
Ylms=fl.ComputeYlms(Lmax,INCL,-PHIREF,selected_modes=list(hlms.keys()))
distMpc=DIST/(lsu.lsu_PC*1e6);invD=fl.distMpcRef/distMpc
npts=400
# CONVENTION-FREE REFERENCE.  An earlier version of this brute force pushed the F(t) modulation
# onto the DATA for term1 (<F h|d> == <h|conj(F) d>) and evaluated term2 for the template pinned
# at event_time.  Both shortcuts are exactly the ones the likelihood used to take, so the test
# agreed to 3e-10 while BOTH were wrong -- the failure mode SLOWROT_HANDOFF.md calls out as the
# critical lesson.  The first identity holds only for the UNWEIGHTED overlap (a frequency shift
# does not commute with the 1/S(f) band weight); the second drops the arrival-time post-phase
# exp(i n Omega (t-tref)).
#
# So this reference now shares nothing with the implementation: it builds the real strain
#     h(t') = invD * Re[ F(t') * hY(t' - t_arr) ]
# in the time domain at EACH arrival sample and takes both inner products of that one series.
# It is therefore a genuine Cauchy-Schwarz-respecting likelihood by construction.
def bf_lnLt(det):
    data=data_dict[det];psd=psd_dict[det];n=data.data.length;dt=1./(n*data.deltaF)
    t_det=fl.ComputeArrivalTimeAtDetector(det,RA,DEC,event_time)
    rho_epoch=data.epoch-hlms[list(hlms.keys())[0]].epoch
    t_shift=float(float(t_det)-float(t_window)-float(rho_epoch));N_shift=int(t_shift/deltaT+0.5)
    rr_epoch=float(rho_epoch)+N_shift*deltaT
    # The arrival samples the NoLoop lands on, as array shifts of the template against the data.
    # The origin is rho_epoch = data.epoch - hlms.epoch, NOT event_time: the data and the modes
    # are generated separately here and their epochs differ by ~0.36 s, so index m of the data
    # holds intrinsic template time (m + (rho_epoch - t_arr)/deltaT).
    ifirst=int(round((float(t_det)-0.02-rr_epoch)/deltaT)+0.5)
    kvals=[N_shift+ifirst+j for j in range(npts)]
    Fd=Fsample(det,float(data.epoch),n,dt)          # F(t') on the absolute data time axis
    hY=np.zeros(n,dtype=complex)
    for m in hlms: hY+=Ylms[m]*np.array(to_td(hlms[m]).data.data)
    IP=lsu.ComplexIP(fmin,fmax,fNyq,data.deltaF,psd,True,False,0.)
    out=np.zeros(npts)
    for j,k in enumerate(kvals):
        hs=lal.CreateCOMPLEX16TimeSeries("h",data.epoch,0.,dt,lal.DimensionlessUnit,n)
        hs.data.data[:]=np.real(Fd*np.roll(hY,k))*invD
        hf=lsu.DataFourier(hs)
        out[j]=IP.ip(hf,data).real-0.5*IP.ip(hf,hf).real
    return out
bf=sum(bf_lnLt(det) for det in data_dict)
m=np.max(bf);bf_marg=m+np.log(np.trapz(np.exp(bf-m),dx=deltaT))
# ---- vec rotation, real Omega, same window ----
rint,ct,ctV,rho,meta=flwr.PrecomputeLikelihoodTermsWithRotation(event_time,t_window,Psig,data_dict,psd_dict,Lmax,fmax,harmonics=HARM,p_max=0,f_sidereal=flwr.F_SIDEREAL,analyticPSD_Q=True,verbose=False,quiet=True,skip_interpolation=False)
lk,rbn,ubn,vbn,ep=flwr.pack_rotation_arrays(meta,rho,ct,ctV)
Pv=Psig.manual_copy()
for k,v in [('phi',RA),('theta',DEC),('incl',INCL),('phiref',PHIREF),('psi',PSI),('dist',DIST)]: setattr(Pv,k,np.ones(1)*v)
Pv.tref=event_time;Pv.deltaT=deltaT
tvals=np.arange(npts)*deltaT-0.02
vec=flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(tvals,Pv,meta,lk,rbn,ubn,vbn,ep,Lmax=Lmax,array_output=True)[0]
mv=np.max(vec);vec_marg=mv+np.log(np.trapz(np.exp(vec-mv),dx=deltaT))
worst = float(np.max(np.abs(vec - bf)))
print("brute-force(real F(t)) : peak=%.4f marg=%.4f" % (np.max(bf), bf_marg))
print("vec rotation           : peak=%.4f marg=%.4f" % (np.max(vec), vec_marg))
print("max|vec_lnL_t - bruteforce_lnL_t| over window = %.3e" % worst)
assert worst < 1e-6, "vectorized rotation NoLoop disagrees with brute-force time-varying response: %g" % worst
print("PASS: vectorized rotation NoLoop == brute-force time-varying-response likelihood")
