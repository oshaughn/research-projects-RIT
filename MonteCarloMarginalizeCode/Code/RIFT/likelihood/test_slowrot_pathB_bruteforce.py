from __future__ import print_function, division
import numpy as np, lal, lalsimulation as lalsim
from scipy.interpolate import CubicSpline
import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.factored_likelihood_with_rotation as flwr
import RIFT.likelihood.slowrot_response as srr
event_time=1e9; Lmax=2; t_window=0.1; det='H1'
psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower; apx=lalsim.GetApproximantFromString("IMRPhenomD")
import os
OMEGA_INF=flwr.OMEGA_EARTH*float(os.environ.get("INFL","340")); FSID_INF=OMEGA_INF/(2*np.pi)
def _ifft(hf_d):
    o={}
    for lm,hf in hf_d.items():
        n=hf.data.length; dt=1./(n*hf.deltaF); ht=lal.CreateCOMPLEX16TimeSeries("h",hf.epoch,0.,dt,lal.DimensionlessUnit,n)
        lal.COMPLEX16FreqTimeFFT(ht,hf,lal.CreateReverseCOMPLEX16FFTPlan(n,0)); o[lm]=ht
    return o
def _to_fd(re,epoch,dt,N):
    ht=lal.CreateCOMPLEX16TimeSeries("h",epoch,0.,dt,lal.DimensionlessUnit,N); ht.data.data[:]=re[:N]
    hf=lal.CreateCOMPLEX16FrequencySeries("hf",epoch,0.,1./dt/N,lsu.lsu_HertzUnit,N)
    lal.COMPLEX16TimeFreqFFT(hf,ht,lal.CreateForwardCOMPLEX16FFTPlan(N,0)); return hf
from scipy.interpolate import InterpolatedUnivariateSpline
def _peak(lt):
    lt=np.asarray(lt,float); x=np.arange(len(lt)); sp=InterpolatedUnivariateSpline(x,lt,k=4)
    xs=np.linspace(0,len(lt)-1,len(lt)*32); return float(np.max(sp(xs)))
fmin,fmax,deltaT,seglen=25.,512.,1/2048.,16.; deltaF=1./seglen; fNyq=1/2./deltaT; N=int(round(seglen/deltaT))
RA,DEC,PSI,INCL,PHIREF=1.2,0.3,0.5,0.4,0.0; DLOUD=fl.distMpcRef*1e6*lsu.lsu_PC/30.
Psig=lsu.ChooseWaveformParams(fmin=fmin,radec=True,incl=INCL,phiref=PHIREF,theta=DEC,phi=RA,psi=PSI,
    m1=2.2*lal.MSUN_SI,m2=1.8*lal.MSUN_SI,detector=det,dist=200e6*lal.PC_SI,deltaT=deltaT,tref=event_time,deltaF=deltaF); Psig.approx=apx
Pm=Psig.manual_copy(); Pm.dist=DLOUD
hlms_fd,_=fl.internal_hlm_generator(Pm,Lmax,verbose=False,quiet=True); hlmsT=_ifft(hlms_fd)
lm0=list(hlmsT.keys())[0]; nn=hlmsT[lm0].data.length; dt=hlmsT[lm0].deltaT; ep=float(hlmsT[lm0].epoch); tt=ep+np.arange(nn)*dt
Sig=np.zeros(nn,complex)
for lm in hlmsT: Sig+=hlmsT[lm].data.data*lal.SpinWeightedSphericalHarmonic(INCL,-PHIREF,-2,lm[0],lm[1])
reS=CubicSpline(tt,Sig.real,extrapolate=False); imS=CubicSpline(tt,Sig.imag,extrapolate=False)
lald=lalsim.DetectorPrefixToLALDetector(det); g_ev=lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(event_time))-RA
A=srr.antenna_harmonics(lald.response,DEC,PSI); At={k:A[k]*np.exp(1j*k*g_ev) for k in A}
B=srr.delay_harmonics(lald.location,DEC); Bt={k:B[k]*np.exp(1j*k*g_ev) for k in B}
tau_t=np.real(sum(Bt[k]*np.exp(1j*k*OMEGA_INF*tt) for k in Bt))
F_t=sum(At[k]*np.exp(1j*k*OMEGA_INF*tt) for k in At)
Sig_d=np.nan_to_num(reS(tt-tau_t)+1j*imS(tt-tau_t))
data=_to_fd(np.real(F_t*Sig_d),lal.LIGOTimeGPS(float(hlmsT[lm0].epoch)+event_time),dt,N); data_dict={det:data}; psd_dict={det:psd}
IPc=lsu.ComplexIP(fmin,fmax,fNyq,data.deltaF,psd,True,False,0.); HALF_DD=0.5*IPc.ip(data,data).real
print("inflated seglen=%.0fs 0.5<d|d>=%.4f"%(seglen,HALF_DD))
Pv=Psig.manual_copy()
for k,v in [('phi',RA),('theta',DEC),('incl',INCL),('phiref',PHIREF),('psi',PSI),('dist',DLOUD)]: setattr(Pv,k,np.ones(1)*v)
Pv.tref=event_time; Pv.deltaT=deltaT; Nw=int(0.02/deltaT); tvals=np.arange(-Nw,Nw)*deltaT
# INFL=340 reproduces the Omega*T of the worst physical case -- a 90-minute (5400 s) BNS at the
# true sidereal rate -- on this 16 s segment (5400/16 = 337.5 ~ 340).  So INFL/340 is the rotation
# rate as a multiple of that worst physical case; it is the quantity the paper quotes.
PHYS_INFL=340.0
lnL_by_pmax={}; deficit_by_pmax={}
for pmax in [0,1,2,3]:
    nh=2+pmax
    bk=flwr.PrecomputeLikelihoodTermsWithRotation(event_time,t_window,Psig,data_dict,psd_dict,Lmax,fmax,harmonics=tuple(range(-nh,nh+1)),p_max=pmax,f_sidereal=FSID_INF,analyticPSD_Q=True,verbose=False,quiet=True,skip_interpolation=True)
    lk,rbn,ubn,vbn,epd=flwr.pack_rotation_arrays(bk[4],bk[3],bk[1],bk[2])
    lnL=_peak(flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(tvals,Pv,bk[4],lk,rbn,ubn,vbn,epd,Lmax=Lmax,array_output=True)[0])
    lnL_by_pmax[str(pmax)]=float(lnL); deficit_by_pmax[str(pmax)]=float(HALF_DD-lnL)
    print("  p_max=%d : lnL=%.5f  deficit=%.5f"%(pmax,lnL,HALF_DD-lnL))
# Opt-in persistence: set OUT=<path>.json.  Default behaviour (print only) is unchanged.
_out=os.environ.get("OUT")
if _out:
    import json
    with open(_out,"w") as _fh:
        json.dump({"infl":float(os.environ.get("INFL","340")),
                   "infl_physical_reference":PHYS_INFL,
                   "omega_ratio_vs_physical":float(os.environ.get("INFL","340"))/PHYS_INFL,
                   "half_dd":float(HALF_DD),
                   "deficit_by_pmax":deficit_by_pmax,"lnL_by_pmax":lnL_by_pmax,
                   "seglen":float(seglen),"fmin":float(fmin),"fmax":float(fmax),
                   "m1":float(Psig.m1/lal.MSUN_SI),"m2":float(Psig.m2/lal.MSUN_SI)},_fh,indent=2)
    print("wrote %s"%_out)
