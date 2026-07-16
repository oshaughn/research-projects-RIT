from __future__ import print_function, division
import sys
import numpy as np
import lal, lalsimulation as lalsim
import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.factored_likelihood_with_rotation as flwr
event_time=1e9; Lmax=2; t_window=0.1; det='H1'
psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower
apx=lalsim.GetApproximantFromString("IMRPhenomD")
SCALE=40.   # loudness: data-mode distance = distMpcRef/SCALE

def _ifft(hlms_fd):
    out={}
    for lm,hf in hlms_fd.items():
        n=hf.data.length; dt=1./(n*hf.deltaF)
        ht=lal.CreateCOMPLEX16TimeSeries("h",hf.epoch,0.,dt,lal.DimensionlessUnit,n)
        lal.COMPLEX16FreqTimeFFT(ht,hf,lal.CreateReverseCOMPLEX16FFTPlan(n,0)); out[lm]=ht
    return out
def _fd(hk,TDlen):
    if hk.data.length!=TDlen: hk=lal.ResizeREAL8TimeSeries(hk,0,TDlen)
    n=TDlen
    htC=lal.CreateCOMPLEX16TimeSeries("h",hk.epoch,hk.f0,hk.deltaT,hk.sampleUnits,n); htC.data.data[:]=hk.data.data
    hf=lal.CreateCOMPLEX16FrequencySeries("hf",hk.epoch,hk.f0,1./hk.deltaT/n,lsu.lsu_HertzUnit,n)
    lal.COMPLEX16TimeFreqFFT(hf,htC,lal.CreateForwardCOMPLEX16FFTPlan(n,0)); return hf
def _peak(lt,up=16):
    lt=np.asarray(lt,dtype=float); N=len(lt)
    L=np.fft.rfft(lt); Lp=np.zeros(N*up//2+1,dtype=complex); Lp[:len(L)]=L
    return float(np.max(np.fft.irfft(Lp,N*up)*up))

def run(mode="short"):
    if mode=="long": fmin,fmax,deltaT,seglen,m1,m2=25.,256.,1/2048.,32.,2.2*lal.MSUN_SI,1.8*lal.MSUN_SI
    else: fmin,fmax,deltaT,seglen,m1,m2=30.,1700.,1/4096.,4.,30*lal.MSUN_SI,25*lal.MSUN_SI
    deltaF=1./seglen; fNyq=1/2./deltaT; TDlen=int(round(seglen/deltaT))
    DLOUD=fl.distMpcRef*1e6*lsu.lsu_PC/SCALE
    Psig=lsu.ChooseWaveformParams(fmin=fmin,radec=True,incl=0.4,phiref=0.0,theta=0.3,phi=1.2,psi=0.5,
        m1=m1,m2=m2,detector=det,dist=200e6*lal.PC_SI,deltaT=deltaT,tref=event_time,deltaF=deltaF); Psig.approx=apx
    Pm=Psig.manual_copy(); Pm.dist=DLOUD
    hlms_fd,_=fl.internal_hlm_generator(Pm,Lmax,verbose=False,quiet=True)
    hlmsT=_ifft(hlms_fd)
    data=_fd(lsu.hoft_from_hlm(hlmsT,Psig.manual_copy()),TDlen)
    data_dict={det:data}; psd_dict={det:psd}
    IPc=lsu.ComplexIP(fmin,fmax,fNyq,data.deltaF,psd,True,False,0.); HALF_DD=0.5*IPc.ip(data,data).real
    print("[%s] seglen=%.0fs data_len=%d 0.5<d|d>=%.4f"%(mode,seglen,data.data.length,HALF_DD))
    RA,DEC,PSI,INCL,PHIREF,DIST=1.2,0.3,0.5,0.4,0.0,DLOUD
    ri,ct,ctV,rho,snr,rest=fl.PrecomputeLikelihoodTerms(event_time,t_window,Psig,data_dict,psd_dict,Lmax,fmax,analyticPSD_Q=True,verbose=False,quiet=True,ignore_threshold=None)
    Pe=Psig.manual_copy(); Pe.incl=INCL;Pe.phiref=PHIREF;Pe.psi=PSI;Pe.phi=RA;Pe.theta=DEC;Pe.dist=DIST
    hk_gt=_fd(lsu.hoft_from_hlm(hlmsT,Pe),TDlen); lnL_gt=IPc.ip(data,hk_gt).real-0.5*IPc.ip(hk_gt,hk_gt).real
    Pv=Psig.manual_copy()
    for k,v in [('phi',RA),('theta',DEC),('incl',INCL),('phiref',PHIREF),('psi',PSI),('dist',DIST)]:
        setattr(Pv,k,np.ones(1)*v)
    Pv.tref=event_time; Pv.deltaT=deltaT
    Nw=int(0.02/deltaT); tvals=np.arange(-Nw,Nw)*deltaT
    lkB={};rAB={};cuB={};cvB={};epB={}
    for d in data_dict:
        a,b,c,U,V,rA,rI,e=fl.PackLikelihoodDataStructuresAsArrays(list(rho[d].keys()),None,rho[d],ct[d],ctV[d])
        lkB[d]=a;rAB[d]=rA;cuB[d]=U;cvB[d]=V;epB[d]=e
    lnL_base=_peak(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(tvals,Pv,lkB,rAB,cuB,cvB,epB,Lmax=Lmax,xpy=np,return_lnLt=True)[0])
    def rot(pmax):
        nh=2+pmax
        bk=flwr.PrecomputeLikelihoodTermsWithRotation(event_time,t_window,Psig,data_dict,psd_dict,Lmax,fmax,harmonics=tuple(range(-nh,nh+1)),p_max=pmax,f_sidereal=flwr.F_SIDEREAL,analyticPSD_Q=True,verbose=False,quiet=True,skip_interpolation=True)
        lk,rbn,ubn,vbn,ep=flwr.pack_rotation_arrays(bk[4],bk[3],bk[1],bk[2])
        return _peak(flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(tvals,Pv,bk[4],lk,rbn,ubn,vbn,ep,Lmax=Lmax,array_output=True)[0])
    lnL_A=rot(0); lnL_B=rot(1)
    print("  ground truth (SimDetStrain) lnL = %.5f"%lnL_gt)
    print("  baseline NoLoop  peak        lnL = %.5f  deficit=%.5f"%(lnL_base,lnL_gt-lnL_base))
    print("  Path A NoLoop (F(t))         lnL = %.5f  deficit=%.5f"%(lnL_A,lnL_gt-lnL_A))
    print("  Path B NoLoop (F(t)+p=1)     lnL = %.5f  deficit=%.5f"%(lnL_B,lnL_gt-lnL_B))
if __name__=="__main__": run(sys.argv[1] if len(sys.argv)>1 else "short")
