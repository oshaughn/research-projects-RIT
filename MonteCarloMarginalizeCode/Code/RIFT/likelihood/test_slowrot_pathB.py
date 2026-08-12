"""
test_slowrot_pathB : Path B (delay-derivative) sanity checks.

  (1) scalar Path B (p_max=1) at f_sidereal=0 reduces EXACTLY to the baseline
      FactoredLogLikelihood (the p>=1 coefficient sums vanish at Omega=0).
  (2) scalar Path B (p_max=1) at the real sidereal rate respects the Cauchy-Schwarz
      bound lnL <= 0.5<d|d> at the injection parameters.
  (3) vectorized Path B (p_max=1) at f_sidereal=0 reduces to the baseline NoLoop.

Rigorous validation of the p>=1 delay physics vs a brute force with TIME-VARYING delay on
a LONG signal is deferred to the systematic final-validation pass.
Run: PYTHONPATH=.../Code python RIFT/likelihood/test_slowrot_pathB.py
"""
from __future__ import print_function, division
import numpy as np, lal, lalsimulation as lalsim
import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.factored_likelihood_with_rotation as flwr

fmin=30.;fmax=1700.;event_time=1e9;t_window=0.1;Lmax=2;deltaT=1/4096.;deltaF=1/4.
HB=tuple(range(-3,4))
Psig=lsu.ChooseWaveformParams(fmin=fmin,radec=True,incl=0.3,phiref=0.0,theta=0.2,phi=1.0,psi=0.4,
    m1=30*lal.MSUN_SI,m2=25*lal.MSUN_SI,detector='H1',dist=200e6*lal.PC_SI,deltaT=deltaT,tref=event_time,deltaF=deltaF)
data_dict={}
for d in ["H1","L1","V1"]:
    P=Psig.manual_copy();P.detector=d;data_dict[d]=lsu.non_herm_hoff(P)
psd_dict={d:lalsim.SimNoisePSDaLIGOZeroDetHighPower for d in data_dict}

def test_scalar_reduces_to_baseline():
    extr=lsu.ChooseWaveformParams(radec=True,phi=1.0,theta=0.2,psi=0.5,incl=0.7,phiref=0.9,tref=event_time,dist=300e6*lal.PC_SI)
    rb=fl.PrecomputeLikelihoodTerms(event_time,t_window,Psig,data_dict,psd_dict,Lmax,fmax,analyticPSD_Q=True,verbose=False,quiet=True,ignore_threshold=None)
    lnLb=fl.FactoredLogLikelihood(extr,rb[3],rb[0],rb[1],rb[2],Lmax)
    rr=flwr.PrecomputeLikelihoodTermsWithRotation(event_time,t_window,Psig,data_dict,psd_dict,Lmax,fmax,harmonics=HB,p_max=1,f_sidereal=0.0,analyticPSD_Q=True,verbose=False,quiet=True,skip_interpolation=False)
    lnLB=flwr.FactoredLogLikelihoodWithRotation(extr,rr[0],rr[1],rr[2],rr[4],Lmax)
    print("(1) scalar PathB(p1,fsid0)=%.8f baseline=%.8f |d|=%.2e"%(lnLB,lnLb,abs(lnLB-lnLb)))
    assert abs(lnLB-lnLb)<1e-6*(1+abs(lnLb))

def test_scalar_respects_bound():
    extr=lsu.ChooseWaveformParams(radec=True,phi=1.0,theta=0.2,psi=0.4,incl=0.3,phiref=0.0,tref=event_time,dist=200e6*lal.PC_SI)
    rr=flwr.PrecomputeLikelihoodTermsWithRotation(event_time,t_window,Psig,{'H1':data_dict['H1']},{'H1':psd_dict['H1']},Lmax,fmax,harmonics=HB,p_max=1,f_sidereal=flwr.F_SIDEREAL,analyticPSD_Q=True,verbose=False,quiet=True,skip_interpolation=False)
    lnL=flwr.FactoredLogLikelihoodWithRotation(extr,rr[0],rr[1],rr[2],rr[4],Lmax)
    IP=lsu.ComplexIP(fmin,fmax,1/2./deltaT,deltaF,psd_dict['H1'],True,False,0.);dd=IP.ip(data_dict['H1'],data_dict['H1']).real
    print("(2) scalar PathB(p1,real,H1) lnL=%.4f  0.5<d|d>=%.4f"%(lnL,0.5*dd))
    assert lnL<=0.5*dd

def test_vec_reduces_to_baseline():
    tvals=np.arange(400)*deltaT-0.02
    rb=fl.PrecomputeLikelihoodTerms(event_time,t_window,Psig,data_dict,psd_dict,Lmax,fmax,analyticPSD_Q=True,verbose=False,quiet=True,ignore_threshold=None)
    lk={};ra={};cu={};cv={};ep={}
    for d in data_dict:
        a,b,c,U,V,rA,rI,e=fl.PackLikelihoodDataStructuresAsArrays(list(rb[3][d].keys()),None,rb[3][d],rb[1][d],rb[2][d]);lk[d]=a;ra[d]=rA;cu[d]=U;cv[d]=V;ep[d]=e
    Pv=Psig.manual_copy()
    for k,v in [('phi',1.0),('theta',0.2),('incl',0.7),('phiref',0.9),('psi',0.5),('dist',300e6*lal.PC_SI)]: setattr(Pv,k,np.ones(1)*v)
    Pv.tref=event_time;Pv.deltaT=deltaT
    bl=fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(tvals,Pv,lk,ra,cu,cv,ep,Lmax=Lmax,xpy=np)[0]
    rr0=flwr.PrecomputeLikelihoodTermsWithRotation(event_time,t_window,Psig,data_dict,psd_dict,Lmax,fmax,harmonics=HB,p_max=1,f_sidereal=0.0,analyticPSD_Q=True,verbose=False,quiet=True,skip_interpolation=False)
    lkr,rar,uar,var,epr=flwr.pack_rotation_arrays(rr0[4],rr0[3],rr0[1],rr0[2])
    vv=flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(tvals,Pv,rr0[4],lkr,rar,uar,var,epr,Lmax=Lmax,array_output=False)[0]
    print("(3) vec PathB(p1,fsid0)=%.8f baseline NoLoop=%.8f |d|=%.2e"%(vv,bl,abs(vv-bl)))
    assert abs(vv-bl)<1e-7*(1+abs(bl))

if __name__=="__main__":
    test_scalar_reduces_to_baseline()
    test_scalar_respects_bound()
    test_vec_reduces_to_baseline()
    print("ALL SLOWROT PATH B CHECKS PASSED")
