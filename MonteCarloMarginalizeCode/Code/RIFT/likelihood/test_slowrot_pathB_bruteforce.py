from __future__ import print_function, division
import numpy as np, lal, lalsimulation as lalsim
from scipy.interpolate import CubicSpline
import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.factored_likelihood_with_rotation as flwr
import RIFT.likelihood.slowrot_response as srr
import os
event_time=1e9; Lmax=2; t_window=0.1; det='H1'
psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower; # APPROX: default IMRPhenomD is an FD model, which routes through hlmoft_FromFD_dict ->
# SimInspiralTDModesFromPolarizations and inherits LAL's minimal post-ringdown pad (~9 ms
# after the peak), bypassing RIFT's own fd_centering_factor=0.9 (which would reserve 10% of
# the segment).  That 9 ms is SHORTER than the Earth light-crossing delay, so the delayed
# lookup clips the loudest samples.  Use a TD model (TaylorT4) for development.
apx=lalsim.GetApproximantFromString(os.environ.get("APPROX","IMRPhenomD"))
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
def _peak_bandlimited(lt,upsample=64):
    """Peak of lnL(t) by band-limited (sinc) interpolation -- exact, not approximate.

    lnL(t) = Re[sum_a conj(C_a) sum_lm conj(Ylm) Q^a_lm(t)] - term2, term2 is time independent,
    and every Q^a_lm(t) is the inverse transform of something supported on [fmin,fmax].  So lnL(t)
    is BAND-LIMITED to fmax.  Sampled at 1/deltaT >> 2 fmax it is heavily oversampled (16x at
    srate 16384, fmax 512), so zero-padding in frequency IS Whittaker-Shannon interpolation and is
    exact -- unlike the order-4 spline in _peak, whose reconstruction error set the old resolution
    floor (1.8 nats of interpolation at srate 2048, and the deficit could go negative on it).

    A LINEAR baseline through the endpoints is removed before the transform and restored after, so
    the implicit periodic extension has neither a step nor a slope discontinuity to ring on.
    """
    lt=np.asarray(lt,float); n=lt.size
    if n<4: return float(np.max(lt))
    x=np.arange(n); slope=(lt[-1]-lt[0])/(n-1.); base=lt[0]+slope*x
    Y=np.fft.rfft(lt-base); m=n*upsample
    Yp=np.zeros(m//2+1,dtype=complex); Yp[:Y.size]=Y
    if n%2==0 and Y.size<=m//2: Yp[n//2]*=0.5   # split the Nyquist bin when zero-padding
    yp=np.fft.irfft(Yp,m)*upsample
    xp=np.arange(m)/float(upsample)
    return float(np.max(yp+lt[0]+slope*xp))
# SRATE (default 2048) and FMAXHZ (default 512) are knobs for diagnosing the deficit floor:
# raising SRATE refines the lnL time grid (tvals spacing is locked to deltaT by the NoLoop
# window logic) and lowers f/f_s for the cubic interpolator.
_SRATE=float(os.environ.get('SRATE','2048')); _FMAX=float(os.environ.get('FMAXHZ','512'))
# SEGLEN/FMINHZ: the DEFAULTS ARE UNPHYSICAL and are kept only for continuity with earlier
# results.  A 2.2+1.8 Msun binary from 25 Hz lasts ~48.5 s; in a 16 s segment it is wrapped,
# with the merger landing ~10 ms from the segment edge.  Never trust a number from a
# configuration where the signal does not fit: use SEGLEN=64 (fits at fmin=25) or FMINHZ=50
# (fits in 16 s).  The script warns when the chirp time exceeds the segment.
_SEGLEN=float(os.environ.get('SEGLEN','16')); _FMIN=float(os.environ.get('FMINHZ','25'))
fmin,fmax,deltaT,seglen=_FMIN,_FMAX,1/_SRATE,_SEGLEN; deltaF=1./seglen; fNyq=1/2./deltaT; N=int(round(seglen/deltaT))
RA,DEC,PSI,INCL,PHIREF=1.2,0.3,0.5,0.4,0.0; DLOUD=fl.distMpcRef*1e6*lsu.lsu_PC/30.
Psig=lsu.ChooseWaveformParams(fmin=fmin,radec=True,incl=INCL,phiref=PHIREF,theta=DEC,phi=RA,psi=PSI,
    m1=2.2*lal.MSUN_SI,m2=1.8*lal.MSUN_SI,detector=det,dist=200e6*lal.PC_SI,deltaT=deltaT,tref=event_time,deltaF=deltaF); Psig.approx=apx
_mt=(2.2+1.8)*lal.MSUN_SI*lal.G_SI/lal.C_SI**3; _eta=2.2*1.8/(2.2+1.8)**2
_tchirp=5./256.*_mt/(_eta*(np.pi*_mt*fmin)**(8./3.))
print("seglen=%.0fs fmin=%.0fHz chirp_time=%.1fs  FITS=%s"%(seglen,fmin,_tchirp,_tchirp<seglen))
if _tchirp>=seglen: print("  *** WARNING: signal is TRUNCATED/WRAPPED in this segment ***")
Pm=Psig.manual_copy(); Pm.dist=DLOUD
# GWSIGNAL path: the SEOBNRv5 family is NOT exposed through GetApproximantFromString at all,
# only through the gwsignal generator interface -- which is also the only route that accepts
# lmax_nyquist.  lmax_nyquist=1 disables the ringdown-vs-Nyquist check entirely (no mode has
# l<2), which is what lets a light system run below srate 16384.  Requires a py>=3.9 env with
# gwsignal importable (e.g. ~/.conda/envs/junior_rift); it will NOT import under RIFT_develUWM.
# The SAME kwargs go to the precompute, so data and template use the same generator -- passing
# them to only one would silently compare a v5 signal against a default-approximant template.
HLM_KW={}
if os.environ.get("GWSIG"):
    HLM_KW=dict(use_gwsignal=True, use_gwsignal_approx=os.environ.get("GWSIG_APPROX","SEOBNRv5PHM"),
                extra_waveform_kwargs={"lmax_nyquist":int(os.environ.get("LMAXNYQ","1"))})
    print("gwsignal: approx=%s lmax_nyquist=%s"%(HLM_KW["use_gwsignal_approx"],
                                                 HLM_KW["extra_waveform_kwargs"]["lmax_nyquist"]))
if HLM_KW:
    # pyseobnr rejects f_ref=0, which is RIFT's default.  Set it ONLY on this path so the
    # previously measured non-gwsignal configurations are bit-for-bit unperturbed.  Both Psig
    # (template, via the precompute) and Pm (data) get it, or they would disagree.
    Psig.fref=fmin; Pm.fref=fmin
hlms_fd,_=fl.internal_hlm_generator(Pm,Lmax,verbose=False,quiet=True,**HLM_KW); hlmsT=_ifft(hlms_fd)
lm0=list(hlmsT.keys())[0]; nn=hlmsT[lm0].data.length; dt=hlmsT[lm0].deltaT; ep=float(hlmsT[lm0].epoch); tt=ep+np.arange(nn)*dt
Sig=np.zeros(nn,complex)
for lm in hlmsT: Sig+=hlmsT[lm].data.data*lal.SpinWeightedSphericalHarmonic(INCL,-PHIREF,-2,lm[0],lm[1])
# NOTE (unresolved): NEITHER generator route leaves room after the peak for the delay lookup.
# IMRPhenomD (FD -> hlmoft_FromFD_dict -> SimInspiralTDModesFromPolarizations) inherits LAL's
# ~9.28 ms post-ringdown pad; TaylorT4 (TD) terminates at ISCO with a 0 ms gap.  max|tau| is
# ~9.5 ms, so in both cases the delayed lookup reads past the end and nan_to_num deletes the
# loudest samples -- 2.6e-3 of the power for PhenomD (floor 0.207), 1.3e-2 for TaylorT4 (floor
# 1.49).  RIFT's own FD-modes path reserves 10% of the segment (fd_centering_factor=0.9,
# fd_alignment_postevent_time) but this route never reaches it.
# Rolling the array to make trailing room is NOT a valid fix: it wraps the head around, and the
# head is quiet only in configurations that do not have the problem in the first place.  The fix
# is to ZERO-EXTEND after the merger (grow the array past the peak, keeping the epoch), with
# Psig.deltaF kept consistent so the template is built on the same grid.  Not yet implemented.
reS=CubicSpline(tt,Sig.real,extrapolate=False); imS=CubicSpline(tt,Sig.imag,extrapolate=False)
lald=lalsim.DetectorPrefixToLALDetector(det); g_ev=lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(event_time))-RA
A=srr.antenna_harmonics(lald.response,DEC,PSI); At={k:A[k]*np.exp(1j*k*g_ev) for k in A}
B=srr.delay_harmonics(lald.location,DEC); Bt={k:B[k]*np.exp(1j*k*g_ev) for k in B}
tau_t=np.real(sum(Bt[k]*np.exp(1j*k*OMEGA_INF*tt) for k in Bt))
F_t=sum(At[k]*np.exp(1j*k*OMEGA_INF*tt) for k in At)
# EDGE=nan (default)|wrap.  'nan' is the original construction: extrapolate=False makes
# Sig(t-tau) NaN wherever the delayed time leaves the sampled span, and nan_to_num ZEROES it,
# deleting a ~|tau| sliver (~9.5 ms here) from the data that the model still contains.  'wrap'
# resamples from a periodic extension instead, which is what the FD model actually assumes.
if os.environ.get("EDGE","nan")=="wrap":
    _pad=int(np.ceil((np.abs(tau_t).max()+10*dt)/dt))
    _tte=np.concatenate([tt[0]-dt*np.arange(_pad,0,-1),tt,tt[-1]+dt*np.arange(1,_pad+1)])
    _sge=np.concatenate([Sig[-_pad:],Sig,Sig[:_pad]])
    _re=CubicSpline(_tte,_sge.real,extrapolate=False); _im=CubicSpline(_tte,_sge.imag,extrapolate=False)
    Sig_d=np.nan_to_num(_re(tt-tau_t)+1j*_im(tt-tau_t))
else:
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
# The invariant that matters is Omega*T over the SIGNAL, not the segment.  The worst physical
# case is a 90-minute (5400 s) BNS at the true sidereal rate, so the equivalent inflation is
# 5400/T_signal, with T_signal = min(chirp_time, seglen) -- the chirp if it fits, the segment if
# it is truncated.  At the old defaults (fmin=25, chirp 48.5 s truncated to 16 s) that gives 337.5
# ~ 340, which is where the historical anchor came from; at fmin=50 (7.6 s chirp) it is ~711.
T_SIGNAL=min(_tchirp,seglen); PHYS_INFL=5400.0/T_SIGNAL
# TINTERP=nearest (default)|cubic -- the sub-bin time sampling used for the data term.  'nearest'
# leaves a ~0.2 nat peak-resolution floor on the deficit; 'cubic' is the calmarg_in_loop
# interpolation and should remove it.
TINTERP=os.environ.get("TINTERP","nearest")
lnL_by_pmax={}; deficit_by_pmax={}; lnL_raw_by_pmax={}; overshoot_by_pmax={}; lnL_spline_by_pmax={}; lnL_bl_by_pmax={}
for pmax in [0,1,2,3]:
    nh=2+pmax
    bk=flwr.PrecomputeLikelihoodTermsWithRotation(event_time,t_window,Psig,data_dict,psd_dict,Lmax,fmax,harmonics=tuple(range(-nh,nh+1)),p_max=pmax,f_sidereal=FSID_INF,analyticPSD_Q=True,verbose=False,quiet=True,skip_interpolation=True,**HLM_KW)
    lk,rbn,ubn,vbn,epd=flwr.pack_rotation_arrays(bk[4],bk[3],bk[1],bk[2])
    _lt=flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(tvals,Pv,bk[4],lk,rbn,ubn,vbn,epd,Lmax=Lmax,array_output=True,time_interp=TINTERP)[0]
    # _peak() splines (k=4) and oversamples 32x, which can OVERSHOOT the sampled maximum and
    # push the deficit negative -- a Cauchy-Schwarz 'violation' that is the estimator, not the
    # likelihood.  Record the raw grid max too so the overshoot is visible rather than folded in.
    lnL_spline=_peak(_lt); lnL_bl=_peak_bandlimited(_lt)
    lnL=lnL_bl if os.environ.get('PEAK','bandlimited')=='bandlimited' else lnL_spline
    lnL_raw=float(np.max(np.asarray(_lt,float)))
    lnL_raw_by_pmax[str(pmax)]=lnL_raw; overshoot_by_pmax[str(pmax)]=lnL-lnL_raw
    lnL_spline_by_pmax[str(pmax)]=lnL_spline; lnL_bl_by_pmax[str(pmax)]=lnL_bl
    lnL_by_pmax[str(pmax)]=float(lnL); deficit_by_pmax[str(pmax)]=float(HALF_DD-lnL)
    print("  p_max=%d : lnL=%.5f  deficit=%.5f"%(pmax,lnL,HALF_DD-lnL))
# Opt-in persistence: set OUT=<path>.json.  Default behaviour (print only) is unchanged.
_out=os.environ.get("OUT")
if _out:
    import json
    with open(_out,"w") as _fh:
        json.dump({"time_interp":TINTERP,"srate":_SRATE,"edge":os.environ.get("EDGE","nan"),
                   "infl":float(os.environ.get("INFL","340")),
                   "infl_physical_reference":PHYS_INFL,
                   "omega_ratio_vs_physical":float(os.environ.get("INFL","340"))/PHYS_INFL,
                   "t_signal":float(T_SIGNAL),
                   "half_dd":float(HALF_DD),
                   "deficit_by_pmax":deficit_by_pmax,"lnL_by_pmax":lnL_by_pmax,
                   "lnL_raw_by_pmax":lnL_raw_by_pmax,"peak_overshoot_by_pmax":overshoot_by_pmax,
                   "peak_estimator":os.environ.get("PEAK","bandlimited"),
                   "lnL_spline_by_pmax":lnL_spline_by_pmax,"lnL_bandlimited_by_pmax":lnL_bl_by_pmax,
                   "approx":(HLM_KW.get("use_gwsignal_approx") or os.environ.get("APPROX","IMRPhenomD")),
                   "lmax_nyquist":HLM_KW.get("extra_waveform_kwargs",{}).get("lmax_nyquist"),
                   "epoch_s":float(ep),"peak_frac":float(int(np.argmax(np.abs(Sig)))/float(nn)),"seglen":float(seglen),"fmin":float(fmin),"chirp_time":float(_tchirp),"signal_fits":bool(_tchirp<seglen),"fmax":float(fmax),
                   "m1":float(Psig.m1/lal.MSUN_SI),"m2":float(Psig.m2/lal.MSUN_SI)},_fh,indent=2)
    print("wrote %s"%_out)
