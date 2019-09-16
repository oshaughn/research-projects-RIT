from factored_likelihood import *
from matplotlib import pylab as plt

m1 = 10.*lal.LAL_MSUN_SI
m2 = 10.*lal.LAL_MSUN_SI
fmin_sig = 10.
fmin_tmplt = 40.
Nsamp = 500

# First determine the length of the waveform
# and choose P.tref so geocenter data starts at time 'gc_epoch'
gc_epoch = 100.0
Psig = ChooseWaveformParams(fmin=fmin_sig, radec=False, m1=m1, m2=m2)
df = findDeltaF(Psig)
Psig.deltaF = df
htest = hoft(Psig)
print("Length of the signal waveform is:", float(htest.epoch))
gc_tref = gc_epoch - float(htest.epoch)

IP = Overlap(fLow=40., deltaF=df, psd=lal.LIGOIPsd, full_output=True)

# Signal comes up through South pole, therefore, should arrive at
# geocenter, L1, V1, H1 in that order
RA = 0.
DEC = - lal.LAL_PI_2
Psig = ChooseWaveformParams(fmin=fmin_sig, radec=True, m1=m1, m2=m2,
        theta=DEC, phi=RA, detector='H1', tref=gc_tref, deltaF=df)

# Make a template of same masses, but shorter
Ptmplt = ChooseWaveformParams(fmin=fmin_tmplt, m1=m1, m2=m2, deltaF=df)

# Compute TD and normalized FD signals and template
htH1 = hoft(Psig)
hfH1 = norm_hoff(Psig, IP)
Psig.detector = 'L1' 
htL1 = hoft(Psig)
hfL1 = norm_hoff(Psig, IP)
Psig.detector = 'V1'
htV1 = hoft(Psig)
hfV1 = norm_hoff(Psig, IP)

htTMPLT = hoft(Ptmplt)
hfTMPLT = norm_hoff(Ptmplt, IP)

# Make array of discrete time steps for each detector
tH1 = float(htH1.epoch) + np.arange(htH1.data.length) * htH1.deltaT
tL1 = float(htL1.epoch) + np.arange(htL1.data.length) * htL1.deltaT
tV1 = float(htV1.epoch) + np.arange(htV1.data.length) * htV1.deltaT
tTMPLT = gc_epoch + np.arange(htTMPLT.data.length) * htTMPLT.deltaT

print("Template length is:", -float(htTMPLT.epoch))
print("H1 epoch is:", float(htH1.epoch))
print("\tso max should be at time:", float(htH1.epoch) - float(htest.epoch)\
        + float(htTMPLT.epoch))
print("L1 epoch is:", float(htL1.epoch))
print("\tso max should be at time:", float(htL1.epoch) - float(htest.epoch)\
        + float(htTMPLT.epoch))
print("V1 epoch is:", float(htV1.epoch))
print("\tso max should be at time:", float(htV1.epoch) - float(htest.epoch)\
        + float(htTMPLT.epoch))

rhoMaxH1, rhoH1, rhoIdxH1, rhoArgH1 = IP.ip(hfH1, hfTMPLT)
rhoMaxL1, rhoL1, rhoIdxL1, rhoArgL1 = IP.ip(hfL1, hfTMPLT)
rhoMaxV1, rhoV1, rhoIdxV1, rhoArgV1 = IP.ip(hfV1, hfTMPLT)

print("Max SNRs are:")
print("H1 max =", rhoMaxH1, "at index", rhoIdxH1, "or time", tH1[rhoIdxH1])
print("L1 max =", rhoMaxL1, "at index", rhoIdxL1, "or time", tL1[rhoIdxL1])
print("V1 max =", rhoMaxV1, "at index", rhoIdxV1, "or time", tV1[rhoIdxV1])

firstH1 = int(rhoIdxH1 - Nsamp)
lastH1 = rhoIdxH1 + Nsamp
firstL1 = int(rhoIdxL1 - Nsamp)
lastL1 = rhoIdxL1 + Nsamp
firstV1 = int(rhoIdxV1 - Nsamp)
lastV1 = rhoIdxV1 + Nsamp
rhoH1cut = lal.CutCOMPLEX16TimeSeries(rhoH1, firstH1, 2*Nsamp+1)
rhoL1cut = lal.CutCOMPLEX16TimeSeries(rhoL1, firstL1, 2*Nsamp+1)
rhoV1cut = lal.CutCOMPLEX16TimeSeries(rhoV1, firstV1, 2*Nsamp+1)

tH1cut = tH1[firstH1:lastH1+1]
tL1cut = tL1[firstL1:lastL1+1]
tV1cut = tV1[firstV1:lastV1+1]

print("Cut length", rhoH1cut.data.length)
print("Length of time steps", len(tH1cut))

# Plot TD template and H1 waveform
plt.figure(1)
plt.plot(tH1, htH1.data.data, 'b-')
plt.plot(tL1, htL1.data.data, 'g-')
plt.plot(tV1, htV1.data.data, 'm-')
plt.plot(tTMPLT, htTMPLT.data.data, 'k-')

# Plot discrete overlap vs time
plt.figure(2)
plt.plot(tH1, np.abs(rhoH1.data.data), 'b-')
plt.plot(tL1, np.abs(rhoL1.data.data), 'g-')
plt.plot(tL1cut, np.abs(rhoL1cut.data.data), 'k--')
plt.plot(tV1, np.abs(rhoV1.data.data), 'm-')

# interpolate the discrete overlaps
tintpH1 = np.arange(float(htH1.epoch),float(htH1.epoch)+1./df,10**-5)
tintpL1 = np.arange(float(htL1.epoch),float(htL1.epoch)+1./df,10**-5)
tintpV1 = np.arange(float(htV1.epoch),float(htV1.epoch)+1./df,10**-5)
ampH1 = np.abs(rhoH1.data.data)
phaseH1 = unwind_phase( np.angle(rhoH1.data.data) )
ampintpH1 = interpolate.InterpolatedUnivariateSpline(tH1, ampH1, k=3)
phaseintpH1 = interpolate.InterpolatedUnivariateSpline(tH1, phaseH1, k=3)
rhointpH1 = lambda x: ampintpH1(x) * np.exp( 1j * phaseintpH1(x) )
ampL1 = np.abs(rhoL1.data.data)
phaseL1 = unwind_phase( np.angle(rhoL1.data.data) )
ampintpL1 = interpolate.InterpolatedUnivariateSpline(tL1, ampL1, k=3)
phaseintpL1 = interpolate.InterpolatedUnivariateSpline(tL1, phaseL1, k=3)
rhointpL1 = lambda x: ampintpL1(x) * np.exp( 1j * phaseintpL1(x) )
ampV1 = np.abs(rhoV1.data.data)
phaseV1 = unwind_phase( np.angle(rhoV1.data.data) )
ampintpV1 = interpolate.InterpolatedUnivariateSpline(tV1, ampV1, k=3)
phaseintpV1 = interpolate.InterpolatedUnivariateSpline(tV1, phaseV1, k=3)
rhointpV1 = lambda x: ampintpV1(x) * np.exp( 1j * phaseintpV1(x) )

plt.figure(3)
plt.plot(tintpH1, np.abs( rhointpH1(tintpH1) ), 'b-')
plt.plot(tintpL1, np.abs( rhointpL1(tintpL1) ), 'g-')
plt.plot(tintpV1, np.abs( rhointpV1(tintpV1) ), 'm-')

# interpolate the truncated overlaps
tintpH1cut = np.arange(tH1cut[0],tH1cut[-1],10**-5)
tintpL1cut = np.arange(tL1cut[0],tL1cut[-1],10**-5)
tintpV1cut = np.arange(tV1cut[0],tV1cut[-1],10**-5)
ampH1cut = np.abs(rhoH1cut.data.data)
phaseH1cut = unwind_phase( np.angle(rhoH1cut.data.data) )
ampintpH1cut = interpolate.InterpolatedUnivariateSpline(tH1cut, ampH1cut, k=3)
phaseintpH1cut = interpolate.InterpolatedUnivariateSpline(tH1cut, phaseH1cut, k=3)
rhointpH1cut = lambda x: ampintpH1cut(x) * np.exp( 1j * phaseintpH1cut(x) )
ampL1cut = np.abs(rhoL1cut.data.data)
phaseL1cut = unwind_phase( np.angle(rhoL1cut.data.data) )
ampintpL1cut = interpolate.InterpolatedUnivariateSpline(tL1cut, ampL1cut, k=3)
phaseintpL1cut = interpolate.InterpolatedUnivariateSpline(tL1cut, phaseL1cut, k=3)
rhointpL1cut = lambda x: ampintpL1cut(x) * np.exp( 1j * phaseintpL1cut(x) )
ampV1cut = np.abs(rhoV1cut.data.data)
phaseV1cut = unwind_phase( np.angle(rhoV1cut.data.data) )
ampintpV1cut = interpolate.InterpolatedUnivariateSpline(tV1cut, ampV1cut, k=3)
phaseintpV1cut = interpolate.InterpolatedUnivariateSpline(tV1cut, phaseV1cut, k=3)
rhointpV1cut = lambda x: ampintpV1cut(x) * np.exp( 1j * phaseintpV1cut(x) )

plt.figure(4)
plt.plot(tintpH1cut, np.abs( rhointpH1cut(tintpH1cut) ), 'b-')
plt.plot(tintpL1cut, np.abs( rhointpL1cut(tintpL1cut) ), 'g-')
plt.plot(tintpV1cut, np.abs( rhointpV1cut(tintpV1cut) ), 'm-')
plt.show()
