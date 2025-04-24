#!/usr/bin/env python
#  based on 'fast_tutorial.py' from https://git.ligo.org/lscsoft/bilby/-/blob/master/examples/gw_examples/injection_examples/fast_tutorial.py
#  and https://git.ligo.org/lscsoft/bilby/-/blob/master/examples/gw_examples/injection_examples/plot_time_domain_data.py
#
# INTENT:
#    tests rift_source, for use in calmarg
# EXAMPLE
#     python compare_in_ifo.py --inj ../../mdc.xml.gz --rift-wf --instrument H1

import bilby
import lal
from matplotlib import pyplot as plt

import numpy as np
#from gwpy.timeseries import TimeSeries
#import glob
from RIFT.calmarg.rift_source import RIFT_lal_binary_black_hole
import RIFT.lalsimutils as lalsimutils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--inj",default=None)
parser.add_argument("--approx", default="IMRPhenomPv2",help="Use H1, L1,V1")
parser.add_argument("--instrument", default=[],action='append',help="Use H1, L1,V1")
parser.add_argument("--rift-wf", action='store_true',help="Use rift wf generator")
parser.add_argument("--guess-offset",default=None, type=float,help="Hack: sign flip the polarization change")
parser.add_argument("--guess-sign-flip", action='store_true',help="Hack: sign flip the polarization change")
parser.add_argument("--seglen",default=4,type=int)
parser.add_argument("--fmin",default=25,type=float)
opts=  parser.parse_args()


ifo_list =['H', 'L', 'V']
my_data = {}
event_time = 1000000010.000
start_time = event_time-2
end_time =event_time +2


current_ifo=opts.instrument




# Set the duration and samplingg frequency of the data segment that we're
# going to inject the signal into
duration = float(opts.seglen)
sampling_frequency = 2048.0
minimum_frequency = opts.fmin   # avoid 2s duration before trigger time!


if opts.inj:
    P = lalsimutils.xml_to_ChooseWaveformParams_array(opts.inj)[0]
else:
    P = lalsimutils.ChooseWaveformParams()
    P.m1 = 25*lal.MSUN_SI
    P.m2 = 20*lal.MSUN_SI
    P.theta= 1.3
    P.phi = -1.57
    P.incl = 1.57
    P.psi = 0.5
    P.phiref = 1
    P.dist = 1*lal.PC_SI*1e6
P.radec=True # real injection
P.deltaF = 1./duration
P.deltaT = 1./sampling_frequency
P.fmin= minimum_frequency
    


# Specify the output directory and the name of the simulation.
outdir = "outdir_plots_"+''.join(current_ifo)
label = "from_file"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
bilby.core.utils.random.seed(88170235)

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
# SHOULD read parameters from mdc.xml.gz file
injection_parameters = dict(
    mass_1=P.m1/lal.MSUN_SI,
    mass_2=P.m2/lal.MSUN_SI,
    a_1=0.0,
    a_2=0.0,
    tilt_1=0,
    tilt_2=1.0,
    phi_12=0,
    phi_jl=0.0,
    luminosity_distance=P.dist/(lal.PC_SI*1e6), # force same distance
    theta_jn=np.pi/2,
    psi=P.psi,
    phase=P.phiref,
    geocent_time=1000000010.0001,
    ra=P.phi,
    dec=P.theta,
)
print(injection_parameters)
P.print_params()   # DOES NOT MATCH UP
if opts.guess_sign_flip:
    injection_parameters['psi'] = -injection_parameters['psi']
if opts.guess_offset:
    injection_parameters['psi'] += opts.guess_offset*np.pi # scale
    

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(current_ifo)
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)
# set to zero
for ifo in ifos:
    ifo.frequency_domain_strain = np.zeros(len(ifo.frequency_domain_strain),dtype=np.complex128)



# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant=opts.approx,
    reference_frequency=50.0,
    minimum_frequency=minimum_frequency,
    maximum_frequency=1024,
    pn_amplitude_order=0
)

source_model=bilby.gw.source.lal_binary_black_hole
conversion_function = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
if opts.rift_wf:
    source_model=RIFT_lal_binary_black_hole
    def clean_converter(*args,**kwargs):
        out = bilby.gw.conversion.generate_all_bbh_parameters(*args, **kwargs)
        if isinstance(out,dict):
            return (out, [])   # hack adding second return value, if it is not working correctlyx
        else:
            return out
    conversion_function =    clean_converter


# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=source_model,
    parameter_conversion=conversion_function,
    waveform_arguments=waveform_arguments,
)

# inject into each interferometer. Remember it is frequency-domain data
hT_dict={}
hf_signal = waveform_generator.frequency_domain_strain(injection_parameters)
for ifo in ifos:
    ifo.minimum_frequency=minimum_frequency
    print(ifo)
    ifo.inject_signal(injection_polarizations=hf_signal, parameters=injection_parameters)
    P.detector=ifo.name
    hT_dict[ifo.name] = lalsimutils.hoft(P)
    
# extract the data using self.time_domain_strain, based on bilby.gw.detector.inferferometer
extra_name =''
if opts.guess_sign_flip:
    extra_name += '-flipped'
for ifo in ifos:
    tvals = ifo.strain_data.time_array
    hvals = ifo.strain_data.time_domain_strain
    plt.plot(tvals - injection_parameters['geocent_time'],hvals)
    tvals_R =lalsimutils.evaluate_tvals( hT_dict[ifo.name])
    plt.xlim(-0.5,0.1)
    plt.plot(tvals_R -  injection_parameters['geocent_time'], hT_dict[ifo.name].data.data)
    plt.savefig("my_compare_{}{}.png".format(ifo.name,extra_name))
    plt.clf()

exit(0)
### additional framework
# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)


# using transient code, to get raw waveform data
#waveform_polarizations = \
#    likelihood.waveform_generator.frequency_domain_strain(self.parameters)


