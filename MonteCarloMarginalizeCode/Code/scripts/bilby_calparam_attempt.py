#!/home/richard.oshaughnessy/.conda/envs/testme_redo_5/bin/python
#  EXAMPLE
#   bilby_calparam_attempt.py --posterior_sample_file reweighted_posterior_samples.dat  --start_index 0 --end_index 10 --use-gwsignal --waveform_approximant SEOBNRv5PHM --use_rift_samples=True  --data_dump_file /home/pe.o4/GWTC5-HLV/project/working/S241009em/rift-v5PHM-calmarg-4096/calmarg/data/calmarg_data_dump.pickle --fmin 13.33 --fref 20 --number_of_calibration_curves 300
"""
Based on 'calibration_reweighting.py' by Ethan Payne

"""

import os
import sys


import h5py
import argparse
from copy import deepcopy, copy

import numpy as np
import pandas as pd
import pickle

import bilby
import bilby_pipe

from bilby.core.result import read_in_result
from bilby_pipe.data_analysis import (
    create_analysis_parser,
    log_version_information,
    parse_args,
    DataAnalysisInput,
)


# So I can import the RIFT source while not making this some setup.py-able package
# TODO this should not be a hardcoded path!

import RIFT.calmarg.rift_source as rift_source

from  bilby.core.utils import logger



parser = argparse.ArgumentParser(description='calibration marginalization via reweighting of posterior samples')
parser.add(
    "--posterior_sample_file", default=None,
    help="Bilby result file or text file with posterior samples to reweight")
parser.add(
    "--data_dump_file", default=None,
    help="Bilby pipe data_dump file. This stores all the relevant data for PE")
parser.add(
    "--waveform_approximant", default=None,
    help="Approximant. Override what is in the pickle file (as expected if running across settings)")
parser.add(
    "--number_of_calibration_curves", type=int, default=1000,
    help="Number of calibration curve realizations to use to reweight samples.")
parser.add(
    "--time_marginalization", type=bool, default=False,
    help="Uses time maginalization when evaluating the likelihoods. Defaults to False")
parser.add(
    "--phase_marginalization", type=bool, default=False,
    help="Uses phase maginalization when evaluating the likelihoods. Defaults to False")
parser.add(
    "--reevaluate_likelihood", type=bool, default=True,
    help="Reevaluates the likelihood of the given samples using the likelihood from the data_dump.")
parser.add(
    "--use_nested_samples", type=bool, default=False,
    help="Use nested samples with their weights to generate the samples. This significantly improves efficiency.")
parser.add(
    "--data-integration-window-half", default=0.1, type=float,
    help="Half-Interval over which the time marginalization is undertaken")
parser.add(
    "--use_rift_samples", type=bool, default=False,
    help="Uses a different waveform function if using RIFT samples. This matches the phase definitions in RIFT")
parser.add(
    "--h_method", type=str, default='hlmoft',
    help="For RIFT, uses two different options for h(t) reconstruction ")
parser.add("--extra-waveform-kwargs",type=str,default=None,help="Generic mechanism to pass other arguments")
parser.add("--internal-waveform-fd-L-frame",action='store_true',help='If true, passes extra_waveform_kwargs = {fd_L_frame=True} to lalsimutils hlmoft. Impacts outputs of ChooseFDWaveform calls only.')
parser.add("--internal-waveform-fd-no-condition",action='store_true',help='If true, adds extra_waveform_kwargs = {no_condition=True} to lalsimutils hlmoft. Impacts outputs of ChooseFDWaveform calls only. Provided to enable controlled tests of conditioning impact on PE')
parser.add("--use-gwsignal",default=False,action='store_true',help='Use gwsignal. In this case the approx name is passed as a string to the lalsimulation.gwsignal interface')
parser.add("--use-eccentricity",default=False,action='store_true',help='Use for eccentricity and mean anomali. Currently can only use with Rossellas repo of bilby: https://git.ligo.org/rossella.gamba/bilby. Once merge into master, can get rid of this option.')
parser.add("--use-gwsignal-lmax-nyquist",default=None,type=int,help='Passes lmax_nyquist integer to the gwsignal waveform interface')
parser.add("--fmin", default=None, type=float, help="Minimum frequency for waveform generation")
parser.add("--fref", default=None, type=float, help="Reference frequency for waveform generation (if applicable)")
parser.add("--l-max", default=4, type=int, help="Maximum L used (for RIFT-based calls)")
parser.add("--start_index", default=None, type=int, help="Starting row in posterior file, where the first line is zero ")
parser.add("--end_index", default=None, type=int, help="Ending index in posterior file, noninclusive")
parser.add("--internal-use-normal-reweight", default=None, type=float)
parser.add("--internal-use-common-cal-draws", action='store_true',help="By default, the code will use a different set of draws for each 'start_index', to avoid collisions accessing the same h5 file")
args = parser.parse_args()

with open(args.data_dump_file, "rb") as data_file:
    data = pickle.load(data_file)



start_index = args.start_index
end_index = args.end_index

# read in the posterior samples for reweighting
if args.posterior_sample_file.split(".")[-1] == 'json' or args.posterior_sample_file.split(".")[-1]=='hdf5':
    result = bilby.core.result.read_in_result(args.posterior_sample_file)
    if end_index is not None:
        if end_index > len(result.posterior):
            end_index = len(result.posterior)
    result.posterior = result.posterior.iloc[start_index:end_index].reset_index(drop=True)
elif (args.posterior_sample_file.split(".")[-1] == 'txt') or (args.posterior_sample_file.split(".")[-1] == 'dat'):
    result = bilby.core.result.Result()
    result.posterior = pd.DataFrame(np.genfromtxt(args.posterior_sample_file, names=True))
    if end_index is not None:
        if end_index > len(result.posterior):
            end_index = len(result.posterior)
    result.posterior = result.posterior.iloc[start_index:end_index].reset_index(drop=True)
    result.meta_data = {}

    if args.use_rift_samples:
        if 'ps' in result.posterior:
          result.posterior  = result.posterior.drop(columns=['lnL','ps'])
        if 'p' in result.posterior:
          result.posterior['p'] = np.log(result.posterior['p'])  # not sure if used, but if so define correctly
        # The key_sap_dict does not have an 'eccentricity' key since both RIFT and Bilby use "eccentricity" as the key.
        # DO NOT ADD THIS TO THE key_swap_dict; THIS WILL CAUSE AN ERROR USING ECCENTRIC WAVEFORMS
        if args.use_eccentricity:
            key_swap_dict = {'m1':'mass_1', 'm2':'mass_2', 'a1x':'spin_1x', 'a1y':'spin_1y', 'a1z':'spin_1z',
                             'a2x':'spin_2x', 'a2y':'spin_2y', 'a2z':'spin_2z', 'incl':'iota', 'time':'geocent_time',
                             'phiorb':'phase', 'p':'log_prior', 'distance':'luminosity_distance', 'lambda1':'lambda_1', 'lambda2':'lambda_2',
                             'meanPerAno':'mean_per_ano'}
        else:
            key_swap_dict = {'m1':'mass_1', 'm2':'mass_2', 'a1x':'spin_1x', 'a1y':'spin_1y', 'a1z':'spin_1z',
                             'a2x':'spin_2x', 'a2y':'spin_2y', 'a2z':'spin_2z', 'incl':'iota', 'time':'geocent_time',
                             'phiorb':'phase', 'p':'log_prior', 'distance':'luminosity_distance', 'lambda1':'lambda_1', 'lambda2':'lambda_2'
                             }
            

        names = list(result.posterior.keys())  # dangerous to have iterator tied to changing structure
        for old_key in names:
            if old_key in key_swap_dict:
                result.posterior[key_swap_dict[old_key]] = result.posterior[old_key]
                del result.posterior[old_key]
        # add tides if not present
        if not('lambda_1' in result.posterior):
            print(" Populating empty tidal params to avoid hang")
            if end_index:
              result.posterior['lambda_1'] = np.zeros(end_index-start_index)
              result.posterior['lambda_2'] = np.zeros(end_index-start_index)
            else:
              result.posterior['lambda_1'] = np.zeros(len(result.posterior['mass_1']))
              result.posterior['lambda_2'] = np.zeros(len(result.posterior['mass_1']))

outdir = os.path.dirname(os.path.abspath(args.posterior_sample_file))

ifos = data.interferometers
time_marginalization_interval = args.data_integration_window_half #args.time_marginalization_interval

spline_calibration_envelope_dict = bilby_pipe.utils.convert_string_to_dict(
                data.meta_data['command_line_args']['spline_calibration_envelope_dict'])
ifos_for_reweighting = deepcopy(ifos)
if args.use_rift_samples:
 for ifo in ifos: # removes any model for the calibration that was set up in the file
    ifo.calibration_model = bilby.gw.calibration.Recalibrate()

if args.data_integration_window_half: #args.time_marginalization:
    result.posterior['geocent_time'] = ifos.start_time * np.ones(len(result.posterior))
    if 'time_jitter' not in result.posterior.keys():
        result.posterior['time_jitter'] = np.zeros(len(result.posterior))

if args.phase_marginalization:
    if not args.use_rift_samples:
        result.posterior['phase'] = np.zeros(len(result.posterior))
if args.use_gwsignal_lmax_nyquist:
    args.use_gwsignal = True  # force setting, in case of fallthrough
if args.use_gwsignal:
    args.h_method = 'gws_hlmoft' 

# Setting up the waveform generator using the data dump features
fref = data.meta_data['command_line_args']['reference_frequency']
if args.fref:
    print(" Reference frequency on *this* command line overrides value in pickle: old, new   = ", fref, args.fref)
    fref = args.fref
waveform_arguments = dict(
    reference_frequency=fref,
    minimum_frequency=args.fmin,
    waveform_approximant=data.meta_data['command_line_args']['waveform_approximant'],
    sampling_frequency=ifos.sampling_frequency,
    h_method=args.h_method)

extra_waveform_kwargs={}
if args.internal_waveform_fd_L_frame:
    extra_waveform_kwargs = {'fd_L_frame':True}
if args.internal_waveform_fd_no_condition:
    extra_waveform_kwargs['no_condition'] = True
if args.use_gwsignal_lmax_nyquist:
    extra_waveform_kwargs['lmax_nyquist'] = int(args.use_gwsignal_lmax_nyquist)
if args.extra_waveform_kwargs:
    my_arg_dict = eval(args.extra_waveform_kwargs)
    # dictionary may be malformed (eg not properly quoted) or render as string
    if not(isinstance(my_arg_dict, dict)):
        base_list = my_arg_dict[1:-1].split(',') # remove {} at end, assume string
        base_dict = {}
        for item in base_list:
            if item:
                key,value =item.split(':')
                key = key.lstrip()
                base_dict[key] = value
        my_arg_dict = base_dict
    extra_waveform_kwargs.update(my_arg_dict)
waveform_arguments['extra_waveform_kwargs'] = extra_waveform_kwargs
if args.waveform_approximant:
    waveform_arguments['waveform_approximant'] = args.waveform_approximant

if args.use_rift_samples:
    waveform_arguments['Lmax'] = args.l_max
#    waveform_arguments['waveform_approximant'] = 'SEOBNRv4PHM'
    if args.use_eccentricity:
        wf_func = rift_source.RIFT_lal_eccentric_binary_black_hole
    else:
        wf_func = rift_source.RIFT_lal_binary_black_hole
else:
    wf_func = eval('bilby.gw.source.'+data.meta_data['command_line_args']['frequency_domain_source_model'])
wf_func_alt = eval('bilby.gw.source.'+data.meta_data['command_line_args']['frequency_domain_source_model'])

print(data.meta_data['command_line_args']['frequency_domain_source_model'])

waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            frequency_domain_source_model=wf_func,
            sampling_frequency=ifos.sampling_frequency,
            duration=ifos.duration,
            start_time=ifos.start_time,
            waveform_arguments=waveform_arguments)
waveform_generator_alt =  bilby.gw.waveform_generator.WaveformGenerator(
            frequency_domain_source_model=copy(wf_func),
            sampling_frequency=ifos.sampling_frequency,
            duration=ifos.duration,
            start_time=ifos.start_time,
            waveform_arguments=waveform_arguments)
# if False: #args.use_gwsignal and hasattr(bilby.gw.waveform_generator, 'GWSignalWaveformGenerator'):
#   waveform_generator_alt = bilby.gw.waveform_generator.GWSignalWaveformGenerator(
#             sampling_frequency=ifos.sampling_frequency,
#             duration=ifos.duration,
#             start_time=ifos.start_time,
#             waveform_arguments=waveform_arguments)
# elif False: #args.use_gwsignal:
#     args_here = copy(waveform_arguments)
#     del(args_here['sampling_frequency'])
#     del(args_here['h_method'])
#     del(args_here['extra_waveform_kwargs'])
#     del(args_here['Lmax'])
#     waveform_generator_alt =  bilby.gw.waveform_generator.WaveformGenerator(
#             frequency_domain_source_model=bilby.gw.source.gwsignal_binary_black_hole,
#             #sampling_frequency=ifos.sampling_frequency,
#             duration=ifos.duration,
#             start_time=ifos.start_time,
#             waveform_arguments=args_here)
# else:
#     print(" FAILED ATTEMPT TO SET GWSIGNAL ALTERNATE")
print(waveform_generator)
#print(waveform_generator_alt)

# Setting up the calibration priors given the spline files for each interferometer
# This is used to generate draws of the calibration uncertainty from their prior
priors = bilby.core.prior.PriorDict()
for ifo in ifos_for_reweighting:
    calibration_file_path = f'{spline_calibration_envelope_dict[ifo.name]}'
    ifo_calibration_priors = bilby.gw.prior.CalibrationPriorDict.from_envelope_file(
        calibration_file_path, ifo.minimum_frequency, ifo.maximum_frequency, 10, ifo.name)

    # TODO FOR DEBUGGING PURPOSES
    # for key in ifo_calibration_priors.keys():
    #     if 'amp' in key:
    #         ifo_calibration_priors[key] = bilby.core.prior.DeltaFunction(peak=0)
    #     elif 'phase' in key:
    #         ifo_calibration_priors[key] = bilby.core.prior.DeltaFunction(peak=0)

    priors.update(ifo_calibration_priors)

# Documentation: priors used for cal marg!  
#   Important sanity check
#print(" CALMARG PRIORS")
#for name in priors:
#    print(name, priors[name])



marg_priors = bilby.core.prior.PriorDict() # priors for the non-calibration likelihood


if args.data_integration_window_half: # args.time_marginalization:
    priors['geocent_time'] = bilby.core.prior.Uniform(
        float(data.meta_data['command_line_args']['trigger_time']) - time_marginalization_interval,
        float(data.meta_data['command_line_args']['trigger_time']) + time_marginalization_interval)

    marg_priors['geocent_time'] = bilby.core.prior.Uniform(
        float(data.meta_data['command_line_args']['trigger_time']) - time_marginalization_interval,
        float(data.meta_data['command_line_args']['trigger_time']) + time_marginalization_interval)

if args.phase_marginalization:
    priors['phase'] = bilby.core.prior.Uniform(0, 2 * np.pi)
    marg_priors['phase'] = bilby.core.prior.Uniform(0, 2 * np.pi)

# Setting up the INPUT for the calibration draw files
calibration_lookup_table = {}
for ifo in ifos:
    extra_string='-{}-'.format(start_index)
    calibration_lookup_table[ifo.name] =\
                        f'{outdir}/{ifo.name}{extra_string}_calibration_file.h5'

# WARNING: time marginalization is done with *bool*, so actual value of time window  is NOT USED.
original_likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator,
    time_marginalization=args.data_integration_window_half,
    phase_marginalization=args.phase_marginalization, priors=marg_priors)

calibration_likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos_for_reweighting, waveform_generator=waveform_generator,
    number_of_response_curves=args.number_of_calibration_curves, calibration_marginalization=True,
    priors=priors, time_marginalization=args.data_integration_window_half, phase_marginalization=args.phase_marginalization,
    calibration_lookup_table=calibration_lookup_table)


# Generate random calibration draws for each event. Note we have ALREADY done lookup to correctly identify files needed
recal_indx_array = np.zeros(len(result.posterior),dtype=int)
for indx in range(len(result.posterior)):
    dict_samples = {key: result.posterior[key][indx] for key in result.posterior}
    calibration_likelihood.parameters = dict_samples
    recal_indx_array[indx] = int(calibration_likelihood.generate_calibration_sample_from_marginalized_likelihood())

# n [8]: import h5py
#    ...: with h5py.File("/home/pe.o4/GWTC5-HLV/project/working/S240520cv/rift-v5PHM-calmarg/L1-31800-_calibr
#       ⋮ ation_file.h5", "r") as ff:
#    ...:     print(ff["CalParams"]["table"]["recalib_L1_amplitude_0"])
# with h5py.File("/home/pe.o4/GWTC5-HLV/project/working/S240520cv/rift-v5PHM-calmarg/L1-31800-_calibr
#       ⋮ ation_file.h5", "r") as ff:
#    ...:     params = ff["CalParams"]
#    ...:     print(params["table"].dtype)
#    ...: 

   
    
# open all files
# create space in cal file for data (zeros)
recal_file_dict = {}
new_posterior = copy(result.posterior)
cal_names_for = {}
for ifo in ifos:
    ifo_name = ifo.name
    recal_file_dict[ifo_name] = h5py.File(calibration_lookup_table[ifo_name], 'r')
    cal_param_names = recal_file_dict[ifo_name]["CalParams"]["table"].dtype.names
    cal_param_names = [x for x in cal_param_names if 'recalib' in x]
    cal_param_names_freq = [x for x in cal_param_names if 'frequency' in x]
    cal_param_names_rest = list( set(cal_param_names) - set(cal_param_names_freq) )
    cal_names_for[ifo_name] =  cal_param_names_rest
    # assign blank entries for remaining parameters
    args = dict(zip( cal_param_names_rest, [ np.zeros(len(new_posterior))  for x in cal_param_names_rest] ) )
#    print(args)
    new_posterior = new_posterior.assign(**args) # empty arrays
    freq_values = [ recal_file_dict[ifo_name]["CalParams"]["table"][name][0]  for name in cal_param_names_freq]
    args = dict(zip( cal_param_names_freq, [ freq_values[indx]*np.ones(len(new_posterior))  for indx in range(len(freq_values))] )) # frequency values
#    print(args)
    new_posterior = new_posterior.assign(**args)
    
    # for name in cal_param_names:
    #     if 'recalib' in name:
    #         if 'frequency' in name:
    #             result.posterior[name ] = recal_file_dict[ifo_name]["CalParams"]["table"][name][0] *np.ones(len(result.posterior))
    #         else:
    #             result.posterior[name ] = np.zeros(len(result.posterior)) # zero out
for ifo in ifos:
    ifo_name = ifo.name
    # loop over draws and assign parameters using the index, for non-frequency arrays
    for name in cal_names_for[ifo_name]:
        for indx_event in range(len(result.posterior)):
#            new_posterior[name][indx_event] = recal_file_dict[ifo_name]["CalParams"]["table"][name][    recal_indx_array[indx_event]]
            new_posterior.loc[indx_event,name] = recal_file_dict[ifo_name]["CalParams"]["table"][name][    recal_indx_array[indx_event]]

                

print(new_posterior)
#print(original_likelihood.noise_log_likelihood())

# save result file.
# Note not sure if we need to do any recalculation
new_posterior.to_csv('test.txt',sep=' ',index=False)
