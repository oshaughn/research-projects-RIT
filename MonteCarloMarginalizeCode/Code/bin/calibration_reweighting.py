#! /usr/bin/env python
"""
Author: Ethan Payne (ethan.payne@ligo.org)

Calibration marginalization of any generic set of samples. Note that this needs to be used within reason.
The calibration marginalization generates N_ifo matrices of size N_freq_bins by N_calibration_curves.
This can be very memory intensive.

We assume that the likelihood required is only the GWTransientLikelihood with and without calibration marginalization.
No other marginalizations are included. The calibration marginalized likelihood can work with both distance and phase
marginalization if desired.

Also note that the rejection sampling used can be quite lossy - even though it does not significantly change the
posterior distribution.

Inputs
------
posterior_sample_file: Bilby result .json or .dat file
    Either a Bilby result file or a .dat file with headers. Contains the posterior samples of the distribution to be
    used in the reweighting calculation.
data_dump_file: .pickle file
    File produced from bilby_pipe_generation config.ini. This gives an easy way to import in all the required data, and
    in the case of bilby_pipe runs, ensure the same data is used between the original PE and the reweighting.
number_of_calibration_curves: int
    Number of draws of the calibration model which we marginalize over.
    Default: 100.
reevaluate_likelihood: bool
    Sets whether we want to also reevaluate the likelihood without calibration uncertainty. This can ensure that the
    likelihood definitions are consistent (i.e. log_likelihood or log_likelihood - log_noise_evidence).
    Default: True

Output
------
reweighted_posterior_samples.dat:
    Data file with the rejection sampled posterior samples from the supplied posterior file
weights.dat:
    File with the weight for each sample in the supplied posterior file. This allows the user to plot results using the
    the weights as opposed to the rejection sampling posterior samples.
"""

import os
import sys

import argparse
from copy import deepcopy, copy

import numpy as np
import pandas as pd
import pickle

import bilby
import bilby_pipe

# So I can import the RIFT source while not making this some setup.py-able package
# TODO this should not be a hardcoded path!

import RIFT.calmarg.rift_source as rift_source

from  bilby.core.utils import logger


# Alternate implementation of bilby.core.result.reweight
#   - we are using draws, not the prior!
#   - sometimes we get delta-function defailt cal priors, so we get infinities//nan
def alt_reweight(result, label=None, new_likelihood=None, new_prior=None,
             old_likelihood=None, old_prior=None, conversion_function=None, npool=1,
             verbose_output=False, resume_file=None, n_checkpoint=5000,
             use_nested_samples=False):
    """ Reweight a result to a new likelihood/prior using rejection sampling

    Parameters
    ==========
    label: str, optional
        An updated label to apply to the result object
    new_likelihood: bilby.core.likelood.Likelihood, (optional)
        If given, the new likelihood to reweight too. If not given, likelihood
        reweighting is not applied
    new_prior: bilby.core.prior.PriorDict, (optional)
        If given, the new prior to reweight too. If not given, prior
        reweighting is not applied
    old_likelihood: bilby.core.likelihood.Likelihood, (optional)
        If given, calculate the old likelihoods from this object. If not given,
        the values stored in the posterior are used.
    old_prior: bilby.core.prior.PriorDict, (optional)
        If given, calculate the old prior from this object. If not given,
        the values stored in the posterior are used.
    conversion_function: function, optional
        Function which adds in extra parameters to the data frame,
        should take the data_frame, likelihood and prior as arguments.
    npool: int, optional
        Number of threads with which to execute the conversion function
    verbose_output: bool, optional
        Flag determining whether the weight array and associated prior and
        likelihood evaluations are output as well as the result file
    resume_file: string, optional
        filepath for the resume file which stores the weights
    n_checkpoint: int, optional
        Number of samples to reweight before writing a resume file
    use_nested_samples: bool, optional
        If true reweight the nested samples instead. This can greatly improve reweighting efficiency, especially if the
        target distribution has support beyond the proposal posterior distribution.

    Returns
    =======
    result: bilby.core.result.Result
        A copy of the result object with a reweighted posterior
    new_log_likelihood_array: array, optional (if verbose_output=True)
        An array of the natural-log likelihoods from the new likelihood
    new_log_prior_array: array, optional (if verbose_output=True)
        An array of the natural-log priors from the new likelihood
    old_log_likelihood_array: array, optional (if verbose_output=True)
        An array of the natural-log likelihoods from the old likelihood
    old_log_prior_array: array, optional (if verbose_output=True)
        An array of the natural-log priors from the old likelihood

    """
    from scipy.special import logsumexp

    result = copy(result)

    if use_nested_samples:
        result.posterior = result.nested_samples

    nposterior = len(result.posterior)
    logger.info("Reweighting posterior with {} samples".format(nposterior))

    ln_weights, new_log_likelihood_array, new_log_prior_array, old_log_likelihood_array, old_log_prior_array =\
        bilby.core.result.get_weights_for_reweighting(
            result, new_likelihood=new_likelihood, new_prior=new_prior,
            old_likelihood=old_likelihood, old_prior=old_prior,
            resume_file=resume_file, n_checkpoint=n_checkpoint, npool=npool)

    # ONLY USING LIKEILHOOD: we are NOT changing priors, just using cal draws
    
    ln_weights = new_log_likelihood_array - old_log_likelihood_array

    weights = np.exp(ln_weights)

    # Overwrite the likelihood and prior evaluations
    result.posterior["log_likelihood"] = new_log_likelihood_array
    result.posterior["log_prior"] = new_log_prior_array

    result.posterior = bilby.core.result.rejection_sample(result.posterior, weights=weights)
    result.posterior = result.posterior.reset_index(drop=True)
    logger.info("Rejection sampling resulted in {} samples".format(len(result.posterior)))
    result.meta_data["reweighted_using_rejection_sampling"] = True

    if use_nested_samples:
        result.log_evidence += np.log(np.sum(weights))
    else:
        result.log_evidence += logsumexp(ln_weights) - np.log(nposterior)

    if new_prior is not None:
        for key, prior in new_prior.items():
            result.priors[key] = prior

    if conversion_function is not None:
        data_frame = result.posterior
        if "npool" in inspect.signature(conversion_function).parameters:
            data_frame = conversion_function(data_frame, new_likelihood, new_prior, npool=npool)
        else:
            data_frame = conversion_function(data_frame, new_likelihood, new_prior)
        result.posterior = data_frame

    if label:
        result.label = label
    else:
        result.label += "_reweighted"

    if verbose_output:
        return result, weights, new_log_likelihood_array, \
            new_log_prior_array, old_log_likelihood_array, old_log_prior_array
    else:
        return result



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
parser.add("--internal-waveform-fd-L-frame",action='store_true',help='If true, passes extra_waveform_kwargs = {fd_L_frame=True} to lalsimutils hlmoft. Impacts outputs of ChooseFDWaveform calls only.')
parser.add("--internal-waveform-fd-no-condition",action='store_true',help='If true, adds extra_waveform_kwargs = {no_condition=True} to lalsimutils hlmoft. Impacts outputs of ChooseFDWaveform calls only. Provided to enable controlled tests of conditioning impact on PE')
parser.add("--use-gwsignal",default=False,action='store_true',help='Use gwsignal. In this case the approx name is passed as a string to the lalsimulation.gwsignal interface')
parser.add("--use-gwsignal-lmax-nyquist",default=None,type=int,help='Passes lmax_nyquist integer to the gwsignal waveform interface')
parser.add("--fmin", default=None, type=float)
parser.add("--l-max", default=4, type=int)
parser.add("--start_index", default=None, type=int)
parser.add("--end_index", default=None, type=int)
parser.add("--internal-use-normal-reweight", default=None, type=float)
args = parser.parse_args()

with open(args.data_dump_file, "rb") as data_file:
    data = pickle.load(data_file)

start_index = args.start_index
end_index = args.end_index

# read in the posterior samples for reweighting
if args.posterior_sample_file.split(".")[-1] == 'json':
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
        result.posterior  = result.posterior.drop(columns=['lnL','ps'])
        result.posterior['p'] = np.log(result.posterior['p'])  # not sure if used, but if so define correctly
        key_swap_dict = {'m1':'mass_1', 'm2':'mass_2', 'a1x':'spin_1x', 'a1y':'spin_1y', 'a1z':'spin_1z',
            'a2x':'spin_2x', 'a2y':'spin_2y', 'a2z':'spin_2z', 'incl':'iota', 'time':'geocent_time',
        'phiorb':'phase', 'p':'log_prior', 'distance':'luminosity_distance', 'lambda1':'lambda_1', 'lambda2':'lambda_2'}

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
for ifo in ifos: # removes any model for the calibration that was set up in the file
    ifo.calibration_model = bilby.gw.calibration.Recalibrate()

if args.data_integration_window_half: #args.time_marginalization:
    result.posterior['geocent_time'] = ifos.start_time * np.ones(len(result.posterior))
    if 'time_jitter' not in result.posterior.keys():
        result.posterior['time_jitter'] = np.zeros(len(result.posterior))

if args.phase_marginalization:
    if not args.use_rift_samples:
        result.posterior['phase'] = np.zeros(len(result.posterior))
if args.use_gwsignal:
    args.h_method = 'gws_hlmoft' 

# Setting up the waveform generator using the data dump features
waveform_arguments = dict(
    reference_frequency=data.meta_data['command_line_args']['reference_frequency'],
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
waveform_arguments['extra_waveform_kwargs'] = extra_waveform_kwargs
if args.waveform_approximant:
    waveform_arguments['waveform_approximant'] = args.waveform_approximant

if args.use_rift_samples:
    waveform_arguments['Lmax'] = args.l_max
#    waveform_arguments['waveform_approximant'] = 'SEOBNRv4PHM'
    wf_func = rift_source.RIFT_lal_binary_black_hole
else:
    wf_func = eval('bilby.gw.source.'+data.meta_data['command_line_args']['frequency_domain_source_model'])

waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            frequency_domain_source_model=wf_func,
            sampling_frequency=ifos.sampling_frequency,
            duration=ifos.duration,
            start_time=ifos.start_time,
            waveform_arguments=waveform_arguments)

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
print(" CALMARG PRIORS")
for name in priors:
    print(name, priors[name])


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

# Setting up the output for the calibration draw files
calibration_lookup_table = {}
for ifo in ifos:
    calibration_lookup_table[ifo.name] =\
                        f'{outdir}/{ifo.name}_calibration_file.h5'

# WARNING: time marginalization is done with *bool*, so actual value of time window  is NOT USED.
original_likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator,
    time_marginalization=args.data_integration_window_half,
    phase_marginalization=args.phase_marginalization, priors=marg_priors)

calibration_likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos_for_reweighting, waveform_generator=copy(waveform_generator),
    number_of_response_curves=args.number_of_calibration_curves, calibration_marginalization=True,
    priors=priors, time_marginalization=args.data_integration_window_half, phase_marginalization=args.phase_marginalization,
    calibration_lookup_table=calibration_lookup_table)

print(f'Log noise evidence: {original_likelihood.noise_log_likelihood()}')

# TODO have not yet implemented the parameter reconstruction in the case where
# time and calibration marginalization are used
if args.data_integration_window_half: # time_marginalization:
    conversion_function = None
else:
    conversion_function = bilby.gw.conversion.generate_all_bbh_parameters

# Setting up the resume and weight file names
resume_file = None
if (start_index != None) and (start_index != None):
    weights_file = f'{outdir}/weight_files/weights_s{start_index}e{end_index}.dat'
elif start_index != None:
    weights_file = f'{outdir}/weight_files/weights_s{start_index}eNone.dat'
elif end_index != None:
    weights_file = f'{outdir}/weight_files/weights_s0e{end_index}.dat'
else:
    resume_file = f'{outdir}/reweighting_resume.dat'
    weights_file = f'{outdir}/weights.dat'

reweight_func = alt_reweight
if args.internal_use_bilby_reweight:
    reweight_func = bilby.core.result.reweight

if args.reevaluate_likelihood:
    result_reweighted, weights, _, _, _, _ = reweight_func(
        result, new_likelihood=calibration_likelihood, old_likelihood=original_likelihood,
        conversion_function=conversion_function, verbose_output=True,
        resume_file=resume_file, n_checkpoint=5000,
        use_nested_samples=args.use_nested_samples)
else:
    result_reweighted, weights, _, _, _, _ = reweight_func(
        result, new_likelihood=conversion_function,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters, verbose_output=True,
        resume_file=resume_file, n_checkpoint=5000,
        use_nested_samples=args.use_nested_samples)

# Save the weights to a file, with the sample index in the name
if not os.path.exists(f'{outdir}/weight_files/'):
    os.makedirs(f'{outdir}/weight_files/')

np.savetxt(weights_file, weights)

if (start_index == None) and (end_index == None):
    result_reweighted.save_posterior_samples(filename=outdir+'/reweighted_posterior_samples.dat', outdir=outdir)
