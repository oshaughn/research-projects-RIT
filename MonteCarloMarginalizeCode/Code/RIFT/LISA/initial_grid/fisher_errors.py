#!/usr/bin/env python

"""The purpose of this code is to give rough estimates of width of posteriors for Mc, eta, spin and skylocation for initial grid generation."""
import numpy as np
import RIFT.lalsimutils as lsu
from RIFT.LISA.response.LISA_response import *
from argparse import ArgumentParser
from scipy.interpolate import interp1d
import os
###########################################################################################
# Functions to generate 2-PN waveforms as per http://arxiv.org/abs/gr-qc/9502040.
###########################################################################################
def amplitude_2PN(fvals, Mc):
    """Returns amplitude in frequency domains for a 2-PN waveform"""
#    print(f"amplitude_2PN: {locals()}")
    return 1 * Mc**(5/6) * fvals**(-7/6)

def phase_2PN(fvals, Mc, eta, sigma, beta, coa_phase=0, coa_time=0):
    """Returns phase in frequency domains for a 2-PN waveform"""
#    print(f"phase_2PN: {locals()}")
    M = Mc/eta**(3/5)
    fac = np.pi*M*fvals
    newtonian = 3/128 * (np.pi*Mc*fvals)**(-5/3)
    one_PN = 20/9 * (743/336 + 11/4* eta)*(fac)**(2/3)
    one_five_PN = 4 * (4*np.pi - beta)*fac
    two_PN = 10*(3058673/1016064 + 5429/1008*eta + 617/144*eta**2 - sigma)*(fac)**(4/3)
    phase_vals = 2*np.pi*fvals*coa_time - coa_phase - np.pi/4 + newtonian * (1 + one_PN  - one_five_PN + two_PN)
    return phase_vals

def time_2PN(fvals, Mc, eta, sigma, beta, coa_time=0):
    """Returns time corresponding to input frequency for a 2-PN waveform"""
    M = Mc/eta**(3/5)
    fac = np.pi*M*fvals
    time_vals = coa_time - 5/256* Mc*(np.pi*Mc*fvals)**(-8/3) * (1 + 4/3 * (743/336 + 11/4 * eta)*(fac)**(2/3) - 8/5 *(4*np.pi - beta)*fac + 2*(3058673/1016064 + 5429/1008 * eta + 617/144*eta**2 - sigma)*(fac)**(4/3))
    return time_vals

def get_sigma_beta(Mc, eta, a1z, a2z):
    """Returns sigma and beta parameters that are used in construction of the waveform. These parameters contain information about spin."""
    alpha = Mc / eta**(3/5)
    beta = Mc**2 / eta**(1/5)
    m1 = 0.5 * (alpha + np.sqrt(alpha**2 - 4*beta))
    m2 = 0.5 * (alpha - np.sqrt(alpha**2 - 4*beta))
    M = m1 + m2
    sigma_val = eta/48 * (-247 * a1z*a1z + 721 * a1z*a2z)
    beta_val = 1/12 * ((113 * (m1/M)**2 + 75 * eta)*a1z + (113 * (m2/M)**2 + 75 * eta)*a2z )
    return sigma_val, beta_val

def get_derivative(fvals, Mc, eta, sigma, beta, wf):
    """Derivates of the waveform with respect to coalescence time, coalescence phase, Mc, eta, sigma, beta"""
#    print(f"get_derivative: {locals()}")
    M = Mc/eta**(3/5)
    v = (np.pi*M*fvals)**(1/3)

    A4 = 4/3 * (743/336 + 11/4 * eta)
    B4 = 8/5 * (4*np.pi - beta)
    C4 = 2 * (3058673/1016064 + 5429/1008 * eta + 617/144 * eta**2 - sigma)

    A5 = 743/168 - 33/4 * eta
    B5 = 27/5 * (4*np.pi - beta)
    C5 = 18 * (3058673/1016064 - 5429/4032 * eta - 617/96*eta**2 -  sigma)

    d_tc = 2*np.pi*1j*(fvals) * wf
    d_phi = -1j * wf

    d_log_mc =   -1j*(5/128 * (np.pi*Mc*fvals)**(-5/3) * (1 + A4*v**2 - B4*v**3 + C4*v**4)) * wf
    d_log_eta =  -1j*(1/96  * (np.pi*Mc*fvals)**(-5/3) * (    A5*v**2 - B5*v**3 + C5*v**4)) * wf

    d_beta = 1j * 3/32 * eta**(-3/5) * (np.pi*Mc*fvals)**(-2/3) * wf
    d_sigma = -1j * 15/64 * eta**(-4/5) * (np.pi*Mc*fvals)**(-1/3) * wf
    return np.array([d_tc, d_phi, d_log_mc, d_log_eta, d_beta, d_sigma])

def get_wf(fvals, Mc, eta, sigma, beta, psd_vals, coa_phase=0, coa_time=0, snr=None, LISA_response=False, skylocation = None):
    """Generate a 2-PN waveform"""
    phase = phase_2PN(fvals, Mc, eta, sigma, beta, coa_phase, coa_time)
    amp = amplitude_2PN(fvals, Mc)
    wf = amp * np.exp(1j*phase)
    if LISA_response:
         H = transformed_Hplus_Hcross(skylocation[0], skylocation[1], 0.0, 0.0, 0.0, 2, 2)
         time = time_2PN(fvals, Mc, eta, sigma, beta, coa_time)
         A, E, T = Evaluate_Gslr(time + coa_time, fvals, H, skylocation[0], skylocation[1])
         wf = wf * A
    if snr:
        # bring the source closer or further, depending on SNR
        deltaF = np.diff(fvals)[0]
        snr_fiducial = np.sqrt(get_inner_product(wf, wf, psd_vals, deltaF))
        correction = snr / snr_fiducial
        wf = correction * wf
    return wf

###########################################################################################
# Utilities 
###########################################################################################
def load_psd(psdf, fvals):
    """Loads in PSD"""
    # load in psd
    psd_dict = {}
    inst = "A"
    print( "Reading PSD for instrument %s from %s" % (inst, psdf))
    psd_dict[inst] = lsu.get_psd_series_from_xmldoc(psdf, inst)
    psd_fvals = psd_dict[inst].f0 + psd_dict[inst].deltaF*np.arange(psd_dict[inst].data.length)
    interp_func = interp1d(psd_fvals, psd_dict[inst].data.data)
    return interp_func(fvals)

def get_mass_from_mc_eta(mc, eta):
    """Returns m1, m2 from mc and eta."""
    alpha = mc / eta**(3/5)
    beta = mc**2 / eta**(1/5)
    m1 = 0.5 * (alpha + np.sqrt(alpha**2 - 4*beta))
    m2 = 0.5 * (alpha - np.sqrt(alpha**2 - 4*beta)) 
    return m1, m2

def get_mc_eta_from_mass(m1, m2):
    """Returns m1, m2 from mc and eta."""
    mc = (m1*m2)**(3/5) / (m1+m2)**(1/5)
    eta = (m1*m2) / (m1+m2)**(2)
    if eta==0.25:
        eta=0.24999
    return mc, eta

def get_inner_product(wf1, wf2, psd_vals, deltaF):
    """Calculate inner product"""
    assert len(wf1) == len(wf2) == len(psd_vals)
    weight = 1/psd_vals
    intgd = np.sum(np.conj(wf1) * wf2 * weight) * deltaF
    return 4 * np.real(intgd)

def get_massratio_error(eta, q, eta_error):
    return eta_error * (1 + q)**3 / (1 - q)

def get_spin_error(eta, q, a1z, a2z, eta_error, beta_error, sigma_error):
    q_error = get_massratio_error(eta, q, eta_error)
    if round(a1z,4)==round(a2z,4):
        a1z = a1z + 0.001 * a1z
        a2z = a2z - 0.001 * a2z

    c1 = sigma_error - eta_error/48 * (474*a1z*a2z)
    a1 = eta/48 * 474 * a2z
    b1 = eta/48 * 474 * a1z
    
    c2 = beta_error - eta_error/12 * ( (113/q + 75)*a1z + (113*q + 75)*a2z) - eta/12 * ( (-113/q**2)*a1z*q_error + 113*a2z*q_error)
    a2 = eta/12 * (113/q + 75)
    b2 = eta/12 * (113*q + 75)
    
    coefficients = np.array([ [a1, a2], [b1, b2] ])
    dependents = np.array( [c1, c2] )
    answers = np.abs(np.linalg.solve(coefficients, dependents))
    return [np.min(answers), np.max(answers)]

###########################################################################################
# Fisher matrix
###########################################################################################
def get_fisher_matrix(Mc, eta, sigma, beta, fvals, psd_vals, deltaF, wf):
    """Get fisher information matrix"""
    derivatives = get_derivative(fvals, Mc, eta, sigma, beta, wf)
    N = 6
    tau_ij = np.zeros((N,N))
    for i in np.arange(0, N):
        for j in np.arange(0, N):
            tau_ij[i,j] = get_inner_product(derivatives[i], derivatives[j], psd_vals, deltaF)
    inv_tau_ij = (np.linalg.inv(tau_ij))
    return tau_ij, inv_tau_ij

def get_error_bounds(P_inj, snr, psd_path):
    response=True # use LISA response
    deltaF = 0.00001 # hardcoded deltaF 
    mc, eta = get_mc_eta_from_mass(P_inj.m1/lsu.lsu_MSUN, P_inj.m2/lsu.lsu_MSUN)
    q = P_inj.m2/P_inj.m1

    if q == 1:
        q = 0.9
    
    # convert chirp mass to seconds
    Mc = mc * 5 * 10**(-6)
    M = Mc/eta**(3/5)
    sigma, beta = get_sigma_beta(Mc, eta, P_inj.s1z, P_inj.s2z)
    fmax = 6**(-3/2) / np.pi/ M
    print(f"Fmax is = {fmax} Hz")
    fvals = np.arange(float(opts.snr_fmin), fmax, deltaF)

    # Load psd
    psd_vals = load_psd(opts.psd_path, fvals)

    # generate waveform
    wf = get_wf(fvals, Mc, eta, sigma, beta, psd_vals, 0, float(P_inj.tref), snr=float(opts.snr), LISA_response=response, skylocation=[P_inj.theta, P_inj.phi])
    print(f"SNR of generated waveform is = {np.sqrt(get_inner_product(wf,wf,psd_vals, deltaF))}")

    # Calculate fisher matrix
    tau_ij, inv_tau_ij = get_fisher_matrix(Mc, eta, sigma, beta, fvals, psd_vals, deltaF, wf)
    factor_eta = 12.5
    factor_mc = 50
    factor_spin1 = 60
    factor_spin2 = 60
    spin_bounds = get_spin_error(eta, q, P_inj.s1z, P_inj.s2z, (np.sqrt(1/tau_ij[3,3]))*eta, np.sqrt(1/tau_ij[4,4]), np.sqrt(1/tau_ij[5,5]))
    
    print(f"Mc span = {2*factor_mc*np.sqrt(1/tau_ij[2,2])*mc}, eta span = {2*np.sqrt(1/tau_ij[3,3])*eta*factor_eta}, s1z span = {2*factor_spin1*spin_bounds[0]}, s2z span = {2*factor_spin2*spin_bounds[1]}, beta span = {0.036*(210/snr)**2}, lambda span = {0.044*(210/snr)**2}")


    return np.array([ mc - factor_mc*np.sqrt(1/tau_ij[2,2])*mc, mc + factor_mc*np.sqrt(1/tau_ij[2,2])*mc, eta-(np.sqrt(1/tau_ij[3,3]))*eta*factor_eta, eta+(np.sqrt(1/tau_ij[3,3]))*eta*factor_eta, P_inj.s1z - factor_spin1*spin_bounds[0], P_inj.s1z + factor_spin1*spin_bounds[0], P_inj.s2z - factor_spin2*spin_bounds[1],  P_inj.s2z + factor_spin2*spin_bounds[1], P_inj.theta - 0.018*(210/snr), P_inj.theta + 0.018*(210/snr),  P_inj.phi - 0.022*(210/snr), P_inj.phi + 0.022*(210/snr)])


###########################################################################################

if __name__ =='__main__':
    ###########################################################################################
    parser=ArgumentParser()
    parser.add_argument("--inj", help="Full path to mdc.xml.gz")
    parser.add_argument("--psd-path", help="Full path to A-psd.xm.gz")
    parser.add_argument("--snr", help="SNR of the signal")
    parser.add_argument("--snr-fmin", help="fmin used in snr calculations", default=0.0001)
    opts = parser.parse_args()
    print(f"Loading file:\n {opts.inj}")
    P_inj_list = lsu.xml_to_ChooseWaveformParams_array(opts.inj)
    P_inj = P_inj_list[0]
    print("######")
    print(f"m1 = {P_inj.m1/lsu.lsu_MSUN}, m2 = {P_inj.m2/lsu.lsu_MSUN}, s1z = {P_inj.s1z}, s2z = {P_inj.s2z}, beta = {P_inj.theta}, lambda = {P_inj.phi}, tref = {P_inj.tref} s")
    print("######")
    error_bounds = get_error_bounds(P_inj, float(opts.snr), opts.psd_path)
    print(f"Mc bounds = [{error_bounds[0]:0.2f}, {error_bounds[1]:0.2f}]")
    print(f"eta bounds = [{error_bounds[2]:0.8f}, {error_bounds[3]:0.8f}]")
    print(f"s1z bounds = [{error_bounds[4]:0.6f}, {error_bounds[5]:0.6f}]")
    print(f"s2z bounds = [{error_bounds[6]:0.6f}, {error_bounds[7]:0.6f}]")
    print(f"beta bounds = [{error_bounds[8]:0.6f}, {error_bounds[9]:0.6f}]")
    print(f"lambda bounds = [{error_bounds[10]:0.6f}, {error_bounds[11]:0.6f}]")







