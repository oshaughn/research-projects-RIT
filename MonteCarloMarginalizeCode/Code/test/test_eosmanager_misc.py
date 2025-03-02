


import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
import argparse


import lal
import RIFT.lalsimutils as lalsimutils
import RIFT.physics.EOSManager as EOSManager
import lalsimulation as lalsim
from scipy.integrate import nquad
from RIFT.plot_utilities.EOSPlotUtilities import render_eos

DENSITY_CGS_IN_MSQUARED=1000*lal.G_SI/lal.C_SI**2  # g/cm^3 -> 1/m^2 //GRUnits. Multiply by this to convert from CGS -> 1/m^2 units (_geom). lal.G_SI/lal.C_SI**2 takes kg/m^3 -> 1/m^2  ||  https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_neutron_star_8h_source.html
PRESSURE_CGS_IN_MSQUARED = DENSITY_CGS_IN_MSQUARED/(lal.C_SI*100)**2

matplotlib.rcParams.update({'font.size': 12.0,  'mathtext.fontset': 'stix'})
matplotlib.rcParams['figure.figsize'] = (9.0, 7.0)
matplotlib.rcParams['xtick.labelsize'] = 15.0
matplotlib.rcParams['ytick.labelsize'] = 15.0
matplotlib.rcParams['axes.labelsize'] = 25.0
matplotlib.rcParams['lines.linewidth'] = 2.0
plt.style.use('seaborn-v0_8-whitegrid')


# File location
eos_base_dir = os.environ['LALSIMULATION_DATADIR']


# Demo 1: Tabular i/o
#    - key point 1: does not fill in data at lower density: finite range
#    - key point 2: data format follows lalsuite standard, in geometric units
eos_names =  ['LALSimNeutronStarEOS_AP4.dat',
              'LALSimNeutronStarEOS_WFF1.dat']
for name in eos_names:
    data = np.loadtxt(eos_base_dir+"/"+name, delimiter = "\t")
    neweos = EOSManager.EOSFromTabularData(eos_data=data, reject_phase_transitions=True)
    plt.loglog(data[:,1]/DENSITY_CGS_IN_MSQUARED,data[:,0]/PRESSURE_CGS_IN_MSQUARED, label = name+' raw', ls = 'dashed')
    plot_render = render_eos(eos=neweos.eos, xvar='energy_density', yvar='pressure')
    plt.legend()
    plt.title("EOSFromTabularData comparison to raw data")
plt.savefig("fig_demo_EOSFromTabular.pdf"); plt.clf()

# Demo 2: Conventional EOS names with lalsuite
#   - key point: really also using the same low-level interface as tabular data
#   - key point: LALSimulation interface does NOT paste on low density EOS model
my_eos = EOSManager.EOSLALSimulation('SLy')
render_eos(my_eos.eos,'energy_density', 'pressure')
plt.savefig('fig_demo_EOSLALSimulation.pdf'); plt.clf()

# Demo 3: MR and M/Lambda calculations, for the same fixed EOS
m_r_L_data = EOSManager.make_mr_lambda_lal(my_eos.eos, n_bins=200)
plt.plot(m_r_L_data[:,1], m_r_L_data[:,0])
plt.xlabel('R (km)'); plt.ylabel('M (Msun)');
plt.savefig("fig_demo_MR.pdf"); plt.clf()
plt.plot(m_r_L_data[:,0], m_r_L_data[:,2])
plt.yscale('log')
plt.xlabel('M'); plt.ylabel('Lambda')
plt.savefig("fig_demo_MLambda.pdf"); plt.clf()


# Demo 4: EOS from tabular information access
fname_tabular_test = "LCEHL_EOS_posterior_samples_PSR.h5"
if os.path.exists(fname_tabular_test):
    my_eos_sequence = EOSManager.EOSSequenceLandry(fname=fname_tabular_test,load_ns=False,load_eos=True,verbose=True,eos_tables_units='cgs')
    plt.xlabel(r'$\rho$ (cgs)')
    plt.ylabel(r'$c_s/c$')
    for indx in range(5):
        my_single_eos = my_eos_sequence.extract_one_eos_object(indx=indx)
        plot_render = render_eos(eos=my_single_eos.eos, xvar='energy_density', yvar='sound_speed_over_c')  # could be anything
    plt.savefig("fig_demo_TabularSoundSpeed.pdf")
