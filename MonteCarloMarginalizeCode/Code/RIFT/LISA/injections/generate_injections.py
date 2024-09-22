from RIFT.LISA.injections.LISA_injections import *

param_dict = {}


param_dict.update({'m1':1757638.40382,
                   'm2':1300396.8986,
                   's1z':0.50977418,
                   's2z':0.13947856,
                   'beta':0.29339096,
                   'lambda':2.27853319,
                   'dist':33703.05400678,
                   'inclination':1.83510232,
                   'phi_ref':4.22856809,
                   'psi':1.24272195,
                   'tref':6712404.50294161,
                   'deltaT':5,
                   'deltaF':float(1.0/(41943040)),
                   'fmin':0.00008,
                   'fref':None,
                   'approx':"IMRPhenomD",
                   'modes':[(2,2)],
                   'save_path':"/Users/aasim/Desktop/Research/Projects/RIFT_LISA/Development/Testing/bias_in_sangria/frames_new",
                   'path_to_NR_hdf5':None, 
                   'psd_path':"/Users/aasim/Desktop/Research/Projects/RIFT_LISA/Development/Testing/bias_in_sangria/psds",
                   'snr_fmin':0.0001,
                   'snr_fmax':0.1})

data_dict = generate_lisa_TDI_dict(param_dict)
generate_lisa_injections(data_dict, param_dict, get_snr=True)