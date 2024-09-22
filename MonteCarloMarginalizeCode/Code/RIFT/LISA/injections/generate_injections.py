from RIFT.LISA.injections.LISA_injections import *

param_dict = {}

param_dict.update({'m1':,
                   'm2':,
                   's1z':,
                   's2z':,
                   'beta':,
                   'lambda':,
                   'dist':,
                   'inclination':
                   'phi_ref':,
                   'psi':,
                   'tref':,
                   'deltaT':,
                   'deltaF':,
                   'fmin':,
                   'fref':,
                   'approx':,
                   'modes':,
                   'path_to_NR_hdf5':,
                   'save_path':,
                   'snr_fmin':,
                   'snr_fmax':})

data_dict = generate_lisa_TDI_dict(param_dict)
generate_lisa_injections(data_dict, param_dict, calculate_snr=True)