#
# efficient_save_sklearn_gp.py
#
# GOAL
#   - sklearn gp objects I use always have a simple form
#   - this saves a json file holding *strings* describing the kernel, AND the X_train_ and y_train_ data used in the GP
#   - we *may* still  use a pickle for now, but it is *much* more efficient -- we're just saving the data we need
#   - this lets me change the data format to make it *portable*

import json

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

def gp2json(gp):
    """
    Converts gp object to (a) json object and (b) data for X_train_ and y_train
    """
    out_dict = {}
    out_dict['kernel'] = str(gp.kernel_)
    out_dict['kernel_params'] = {}
    dict_params = gp.kernel_.get_params()
    for name in dict_params:
        out_dict['kernel_params'][name] = str(dict_params[name])   # gives me ability to set more parameters in greater detail
    out_dict['y_train_std'] = str(gp._y_train_std)
    out_dict['y_train_mean'] = str(gp._y_train_mean)
    return [out_dict, gp.X_train_, gp.y_train_,gp.alpha_]

def export_gp_compact(fname_base,gp):
    fname_json = fname_base+".json"
    fname_dat_X = fname_base+"_X.dat"
    fname_dat_y = fname_base+"_y.dat"
    fname_dat_alpha = fname_base+"_alpha.dat"
    my_json, my_X,my_y, my_alpha = gp2json(gp) 
    np.savetxt(fname_dat_X,my_X)
    np.savetxt(fname_dat_y,my_y)
    np.savetxt(fname_dat_alpha,my_alpha)
    with open(fname_json,'w') as f:
        json.dump(my_json, f)
    return None


def load_gp(fname_base):
    kernel=None
    with open(fname_base+".json",'r') as f:
        my_json = json.load(f)
    my_X = np.loadtxt(fname_base+"_X.dat")
    my_y = np.loadtxt(fname_base+"_y.dat")
    my_alpha = np.loadtxt(fname_base+"_alpha.dat")
    dict_params = my_json['kernel_params']
    eval("kernel = "+my_json['kernel'])
    gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0)
    gp.kernel_ = kernel
    dict_params_eval = {}
    for name in dict_params:
        if not('length' in name   or 'constant' in name):
            continue
        if name =="k2__k2__length_scale":
            one_space = ' '.join(dict_params[name].split())
            dict_params_eval[name] = eval(one_space.replace(' ',','))
        else:
            dict_params_eval[name] = eval(dict_params[name])
    gp.kernel_.set_params(dict_params_eval)
    gp.X_train_ = my_X
    gp.y_train_ = my_y
    gp.alpha_ = my_alpha
    gp._y_train_std = float(my_json['y_train_std'])
    gp._y_train_mean = float(my_json['y_train_mean'])
    return gp

                            
