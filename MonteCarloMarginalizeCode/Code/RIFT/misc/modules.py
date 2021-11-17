
# modules.py: from rapid_pe (Caitlin ?)

import ast,sys,os
import numpy as np

def convert_section_args_to_dict(cfg,section):
    #in principle this can be done with a single line, but i want to include sanity checks. 
#    string_of_dict = json.dumps(dict(cfg.items(section)))
    if not cfg.has_section(section):
        return {}

    dictin = dict(cfg.items(section))

    for opt,arg in dictin.items():
#        print opt,arg
        if len(arg) > 1 and arg[0] == "[" and arg[-1] == "]":
            dictin[opt] = correct_list_string_formatting(arg)
        else:
            dictin[opt] = arg

    return dictin


def convert_dict_string_to_dict(input_dict_string):
    #in principle this can be done with a single line, but i want to include sanity checks. 
#    string_of_dict = json.dumps(dict(cfg.items(section)))

    dictin = ast.literal_eval(input_dict_string)

    for opt,arg in dictin.items():
#        print opt,arg
        if isinstance(arg,str) and arg[0] == "[" and arg[-1] == "]":
            dictin[opt] = correct_list_string_formatting(arg)

    return dictin

    
#convert cfg options directly to command line arguments              
def convert_cfg_section_to_cmd_line(cfg,cfg_section):
    cmds_dict = dict(cfg.items(cfg_section))
    cmd_line = ""
    for name,arg in cmds_dict.items():
        #check if it's meant to be a list and if yes convert it to format for ArgumentParser append option
        if len(arg) > 1 and arg[0] == "[" and arg[-1] == "]":
            arglist = ast.literal_eval(correct_list_string_formatting(arg))
            for subarg in arglist:
                cmd_line += " --"+name+"="+subarg
        else:
            cmd_line += " --"+name
            if len(arg) > 0:
                cmd_line +="="+arg
    return cmd_line

def convert_dict_to_cmd_line(cmds_dict):
    cmd_line = ""
    for name,arg in cmds_dict.iteritems():
        #check if it's meant to be a list and if yes convert it to format for ArgumentParser append option
        if len(arg) > 1 and arg[0] == "[" and arg[-1] == "]":
            arglist = ast.literal_eval(correct_list_string_formatting(arg))
            for subarg in arglist:
                cmd_line += " --"+name+"="+subarg
        else:
            cmd_line += " --"+name
            if len(arg) > 0:
                cmd_line +="="+arg
    return cmd_line


def convert_list_string_to_dict(list_string):
    output_dict = {}
    list_string = correct_list_string_formatting(list_string)
#    print "ls",list_string
    for subarg in ast.literal_eval(list_string):
#        print subarg
        tmp=subarg.split("=")
        output_dict[tmp[0]] = tmp[1]
    return output_dict

#make sure the format is ['arg1=opt1','arg2=opt2'], otherwise ast.literal eval wont know how to split it
#If the arg is a list, make sure the items wihtin the list are strings, otherwise each character will be split later 
def correct_list_string_formatting(list_string):
    output_str= ""
    list_string = list_string.replace("\n","")
    if not (len(list_string) > 1 and list_string[0] == "[" and list_string[-1] == "]"):
        #Check if theres just one arg=opt
        if list_string.count("=") == 1 and list_string.count("[") == 0  and list_string.count("]") == 0:
            if not "'" in list_string and not '"' in list_string:
                output_str = "['"+list_string+"']"
            else:
                output_str = "["+list_string+"]"
        else:
            print (len(list_string),list_string[0], list_string[-1:],list_string.split(),list_string.replace(" ","")[-1:])
            sys.exit("ERROR: incorrect format. The variables should be in the form '[arg1=opt1,arg2=opt2]', not "+list_string)
    else:
        #First, make sure the format is ['arg1=opt1','arg2=opt2'], otherwise ast.literal eval wont know how to split it
        arg2 = list_string
        if not '"' in arg2 and not "'" in arg2:
            arg2 = arg2.replace('[','["')
            arg2 = arg2.replace(',','","')
            arg2 = arg2.replace(']','"]')
        output_str = arg2
    return output_str
            
def get_html_str(plot_name_no_path):
    return '<img src="'+plot_name_no_path+'" style="float: left; width: 45%; margin-right: 1%; margin-bottom: 0.5em;">'
#    return '<img src="'+plot_name_no_path+'" style="float: left; width: 100%; margin-right: 1%; margin-bottom: 0.5em;">'

def html_newline():
    return '<p style="clear: both;">\n'

def check_switch_m1m2s1s2(intr_prm):
    if intr_prm["mass1"] < intr_prm["mass2"]:
        copy = intr_prm.copy()
        copy["mass2"] = intr_prm["mass1"]
        copy["mass1"] = intr_prm["mass2"]
        if "spin1z" in intr_prm:
            copy["spin2z"] = intr_prm["spin1z"]
            copy["spin1z"] = intr_prm["spin2z"]
        return copy
    return intr_prm
    

def get_mchirp_eta(m1,m2):
    mc = (((m1*m2)**(3./5.))/((m1+m2)**(1./5.)))
    eta = ((m1*m2)/((m1+m2)**2.))
    return mc,eta

def get_m1m2_from_mceta(Mc,eta):
    m1 = 0.5*Mc*eta**(-3./5.)*(1. + norm_sym_ratio(eta))
    m2 = 0.5*Mc*eta**(-3./5.)*(1. - norm_sym_ratio(eta))
    return m1,m2

def transform_s1zs2z_chi(m1, m2, s1z, s2z):
    return (m1 * s1z + m2 * s2z) / (m1 + m2)

def transform_s1zs2z_chi_eff_chi_a(mass1, mass2, spin1z, spin2z):
    #Copied from pycbc https://github.com/gwastro/pycbc/blob/master/pycbc/conversions.py         
    """ Returns the aligned mass-weighted spin difference from mass1, mass2,                     
    spin1z, and spin2z.                                                                          
    """
    chi_eff = (spin1z * mass1 + spin2z * mass2) / (mass1 + mass2)
    chi_a = (spin2z * mass2 - spin1z * mass1) / (mass2 + mass1)
    return chi_eff,chi_a

def transform_chi_eff_chi_a_s1zs2z(mass1, mass2, chi_eff, chi_a):
    """Returns spin1z.                                                                           
    """
    spin1z = (mass1 + mass2) / (2.0 * mass1) * (chi_eff - chi_a)
    spin2z = (mass1 + mass2) / (2.0 * mass2) * (chi_eff + chi_a)
    return spin1z,spin2z

def norm_sym_ratio(eta):
    assert np.all(eta <= 0.25)

    return np.sqrt(1 - 4. * eta)

def convert_injections_txt_to_objects(inj_file):
    output = []
    inj = np.genfromtxt(inj_file,unpack=True,names=True)
    #load the injections as a list of objects so they can easily be accessed in the same way
    # as SimInspiralTables.

    for ei,e in enumerate(inj["mass1"]):
        output.append(Injection(inj,ei))

    return output

def get_priors(param,val,prior_limits={}):

    islist = 1 if (isinstance(val,list) or isinstance(val,np.ndarray)) else 0

    weight = 1.0
    if param == "distance":
        min_dist_mpc = prior_limits["distance"][0] if "distance" in prior_limits else 0
        max_dist_mpc = prior_limits["distance"][1] if "distance" in prior_limits else 2000.0
        print ("Prior limits on distance",min_dist_mpc,max_dist_mpc)
        return val**2/(max_dist_mpc**3/3. - min_dist_mpc**3/3.)
    elif param == "inclination":
        return np.sin(val)/2.0
    elif param == "latitude":
        return np.cos(val)/2.0
    elif param == "phase" or param == "longitude" or param == "psi0":
        weight = 1.0/(2.0*np.pi)
    elif param == "polarization" or param == "phi0":
        weight = 1.0/(np.pi)
    elif param == "mchirp":
        minmc = 0.0
        maxmc=200.0
        if "mchirp" in prior_limits:
            minmc = prior_limits["mchirp"][0]
            maxmc = prior_limits["mchirp"][1]
        print ("Prior limits on mchirp",minmc,maxmc)
        weight = np.asarray([uniform_samp_prior(minmc,maxmc,e) for e in val]) if islist else uniform_samp_prior(minmc,maxmc,val)
        islist = 0
    elif param == "eta":
        weight = 4.0
    else:
        weight = 1.0

    return np.asarray([weight for e in val]) if islist else weight

def uniform_samp_prior(a, b, x):
    if x > a and x < b:
        return 1.0/(b-a)
    else:
        return 0

class Injection:
    def __init__(self,e,ei):
        self.mass1 = e["mass1"][ei]
        self.mass2 = e["mass2"][ei]
        self.spin1z = e["spin1z"][ei]
        self.spin2z = e["spin2z"][ei]
        self.latitude = e["latitude"][ei]
        self.longitude = e["longitude"][ei]
        self.distance = e["distance"][ei]
        self.inclination = e["inclination"][ei]
        self.psi0 = e["psi0"][ei]
        self.phi0 = e["phi0"][ei]
        if "geocent_end_time" in e:
            #WARNING: make sure the event_time saved in the txt file isn't missing the nanoseconds part
            self.geocent_end_time = e["geocent_end_time"][ei]
            self.geocent_end_time_ns = -1
        else:
            self.geocent_end_time = e["geocent_end_time_part1"][ei]
            self.geocent_end_time_ns = e["geocent_end_time_ns_part2"][ei]
        self.alpha6 = int(e["original_index"][ei])


def run_cmd(cmd,verbose=1):
    if verbose:
        print (cmd)
    exit_status = os.system(cmd)
    if exit_status != 0:
        print (cmd)
        sys.exit("ERROR: non zero exit status"+str(exit_status))


def construct_event_time_string(t_s,t_ns):
    if t_ns < 0:
        #if ns has been set to a value less than 0, it is because we shouldn't use it
        return str(t_s)
    t_s_s = str(t_s)
    t_ns_s = str(t_ns)
    while len(t_ns_s) < 9:
        t_ns_s += "0"
    t_str = t_s_s+"."+t_ns_s
    return t_str


