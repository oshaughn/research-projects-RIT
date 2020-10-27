#
# Monica Rizzo
#

import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lal
import numpy as np

def calc_lambda_from_m(m, eos_fam):
    if m<10**15:
       m=m*lal.MSUN_SI

    k2=lalsim.SimNeutronStarLoveNumberK2(m, eos_fam)
    r=lalsim.SimNeutronStarRadius(m, eos_fam)

    m=m*lal.G_SI/lal.C_SI**2
    lam=2./(3*lal.G_SI)*k2*r**5
    dimensionless_lam=lal.G_SI*lam*(1/m)**5

    return dimensionless_lam


def append_lambda_to_xml(file_name, eos_name,file_name_out=None):
    param_list=lalsimutils.xml_to_ChooseWaveformParams_array(file_name)
    
    from gwemlightcurves.KNModels import table

    eos,eos_fam=table.get_lalsim_eos(eos_name)

    print("Writing to xml:")
    print("Event  m1  m2  lambda1  lambda2")
    for i in np.arange(len(param_list)):
       m1=param_list[i].m1
       m2=param_list[i].m2
       param_list[i].lambda1=calc_lambda_from_m(m1,eos_fam)
       param_list[i].lambda2=calc_lambda_from_m(m2,eos_fam)
       print(i,"[",param_list[i].m1/lal.MSUN_SI,param_list[i].m2/lal.MSUN_SI,param_list[i].lambda1,param_list[i].lambda2,"]")


    if file_name_out is None:
        file_name_out = file_name.replace(".xml.gz","") + "_"+ eos_name
    lalsimutils.ChooseWaveformParams_array_to_xml(param_list,fname=file_name_out)

if __name__ == "__main__":
#   append_lambda_to_xml("overlap-grid.xml.gz", 'sly')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-in", default="overlap-grid.xml.gz", help="input XML")
    parser.add_argument("--xml-out", default=None, help="Target output xml. Default is add eos name afterwards")
    parser.add_argument("--use-eos", default='sly', help="Equation of state to determine lambdas for given mass ranges. If a file name, will load EOS from that file; otherwise, will look for that name in lalsimulation")
    opts=  parser.parse_args()

    append_lambda_to_xml(opts.xml_in,opts.use_eos, file_name_out=opts.xml_out)
    




