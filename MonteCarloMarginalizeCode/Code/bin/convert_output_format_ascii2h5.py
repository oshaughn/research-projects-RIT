#!/home/jacob.lange/CalMarg-patch2/venv-teobresumsgeneral/bin/python3

from pesummary.io import read
from optparse import OptionParser
optp = OptionParser()
optp.add_option("--output-file",default="posterior_samples.h5",type=str,help="Output samples in the h5 format.")
optp.add_option("--posterior-samples",default="extrinsic_posterior_samples.dat",type=str,help="Input samples in the ascii format.")
opts, args = optp.parse_args()
f = read(opts.posterior_samples)
f.write(file_format="pesummary", filename=opts.output_file, label="label")
