#! /usr/bin/env python
#
#  Probably borrowed from someone like Kipp, Leo, or Duncan M, can't track down the original source at the moment

import argparse
import json

from glue.ligolw import ilwd
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import table
from glue.ligolw import utils

import RIFT.lalsimutils as lalsimutils
import lal

parser = argparse.ArgumentParser()
parser.add_argument("--sim-xml",help="input file")
parser.add_argument("--event",type=int,default=0,help="input file")
parser.add_argument("--ifo", action='append',help="input file")
parser.add_argument("--output",default=None,type=str)
opts= parser.parse_args()


P_list = lalsimutils.xml_to_ChooseWaveformParams_array(str(opts.sim_xml))
print(len(P_list))
if len(P_list) < opts.event:
    print( " Event out of range; hard exit")
    import sys
    sys.exit(1)
P = P_list[opts.event]

class DefaultContentHandler(lsctables.ligolw.LIGOLWContentHandler):
    pass
lsctables.use_in(DefaultContentHandler)

def _empty_row(obj):
    """Create an empty sim_inspiral or sngl_inspiral row where the columns have
    default values of 0.0 for a float, 0 for an int, '' for a string. The ilwd
    columns have a default where the index is 0.
    """

    # check if sim_inspiral or sngl_inspiral
    if obj == lsctables.SimInspiral:
        row = lsctables.SimInspiral()
        cols = lsctables.SimInspiralTable.validcolumns
    else:
        row = lsctables.SnglInspiral()
        cols = lsctables.SnglInspiralTable.validcolumns

    # populate columns with default values
    for entry in cols.keys():
        if cols[entry] in ['real_4','real_8']:
            setattr(row,entry,0.)
        elif cols[entry] == 'int_4s':
            setattr(row,entry,0)
        elif cols[entry] == 'lstring':
            setattr(row,entry,'')
        elif entry == 'process_id':
            row.process_id = ilwd.ilwdchar("sim_inspiral:process_id:0")
        elif entry == 'simulation_id':
            row.simulation_id = ilwd.ilwdchar("sim_inspiral:simulation_id:0")
        elif entry == 'event_id':
            row.event_id = ilwd.ilwdchar("sngl_inspiral:event_id:0")
        else:
            raise ValueError("Column %s not recognized." %(entry) )

    return row


# Create sngl_inspiral table
sngl_table = lsctables.New(lsctables.SnglInspiralTable,
                            columns=lsctables.SnglInspiralTable.validcolumns)

outdoc = ligolw.Document()
outdoc.appendChild(ligolw.LIGO_LW())

# add sngl_inspiral table to output XML document
outdoc.childNodes[0].appendChild(sngl_table)

if not(opts.ifo):
    opts.ifo = ["H1","L1"]  # default

# Create one row
for indx in range(len(opts.ifo)):
    sngl = _empty_row(lsctables.SnglInspiral)
    # add column values
    # note NOT all columns can be popoulated: the key thing is the event time
    sngl.ifo = opts.ifo[indx]
    sngl.end_time = int(P.tref)
    sngl.end_time_ns = int(1e9*(P.tref-int(P.tref)))
    sngl.mass1 = P.m1/lal.MSUN_SI
    sngl.mass2 = P.m2/lal.MSUN_SI
    sngl.mtotal = sngl.mass1 + sngl.mass2
    sngl.mchirp = lalsimutils.mchirp(sngl.mass1,sngl.mass2)
    sngl.eta = lalsimutils.symRatio(sngl.mass1,sngl.mass2)
    sngl.coa_phase = 0.
    sngl.spin1z = P.s1z
    sngl.spin2z = P.s2z
    sngl.eff_distance = P.dist/(1e6*lal.PC_SI)
    sngl.snr = 20.  # made up, needed for some algorithms to work

    # add to table
    sngl_table.append(sngl)

# write the output XML file
output_file='coinc.xml'
if opts.output:
    output_file = opts.output
utils.write_filename(outdoc,output_file, gz=output_file.endswith('gz'))


