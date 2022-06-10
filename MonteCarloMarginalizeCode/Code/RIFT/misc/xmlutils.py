import types
import sqlite3
from collections import namedtuple

from functools import reduce

import numpy

from glue.lal import LIGOTimeGPS
from ligo.lw import ligolw, lsctables, table #, ilwd
#from glue.ligolw.utils import process

def assign_id(row, i):
    row.simulation_id = i # ilwd.ilwdchar("sim_inspiral_table:sim_inspiral:%d" % i)

def assign_time(row, t):
    setattr(row, "geocent_end_time",( int(t)))
    setattr(row, "geocent_end_time_ns",int( (t-int(t))*1e9 ) )

CMAP = { "right_ascension": "longitude",
    "longitude":"longitude",
    "latitude":"latitude",
    "declination": "latitude",
    "inclination": "inclination",
    "polarization": "polarization",
    "t_ref": assign_time,#r.set_time_geocent(LIGOTimeGPS(float(t))),
    "coa_phase": "coa_phase",
    "distance": "distance",
    "mass1": "mass1",
    "mass2": "mass2",
    # SHOEHORN ALERT
    "sample_n": assign_id,
    "alpha1":"alpha1",
    "alpha2":"alpha2",
    "alpha3":"alpha3",
    "alpha4":"alpha4",
    "loglikelihood": "alpha1",
    "joint_prior": "alpha2",
    "joint_s_prior": "alpha3",
    "eccentricity":"alpha4",
    "spin1x":"spin1x",
    "spin1y":"spin1y",
    "spin1z":"spin1z",
    "spin2x":"spin2x",
    "spin2y":"spin2y",
    "spin2z":"spin2z"
}

# FIXME: Find way to intersect given cols with valid cols when making table.
# Otherwise, we'll have to add them manually and ensure they all exist
sim_valid_cols = ["simulation_id", "inclination", "longitude", "latitude", "polarization", "geocent_end_time", "geocent_end_time_ns", "coa_phase", "distance", "mass1", "mass2", "alpha1", "alpha2", "alpha3", "alpha4", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z"]
sngl_valid_cols = [ "event_id", "snr", "tau0", "tau3"]
multi_valid_cols = ["process_id", "event_id", "snr"]

def append_samples_to_xmldoc(xmldoc, sampdict):
    try: 
        si_table = lsctables.SimInspiralTable.get_table(xmldoc)
        new_table = False
    # Warning: This will also get triggered if there is *more* than one table
    except ValueError:
        si_table = lsctables.New(lsctables.SimInspiralTable, sim_valid_cols)
        new_table = True
    
    keys = list(sampdict.keys())
    # Just in case the key/value pairs don't come out synchronized
    values = numpy.array([sampdict[k] for k in keys], object)
    
    # Flatten the keys
    import collections
    keys = reduce(list.__add__, [list(i) if isinstance(i, tuple) else [i] for i in keys])

    # Get the process
    # FIXME: Assumed that only we have appended information
    procid = lsctables.ProcessTable.get_table(xmldoc)[-1].process_id
    
    # map the samples to sim inspiral rows
    # NOTE :The list comprehension is to preserve the grouping of multiple 
    # parameters across the transpose operation. It's probably not necessary,
    # so if speed dictates, it can be reworked by flattening before arriving 
    # here
    for vrow in numpy.array(list(zip(*[vrow_sub.T for vrow_sub in values])), dtype=numpy.object):
        #si_table.append(samples_to_siminsp_row(si_table, **dict(zip(keys, vrow.flatten()))))
        vrow = reduce(list.__add__, [list(i) if isinstance(i, collections.Iterable) else [i] for i in vrow])
        si_table.append(samples_to_siminsp_row(si_table, **dict(list(zip(keys, vrow)))))
        si_table[-1].process_id = procid

    if new_table:
        xmldoc.childNodes[0].appendChild(si_table)
    return xmldoc

def append_likelihood_result_to_xmldoc(xmldoc, loglikelihood, neff=0, converged=False,**cols):
    try: 
        si_table = lsctables.SnglInspiralTable.get_table(xmldoc)
        new_table = False
        # NOTE: MultiInspiralTable has no spin columns
        #si_table = table.get_table(xmldoc, lsctables.MultiInspiralTable.tableName)
    # Warning: This will also get triggered if there is *more* than one table
    except ValueError:
        si_table = lsctables.New(lsctables.SnglInspiralTable, sngl_valid_cols + list(cols.keys()))
        new_table = True
        # NOTE: MultiInspiralTable has no spin columns
        #si_table = lsctables.New(lsctables.MultiInspiralTable, multi_valid_cols + cols.keys())

    # Get the process
    # FIXME: Assumed that only we have appended information
    procid = lsctables.ProcessTable.get_table(xmldoc)[-1].process_id
    
    # map the samples to sim inspiral rows
    si_table.append(likelihood_to_snglinsp_row(si_table, loglikelihood, neff, converged,**cols))
    si_table[-1].process_id = procid

    if new_table:
        xmldoc.childNodes[0].appendChild(si_table)

    return xmldoc

def samples_to_siminsp_row(table, colmap={}, **sampdict):
    row = table.RowType()
    row.simulation_id = table.get_next_id()
    for key, col in list(CMAP.items()):
        if key not in sampdict:
            continue
        if isinstance(col, types.FunctionType):
            col(row, sampdict[key])
        else:
            setattr(row, col, sampdict[key])

    return row

def likelihood_to_snglinsp_row(table, loglikelihood, neff=0, converged=False, **cols):
    row = table.RowType()
    row.event_id = table.get_next_id()
    for col in cols:
            setattr(row, col, cols[col])
    row.snr = loglikelihood
    row.tau0 = neff
    row.tau3 = int(converged)

    return row

def db_identify_param(db_fname, process_id, param):
    """
    Extract the event time for a given process ID. This may fail in the case that the event time was not given on the command line (rather in a pointer to a XML file)
    NOTE: This is definitely not the best way to do this.
    """

    cmd_prm = "--" + param.replace("_", "-")

    sql = """select value from process_params where process_id = "%s" and param = "%s" """ % (str(process_id), cmd_prm)

    try:
        connection = sqlite3.connect(db_fname)
        result = list(connection.execute(sql))[0][0]
    finally:
        connection.close()
    return result

def db_to_samples(db_fname, tbltype, cols):
    """
    Pull samples from db_fname and return object that resembles a row from an XML table.
    """
    if "geocent_end_time" in cols:
        cols.append("geocent_end_time_ns")

    # FIXME: Get columns from db
    #if cols is None:
        #colsspec = "*"
    #else:
    colsspec = ", ".join(cols)

    if tbltype == lsctables.SimInspiralTable:
        sql = """select %s from sim_inspiral""" % colsspec
    elif tbltype == lsctables.SnglInspiralTable:
        sql = """select %s from sngl_inspiral""" % colsspec
    else:
        raise ValueError("Don't know SQL for table %s" % tbltype.tableName)

    Sample = namedtuple("Sample", cols)
    samples = []

    try:
        connection = sqlite3.connect(db_fname)
        connection.row_factory = sqlite3.Row
        for row in connection.execute(sql):
            # FIXME: UGH!
            res = dict(list(zip(cols, row)))
            if "geocent_end_time" in list(res.keys()):
                res["geocent_end_time"] += res["geocent_end_time_ns"]*1e-9

            samples.append(Sample(**res))
    finally:
        connection.close()

    return samples

# TESTING
import sys
if __file__ == sys.argv[0]:
    import numpy

    # Not used yet
    del CMAP["int_var"]
    del CMAP["int_val"]
    del CMAP["sample_n"]

    # Reworked to resemble usage in pipeline
    del CMAP["mass1"]
    del CMAP["mass2"]
    CMAP[("mass1", "mass2")] = ("mass1", "mass2")
    ar = numpy.random.random((len(CMAP), 10))
    samp_dict = dict(list(zip(CMAP, ar)))
    ar = samp_dict[("mass1", "mass2")]
    samp_dict[("mass1", "mass2")] = numpy.array([ar, ar])
    del CMAP[("mass1", "mass2")]
    CMAP["mass1"] = "mass1"
    CMAP["mass2"] = "mass2"

    samp_dict["samp_n"] = numpy.array(list(range(0,10)))
    CMAP["sample_n"] = "sample_n"
    
    xmldoc = ligolw.Document()
    xmldoc.appendChild(ligolw.LIGO_LW())
    process.register_to_xmldoc(xmldoc, sys.argv[0], {})

    append_samples_to_xmldoc(xmldoc, samp_dict)

    def gaussian(x, mu=0, std=1):
        return 1/numpy.sqrt(numpy.pi*2)/std * numpy.exp(-(x-mu)**2/2/std**2)

    m1m, m2m = 1.4, 1.5
    m1, m2 = numpy.random.random(2000).reshape(2,1000)*1.0+1.0
    loglikes = [gaussian(m1i, m1m)*gaussian(m2i, m2m) for m1i, m2i in zip(m1, m2)]
    #loglikelihood = - 7.5**2/2
    #append_likelihood_result_to_xmldoc(xmldoc, loglikelihood, **{"mass1": 1.4, "mass2": 1.4, "ifos": "H1,L1,V1"})
    for m1i, m2i, loglikelihood in zip(m1, m2, loglikes):
        append_likelihood_result_to_xmldoc(xmldoc, loglikelihood, **{"mass1": m1i, "mass2": m2i})

    from ligo.lw import utils
    utils.write_filename(xmldoc, "iotest.xml.gz", gz=True)
