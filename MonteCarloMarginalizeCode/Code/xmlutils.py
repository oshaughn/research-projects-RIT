import types

import numpy

from glue.lal import LIGOTimeGPS
from glue.ligolw import ligolw, lsctables, table
from glue.ligolw.utils import process

CMAP = { "right_ascension": "longitude",
    "declination": "latitude",
    "inclination": "inclination",
    "polarization": "polarization",
    "t_ref": lambda r, t: r.set_time_geocent(LIGOTimeGPS(float(t))),
    "coa_phase": "coa_phase",
    "distance": "distance",
    "mass1": "mass1",
    "mass2": "mass2"
}

sim_valid_cols = ["process_id", "simulation_id", "longitude", "latitude", "polarization", "geocent_end_time", "geocent_end_time_ns", "coa_phase", "distance", "mass1", "mass2"]
sngl_valid_cols = ["process_id", "event_id", "snr"]
multi_valid_cols = ["process_id", "event_id", "snr"]

def append_samples_to_xmldoc(xmldoc, sampdict):
    try: 
        si_table = table.get_table(xmldoc, lsctables.SimInspiralTable.tableName)
        new_table = False
    # Warning: This will also get triggered if there is *more* than one table
    except ValueError:
        si_table = lsctables.New(lsctables.SimInspiralTable, sim_valid_cols)
        new_table = True
    
    keys = sampdict.keys()
    # Just in case the key/value pairs don't come out synchronized
    values = numpy.array([sampdict[k] for k in keys], object)
    
    # Flatten the keys
    keys = reduce(list.__add__, [list(i) if isinstance(i, tuple) else [i] for i in keys])

    # Get the process
    # FIXME: Assumed that only we have appended information
    procid = table.get_table(xmldoc, lsctables.ProcessTable.tableName)[-1].process_id
    
    # map the samples to sim inspiral rows
    for vrow in values.T:
        si_table.append(samples_to_siminsp_row(si_table, **dict(zip(keys, vrow.flatten()))))
        si_table[-1].process_id = procid

    if new_table:
        xmldoc.childNodes[0].appendChild(si_table)
    return xmldoc

def append_likelihood_result_to_xmldoc(xmldoc, loglikelihood, **cols):
    try: 
        si_table = table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)
        new_table = False
        # NOTE: MultiInspiralTable has no spin columns
        #si_table = table.get_table(xmldoc, lsctables.MultiInspiralTable.tableName)
    # Warning: This will also get triggered if there is *more* than one table
    except ValueError:
        si_table = lsctables.New(lsctables.SnglInspiralTable, sngl_valid_cols + cols.keys())
        new_table = True
        # NOTE: MultiInspiralTable has no spin columns
        #si_table = lsctables.New(lsctables.MultiInspiralTable, multi_valid_cols + cols.keys())

    # Get the process
    # FIXME: Assumed that only we have appended information
    procid = table.get_table(xmldoc, lsctables.ProcessTable.tableName)[-1].process_id
    
    # map the samples to sim inspiral rows
    si_table.append(likelihood_to_snglinsp_row(si_table, loglikelihood, **cols))
    si_table[-1].process_id = procid

    if new_table:
        xmldoc.childNodes[0].appendChild(si_table)

    return xmldoc

def samples_to_siminsp_row(table, colmap={}, **sampdict):
    row = table.RowType()
    row.simulation_id = table.get_next_id()
    #mapping = CMAP.update(colmap)
    for key, col in CMAP.iteritems():
        if isinstance(col, types.FunctionType):
            col(row, sampdict[key])
        else:
            setattr(row, col, sampdict[key])

    return row

def likelihood_to_snglinsp_row(table, loglikelihood, **cols):
    row = table.RowType()
    row.event_id = table.get_next_id()
    for col in cols:
            setattr(row, col, cols[col])
    row.snr = numpy.sqrt(2*numpy.abs(loglikelihood))

    return row

# TESTING
import sys
if __file__ == sys.argv[0]:
    CMAP[("mass1", "mass2")] = ("mass1", "mass2")
    ar = numpy.random.random((len(CMAP), 10))
    samp_dict = dict(zip(CMAP, ar))
    ar = samp_dict[("mass1", "mass2")]
    samp_dict[("mass1", "mass2")] = numpy.array([ar, ar])
    del CMAP[("mass1", "mass2")]
    CMAP["mass1"] = "mass1"
    CMAP["mass2"] = "mass2"
    
    xmldoc = ligolw.Document()
    xmldoc.appendChild(ligolw.LIGO_LW())
    process.register_to_xmldoc(xmldoc, sys.argv[0], {})

    append_samples_to_xmldoc(xmldoc, samp_dict)

    def gaussian(x, mu=0, std=1):
        return 1/numpy.sqrt(numpy.pi*2)/std * numpy.exp(-(x-mu)**2/2/std**2)

    import numpy
    m1m, m2m = 1.4, 1.5
    m1, m2 = numpy.random.random(2000).reshape(2,1000)*1.0+1.0
    loglikes = [gaussian(m1i, m1m)*gaussian(m2i, m2m) for m1i, m2i in zip(m1, m2)]
    #loglikelihood = - 7.5**2/2
    #append_likelihood_result_to_xmldoc(xmldoc, loglikelihood, **{"mass1": 1.4, "mass2": 1.4, "ifos": "H1,L1,V1"})
    for m1i, m2i, loglikelihood in zip(m1, m2, loglikes):
        append_likelihood_result_to_xmldoc(xmldoc, loglikelihood, **{"mass1": m1i, "mass2": m2i})

    from glue.ligolw import utils
    utils.write_filename(xmldoc, "iotest.xml.gz", gz=True)
