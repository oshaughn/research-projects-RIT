#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  util_AMRGrid.py  based on rapid_pe_compute_intrinsic_grid https://github.com/CaitlinRose/rapidpe/blob/master/bin/rapidpe_compute_intrinsic_grid
# Copyright (C) 2015 Chris Pankow
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Given a set of extrinsic evidence calculations on a given set of intrinsic parameters, refines the grid to do additional calculations.
"""

__author__ = "Chris Pankow <chris.pankow@ligo.org>"

import os
import sys
import glob
import json
import bisect
import re
from collections import defaultdict
from argparse import ArgumentParser
from copy import copy

import h5py
import numpy
from scipy.special import binom
from sklearn.neighbors import BallTree

from ligo.lw import utils, ligolw, lsctables # , ilwd
lsctables.use_in(ligolw.LIGOLWContentHandler)
from ligo.lw.utils import process

import lal
import lalsimulation
from RIFT.misc import amrlib
from RIFT import lalsimutils

remap_rpe2rift = {'m1':'mass1','m2':'mass2','s1z':'spin1z', 's2z':'spin2z'}
def translate_params(param):
    if param in remap_rpe2rift:
        return remap_rpe2rift[param]
    return param


def get_cr_from_grid(cells, weight, cr_thr=0.9, min_n=None, max_n=None,delta_logL_threshold=None):
    """
    Given a set of cells and the weight of that cell, calculate a N% CR including cells which contribute to that probability mass. If n is set, cr_thr is ignored and instead this many points are taken.
    """
    if cr_thr == 0.0:
        return numpy.empty((0,))

    # Arrange them all with their respective weight
    cell_sort = numpy.hstack( (weight[:,numpy.newaxis], cells) )

    # Sort and form the CDF
    cell_sort = cell_sort[cell_sort[:,0].argsort()]

    if delta_logL_threshold == None:
        cell_sort[:,0] = cell_sort[:,0].cumsum()
        cell_sort[:,0] /= cell_sort[-1,0]
    
        idx = cell_sort[:,0].searchsorted(1-cr_thr)
        n_select = cell_sort.shape[0] - idx
        if min_n is not None:
            n_select = max(n_select, min_n)
        if max_n is not None:
            n_select = min(n_select, max_n)
        idx = cell_sort.shape[0] - n_select

        return cell_sort[idx:,1:]
    else:
        cell_sort[:,0] /= cell_sort[-1,0]  #normalize out peak
        cell_sort[:,0] = numpy.log(cell_sort[:,0] + 1e-40)  # go to log.  All will be negative
        idx = cell_sort[:,0].searchsorted(-delta_logL_threshold)
        n_select = cell_sort.shape[0] - idx
        if min_n is not None:
            n_select = max(n_select, min_n)
        if max_n is not None:
            n_select = min(n_select, max_n)
        idx = cell_sort.shape[0] - n_select

        return cell_sort[idx,1:]

def determine_region(pt, pts, ovrlp, ovrlp_thresh, expand_prms={}):
    """
    Given a point (pt) in a set of points (pts), with a function value at those points (ovrlp), return a rectangular hull such that the function exceeds the value ovrlp_thresh.
    """
    sidx = bisect.bisect(ovrlp, ovrlp_thresh)
    print("Found %d neighbors with overlap >= %f" % (len(ovrlp[sidx:]), ovrlp_thresh))
    print("HERE",ovrlp_thresh,ovrlp[sidx:],pts[sidx:],pt)
    

    cell = amrlib.Cell.make_cell_from_boundaries(pt, pts[sidx:])
    for k, lim in expand_prms.items():
        cell._bounds = numpy.vstack((cell._bounds, lim))
        # FIXME: Need to do center?
    #Force eta to be bounded at 0.25
    if opts.distance_coordinates == "mchirp_eta":
        if cell._bounds[1][1] > 0.25:
            cell._bounds[1][1] = 0.25
        if cell._bounds[1][0] >= 0.25:
            cell._bounds[1][0] = 0.25-1e-7
        if cell._bounds[1][0] < 0.01:
            #As of 201905 SEOBNRv4 cant handle mass ratios > 100, so don't let eta frop below 0.01
            cell._bounds[1][0] = 0.01
    #If ranges for Mc are specified, require bins to be in this range
    if opts.distance_coordinates == "mchirp_eta" and (opts.mc_min is not None or opts.mc_max is not None):
        mc_min = -1 if opts.mc_min is None else opts.mc_min
        mc_max = 99999999.0 if opts.mc_max is None else opts.mc_max
        if cell._bounds[0][0] < mc_min:
            cell._bounds[0][0] = mc_min
        if cell._bounds[0][1] > mc_max:
            cell._bounds[0][1] = mc_max
        if cell._bounds[0][1] < mc_min or cell._bounds[0][0] > mc_max:
            sys.exit("ERROR: All grid points are outside the specified mc-range. Are you sure you want to set --mc-min or mc-max? You probably don't.")

    return cell, sidx

def find_olap_index(tree, intr_prms, exact=True, **kwargs):
    """
    Given an object that can retrieve distance via a 'query' function (e.g. KDTree or BallTree), find the index of a point closest to the input point. Note that kwargs is used to get the current known values of the event. E.g.
    intr_prms = {'mass1': 1.4, 'mass2': 1.35}
    find_olap_index(tree, **intr_prms)
    """
    pt = numpy.array([kwargs[k] for k in intr_prms])

    # FIXME: Replace with standard function
    dist, m_idx = tree.query(numpy.atleast_2d(pt), k=1)
    dist, m_idx = dist[0][0], int(m_idx[0][0])

    # FIXME: There's still some tolerance from floating point conversions
    if exact and dist > 0.000001:
        exit("Could not find template in bank, closest pt was %f away" % dist)
    return m_idx, pt, dist

def write_to_xml_new(cells, intr_prms, pin_prms={}, fvals=None, fname=None, verbose=False):
    """
    Write a set of cells, with dimensions corresponding to intr_prms to an XML file as sim_inspiral rows.
    Uses RIFT-compatible syntax
    """
    P_list = []
    # Assume everyhing in intrinsic grid, no pin_prms
    indx_lookup={}
    namelist = []
    if ('mass1' in intr_prms and 'mass2' in intr_prms):
        indx_lookup['m1'] = intr_prms.index('mass1')
        indx_lookup['m2'] = intr_prms.index('mass2')
        namelist = ['m1','m2']
    elif 'mchirp' in intr_prms and 'delta' in intr_prms:
        indx_lookup['mc'] = intr_prms.index('mchirp')
        indx_lookup['delta_mc'] = intr_prms.index('delta')
        namelist = ['mc','delta_mc']
    else:
        indx_lookup['mc'] = intr_prms.index('mchirp')
        indx_lookup['eta'] = intr_prms.index('eta')
        namelist = ['mc','eta']
    if 'spin1z' in intr_prms:
        indx_lookup['s1z'] = intr_prms.index('spin1z')
        indx_lookup['s2z'] = intr_prms.index('spin2z')
        namelist += ['s1z','s2z']
    for indx in numpy.arange(len(cells)):
        P = lalsimutils.ChooseWaveformParams()
        for name in namelist:
            fac_correct = 1
            if name in ['mc', 'm1', 'm2']:
                fac_correct =lal.MSUN_SI
#            setattr(P, name, fac_correct*cells[indx]._center[indx_lookup[name]])
            if hasattr(P, name):
                setattr(P, name, fac_correct*cells[indx]._center[indx_lookup[name]])
            else:
                P.assign_param(name, fac_correct*cells[indx]._center[indx_lookup[name]])
        P_list.append(P)

    fname_out = fname
    if fname is None:
        fname_out="my_grid.xml"
    lalsimutils.ChooseWaveformParams_array_to_xml(P_list,fname_out)


def write_to_xml(cells, intr_prms, pin_prms={}, fvals=None, fname=None, verbose=False):
    """
    Write a set of cells, with dimensions corresponding to intr_prms to an XML file as sim_inspiral rows.
    Note this is NOT COMPATIBLE IN GENERAL with RIFT results in general
    """
    xmldoc = ligolw.Document()
    xmldoc.appendChild(ligolw.LIGO_LW())
    procrow = process.append_process(xmldoc, program=sys.argv[0])
    procid = procrow.process_id
    process.append_process_params(xmldoc, procrow, process.process_params_from_dict(opts.__dict__))

    rows = ["simulation_id", "process_id", "numrel_data"]
    # Override eff_lambda to with psi0, its shoehorn column
    if "eff_lambda" in intr_prms:
        intr_prms[intr_prms.index("eff_lambda")] = "psi0"
    if "deff_lambda" in intr_prms:
        intr_prms[intr_prms.index("deff_lambda")] = "psi3"
    intr_params_revised = [ translate_params(param) for param in intr_prms]
    rows += list(intr_params_revised) # relabel parameters for writing files out
    pin_params_revised = [ translate_params(param) for param in pin_prms]
    rows += list(pin_params_revised)
    if fvals is not None:
        rows.append("alpha1")
    sim_insp_tbl = lsctables.New(lsctables.SimInspiralTable, list(set(rows)))  # remove overlaps/duplicates !
    for itr, intr_prm in enumerate(cells):
        sim_insp = sim_insp_tbl.RowType()
        # FIXME: Need better IDs
        sim_insp.numrel_data = "INTR_SET_%d" % itr
        sim_insp.simulation_id = itr #ilwd.ilwdchar("sim_inspiral:sim_inspiral_id:%d" % itr)
        sim_insp.process_id = procid
        if fvals:
            sim_insp.alpha1 = fvals[itr]
        for p, v in zip(intr_prms, intr_prm._center):
            setattr(sim_insp, p, v)
        for p, v in pin_prms.items():
            setattr(sim_insp, p, v)
        sim_insp_tbl.append(sim_insp)

    xmldoc.childNodes[0].appendChild(sim_insp_tbl)
    if fname is None:
        channel_name = ["H=H", "L=L"]
        ifos = "".join([o.split("=")[0][0] for o in channel_name])
        #start = int(event_time)
        start = 0
        fname = "%s-MASS_POINTS-%d-1.xml.gz" % (ifos, start)
    utils.write_filename(xmldoc, fname, compress="gz", verbose=verbose)

def get_evidence_grid(points, res_pts, intr_prms, exact=False):
    """
    Associate the "z-axis" value (evidence, overlap, etc...) res_pts with its
    corresponding point in the template bank (points). If exact is True, then
    the poit must exactly match the point in the bank.
    """
    grid_tree = BallTree(selected)
    grid_idx = []
    # Reorder the grid points to match their weight indices
    for res in res_pts:
        dist, idx = grid_tree.query(numpy.atleast_2d(res), k=1)
        # Stupid floating point inexactitude...
        #print res, selected[idx[0][0]]
        #assert numpy.allclose(res, selected[idx[0][0]])
        grid_idx.append(idx[0][0])
    return points[grid_idx]

#
# Plotting utilities
#
def plot_grid_cells(cells, color, axis1=0, axis2=1):
    from matplotlib.patches import Rectangle
    from matplotlib import pyplot
    ax = pyplot.gca()
    for cell in cells:
        ext1 = cell._bounds[axis1][1] - cell._bounds[axis1][0]
        ext2 = cell._bounds[axis2][1] - cell._bounds[axis2][0]

        ax.add_patch(Rectangle((cell._bounds[axis1][0], cell._bounds[axis2][0]), ext1, ext2, edgecolor = color, facecolor='none'))

argp = ArgumentParser()
argp.add_argument("--fname",default=None,help="Overrides result-file, for RIFT compatibility")
argp.add_argument("--fname-output-samples",default=None,help="Overrides output-xml-file-name, for RIFT compatibility")
argp.add_argument("--fname-output-integral",default=None,help="Does nothing, for RIFT compatibility")
argp.add_argument("-d", "--distance-coordinates", default=None, help="Coordinate system in which to calculate 'closeness'. Default is tau0_tau3.")
argp.add_argument("-n", "--no-exact-match", action="store_true", help="Loosen criteria that the input intrinsic point must be a member of the input template bank.")
argp.add_argument("-v", "--verbose", action='store_true', help="Be verbose.")
argp.add_argument("--n-max-output", type=int, help="HARD limit on output size, imposed at end, to throttle. Selected AT RANDOM from refinement.")

# FIXME: These two probably should only be for the initial set up. While it
# could work, in theory, for refinement, the procedure would be a bit more
# tricky.
# FIXME: This could be a single value (lock a point in) or a range (adapt across
# this is range). No argument given implies use entire known range (if
# available).
argp.add_argument("-i", "--intrinsic-param", action="append", help="Adapt in this intrinsic parameter. If a pre-existing value is known (e.g. a search template was identified), specify this parameter as -i mass1=1.4 . This will indicate to the program to choose grid points which are commensurate with this value. Note that the mass1, mass2 names are hardcoded by fiat because they are used in the template bank files")
argp.add_argument("-p", "--pin-param", action="append", help="Pin the parameter to this value in the template bank. If spin is not defined, spin1z,spin2z will be pinned to 0. ")
#argp.add_argument( "--fmin-template",default=15.0, help="Min template frequency. Used in some mass transforms.") #Not implemented

grid_section = argp.add_argument_group("initial gridding options", "Options for setting up the initial grid.")
grid_section.add_argument("--setup", help="Set up the initial grid based on template bank overlaps. The new grid will be saved to this argument, e.g. --setup grid will produce a grid.hdf file.")
grid_section.add_argument("--output-xml-file-name",default="", help="Set the name of the output xml file. The default name is HL-MASS_POINTS_LEVEL_x-0-1.xml.gz where x is the  level")
grid_section.add_argument("-t", "--tmplt-bank", help="XML file with template bank.")
grid_section.add_argument("-O", "--use-overlap", action="append",help="Use overlap information to define 'closeness'. If a list of files is given, the script will find the file with the closest template, and select nearby templates only from that file.")
grid_section.add_argument("-T", "--overlap-threshold", default=0.9,type=float, help="Threshold on overlap value.")
grid_section.add_argument("--lnL-threshold", default=None,type=float, help="Threshold on difference betwene lnLmax and lnL for refinement. IF USED, OVERRIDES THE OTHER CHOICE.  Suggested value of 6 to 8")
grid_section.add_argument("-s", "--points-per-side", type=int, default=10, help="Number of points per side, default is 10.")
grid_section.add_argument("-I", "--initial-region", action="append", help="Override the initial region with a custom specification. Specify multiple times like, -I mass1=1.0,2.0 -I mass2=1.0,1.5")
grid_section.add_argument("-D", "--deactivate", action="store_true", help="Deactivate cells initially which have no template within them.")
grid_section.add_argument("-P", "--prerefine", help="Refine this initial grid based on overlap values.")
grid_section.add_argument("--mc-min", type=float, default=None, help="Restrict chirp mass grid points to be > mc-min. This is used when generating pp-plots, so that recovered Mc isn't outside of injected prior range.It should not be used otherwise.")
grid_section.add_argument("--mc-max", type=float, default=None, help="Restrict chirp mass grid points to be < mc-max. This is used when generating pp-plots, so that recovered Mc isn't outside of injected prior range.It should not be used otherwise.")
refine_section = argp.add_argument_group("refine options", "Options for refining a pre-existing grid.")
refine_section.add_argument("--refine", help="Refine a prexisting grid. Pass this option the grid points from previous levels (or the --setup) option.")
refine_section.add_argument("-r", "--result-file", help="Input XML file containing newest result to refine.")
refine_section.add_argument("-M", "--max-n-points",type=int, help="Refine *at most* this many points, can override confidence region thresholds.")
refine_section.add_argument("-m", "--min-n-points", type=int, help="Refine *at least* this many points, can override confidence region thresholds.")


opts = argp.parse_args()

if not (opts.setup or opts.refine or opts.prerefine):
    exit("Either --setup or --refine or --prerefine must be chosen")

# RIFT compatibility
if opts.fname:
    opts.result_file = opts.fname
if opts.fname_output_samples:
    opts.output_xml_file_name = opts.fname_output_samples

def parse_param(popts):
    """
    Parse out the specification of the intrinsic space. Examples:
    >>> parse_param(["mass1=1.4", "mass2", "spin1z=-1.0,10"])
    {'mass1': 1.4, 'mass2': None, 'spin1z': (-1.0, 10.0)}
    """
    if popts is None:
        return {}, {}
    intr_prms, expand_prms = {}, {}
    for popt in popts:
        popt = popt.split("=")
        if len(popt) == 1:
            # Implicit expand in full parameter space -- not yet completely
            # implemented
            intr_prms[popt[0]] = None
        elif len(popt) == 2:
            popt[1] = popt[1].split(",")
            if len(popt[1]) == 1:
                # Fix intrinsic point
                intr_prms[popt[0]] = float(popt[1][0])
            else:
                expand_prms[popt[0]] = tuple(map(float, popt[1]))
    return intr_prms, expand_prms


# Hopefully the point is already present and we can just get it, otherwise it
# could incur an overlap calculation, or suffer from the effects of being close
# only in Euclidean terms

intr_prms, expand_prms = parse_param(opts.intrinsic_param)
pin_prms, _ = parse_param(opts.pin_param)
intr_pt = numpy.array([intr_prms[k] for k in sorted(intr_prms)])
# This keeps the list of parameters consistent across runs
intr_prms = sorted(intr_prms.keys())

#If spin 1 and 2 are not specified, they are pinned. This means the spin columns still appear in the output grid.
spin_transform=None
if not "s1z" in intr_prms or not "s2z" in intr_prms:
    if not "s1z" in intr_prms and not "s2z" in intr_prms:
        if not "s1z" in pin_prms:
            pin_prms["s1z"] = 0.0
        if not "s2z" in pin_prms:
            pin_prms["s2z"] = 0.0
    else:
        sys.exit("spin1z or spin2z is specified but not the other spin. compute intrinsic grid is not setup to search just one")
else:
    if opts.distance_coordinates == "mu1_mu2_q_s2z":
        spin_transform = opts.distance_coordinates
    else:
        spin_transform = "chi_z"

#
# Step 2: Set up metric space
#
# If asked, retrieve bank overlap
# You need to use the overlap if generating the inital grid, there is not other option
# Also, if inital_region = None, it wont be used anywhere
ovrlp = []
if opts.use_overlap is not None:
    # Transform and repack initial point                                                                                                 
    intr_pt = amrlib.apply_transform(intr_pt[numpy.newaxis,:], intr_prms, opts.distance_coordinates,spin_transform)[0]
    intr_pt = dict(zip(intr_prms, intr_pt))

    #Check if there are many overlap files
    overlap_filename = ""
    if len(opts.use_overlap) > 1:
        #If yes, loop over each one, get the closest in each, and 
        dists = []
        for hdf_filename in opts.use_overlap:
            print (hdf_filename)
            h5file = h5py.File(hdf_filename, "r")
            wfrm_fam = list(h5file.keys())[0]
            odata = h5file[wfrm_fam]
            ovrlp = odata["overlaps"]
            
            #This only needs to be done when generating the initial grid, using overlaps, if overlap file doesn't have all the necessary information
            if opts.tmplt_bank is not None:
                sys.exit("ERROR: Not setup to handle reading stuff from template bank when multiple overlap files provided. You shouldn't need to provide the template bank at all at this point, all the info you need is probably in the overlap file. The template bank option is only kept for backwards compatibility")

            pts = numpy.array([odata[a] for a in intr_prms]).T
            pts = amrlib.apply_transform(pts, intr_prms, opts.distance_coordinates,spin_transform)
            tree = BallTree(pts[:ovrlp.shape[0]])
                
            #cant require exact match, because don't know if this is the right file yet
            unused_idx, unused_pt,dist = find_olap_index(tree, intr_prms, False, **intr_pt)
            dists.append(dist)
            
        #get index of filename with min dist
        min_olap_file_index = numpy.where(dists == min(dists))[0][0]
        overlap_filename = opts.use_overlap[min_olap_file_index]
        print ("File with closest template",overlap_filename)

    else:
        overlap_filename = opts.use_overlap[0]


    h5file = h5py.File(overlap_filename, "r")

    # FIXME:
    #wfrm_fam = args.waveform_type
    # Just get the first one
    wfrm_fam = list(h5file.keys())[0]

    odata = h5file[wfrm_fam]
    ovrlp = odata["overlaps"]
    if opts.verbose:
        print("Using overlap data from %s" % wfrm_fam)

    #This only needs to be done when generating the initial grid, using overlaps, if overlap file doesn't have all the necessary information
    tmplt_bank = []
    if opts.tmplt_bank is not None:
        xmldoc_tmplt_bank = utils.load_filename(opts.tmplt_bank, contenthandler=ligolw.LIGOLWContentHandler)
        tmplt_bank = lsctables.SnglInspiralTable.get_table(xmldoc_tmplt_bank)


    if ovrlp.shape[1] != len(tmplt_bank):
        pts = numpy.array([odata[a] for a in intr_prms]).T
    else:
        # NOTE: We use the template bank here because the overlap results might not
        # have all the intrinsic information stored (e.g.: no spins, even though the
        # bank is aligned-spin).
        # FIXME: This is an oversight in the overlap calculator which was rectified
        # but this remains for legacy banks
        #FIXME: this is the only place where template bank would possibly be used.
        pts = numpy.array([tuple(getattr(t, a) for a in intr_prms) for t in tmplt_bank])

    pts = amrlib.apply_transform(pts, intr_prms, opts.distance_coordinates,spin_transform)

    # FIXME: Can probably be moved to point index identification function -- it's
    # not used again
    # The slicing here is a slight hack to work around uberbank overlaps where the
    # overlap matrix is non square. This can be slightly dangerous because it
    # assumes the first N points are from the bank in question. That's okay for now
    # but we're getting increasingly complex in how we do construction, so we should
    # be more sophisticated by matching template IDs instead.
    tree = BallTree(pts[:ovrlp.shape[0]])

    #
    # Step 3: Get the row of the overlap matrix to work with
    #
    m_idx, pt, unused_dist_var = find_olap_index(tree, intr_prms, not opts.no_exact_match, **intr_pt)
    print("HERE2: ",pt,m_idx,unused_dist_var)
#    m_idx = 266

    #
    # Rearrange data to correspond to input point
    #
    sort_order = ovrlp[m_idx].argsort()
    ovrlp = numpy.array(ovrlp[m_idx])[sort_order]

    # DANGEROUS: This assumes the (template bank) points are the same order as the
    # overlaps. While we've taken every precaution to ensure this is true, it may
    # not always be.
    pts = pts[sort_order]
    m_idx = sort_order[m_idx]

#
# Step 1: Retrieve results from previous integration 
# Step 2 is before step 1 becuase I moved around the code to put all the overlap stuff togther
#

# Expanded parameters are now part of the intrinsic set
intr_prms = list(intr_prms) + list(expand_prms.keys())

# Gather any results we may want to use -- this is either the evidence values
# we've calculated, or overlaps of points we've looked at
results = []
if opts.result_file:
    # Default code path: assume XML formatted information, and we point to a glob of result files from ILE
    # If instead we have RIFT-style output files, create a temporary XML file with the correct format
    use_composite = False
    if '.composite' in opts.result_file or '.net' in opts.result_file:
        use_composite=True
        tempfile = "temp_convert_file"
        cmd = "convert_output_format_allnet2xml --fname {} --fname-output-samples temp_convert_file ".format(opts.result_file)
        os.system(cmd)
        opts.result_file = "temp_convert_file.xml.gz"

    for arg in glob.glob(opts.result_file):
        # FIXME: Bad hardcode
        # This is here because I'm too lazy to figure out the glob syntax to
        # exclude the samples files which would be both double counting and
        # slow to load because of their potential size
        if "samples" in arg:
            continue
        xmldoc = utils.load_filename(arg, contenthandler=ligolw.LIGOLWContentHandler)

        # FIXME: The template banks we make are sim inspirals, we should
        # revisit this decision -- it isn't really helping anything
        if opts.prerefine or use_composite:
            results.extend(lsctables.SimInspiralTable.get_table(xmldoc))
        else:
            results.extend(lsctables.SnglInspiralTable.get_table(xmldoc))

    # original code only works if coordinates are in the XML table! 'eta' is not a field in the xml file
    if 'mass1' in intr_prms:
        res_pts = numpy.array([tuple(getattr(t, a) for a in intr_prms) for t in results])
    elif ('mchirp' in intr_prms):
        res_pts = numpy.zeros( (len(results), len(intr_prms)))
        mass_pts = numpy.array([tuple(getattr(t, a) for a in ['mass1','mass2']) for t in results])
        def blank_entry(name):
            if name in ['delta', 'eta']:
                return 'eta'
            return name
        intr_prms_reduced = list(map(blank_entry, intr_prms))
        res_pts = numpy.array([tuple(getattr(t, a) for a in intr_prms_reduced) for t in results]) # can fill spin components if present
        # now overwrite eta values (all filled with 0 since no attr) with correct value
        if 'eta' in intr_prms:
            eta_indx = intr_prms.index('eta')
            res_pts[:,eta_indx] = lalsimutils.symRatio(mass_pts[:,0],mass_pts[:,1])
        if 'delta' in intr_prms:
            delta_indx = intr_prms.index('delta')
            res_pts[:,delta_indx] = (mass_pts[:,0] - mass_pts[:,1])/(mass_pts[:,0]+mass_pts[:,1]) # hardcode definition, easy
        # res_pts = numpy.zeros((len(intr_prms),len(results)))
        # indx_mc = intr_prms.index['mchirp']
        # indx_eta = intr_prms.index['eta']
        # indx_s1z = intr_prms.index['s1z']
        # indx_s2z = intr_prms.index['s2z']
        # rng = numpy.arange(len(results))
        # # as in lalsimutils xml_to_ChooseWaveformParams_array
        # Ps = [ChooseWaveformParams(deltaT=deltaT, fref=fref, lambda1=lambda1,
        #     lambda2=lambda2, waveFlags=waveFlags, nonGRparams=nonGRparams,                                   
        #     detector=detector, deltaF=deltaF, fmax=fmax) for i in rng]
        # [Ps[i].copy_lsctables_sim_inspiral(sim_insp[i]) for i in rng]
        # # could finish this

    res_pts = amrlib.apply_transform(res_pts, intr_prms, opts.distance_coordinates,spin_transform)

    # In the prerefine case, the "result" is the overlap values, which we use as
    # a surrogate for the true evidence value.
    if opts.prerefine:
        # We only want toe overlap values
        # FIXME: this needs to be done in a more consistent way
        results = numpy.array([res.alpha1 for res in results])
    elif use_composite:
        # using composite file information
        # the composite field for lnL is *alpha3*
        maxlnevid = numpy.max([s.alpha3 for s in results])
        total_evid = numpy.exp([s.alpha3 - maxlnevid for s in results]).sum()
        for res in results:
            res.alpha3 = numpy.exp(res.alpha3 - maxlnevid)/total_evid

        # FIXME: this needs to be done in a more consistent way
        results = numpy.array([res.alpha3 for res in results])
    else:
        # Normalize
        # We're gathering the evidence values. We normalize here so as to avoid
        # overflows later on
        # FIXME: If we have more than 1 copies -- This is tricky because we need
        # to pare down the duplicate sngl rows too
        maxlnevid = numpy.max([s.snr for s in results])
        total_evid = numpy.exp([s.snr - maxlnevid for s in results]).sum()
        for res in results:
            res.snr = numpy.exp(res.snr - maxlnevid)/total_evid

        # FIXME: this needs to be done in a more consistent way
        results = numpy.array([res.snr for res in results])

#
# Build (or retrieve) the initial region
#
if opts.refine or opts.prerefine:
    init_region, region_labels = amrlib.load_init_region(opts.refine or opts.prerefine, get_labels=True)
else:
    ####### BEGIN INITIAL GRID CODE #########
    if opts.initial_region is None:
        #This is the only time anything from the overlap file is used anywhere
        init_region, idx = determine_region(pt, pts, ovrlp, opts.overlap_threshold, expand_prms)
        region_labels = intr_prms
#        print "init trgion",len(pts[idx:])
        # FIXME: To be reimplemented in a different way
        #if opts.expand_param is not None:
            #expand_param(init_region, opts.expand_param)
    else:
        # Override initial region -- use with care
        _, init_region = parse_param(opts.initial_region)
        region_labels = list(init_region.keys())
        init_region = amrlib.Cell(numpy.vstack(init_region[k] for k in region_labels))

    # TODO: Alternatively, check density of points in the region to determine
    # the points to a side
    grid, spacing = amrlib.create_regular_grid_from_cell(init_region, side_pts=opts.points_per_side / 2, return_cells=True)
#    print "grid",len(grid)
    # "Deactivate" cells not close to template points
    # FIXME: This gets more and more dangerous in higher dimensions
    # FIXME: Move to function
    tree = BallTree(grid)
    if opts.deactivate:
        get_idx = set()
        for pt in pts[idx:]:
            get_idx.add(tree.query(numpy.atleast_2d(pt), k=1, return_distance=False)[0][0])
        selected = grid[numpy.array(list(get_idx))]
    else:
        selected = grid
#    print "selected",len(selected)
#    sys.exit()

# Make sure all our dimensions line up
# FIXME: We just need to be consistent from the beginning
reindex = numpy.array([list(region_labels).index(l) for l in intr_prms])
intr_prms = list(region_labels)
if opts.refine or opts.prerefine:
    res_pts = res_pts[:,reindex]

extent_str = " ".join("(%f, %f)" % bnd for bnd in map(tuple, init_region._bounds))
center_str = " ".join(map(str, init_region._center))
label_str = ", ".join(region_labels)
print("Initial region (" + label_str + ") has center " + center_str + " and extent " + extent_str)

#### BEGIN REFINEMENT OF RESULTS #########

if opts.result_file is not None:
    (prev_cells, spacing), level, _ = amrlib.load_grid_level(opts.refine or opts.prerefine, -1, True)

    selected = numpy.array([c._center for c in prev_cells])
    selected = amrlib.apply_transform(selected, intr_prms, opts.distance_coordinates,spin_transform)

    selected = get_evidence_grid(selected, res_pts, intr_prms)

    if opts.verbose:
        print("Loaded %d result points" % len(selected))

    if opts.refine:
        # FIXME: We use overlap threshold as a proxy for confidence level
        selected = get_cr_from_grid(selected, results, cr_thr=opts.overlap_threshold, min_n=opts.min_n_points, max_n=opts.max_n_points, delta_logL_threshold=opts.lnL_threshold)
        if not(opts.lnL_threshold):
            print("Selected %d cells from %3.2f%% confidence region" % (len(selected), opts.overlap_threshold*100))
        else:
            print("Selected {} cells from region within {} of max lnL ".format(len(selected), opts.lnL_threshold))

if opts.prerefine:
    print("Performing refinement for points with overlap > %1.3f" % opts.overlap_threshold)
    pt_select = results > opts.overlap_threshold
    selected = selected[pt_select]
    results = results[pt_select]
    grid, spacing = amrlib.refine_regular_grid(selected, spacing, return_cntr=True)

else:
#    print "selected",len(selected)
    grid, spacing = amrlib.refine_regular_grid(selected, spacing, return_cntr=opts.setup)
#    print "refine grid",len(grid)

print("%d cells after refinement" % len(grid))
grid = amrlib.prune_duplicate_pts(grid, init_region._bounds, spacing)
#print "prune grid",len(grid)
#
# Clean up
#

grid = numpy.array(grid)
bounds_mask = amrlib.check_grid(grid, intr_prms, opts.distance_coordinates)
grid = grid[bounds_mask]
print("%d cells after bounds checking" % len(grid))

if len(grid) == 0:
    exit("All cells would be removed by physical boundaries.")

# Convert back to initial coordinate system.  Not needed if grid is *already* in this system?
if not('mchirp' in intr_prms):
    grid = amrlib.apply_inv_transform(grid, intr_prms, opts.distance_coordinates,spin_transform)
#print "inv transform",grid

cells = amrlib.grid_to_cells(grid, spacing)
print("Selected %d cells for further analysis." % len(cells))
if not(opts.n_max_output is None):
    if len(cells) > opts.n_max_output:
        print("Imposing HARD LIMIT on output size of {}".format(opts.n_max_output))
#        indx_ok = numpy.random.choice(len(cells), size=opts.n_max_output,replace=False)
#        cells = cells[indx_ok]
        cells = numpy.random.choice(cells, size=opts.n_max_output,replace=False)

if opts.setup:
    hdf_filename = opts.setup+".hdf" if not ".hdf" in opts.setup else opts.setup
    grid_group = amrlib.init_grid_hdf(init_region, hdf_filename, opts.overlap_threshold, opts.distance_coordinates, intr_prms=intr_prms)
    level = amrlib.save_grid_cells_hdf(grid_group, cells, "mass1_mass2", intr_prms=intr_prms)
else:
    grp = amrlib.load_grid_level(opts.refine, None)
    level = amrlib.save_grid_cells_hdf(grp, cells, "mass1_mass2", intr_prms)


if opts.setup:
    fname = "HL-MASS_POINTS_LEVEL_0-0-1.xml.gz" if opts.output_xml_file_name == "" else opts.output_xml_file_name 
    write_to_xml_new(cells, intr_prms, pin_prms, None, fname, verbose=opts.verbose)
else:
    #m = re.search("LEVEL_(\d+)", opts.result_file)
    #if m is not None:
        #level = int(m.group(1)) + 1
        #fname = "HL-MASS_POINTS_LEVEL_%d-0-1.xml.gz" % level
    #else:
        #fname = "HL-MASS_POINTS_LEVEL_X-0-1.xml.gz"
    fname = "HL-MASS_POINTS_LEVEL_%d-0-1.xml.gz" % level if opts.output_xml_file_name == "" else opts.output_xml_file_name 
    write_to_xml_new(cells, intr_prms, pin_prms, None, fname, verbose=opts.verbose)
