import sys
import lal
import lalsimulation as lalsim
import lalsimutils as lsu
import effectiveFisher as eff
import dag_utils
import numpy as np
import matplotlib.pyplot as plt
from time import clock
from functools import partial
from six.moves import range

import itertools
from glue.ligolw import utils, ligolw, lsctables, table
from glue.ligolw.utils import process

start = clock()
elapsed_time = lambda: clock()-start

pts_per_job = 10 # How many intrinsic points to pass to each condor job

def write_sngl_params(grid, proc_id):
    sngl_insp_table = lsctables.New(lsctables.SnglInspiralTable, ["mass1", "mass2", "event_id", "process_id"])
    sngl_insp_table.sync_next_id()
    for (m1, m2) in itertools.chain(*grid):
        sngl_insp = sngl_insp_table.RowType()
        sngl_insp.event_id = sngl_insp_table.get_next_id()
        #sngl_insp.mass1, sngl_insp.mass2 = m1/lal.LAL_MSUN_SI, m2/lal.LAL_MSUN_SI
        sngl_insp.mass1, sngl_insp.mass2 = m1, m2
        sngl_insp.process_id = proc_id
        sngl_insp_table.append(sngl_insp)

    return sngl_insp_table

# Setup signal and IP class
m1=1.5*lal.LAL_MSUN_SI
m2=1.35*lal.LAL_MSUN_SI
PSIG = lsu.ChooseWaveformParams(m1=m1, m2=m2, approx=lalsim.TaylorT1)
PTEST = PSIG.copy() # find deltaF for lower end of range we're looking in
PTEST.m1 *= 0.9
PTEST.m2 *= 0.9
PSIG.deltaF = lsu.findDeltaF(PTEST)
IP = lsu.Overlap(fLow=40., deltaF=PSIG.deltaF, psd=lal.LIGOIPsd)
PTMPLT = PSIG.copy()
hfSIG = lsu.norm_hoff(PSIG, IP)
McSIG = lsu.mchirp(m1, m2)
etaSIG = lsu.symRatio(m1, m2)
NMcs = 11
NEtas = 11
param_names = ['Mc', 'eta']

# Find appropriate parameter ranges
param_ranges = eff.find_effective_Fisher_region(PSIG, IP, 0.90, param_names,
        [[0.9*McSIG,1.1*McSIG],[0.05,0.25]])
print("Computing amibiguity function in the range:")
for i, param in enumerate(param_names):
    if param=='Mc' or param=='m1' or param=='m2': # rescale output by MSUN
        print("\t", param, ":", np.array(param_ranges[i])/lal.LAL_MSUN_SI,\
                "(Msun)")
    else:
        print("\t", param, ":", param_ranges[i])

elapsed = elapsed_time()
print("Range-finding took:", elapsed, "(s) for", len(param_names),"parameters\n")

# setup uniform parameter grid for effective Fisher
pts_per_dim = [NMcs, NEtas]
Mcpts, etapts = eff.make_regular_1d_grids(param_ranges, pts_per_dim)
etapts = map(lsu.sanitize_eta, etapts)
McMESH, etaMESH = eff.multi_dim_meshgrid(Mcpts, etapts)
McFLAT, etaFLAT = eff.multi_dim_flatgrid(Mcpts, etapts)
dMcMESH = McMESH - McSIG
detaMESH = etaMESH - etaSIG
dMcFLAT = McFLAT - McSIG
detaFLAT = etaFLAT - etaSIG
grid = eff.multi_dim_grid(Mcpts, etapts)

# Change units on Mc
dMcFLAT_MSUN = dMcFLAT / lal.LAL_MSUN_SI
dMcMESH_MSUN = dMcMESH / lal.LAL_MSUN_SI
McMESH_MSUN = McMESH / lal.LAL_MSUN_SI
McSIG_MSUN = McSIG / lal.LAL_MSUN_SI

# Evaluate ambiguity function on the grid
rhos = np.array(eff.evaluate_ip_on_grid(hfSIG, PTMPLT, IP, param_names, grid))
rhogrid = rhos.reshape(NMcs, NEtas)

# Fit to determine effective Fisher matrix
cut = rhos > 0.99
fitgamma = eff.effectiveFisher(eff.residuals2d, rhos[cut], dMcFLAT_MSUN[cut],
        detaFLAT[cut])
# Find the eigenvalues/vectors of the effective Fisher matrix
gam = eff.array_to_symmetric_matrix(fitgamma)
evals, evecs, rot = eff.eigensystem(gam)

elapsed = elapsed_time() - elapsed
print("Time to compute effective Fisher matrix and its eigensystem:", elapsed)
print("For a grid of size:", pts_per_dim, "\n")

# Print information about the effective Fisher matrix
# and its eigensystem
print("Least squares fit finds g_Mc,Mc = ", fitgamma[0])
print("                        g_Mc,eta = ", fitgamma[1])
print("                        g_eta,eta = ", fitgamma[2])

print("\nFisher matrix:")
print("eigenvalues:", evals)
print("eigenvectors:")
print(evecs)
print("rotation taking eigenvectors into Mc, eta basis:")
print(rot)

#
# Distribute points inside predicted ellipsoid of certain level of overlap
#
match_cntr = 0.90
Nrandpts=200
r1 = np.sqrt(2.*(1.-match_cntr)/np.real(evals[0])) # ellipse radii along eigendirections
r2 = np.sqrt(2.*(1.-match_cntr)/np.real(evals[1]))
# Get pts. inside an ellipsoid oriented along eigenvectors...
#cart_grid, sph_grid = eff.uniform_random_ellipsoid(Nrandpts, r1, r2)
Nrad = 10
Nspokes = 40
ph0 = np.arctan(np.abs(r1) * (rot[0,1])/(np.abs(r2) * rot[0,0]) )
#ph0 = 0.
print("angle is:", ph0, r1, r2)
#cart_grid, sph_grid = eff.uniform_spoked_ellipsoid(Nrad,Nspokes, [ph0], r1, r2)
cart_grid, sph_grid = eff.linear_spoked_ellipsoid(Nrad, Nspokes, [ph0], r1, r2)
gridT = np.transpose(cart_grid)
xs = gridT[0]
ys = gridT[1]
# Rotate to get coordinates in parameter basis
cart_grid = np.array([ np.real( np.dot(rot, cart_grid[i]))
    for i in range(len(cart_grid)) ])
gridT = np.transpose(cart_grid)
Xs = gridT[0]
Ys = gridT[1]
# Put in convenient units,
# change from parameter differential (i.e. dtheta)
# to absolute parameter value (i.e. theta = theta_true + dtheta)
rand_dMcs_MSUN, rand_detas = tuple(np.transpose(cart_grid)) # dMc, deta
rand_Mcs = rand_dMcs_MSUN * lal.LAL_MSUN_SI + McSIG # Mc (kg)
rand_etas = rand_detas + etaSIG # eta

# Prune points with unphysical values of eta from cart_grid
rand_etas = np.array(map(partial(lsu.sanitize_eta, exception=np.NAN), rand_etas))
cart_grid = np.transpose((rand_Mcs,rand_etas))
phys_cut = ~np.isnan(cart_grid).any(1) # cut to remove unphysical pts
unphys_cut = np.isnan(cart_grid).any(1) # unphysical pts only
cart_grid = cart_grid[phys_cut]
print("Requested", Nrandpts, "points inside the ellipsoid of",\
        match_cntr, "match.")
print("Kept", len(cart_grid), "points with physically allowed parameters.")

# Save grid of mass points to file
#np.savetxt("Mc_eta_pts.txt", cart_grid)
cart_grid2 = np.array([lsu.m1m2(cart_grid[i][0], cart_grid[i][1]) # convert to m1, m2
        for i in range(len(cart_grid))])
cart_grid2 /= lal.LAL_MSUN_SI
#np.savetxt("m1_m2_pts.txt", cart_grid2)
Njobs = int(np.ceil(len(cart_grid2)/float(pts_per_job)))
cart_grid3 = np.array_split(cart_grid2, Njobs)
for i in range(Njobs):
        fname = "m1_m2_pts_%i.txt" % i
        np.savetxt(fname, cart_grid3[i])

elapsed = elapsed_time() - elapsed
print("Time to distribute points, split and write to file:", elapsed)

#dag_utils.write_integrate_likelihood_extrinsic_sub('test')
#dag_utils.write_extrinsic_marginalization_dag(cart_grid2, 'test.sub')

xmldoc = ligolw.Document()
xmldoc.childNodes.append(ligolw.LIGO_LW())
#proc_id = process.register_to_xmldoc(xmldoc, sys.argv[0], opts.__dict__)
proc_id = process.register_to_xmldoc(xmldoc, sys.argv[0], {})
proc_id = proc_id.process_id
xmldoc.childNodes[0].appendChild(write_sngl_params(cart_grid3, proc_id))
utils.write_filename(xmldoc, "m1m2_grid.xml.gz", gz=True)

#
# N.B. Below here, the real code will divy up cart_grid into blocks of intrinsic
# parameters and compute the marginalized likelihood on each point
# in intrinsic parameter space
#
# For testing purposes, simply evaluate the overlap on cart_grid and plot
# to confirm it is placing points in the expected ellipsoid with the
# proper distribution, and the overlap behaves properly.
#

# Evaluate IP on the grid inside the ellipsoid
rhos2 = eff.evaluate_ip_on_grid(hfSIG, PTMPLT, IP, param_names, cart_grid)

# Plot the ambiguity function, effective Fisher and ellipsoid points
plt.figure(1)
plt.title('Ambiguity function')
cntrs = [.97,0.975,0.98,0.985,0.99,0.995,1.0]
plt.contourf(McMESH_MSUN, etaMESH, rhogrid, cntrs, cmap=plt.cm.jet)
plt.colorbar()
plt.scatter(McSIG_MSUN,etaSIG,marker='x',c='k',s=40)
plt.xlabel("Mc")
plt.ylabel("eta")

plt.figure(2)
plt.title('Effective Fisher contours')
plt.contourf(dMcMESH_MSUN, detaMESH,
        eff.evalfit2d(dMcMESH_MSUN, detaMESH, fitgamma),
        cntrs, cmap=plt.cm.jet)
plt.colorbar()
plt.scatter(0.,0.,marker='x',c='k',s=40)
plt.xlabel("$\Delta M_c$")
plt.ylabel("$\Delta \eta$")

plt.figure(3)
plt.title('Random sampling of 0.97 ellipsoid')
plt.scatter(rand_dMcs_MSUN[phys_cut], rand_detas[phys_cut], c=rhos2, cmap=plt.cm.jet)
plt.colorbar()
plt.xlabel("$\Delta M_c$")
plt.ylabel("$\Delta \eta$")

plt.figure(4)
plt.title('Unrotated ellipse')
plt.scatter(xs, ys)
plt.xlabel("$e_1$")
plt.ylabel("$e_2$")
plt.figure(5)
plt.title('Rotated ellipse')
plt.scatter(Xs, Ys)
plt.xlabel("$e_1$")
plt.ylabel("$e_2$")

plt.figure(6)
plt.title('Intrinsic parameter placement for $(1.5+1.35) M_\odot$')
plt.scatter(McSIG_MSUN + rand_dMcs_MSUN[unphys_cut], etaSIG + rand_detas[unphys_cut], c='b', marker='x', label="unphysical points")
plt.scatter(McSIG_MSUN + rand_dMcs_MSUN[phys_cut], etaSIG + rand_detas[phys_cut], c='r', label="physical points")
plt.axhline(0.25, c='k')
plt.xlabel("$M_c (M_\odot)$")
plt.ylabel("$\eta$")
plt.legend(loc='upper left')
plt.show()
