import lal
import lalsimulation as lalsim
import lalsimutils as lsu
import effectiveFisher as eff
import numpy as np
import matplotlib.pyplot as plt
from time import clock
from functools import partial

start = clock()
elapsed_time = lambda: clock()-start

pts_per_job = 10 # How many intrinsic points to pass to each condor job

# Setup signal and IP class
m1=10.*lal.LAL_MSUN_SI
m2=10.*lal.LAL_MSUN_SI
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
print "Computing amibiguity function in the range:"
for i, param in enumerate(param_names):
    if param=='Mc' or param=='m1' or param=='m2': # rescale output by MSUN
        print "\t", param, ":", np.array(param_ranges[i])/lal.LAL_MSUN_SI,\
                "(Msun)"
    else:
        print "\t", param, ":", param_ranges[i]

elapsed = elapsed_time()
print "Range-finding took:", elapsed, "(s) for", len(param_names),"parameters\n"

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
print "Time to compute effective Fisher matrix and its eigensystem:", elapsed
print "For a grid of size:", pts_per_dim, "\n"

# Print information about the effective Fisher matrix
# and its eigensystem
print "Least squares fit finds g_Mc,Mc = ", fitgamma[0]
print "                        g_Mc,eta = ", fitgamma[1]
print "                        g_eta,eta = ", fitgamma[2]

print "\nFisher matrix:"
print "eigenvalues:", evals
print "eigenvectors:"
print evecs
print "rotation taking eigenvectors into Mc, eta basis:"
print rot

#
# Distribute points inside predicted ellipsoid of certain level of overlap
#
match_cntr = 0.97
Nrandpts=200
r1 = np.sqrt(2.*(1.-match_cntr)/evals[0]) # ellipse radii along eigendirections
r2 = np.sqrt(2.*(1.-match_cntr)/evals[1])
# Get pts. inside an ellipsoid oriented along eigenvectors...
rand_grid = eff.uniform_random_ellipsoid(Nrandpts, r1, r2)
# Rotate to get coordinates in parameter basis
rand_grid = np.array([ np.real(rot.dot(rand_grid[i]))
    for i in xrange(len(rand_grid)) ])
# Put in convenient units,
# change from parameter differential (i.e. dtheta)
# to absolute parameter value (i.e. theta = theta_true + dtheta)
rand_dMcs_MSUN, rand_detas = tuple(np.transpose(rand_grid)) # dMc, deta
rand_Mcs = rand_dMcs_MSUN * lal.LAL_MSUN_SI + McSIG # Mc (kg)
rand_etas = rand_detas + etaSIG # eta

# Prune points with unphysical values of eta from rand_grid
rand_etas = np.array(map(partial(lsu.sanitize_eta, exception=np.NAN), rand_etas))
rand_grid = np.transpose((rand_Mcs,rand_etas))
phys_cut = ~np.isnan(rand_grid).any(1) # cut to remove unphysical pts
rand_grid = rand_grid[phys_cut]
print "Requested", Nrandpts, "points inside the ellipsoid of",\
        match_cntr, "match."
print "Kept", len(rand_grid), "points with physically allowed parameters."

# Save grid of mass points to file
#np.savetxt("Mc_eta_pts.txt", rand_grid)
rand_grid2 = [lsu.m1m2(rand_grid[i][0], rand_grid[i][1]) # convert to m1, m2
        for i in xrange(len(rand_grid))]
#np.savetxt("m1_m2_pts.txt", rand_grid2)
Njobs = int(np.ceil(len(rand_grid2)/float(pts_per_job)))
rand_grid2 = np.array_split(rand_grid2, Njobs)
for i in xrange(Njobs):
        fname = "m1_m2_pts_%i.txt" % i
        np.savetxt(fname, rand_grid2[i])

elapsed = elapsed_time() - elapsed
print "Time to distribute points, split and write to file:", elapsed

#
# N.B. Below here, the real code will divy up rand_grid into blocks of intrinsic
# parameters and compute the marginalized likelihood on each point
# in intrinsic parameter space
#
# For testing purposes, simply evaluate the overlap on rand_grid and plot
# to confirm it is placing points in the expected ellipsoid with the
# proper distribution, and the overlap behaves properly.
#

# Evaluate IP on the grid inside the ellipsoid
rhos2 = eff.evaluate_ip_on_grid(hfSIG, PTMPLT, IP, param_names, rand_grid)

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
plt.show()
