from effectiveFisher import *
import matplotlib.pyplot as plt
from time import clock
from functools import partial

start = clock()
elapsed_time = lambda: clock()-start

# Setup signal and IP class
m1=10.*lal.LAL_MSUN_SI
m2=10.*lal.LAL_MSUN_SI
PSIG = ChooseWaveformParams(m1=m1, m2=m2, approx=lalsim.TaylorT1)
#PSIG.deltaF = findDeltaF(PSIG)
PSIG.deltaF = 1./8.
IP = Overlap(fLow=40., deltaF=PSIG.deltaF, psd=lal.LIGOIPsd)
PTMPLT = PSIG.copy()
hfSIG = norm_hoff(PSIG, IP)
McSIG = mchirp(m1, m2)
etaSIG = symRatio(m1, m2)
NMcs = 11
NEtas = 11
param_names = ['Mc', 'eta']

# Find appropriate parameter ranges
param_ranges = find_effective_Fisher_region(PSIG, IP, 0.90, param_names,
        [[McSIG-0.5*lal.LAL_MSUN_SI,McSIG+0.5*lal.LAL_MSUN_SI],[0.05,0.25]])
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
Mcpts, etapts = make_regular_1d_grids(param_ranges, pts_per_dim)
etapts = map(sanitize_eta, etapts)
McMESH, etaMESH =multi_dim_meshgrid(Mcpts, etapts)
McFLAT, etaFLAT = multi_dim_flatgrid(Mcpts, etapts)
dMcMESH = McMESH - McSIG
detaMESH = etaMESH - etaSIG
dMcFLAT = McFLAT - McSIG
detaFLAT = etaFLAT - etaSIG
grid = multi_dim_grid(Mcpts, etapts)

# Change units on Mc
dMcFLAT_MSUN = dMcFLAT / lal.LAL_MSUN_SI
dMcMESH_MSUN = dMcMESH / lal.LAL_MSUN_SI
McMESH_MSUN = McMESH / lal.LAL_MSUN_SI
McSIG_MSUN = McSIG / lal.LAL_MSUN_SI

# Evaluate ambiguity function on the grid
rhos = np.array(evaluate_ip_on_grid(hfSIG, PTMPLT, IP, param_names, grid))
rhogrid = rhos.reshape(NMcs, NEtas)

# Fit to determine effective Fisher matrix
cut = rhos > 0.99
fitgamma = effectiveFisher(residuals2d, rhos[cut], dMcFLAT_MSUN[cut],
        detaFLAT[cut])
# Find the eigenvalues/vectors of the effective Fisher matrix
gam = array_to_symmetric_matrix(fitgamma)
evals, evecs, rot = eigensystem(gam)

elapsed = elapsed_time() - elapsed
print "Time to compute Fisher matrix and its eigensystem:", elapsed
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
rand_grid = uniform_random_ellipsoid(Nrandpts, r1, r2)
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
rand_etas = np.array(map(partial(sanitize_eta, exception=np.NAN), rand_etas))
rand_grid = np.transpose((rand_Mcs,rand_etas))
phys_cut = ~np.isnan(rand_grid).any(1) # cut to remove unphysical pts
rand_grid = rand_grid[phys_cut]
print "Requested", Nrandpts, "points inside the ellipsoid of",\
        match_cntr, "match."
print "Kept", len(rand_grid), "points with physically allowed parameters."


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
rhos2 = evaluate_ip_on_grid(hfSIG, PTMPLT, IP, param_names, rand_grid)

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
        evalfit2d(dMcMESH_MSUN, detaMESH, fitgamma),
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
