from effectiveFisher import *
import matplotlib.pyplot as plt

# Setup signal and IP class
m1=30.*lal.LAL_MSUN_SI
m2=4.2*lal.LAL_MSUN_SI
PSIG = ChooseWaveformParams(m1=m1, m2=m2, deltaF=1./16., approx=lalsim.EOBNRv2)
#PSIG = ChooseWaveformParams(m1=m1, m2=m2, deltaF=1./16., approx=lalsim.TaylorT1)
IP = Overlap(fLow=40., deltaF=1./16., psd=lal.LIGOIPsd)
PTMPLT = PSIG.copy()
hfSIG = norm_hoff(PSIG, IP)
McSIG = mchirp(m1, m2)
etaSIG = symRatio(m1, m2)
NMcs = 21
NEtas = 21
#minMc = McSIG - 0.04*lal.LAL_MSUN_SI
#maxMc = McSIG + 0.04*lal.LAL_MSUN_SI
#minEta = max(0.01, etaSIG-0.005)
#maxEta = min(0.25, etaSIG+0.005)
param_names = ['Mc', 'eta']

# Find appropriate parameter ranges
param_ranges = find_effective_Fisher_region(PSIG, IP, 0.90, param_names,
        [[McSIG-0.5*lal.LAL_MSUN_SI,McSIG+0.5*lal.LAL_MSUN_SI],[0.05,0.25]])
print "Computing amibiguity function in the range:"
for i, param in enumerate(param_names):
    print "\t", param, ":", param_ranges[i]

# setup parameter grid
#Mcrange = [minMc, maxMc]
#etarange = [minEta, maxEta]
#param_ranges = [Mcrange, etarange]
pts_per_dim = [NMcs, NEtas]
Mcpts, etapts = make_regular_1d_grids(param_ranges, pts_per_dim)
McMESH, etaMESH =multi_dim_meshgrid(Mcpts, etapts)
McFLAT, etaFLAT = multi_dim_flatgrid(Mcpts, etapts)
dMcMESH = McMESH - McSIG
detaMESH = etaMESH - etaSIG
dMcFLAT = McFLAT - McSIG
detaFLAT = etaFLAT - etaSIG
grid = multi_dim_grid(Mcpts, etapts)

# Evaluate ambiguity function on the grid
rhos = np.array(evaluate_ip_on_grid(hfSIG, PTMPLT, IP, param_names, grid))
rhogrid = rhos.reshape(NMcs, NEtas)

# Fit to determine effective Fisher matrix
cut = rhos > 0.99
fitgamma = effectiveFisher(residuals2d, rhos[cut], dMcFLAT[cut], detaFLAT[cut])
print "Least squares fit finds g_Mc,Mc = ", fitgamma[0]
print "                        g_Mc,eta = ", fitgamma[1]
print "                        g_eta,eta = ", fitgamma[2]


plt.figure(1)
cntrs = [.97,0.975,0.98,0.985,0.99,0.995,1.0]
plt.contourf(McMESH, etaMESH, rhogrid, cntrs, cmap=plt.cm.jet)
plt.scatter(McSIG,etaSIG,marker='x',c='k',s=40)
plt.xlabel("Mc")
plt.ylabel("eta")

plt.figure(2)
plt.contourf(dMcMESH, detaMESH,
        evalfit2d(dMcMESH, detaMESH, fitgamma),
        cntrs, cmap=plt.cm.jet)
plt.scatter(0.,0.,marker='x',c='k',s=40)
plt.show()
