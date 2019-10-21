from __future__ import print_function

import regions
import scipy.special
import numpy as np
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interpolate

slowTests=True
verySlowTests=True




### GROUP 0: Circular region.  For illustration
print(" ------- Test group 0: Circular region ")

el = regions.RegionEllipse([['x',-2,1],['y',-1,1]])

print(" -- 0.1: Sample region ")
print("  Plot 1:  a circle, blue point inside, red outside ")
print("  Plot 2:  as above, in polar coordinates (in principal axes of circle) ")
# Test 1: Identify region boundary (default circle)
#   To draw an ellipse, use http://matplotlib.org/api/artist_api.html#matplotlib.patches.Ellipse
#   Related infrastructure test: express 'good' points in associated polar coordinates
if slowTests:
#    pts = el.draw(500)     # do not use the native draw function, so I can find points *outside* the region
    pts = np.random.randn(1000,2)
    pts_test = [el.inside_q(x) for x in pts]
    ptsOk = np.array([pts[k] for k, value in enumerate(pts_test) if value])
    ptsBad = np.array([pts[k] for k, value in enumerate(pts_test) if not value])
    print(len(ptsOk), pts.shape)
    fig = plt.gcf()
    plt.scatter(ptsOk[:,0], ptsOk[:,1])
    if len(ptsBad)>0:
        plt.scatter(ptsBad[:,0], ptsBad[:,1],color='r')
    fig.gca().add_artist(plt.Circle((0,0),1,fill=False))
    plt.xlabel('x')
    plt.ylabel('y')
    fig = plt.figure(2)
    ptsOk_internal = np.array(map(el.convert_global_to_internal, ptsOk))
    plt.scatter(ptsOk_internal[:,0], ptsOk_internal[:,1])
    if len(ptsBad)>0:
        ptsBad_internal = np.array(map(el.convert_global_to_internal, ptsBad))
        plt.scatter(ptsBad_internal[:,0], ptsBad_internal[:,1],color='r')
    plt.xlabel('polar radius (ellipsoidal native)')
    plt.ylabel('polar angle (ellipsoidal native)')
    plt.show()


# Test 2: Integrate functions on this region (1; gaussian; ...), using BRUTE FORCE (cartesian+rejection) method
#    Note the integral will be independent of the region boundaries
print(" -- 0.2: Integrate over region via mcsampler: 1,2, and gaussian ")
mypi,err = el.integrate(lambda x: 1, lambda x:1.,verbose=False)
print("The following quantity should be pi, plus a small error: ",  mypi,err)  # circle of radius 1
mypi,err = el.integrate(lambda x: 1, lambda x:2.,verbose=False)
print("The following quantity should be 2pi, plus a small error: ",  mypi,err)  # circle of radius 1

if slowTests:
    sigma = 0.1
    myunit,err = el.integrate(lambda x: np.exp(-np.dot(x,x)/(2.*sigma**2))/(2*np.pi*sigma**2), lambda x:1.,verbose=False,adaptive=True)
    print("The following quantity should be 1, plus a small error: ",  myunit,err)  # gaussian centered on 0


# Test 3: Marginalize: construct posterior distributions for 1d quantitiees
print(" -- 0.3(a): Marginalized distribution [1 over circle] ")
print("  Plot 1:  cumulative area of unit circle to left of 'x, compared with exact formula' ")
weighted_samples = el.sample_and_marginalize(lambda x: x[0], lambda x: 1, lambda x:1.,verbose=False,sort_by_parameter=True)
# Construct 1d cumulative versus *parameter*.
# Should agree with 'x' for x in [-1,1] (basically, area of circle]
# Limited by finite neff
print("neff for this distribution is", np.sum(weighted_samples[:,1])/np.max(weighted_samples[:,1]))  # neff
xtmp =weighted_samples[:,0]
plt.plot(xtmp,weighted_samples[:,2],label="from samples")                                                     # Cumulative distribution in 1d (scatter)
plt.plot(xtmp,  0.5*((2*(xtmp*np.sqrt(1.-xtmp*xtmp) + np.arcsin(xtmp)))/np.pi+1),label="exact")  # Should recover the area of a circle
plt.legend()
plt.xlabel('x')
plt.ylabel('P(<x)')
plt.show()


if slowTests:
    print(" -- 0.3(b): Marginalized distribution [narrow gaussian in circle] ")
    print("  Plot 1:  cumulative area of gaussian to left of 'x, compared with exact formula' ")
    sigma = 0.1
    weighted_samples = el.sample_and_marginalize(lambda x: x[0], lambda x: np.exp(-np.dot(x,x)/(2.*sigma**2))/(2*np.pi*sigma**2), lambda x:1.,verbose=False, sort_by_parameter=True)
#    print "neff for this distribution is", np.sum(weighted_samples[:,1])/np.max(weighted_samples[:,1])  # neff
    xtmp =weighted_samples[:,0]
    plt.plot(xtmp,weighted_samples[:,2],label="from samples")                  # Cumulative distribution in 1d (scatter).  Note this should agree with the error function when sigma is small
    plt.plot(xtmp,0.5+scipy.special.erf(xtmp/np.sqrt(2)/sigma)*0.5,label="almost exact")  # Error function
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('P(<x)')
    plt.show()


### GROUP 1: Elliptical regions
print(" ------- Test group 1: Elliptical  ")
el = regions.RegionEllipse([['x',-5,5],['y',-5,5]], mtx = [[2,0],[0,0.5]])

# Test 1.1: Identify region boundary (default circle)
#   To draw an ellipse, use http://matplotlib.org/api/artist_api.html#matplotlib.patches.Ellipse
if slowTests:
    pts = np.random.randn(1000,2)
    pts_test = [el.inside_q(x) for x in pts]
    ptsOk = np.array([pts[k] for k, value in enumerate(pts_test) if value])
    ptsBad = np.array([pts[k] for k, value in enumerate(pts_test) if not value])
    print(len(ptsOk))
    plt.scatter(ptsOk[:,0], ptsOk[:,1])
    plt.scatter(ptsBad[:,0], ptsBad[:,1],color='r')
    fig = plt.figure(2)
    ptsOk_internal = np.array(map(el.convert_global_to_internal, ptsOk))
    ptsBad_internal = np.array(map(el.convert_global_to_internal, ptsBad))
    plt.scatter(ptsOk_internal[:,0], ptsOk_internal[:,1])
    plt.scatter(ptsBad_internal[:,0], ptsBad_internal[:,1],color='r')
    plt.xlabel('polar radius (ellipsoidal native)')
    plt.ylabel('polar angle (ellipsoidal native)')

    plt.show()


### GROUP 1c: Elliptical region, with nontrivial correlation
el = regions.RegionEllipse([['x',-5,5],['y',-5,5]], mtx = [[2,0.6],[0.6,0.5]])

# Test 1.1: Identify region boundary (default circle)
#   To draw an ellipse, use http://matplotlib.org/api/artist_api.html#matplotlib.patches.Ellipse
if slowTests:
    pts = np.random.randn(1000,2)
    pts_test = [el.inside_q(x) for x in pts]
    ptsOk = np.array([pts[k] for k, value in enumerate(pts_test) if value])
    ptsBad = np.array([pts[k] for k, value in enumerate(pts_test) if not value])
    print(len(ptsOk))
    plt.scatter(ptsOk[:,0], ptsOk[:,1])
    plt.scatter(ptsBad[:,0], ptsBad[:,1],color='r')
    fig = plt.figure(2)
    ptsOk_internal = np.array(map(el.convert_global_to_internal, ptsOk))
    ptsBad_internal = np.array(map(el.convert_global_to_internal, ptsBad))
    plt.scatter(ptsOk_internal[:,0], ptsOk_internal[:,1])
    plt.scatter(ptsBad_internal[:,0], ptsBad_internal[:,1],color='r')
    plt.xlabel('polar radius (ellipsoidal native)')
    plt.ylabel('polar angle (ellipsoidal native)')
    
    plt.show()




### GROUP 2: Circular region with cutoff: Half-circle

el = regions.RegionEllipse([['x',0,1],['y',-1,1]])

# Test 2.1: Identify region boundary (default circle)
#   To draw an ellipse, use http://matplotlib.org/api/artist_api.html#matplotlib.patches.Ellipse
if slowTests:
    pts = np.random.randn(1000,2)
    pts_test = [el.inside_q(x) for x in pts]
    ptsOk = np.array([pts[k] for k, value in enumerate(pts_test) if value])
    ptsBad = np.array([pts[k] for k, value in enumerate(pts_test) if not value])
    print(len(ptsOk))
    fig = plt.gcf()
    plt.scatter(ptsOk[:,0], ptsOk[:,1])
    plt.scatter(ptsBad[:,0], ptsBad[:,1],color='r')
    fig.gca().add_artist(plt.Circle((0,0),1,fill=False))
    fig = plt.figure(2)
    ptsOk_internal = np.array(map(el.convert_global_to_internal, ptsOk))
    ptsBad_internal = np.array(map(el.convert_global_to_internal, ptsBad))
    plt.scatter(ptsOk_internal[:,0], ptsOk_internal[:,1])
    plt.scatter(ptsBad_internal[:,0], ptsBad_internal[:,1],color='r')
    plt.xlabel('polar radius (ellipsoidal native)')
    plt.ylabel('polar angle (ellipsoidal native)')
    plt.show()

# Test 2.2: Integrate functions on this region (1; gaussian; ...), using BRUTE FORCE (cartesian+rejection) method
#    I use a region boundary to impose a hard cutoff
mypi,err = el.integrate(lambda x: 1, lambda x:1.,verbose=True)
print("The following quantity should be pi/2, plus a small error: ",  mypi,err)  # circle of radius 1



# Test 2.3: Marginalize: construct posterior distributions for 1d quantitiees: Circle and gaussian
print(" Test 2.3 : Construct CDF for area of circle to left of 'x' ")
weighted_samples = el.sample_and_marginalize(lambda x: x[0], lambda x: 1, lambda x:1.,verbose=True,sort_by_parameter=True)
print("neff for this distribution is", np.sum(weighted_samples[:,1])/np.max(weighted_samples[:,1]))  # neff
xtmp =weighted_samples[:,0]
plt.plot(weighted_samples[:,0],weighted_samples[:,2])  # Cumulative distribution in 1d (scatter)
plt.plot(xtmp,  2*(xtmp*np.sqrt(1.-xtmp*xtmp) + np.arcsin(xtmp))/np.pi)  # Should recover the area of a half-circle
plt.xlabel('x')
plt.ylabel('P(<x)')
plt.show()



### GROUP 3:  Load in 'ellipsoid.dat' file produced by the dag-generation code
#   WARNING: CONSISTENT FILES REQUIRED
#       ellipsoid.dat,
#       intrinsic_grid.dat
#       massgrid-coal
#       
#   Format:
#       header='2x2 effective Fisher matrix
#       3rd row: ellipsoid axes
#       4th row: total ellipse area, estimated physical area'
if True: #slowTests:
    print(" -- 3.1: Sample real 'ellipsoid.dat' from CME ")
    import lalsimutils
    dat = np.loadtxt("ellipsoid.dat")
    center = dat[0]
    mtx =np.array([dat[1], dat[2]])
    match_cntr = dat[-1,0]
    radius_ile = np.sqrt(2*(1-match_cntr))            # coordinate radius that ILE will scale to unity in 'intrinsic_grid.dat', below. Based on match = 1-r^2/2 in ellipsoid coords
    mtxSmaller = mtx/((1-match_cntr)*2.)              # Rescale so x.mtxSmaller.x =1 surface corresponds to x.mtx.x = (1-match)*2
    print('Center: ',center)
    print('Matrix: ',mtx)
    print('match threshold:', match_cntr)
    el = regions.RegionEllipse([['mc',center[0]-5./np.sqrt(mtx[0,0]),center[0]+5./np.sqrt(mtx[0,0])],['eta', 0.15,0.25]],mtx=mtxSmaller,center=center)
    el_larger = regions.RegionEllipse([['mc',center[0]-5./np.sqrt(mtx[0,0]),center[0] +5./np.sqrt(mtx[0,0])],['eta', 0.15,0.25]],mtx=mtx,center=center)

    # 3.1 : Draw points natively [using the 'larger' ellipsoid to draw in a sensible place]
    pts = el_larger.draw(500)
    if len(pts) == 0:
        import sys
        sys.exit(0)
    pts_test = [el.inside_q(x) for x in pts]
    ptsOk = np.array([pts[k] for k, value in enumerate(pts_test) if value])
    ptsBad = np.array([pts[k] for k, value in enumerate(pts_test) if not value])
    print(len(ptsOk))
    fig = plt.gcf()
    plt.scatter(ptsOk[:,0], ptsOk[:,1])
    plt.scatter(ptsBad[:,0], ptsBad[:,1],color='r')
    plt.xlabel('Mc (Msun)')
    plt.ylabel('eta')
    fig = plt.figure(2)
    ptsOk_internal = np.array(map(el.convert_global_to_internal, ptsOk))
    ptsBad_internal = np.array(map(el.convert_global_to_internal, ptsBad))
    plt.scatter(ptsOk_internal[:,0], ptsOk_internal[:,1])
    plt.scatter(ptsBad_internal[:,0], ptsBad_internal[:,1],color='r')
    plt.xlabel('polar radius (ellipsoidal native)')
    plt.ylabel('polar angle (ellipsoidal native)')
    plt.show()

    # 3.2 : Import and infer coordinates from mass sample file (intrinsic_grid.dat)
    #    indx  mc eta  r th
    #    Use convert_global_to_intrinsic to reproduce it (=confirm codes consistent)
    #    WARNING: 'intrinsic_grid.dat' uses grid radii that are UNITY at the edge, which is an ARBITRARY match contour
    #    Used to prove CME and this code have the same underlying grid transformations
    print(" -- 3.2: Import intrinsic_grid.dat ")
    print("  Plot 1:  mc, eta coordinates ")
    print("  Plot 2:  intrinsic polar coordinates of ellipsoid ")
    dat_grid = np.loadtxt("intrinsic_grid.dat")
    plt.scatter(dat_grid[:,1],dat_grid[:,2], label='CME grid')
    plt.xlabel("Mc (Msun)")
    plt.ylabel("eta")
    plt.legend()
    plt.figure(2)
    plt.scatter(dat_grid[:,3],dat_grid[:,4])  # note change of radius to 'sensible' radial coordinate
    pts_intrinsic = np.array([el.convert_global_to_internal([dat_grid[k,1],dat_grid[k,2]]) for k in np.arange(len(dat_grid))])
    plt.scatter(pts_intrinsic[:,0],np.mod(pts_intrinsic[:,1], 2*np.pi),color='r')
    plt.xlabel('polar radius (ellipsoidal native)')
    plt.ylabel('polar angle (ellipsoidal native)')
    plt.show()

    # 3.3 : Import and infer coordinates from mass sample file (output of util_MassGrid.py)
    #    indx m1 m2 lnLred neff sigma_{lnLred}
    print(" -- 3.3: Import output of util_MassGrid.py ")
    print("  Test : are all points in the ellipsoid? ")
    print("  Plot 1:  mc, eta coordinates: grid provided by file (blue) and random samples (green) out to 1/2e point ")
    print("  Plot 2:  polar coordinates in inferred from grid")
    print("  Plot 3:  3d plot of same data (no interpolation)")
    print("  Plot 4:  3d plot of same data (no interpolation), polar coordinates")
    dat_grid_physical = np.loadtxt("massgrid-coal-indexed.dat")
    dat_grid = np.array(map(lambda x: [lalsimutils.mchirp(x[1],x[2]), lalsimutils.symRatio(x[1],x[2])],   dat_grid_physical))
    dat_grid_internal = np.array(map(el.convert_global_to_internal, dat_grid) )
    # Test if points in ellipsoid
    for pt in dat_grid:
        if not(el.inside_q(pt)):
            print(" Not inside ellipse! ", pt)
    # Plot
    plt.figure(1)
    plt.scatter(ptsOk[:,0], ptsOk[:,1],color='g',label="random draws from ellipse")
    plt.scatter(dat_grid[:,0],dat_grid[:,1], label='CME grid')
    plt.xlabel("$M_c$ ($M_\odot$)");    plt.ylabel("$\eta$"); plt.legend()
    plt.figure(2)
    plt.scatter(dat_grid_internal[:,0],dat_grid_internal[:,1], label='CME grid')
    plt.xlabel("r (ellipse)");    plt.ylabel("$\theta$ (ellipse)"); plt.legend()
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dat_grid[:,0], dat_grid[:,1], dat_grid_physical[:,3])
    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dat_grid_internal[:,0], dat_grid_internal[:,1], dat_grid_physical[:,3])
    plt.show()

    # 3.4 : Import and infer coordinates from mass sample file; construct interpolation; do integral, with *uniform* mc,eta prior
    #        This grid need *not* be consistent with 'ellipsoid.dat' above, so we redefine the center
    # 3.4.a Interpolate off of an unstructured grid (i.e., in raw mc, eta)
    print(" -- 3.4: Import output of util_MassGrid.py, then integrate over mc eta ")
    print("  Plot 2:  Add output of interpolation")
    center = dat_grid[0]
    lnL_interp_raw = interpolate.interp2d(dat_grid[:,0], dat_grid[:,1], dat_grid_physical[:,3])
    lnL_interp =lambda x,y: lnL_interp_raw(x,y)[0] if el.inside_q(np.array([x,y])) else 0   # force scalar return value, not array. Force 0 outside ellipse
    lnL_interp_polar_raw = interpolate.interp2d(dat_grid_internal[:,0], dat_grid_internal[:,1], dat_grid_physical[:,3])
    lnL_interp_polar =lambda x,y: lnL_interp_polar_raw(x,y)[0] if (x<1 and x>=0.) else 0   # force scalar return value, not array. Force 0 outside ellipse
    print(" Sanity check: lnL at interpolated grid center ", center,  el.inside_q(center), lnL_interp(center[0], center[1]))
    mcg, etag  = np.meshgrid(np.linspace(el.llim[0], el.rlim[0], 50), np.linspace(el.llim[1], el.rlim[1],50))
    mcG = mcg.flatten()
    etaG = etag.flatten()
    plt.figure(1)
    plt.scatter(mcG,etaG,color='b',label="example of grid")
    plt.scatter(ptsOk[:,0], ptsOk[:,1],color='g',label="random draws from ellipse")
    plt.scatter(dat_grid[:,0],dat_grid[:,1], label='CME grid')
    plt.legend()
    plt.show()
    lnL_interp_vals = np.array( [ lnL_interp(mcG[k], etaG[k]) for k in np.arange(len(mcG))])
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mcG,etaG,lnL_interp_vals)
    ax.scatter(dat_grid[:,0], dat_grid[:,1], dat_grid_physical[:,3],color='b')
    ax.set_zlim(0,np.max(dat_grid_physical[:,3]))
    plt.show()
    lnLScale = np.float128(np.max(dat_grid_physical[:,3]))
    fnL = lambda x: np.max([1e-7,np.exp(-lnLScale+lnL_interp(x[0], x[1]))])
    val  = lnLScale + np.log(el.integrate(fnL, lambda x:1, verbose=True))    # Vectorization should be provided by low-level code
    print("Evidence value, uniform prior: Unstructured grid interpolation: ", val)

    # 3.4a.1 : Import and infer coordinates from mass sample file; construct interpolation; construct posterior, with  *uniform* mc,eta prior
    print(" -- 3.5: Import and interpolate output of util_Mass grid.py, then construct marginalized mchirp distribution")
    weighted_samples = el.sample_and_marginalize(lambda x: x[0], fnL, lambda x:1.,verbose=True,sort_by_parameter=True)
    xtmp =weighted_samples[:,0]
    plt.plot(xtmp,weighted_samples[:,2])                                                     # Cumulative distribution in 1d (scatter)
    plt.xlabel('mc')
    plt.ylabel('P(<mc)')
    plt.show()


    # 3.4.b Repeat, using structured grid interpolation


### Infrastructure tests
print(" ------- INFRASTRUCTURE: Hyper-sphere unit vectors ---------- ")
print("Hyper-sphere unit vectors. Note ordering is NOT conventional in low-d. Fix?")
print("Two dimensional unit vector test: xhat, yhat = ", regions.hyperunit([1,0]), regions.hyperunit([1,np.pi/2]))  # should be x axis, y axis
print("Three dimensional unit vector test: xhat, yhat, zhat = ", regions.hyperunit([1,np.pi/2,0]), regions.hyperunit([1,np.pi/2,np.pi/2]), regions.hyperunit([1,0,0]))
print("Four dimensional unit vector test:  = ", end=' ')
print(regions.hyperunit([1,np.pi/2,0,0]))
print(regions.hyperunit([1,np.pi/2,np.pi/2,0]))
print(regions.hyperunit([1,np.pi/2,np.pi/2,np.pi/2]))
print(regions.hyperunit([1,0,0,0]))
