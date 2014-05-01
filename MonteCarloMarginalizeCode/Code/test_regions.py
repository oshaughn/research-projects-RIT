
import regions
import scipy.special
import numpy as np
from matplotlib import pylab as plt

slowTests=True
verySlowTests=False




### GROUP 0: Circular region.  For illustration
print " ------- Test group 0: Circular region "

el = regions.RegionEllipse([['x',-2,1],['y',-1,1]])

print " -- 0.1: Sample region "

# Test 1: Identify region boundary (default circle)
#   To draw an ellipse, use http://matplotlib.org/api/artist_api.html#matplotlib.patches.Ellipse
#   Related infrastructure test: express 'good' points in associated polar coordinates
if slowTests:
#    pts = el.draw(500)     # do not use the native draw function, so I can find points *outside* the region
    pts = np.random.randn(1000,2)
    pts_test = [el.inside_q(x) for x in pts]
    ptsOk = np.array([pts[k] for k, value in enumerate(pts_test) if value])
    ptsBad = np.array([pts[k] for k, value in enumerate(pts_test) if not value])
    print len(ptsOk), pts.shape
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
print " -- 0.2: Integrate over region via mcsampler "
mypi,err = el.integrate(lambda x: 1, lambda x:1.,verbose=False)
print "The following quantity should be pi, plus a small error: ",  mypi,err  # circle of radius 1
mypi,err = el.integrate(lambda x: 1, lambda x:2.,verbose=False)
print "The following quantity should be 2pi, plus a small error: ",  mypi,err  # circle of radius 1

if slowTests:
    sigma = 0.1
    myunit,err = el.integrate(lambda x: np.exp(-np.dot(x,x)/(2.*sigma**2))/(2*np.pi*sigma**2), lambda x:1.,verbose=False,adaptive=True)
    print "The following quantity should be 1, plus a small error: ",  myunit,err  # gaussian centered on 0


# Test 3: Marginalize: construct posterior distributions for 1d quantitiees
print " -- 0.3: Marginalized distribution "
weighted_samples = el.sample_and_marginalize(lambda x: x[0], lambda x: 1, lambda x:1.,verbose=False,sort_by_parameter=True)
# Construct 1d cumulative versus *parameter*.
# Should agree with 'x' for x in [-1,1] (basically, area of circle]
# Limited by finite neff
print "neff for this distribution is", np.sum(weighted_samples[:,1])/np.max(weighted_samples[:,1])  # neff
xtmp =weighted_samples[:,0]
plt.plot(xtmp,weighted_samples[:,2])                                                     # Cumulative distribution in 1d (scatter)
plt.plot(xtmp,  0.5*((2*(xtmp*np.sqrt(1.-xtmp*xtmp) + np.arcsin(xtmp)))/np.pi+1))  # Should recover the area of a circle
plt.xlabel('x')
plt.ylabel('P(<x)')
plt.show()


if slowTests:
    sigma = 0.1
    weighted_samples = el.sample_and_marginalize(lambda x: x[0], lambda x: np.exp(-np.dot(x,x)/(2.*sigma**2))/(2*np.pi*sigma**2), lambda x:1.,verbose=True, sort_by_parameter=True)
    print "neff for this distribution is", np.sum(weighted_samples[:,1])/np.max(weighted_samples[:,1])  # neff
    xtmp =weighted_samples[:,0]
    plt.plot(xtmp,weighted_samples[:,2])                  # Cumulative distribution in 1d (scatter).  Note this should agree with the error function when sigma is small
    plt.plot(xtmp,0.5+scipy.special.erf(xtmp/np.sqrt(2)/sigma)*0.5)  # Error function
    plt.xlabel('x')
    plt.ylabel('P(<x)')
    plt.show()


### GROUP 1: Elliptical regions
print " ------- Test group 1: Elliptical  "
el = regions.RegionEllipse([['x',-5,5],['y',-5,5]], mtx = [[2,0],[0,0.5]])

# Test 1.1: Identify region boundary (default circle)
#   To draw an ellipse, use http://matplotlib.org/api/artist_api.html#matplotlib.patches.Ellipse
if slowTests:
    pts = np.random.randn(1000,2)
    pts_test = [el.inside_q(x) for x in pts]
    ptsOk = np.array([pts[k] for k, value in enumerate(pts_test) if value])
    ptsBad = np.array([pts[k] for k, value in enumerate(pts_test) if not value])
    print len(ptsOk)
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
el = regions.RegionEllipse([['x',-5,5],['y',-5,5]], mtx = [[2,0.1],[0.1,0.5]])

# Test 1.1: Identify region boundary (default circle)
#   To draw an ellipse, use http://matplotlib.org/api/artist_api.html#matplotlib.patches.Ellipse
if slowTests:
    pts = np.random.randn(1000,2)
    pts_test = [el.inside_q(x) for x in pts]
    ptsOk = np.array([pts[k] for k, value in enumerate(pts_test) if value])
    ptsBad = np.array([pts[k] for k, value in enumerate(pts_test) if not value])
    print len(ptsOk)
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
    print len(ptsOk)
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
print "The following quantity should be pi/2, plus a small error: ",  mypi,err  # circle of radius 1



# Test 2.3: Marginalize: construct posterior distributions for 1d quantitiees: Circle and gaussian
weighted_samples = el.sample_and_marginalize(lambda x: x[0], lambda x: 1, lambda x:1.,verbose=True,sort_by_parameter=True)
print "neff for this distribution is", np.sum(weighted_samples[:,1])/np.max(weighted_samples[:,1])  # neff
xtmp =weighted_samples[:,0]
plt.plot(weighted_samples[:,0],weighted_samples[:,2])  # Cumulative distribution in 1d (scatter)
plt.plot(xtmp,  2*(xtmp*np.sqrt(1.-xtmp*xtmp) + np.arcsin(xtmp))/np.pi)  # Should recover the area of a half-circle
plt.xlabel('x')
plt.ylabel('P(<x)')
plt.show()



### GROUP 3:  Load in 'ellipsoid.dat' file produced by the dag-generation code
#   Format:
#       header='2x2 effective Fisher matrix
#       3rd row: ellipsoid axes
#       4th row: total ellipse area, estimated physical area'
if slowTests:
    print " -- 3.1: Sample real 'ellipsoid.dat' from CME "
    import lalsimutils
    dat = np.loadtxt("ellipsoid.dat")
    center = dat[0]
    mtx =np.array([dat[1], dat[2]])
    print 'Center: ',center
    print 'Matrix: ',mtx
    el = regions.RegionEllipse([['mc',center[0]-5./np.sqrt(mtx[0,0]),center[0]+3./np.sqrt(mtx[0,0])],['eta', 0.05,0.25]],mtx=mtx,center=center)
    el_larger = regions.RegionEllipse([['mc',center[0]-5./np.sqrt(mtx[0,0]),center[0] +3./np.sqrt(mtx[0,0])],['eta', 0.05,0.25]],mtx=0.5*mtx,center=center)

    # 3.1 : Draw points natively [using the 'larger' ellipsoid to draw in a sensible place]
    pts = el_larger.draw(500)
    if len(pts) == 0:
        import sys
        sys.exit(0)
    pts_test = [el.inside_q(x) for x in pts]
    ptsOk = np.array([pts[k] for k, value in enumerate(pts_test) if value])
    ptsBad = np.array([pts[k] for k, value in enumerate(pts_test) if not value])
    print len(ptsOk)
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
    dat_grid = np.loadtxt("intrinsic_grid.dat")
    plt.scatter(dat_grid[:,1],dat_grid[:,2])
    plt.figure(2)
    plt.scatter(dat_grid[:,3],dat_grid[:,4])
    pts_intrinsic = np.array([el.convert_global_to_internal(dat_grid[k,1],dat_grid[k,2]) for k in np.arange(len(dat_grid))])
    plt.scatter(pts_intrinsic[:,0],pts_intrinsic[:,1],color='r')
    plt.show()

    # 3.3 : Import and infer coordinates from mass sample file (output of util_MassGrid.py)


    # 3.4 : Import and infer coordinates from mass sample file; construct interpolation; do integral, with *uniform* mc,eta prior


    # 3.4 : Import and infer coordinates from mass sample file; construct interpolation; construct posterior, with  *uniform* mc,eta prior


### Infrastructure tests
print " ------- INFRASTRUCTURE: Hyper-sphere unit vectors ---------- "
print "Hyper-sphere unit vectors. Note ordering is NOT conventional in low-d. Fix?"
print "Two dimensional unit vector test: xhat, yhat = ", regions.hyperunit([1,0]), regions.hyperunit([1,np.pi/2])  # should be x axis, y axis
print "Three dimensional unit vector test: xhat, yhat, zhat = ", regions.hyperunit([1,np.pi/2,0]), regions.hyperunit([1,np.pi/2,np.pi/2]), regions.hyperunit([1,0,0])
print "Four dimensional unit vector test:  = ",
print  regions.hyperunit([1,np.pi/2,0,0])
print regions.hyperunit([1,np.pi/2,np.pi/2,0])
print regions.hyperunit([1,np.pi/2,np.pi/2,np.pi/2])
print regions.hyperunit([1,0,0,0])
