"""Build a ladder-2 JAX-ILE likelihood from the driver, with configurable mode content.

Derived from ~/rift_skyoffset_20260902/sky_probe_new.py, but parameterised on
approximant / l_max because Task 1 must measure the psi-envelope bracket for
HIGHER-MODE content, not only the (2,+-2) fixture.  Kept in this tree so it
loads the tree under test (asserted below), never a neighbour.
"""
import importlib.util, importlib.machinery, os, sys
import numpy as np

SNAP = os.environ["SNAP"]
DRV = os.path.join(SNAP, "bin", "integrate_likelihood_extrinsic_jax")


def load_driver():
    spec = importlib.util.spec_from_loader(
        "ile_jax_drv", importlib.machinery.SourceFileLoader("ile_jax_drv", DRV))
    m = importlib.util.module_from_spec(spec); m.__name__ = "ile_jax_drv"
    sys.modules["ile_jax_drv"] = m; spec.loader.exec_module(m)
    return m


DIST = {40: 633.920, 80: 316.960, 160: 158.480, 320: 79.240, 640: 39.620}


def ladder2_argv(snr, srate, interp, seed=1001, fmax=1700.0, ndist=256, nphi=8,
                 npsi=8, inj_phiref=0.0, angle_marg="grid",
                 approximant="SEOBNRv4", l_max=2):
    a = ["--inj-mode", "--mass1", "35", "--mass2", "30", "--inj-deltaF", "0.0625",
         "--inj-ra", "1.2", "--inj-dec", "0.3", "--inj-psi", "0.5",
         "--inj-incl", "1.05", "--inj-phiref", repr(inj_phiref),
         "--inj-distance", repr(DIST[snr]), "--inj-detectors", "H1,L1,V1",
         "--distance-marginalization", "--distance-grid-points", str(ndist),
         "--mode", "flowmc-phipsimarg", "--n-phi", str(nphi), "--n-psi", str(npsi),
         "--angle-marg-scheme", angle_marg,
         "--time-marginalization", "--n-events-to-analyze", "1",
         "--reference-freq", "100.0", "--fmin-template", "10", "--fmax", repr(fmax),
         "--l-max", str(l_max), "--approximant", approximant,
         "--d-min", "1", "--d-max", "10000", "--srate", str(srate),
         "--seed", str(seed), "--output-file", "/dev/null/unused"]
    if interp is not None:
        a += ["--interp", interp]
    return a


def build(snr, srate=4096, interp=None, verbose=False, iwh=None, **kw):
    import RIFT
    assert os.path.realpath(RIFT.__file__).startswith(os.path.realpath(SNAP)), \
        "RIFT resolves outside $SNAP: %s" % RIFT.__file__
    drv = load_driver(); optp = drv.build_parser()
    argv = ladder2_argv(snr, srate, interp, **kw)
    opts, _ = optp.parse_args(argv)
    drv.record_supplied_options(opts, argv, optp)
    assert opts.event_time is None
    opts.event_time = 1126259462.0
    fid = opts.event_time; opts.verbose = verbose
    if iwh is not None:
        opts.data_integration_window_half = float(iwh)
    deltaT = 1.0 / opts.srate
    P_t, data_dict, psd_dict, dets, aQ = drv.load_injection(opts, fid)
    deltaF = data_dict[dets[0]].deltaF
    P_t.deltaT, P_t.deltaF = deltaT, deltaF
    like_data, extras = drv.build_data_from_precompute(
        P_t.copy(), data_dict, psd_dict, fid,
        opts.internal_data_storage_window_half, opts.data_integration_window_half,
        opts.l_max, opts.fmax, analyticPSD_Q=aQ, verbose=verbose)
    from RIFT.likelihood.jax_ile.wrapper import JAXDistPhiPsiMargLikelihood
    kwargs = dict(nphi=opts.n_phi, npsi=opts.n_psi,
                  n_grid=opts.distance_grid_points, interp=opts.interp,
                  guess_snr=extras["guess_snr"],
                  angle_marg=getattr(opts, "angle_marg_scheme", "grid"))
    tq = getattr(opts, "time_quadrature", None)
    if tq is not None:
        kwargs["time_quadrature"] = tq
    like = JAXDistPhiPsiMargLikelihood(like_data, opts.d_min, opts.d_max, **kwargs)
    prov = dict(
        tree=SNAP, snr=snr, srate=opts.srate, interp=opts.interp, fmax=opts.fmax,
        l_max=opts.l_max, approximant=opts.approximant,
        d_min=opts.d_min, d_max=opts.d_max, n_dist=opts.distance_grid_points,
        event_time=fid, inj_distance=opts.inj_distance, detectors=dets,
        lms=[list(x) for x in like_data.lms],
        guess_snr=float(extras["guess_snr"]),
        JAX_ILE_DISTMARG_GH=os.environ.get("JAX_ILE_DISTMARG_GH", "unset"),
        JAX_ILE_DISTGRID_ADAPTIVE=os.environ.get("JAX_ILE_DISTGRID_ADAPTIVE", "unset"))
    return like, like_data, prov, opts, drv


def sky_cloud(snr, n, seed_tag="smc_seed1001"):
    """Sky/inclination points the sampler ACTUALLY visited, from the bake-off cloud."""
    run = os.path.expanduser(
        "~/rift_costbakeoff_20260826/runs2/snr%d_%s/output_0_samples.dat" % (snr, seed_tag))
    cl = np.loadtxt(run)
    idx = np.linspace(0, len(cl) - 1, n).astype(int)
    return cl[idx, 0], cl[idx, 1], cl[idx, 2]


# Sky/inclination draw used by the peer session's angle_coeff_structure.py
# (paper repo, branch claude/elated-merkle-c4dda4): a Gaussian around the
# MEASURED rho-40.77 whole-sky AV posterior, shrunk as 1/rho, so the sky points
# sit where the campaign's posterior actually is.  Reproduced verbatim so the
# control numbers are comparable point for point.
_PEER_RHO = {40: 40.7691, 80: 81.5383, 160: 163.0766, 320: 326.1531, 640: 652.3062}
_PEER_SKY = dict(RA0=1.206871, RA_SD=0.006317, DEC0=0.299597, DEC_SD=0.015146,
                 INCL0=0.570507, INCL_SD=0.245015)


def sky_gauss(rung, n=16, seed=31):
    rng = np.random.default_rng(seed)
    sc = _PEER_RHO[40] / _PEER_RHO[rung]
    p = _PEER_SKY
    ra = p["RA0"] + rng.normal(0, p["RA_SD"] * sc, n)
    dec = p["DEC0"] + rng.normal(0, p["DEC_SD"] * sc, n)
    incl = np.clip(p["INCL0"] + rng.normal(0, p["INCL_SD"] * sc, n),
                   1e-3, np.pi - 1e-3)
    return ra, dec, incl
