"""Blind full-sky prior draws through the fixed-distance (6-D) bandlimited kernel,
and the driver modes that stop the run on one uncertified row.

Before 2026-09-08 the 6-D kernel applied a 15-nat endpoint gap that returned
NaN on 92 of 256 blind driver-prior draws at a 20 ms half-window and 32 of 256
at 50 ms (rift_O4d d84597c2a), so ``--mode prior-mc / laplace-is / map`` with
``--time-marginalization-quadrature bandlimited`` and no distance
marginalization stopped in the first chunk.  Evidence and decision:
DESIGN_jax_bandlimited_distmarg.md, "The fixed-distance kernel".

TEST-TIMING: the two kernel fixtures build one injection each and refine
4 x 64 rows; the driver runs take ~10 s each on an 8-core ldas-grid slice.
"""
import importlib.util
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
lal = pytest.importorskip("lal")
lalsim = pytest.importorskip("lalsimulation")

import jax.numpy as jnp                                             # noqa: E402
import RIFT.lalsimutils as lalsimutils                              # noqa: E402
from RIFT.likelihood.jax_ile import core                            # noqa: E402
from RIFT.likelihood.jax_ile.wrapper import (                       # noqa: E402
    JAXExtrinsicLikelihood, build_data_from_precompute,
    bandlimited_storage_requirement)

MSUN, PC = lal.MSUN_SI, lal.PC_SI
_HERE = pathlib.Path(__file__).parent
_CODE = pathlib.Path(__file__).parents[2]
_DRIVER = _CODE / "bin" / "integrate_likelihood_extrinsic_jax"

EPOCH = 1126259462.0
DETECTORS = ("H1", "L1")
FMIN, FREF, FMAX = 40.0, 40.0, 300.0
DELTAF = 0.25
SRATE = 1024.0
DIST_INJ = 900.0                  # rho ~ 8.7 in H1L1: the driver test's injection
ANGLES = (1.2, -0.4, 0.7, 0.9, 2.1)
D_MIN, D_MAX = 50.0, 4000.0
# 20 ms cannot contain a wrong-sky arrival shift (2 R_earth / c = 42.6 ms);
# 50 ms is the measured clean value.
HALF_WINDOW_NARROW, HALF_WINDOW_CLEAN = 0.02, 0.05
SEEDS, N_ROWS, N_CHECK = 4, 64, 6
# Measured worst gap-only disagreement: 1.1e-04 nat (20 ms), 6.1e-06 nat (50 ms).
TOL_REFERENCE = 1e-3
# Measured: every gap-only row sat 54-128 nat below its batch maximum.
DEPTH_BELOW_MAX_MIN = 30.0


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The independent numpy reference pieces (periodic FFT upsample, guard taper,
# log-trapezoid) live with the distance-marginalized tests; loaded by path so
# this file does not depend on how pytest names that module.
_ref_pieces = _load_module(_HERE / "test_jax_bandlimited_distmarg.py",
                           "_jax_bl_distmarg_reference_pieces")


def _params(dist_mpc):
    P = lalsimutils.ChooseWaveformParams()
    P.m1, P.m2 = 35.0 * MSUN, 30.0 * MSUN
    P.fmin, P.fref = FMIN, FREF
    P.deltaT, P.deltaF = 1.0 / SRATE, DELTAF
    P.dist = dist_mpc * 1e6 * PC
    P.fmax = 0.0
    P.approx = lalsim.IMRPhenomD
    P.radec = True
    P.tref = EPOCH
    P.phi, P.theta, P.psi, P.incl, P.phiref = ANGLES
    return P


def _build(half_window):
    _, _, gcert = bandlimited_storage_requirement(1.0 / SRATE, half_window)
    ref_guard = 2 * gcert
    storage = half_window + ref_guard / SRATE + 0.05 + 16.0 / SRATE
    P = _params(DIST_INJ)
    data_dict, psd_dict = {}, {}
    for det in DETECTORS:
        Pd = P.copy()
        Pd.detector = det
        data_dict[det] = lalsimutils.non_herm_hoff(Pd)
        psd_dict[det] = lalsim.SimNoisePSDaLIGOZeroDetHighPower
    data, _ = build_data_from_precompute(
        P.copy(), data_dict, psd_dict, EPOCH, storage, half_window, 2, FMAX,
        analyticPSD_Q=True, verbose=False)
    return data, gcert, ref_guard


def _driver_prior_draws(n, seed):
    """The driver's ``sample_prior`` with distance, re-typed on purpose: the
    point is a full-sky, isotropic-orientation, volumetric-distance draw."""
    rng = np.random.default_rng(seed)
    angles = [rng.uniform(0.0, 2 * np.pi, n), np.arcsin(rng.uniform(-1.0, 1.0, n)),
              rng.uniform(0.0, np.pi, n), np.arccos(rng.uniform(-1.0, 1.0, n)),
              rng.uniform(0.0, 2 * np.pi, n)]
    u = rng.uniform(0.0, 1.0, n)
    dist = (D_MIN ** 3 + u * (D_MAX ** 3 - D_MIN ** 3)) ** (1.0 / 3.0)
    return angles, dist


def _kernel(data, gcert, angles, dist, endpoint_log_gap):
    """The shipped refinement on the fixed-distance field with a CHOSEN gap."""
    kappa_u, rho_u = core._accumulate_unit(
        data, *[jnp.asarray(a) for a in angles], core.JAX_INTERP_DEFAULT, False,
        guard=gcert)
    inv = jnp.asarray(float(data.distMpcRef) / np.asarray(dist))
    return np.asarray(core._time_marginalize_reflected_primitive(
        kappa_u * inv[:, None], rho_u * jnp.square(inv)[:, None], data.deltaT,
        False, guard=gcert, endpoint_log_gap=endpoint_log_gap))


def _reference(data, ref_guard, row_angles, dist):
    """Periodic FFT on a twice-wider guard at factor 512, fixed-distance
    reduction, log-trapezoid: independent of the reflected primitive."""
    ang = [jnp.atleast_1d(jnp.asarray(v)) for v in row_angles]
    kappa, rho = core._accumulate_unit(data, *ang, core.JAX_INTERP_DEFAULT, False,
                                       guard=ref_guard)
    inv = float(data.distMpcRef) / float(dist)
    kappa = np.asarray(kappa)[0] * inv
    rho = np.asarray(rho)[0] * inv * inv
    npts, factor = int(data.npts), _ref_pieces.REF_FACTOR
    fine = _ref_pieces._periodic_fft_upsample(
        _ref_pieces._taper_guard(kappa, ref_guard), factor)
    keep = slice(ref_guard * factor, ref_guard * factor + (npts - 1) * factor + 1)
    lnL = fine[keep].real - 0.5 * rho[ref_guard]
    return _ref_pieces._numpy_log_trapezoid(lnL, float(data.deltaT) / factor)


@pytest.fixture(scope="module", params=[HALF_WINDOW_NARROW, HALF_WINDOW_CLEAN],
                ids=["20ms", "50ms"])
def blind(request):
    half_window = request.param
    data, gcert, ref_guard = _build(half_window)
    like = JAXExtrinsicLikelihood(data, time_quadrature="bandlimited")
    seeds = []
    for seed in range(SEEDS):
        angles, dist = _driver_prior_draws(N_ROWS, seed)
        seeds.append(dict(
            angles=angles, dist=dist,
            shipped=np.asarray(like.log_likelihood(*angles, dist)),
            gap_on=_kernel(data, gcert, angles, dist, core._TIME_ENDPOINT_LOG_GAP_MIN),
            gap_off=_kernel(data, gcert, angles, dist, None)))
    return dict(half_window=half_window, data=data, gcert=gcert,
                ref_guard=ref_guard, seeds=seeds)


def _gap_only(row):
    return np.where(np.isnan(row["gap_on"]) & np.isfinite(row["gap_off"]))[0]


# --------------------------------------------------------------------------
# (a) the kernel
# --------------------------------------------------------------------------
def test_shipped_6d_value_is_the_gap_off_refinement(blind):
    """The wrapper's number is the ``endpoint_log_gap=None`` refinement, and
    the gap parameter is still live: passing the old threshold rejects rows
    the shipped path certifies (measured 57 of 256 at 20 ms, 32 at 50 ms)."""
    n_gap_only = 0
    for row in blind["seeds"]:
        assert np.allclose(row["shipped"], row["gap_off"], equal_nan=True)
        n_gap_only += len(_gap_only(row))
    assert n_gap_only > 0, "the endpoint gap no longer rejects anything: revisit the pin"


def test_remaining_failures_are_the_window_not_the_certificates(blind):
    """With the gap off, every row is certified once the half-window contains
    a wrong-sky arrival shift (0 of 256 at 50 ms), and rows still fail at
    20 ms (measured 35 of 256).  The driver's parse-time refusal rests on
    both halves of this."""
    n_nan = sum(int(np.isnan(row["gap_off"]).sum()) for row in blind["seeds"])
    if blind["half_window"] >= HALF_WINDOW_CLEAN:
        assert n_nan == 0, n_nan
    else:
        assert n_nan > 0, "20 ms is now clean: revisit the parse-time window refusal"


def test_gap_only_rows_agree_with_the_independent_reference(blind):
    """The rows the gap rejected alone are converged: they match the periodic
    reference to TOL_REFERENCE and lie far below the batch maximum, so the
    certificate was measuring the row's amplitude, not the quadrature."""
    checked = 0
    for row in blind["seeds"]:
        vmax = np.nanmax(row["gap_off"])
        for i in _gap_only(row)[:N_CHECK]:
            ref = _reference(blind["data"], blind["ref_guard"],
                             [a[i] for a in row["angles"]], row["dist"][i])
            assert abs(row["gap_off"][i] - ref) <= TOL_REFERENCE, (
                blind["half_window"], i, row["gap_off"][i], ref)
            assert row["gap_off"][i] <= vmax - DEPTH_BELOW_MAX_MIN, (i, row["gap_off"][i], vmax)
            checked += 1
    assert checked >= SEEDS


# --------------------------------------------------------------------------
# (b) the driver's parse-time refusal
# --------------------------------------------------------------------------
def _load_driver():
    import importlib.machinery
    loader = importlib.machinery.SourceFileLoader("_jax_bl_6d_blind_driver", str(_DRIVER))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _parse_and_check(drv, extra):
    argv = ["--inj-mode", "--mass1", "35", "--mass2", "30"] + list(extra)
    optp = drv.build_parser()
    argv = drv._normalize_interpolate_time_argv(argv)
    opts, _ = optp.parse_args(argv)
    drv.record_supplied_options(opts, argv, optp)
    drv.resolve_ile_interface_aliases(opts, optp)
    drv.check_critical_and_report(opts, optp)
    return opts


def test_driver_refuses_the_stop_modes_below_the_arrival_bound(capsys):
    """The window bound is 2 R_earth / c, applied only to the modes whose
    full-sky draws go through ``eval_lnL``'s stop; Simpson, a wide window
    and the flowMC pilot (which the 20 ms flowMC test runs) are not refused."""
    drv = _load_driver()
    bound = 2.0 * lal.REARTH_SI / lal.C_SI
    assert drv._BANDLIMITED_FULLSKY_HALF_WINDOW_MIN == bound
    assert 0.042 < bound < 0.043
    for mode in ("prior-mc", "laplace-is", "map", "nuts"):
        with pytest.raises(SystemExit):
            _parse_and_check(drv, ["--mode", mode,
                                   "--time-marginalization-quadrature", "bandlimited",
                                   "--data-integration-window-half", "0.02"])
        err = capsys.readouterr().err
        assert "2 R_earth / c" in err and "--data-integration-window-half" in err, err
        assert "0.05 s" in err, err
    _parse_and_check(drv, ["--mode", "prior-mc",
                           "--time-marginalization-quadrature", "bandlimited",
                           "--data-integration-window-half", "0.05"])
    _parse_and_check(drv, ["--mode", "prior-mc",
                           "--time-marginalization-quadrature", "simpson",
                           "--data-integration-window-half", "0.02"])
    _parse_and_check(drv, ["--mode", "flowmc", "--distance-marginalization",
                           "--time-marginalization-quadrature", "bandlimited",
                           "--data-integration-window-half", "0.02"])


# --------------------------------------------------------------------------
# (c) the driver end to end
# --------------------------------------------------------------------------
def _driver_argv(mode, half_window, out):
    return [sys.executable, str(_DRIVER),
            "--inj-mode", "--mass1", "35", "--mass2", "30",
            "--inj-deltaF", "0.25", "--inj-detectors", "H1,L1",
            "--inj-distance", str(DIST_INJ),
            "--fmin-template", "40", "--reference-freq", "40", "--fmax", "300",
            "--l-max", "2", "--approximant", "IMRPhenomD", "--srate", "1024",
            "--data-integration-window-half", str(half_window),
            "--internal-data-storage-window-half", "0.08",
            "--d-min", str(D_MIN), "--d-max", str(D_MAX),
            "--mode", mode, "--time-marginalization-quadrature", "bandlimited",
            "--n-max", "400", "--seed", "3", "--output-file", str(out)]


def _run_driver(tmp_path, mode, half_window):
    out = tmp_path / "ile"
    env = dict(os.environ, PYTHONPATH=str(_CODE), OMP_NUM_THREADS="1",
               JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    proc = subprocess.run(_driver_argv(mode, half_window, out), capture_output=True,
                          text=True, env=env, cwd=str(tmp_path), timeout=3600)
    return proc, proc.stdout + proc.stderr, out


@pytest.mark.parametrize("mode", ["prior-mc", "map"])
def test_driver_6d_prior_seeded_modes_run_bandlimited(tmp_path, mode):
    """No distance marginalization: the 6-D kernel under the prior-seeded
    modes, at the clean half-window.  Before the endpoint change this command
    stopped in its first chunk (prior-mc: 30 of 400 rows failed, all at the
    gap).

    ``laplace-is`` is not run end to end: at this injection and seed its
    moment-matched proposal walks off the peak and ``require_finite_evidence``
    raises, at ``--n-max`` 400, 1000 and 2000 alike, and the distance-
    marginalized variant does the same on the unchanged base (ldas-grid,
    2026-09-08).  That is an evidence-quality failure, not a certificate one;
    its blind draws take the same ``eval_lnL`` path as ``prior-mc``, and the
    parse-time refusal above covers it."""
    proc, log, out = _run_driver(tmp_path, mode, HALF_WINDOW_CLEAN)
    assert proc.returncode == 0, log[-3000:]
    assert "failed a certificate" not in log, log[-3000:]
    assert "time-marginalization quadrature: bandlimited" in log, log[-3000:]
    row = np.atleast_2d(np.loadtxt(str(out) + "_0_.dat"))
    assert row.shape[1] == 13 and np.isfinite(row[0, 9]), row


def test_driver_refuses_the_narrow_window_before_precompute(tmp_path):
    proc, log, _ = _run_driver(tmp_path, "prior-mc", HALF_WINDOW_NARROW)
    assert proc.returncode != 0
    assert "2 R_earth / c" in log, log[-3000:]
    assert "Building JAX likelihood" not in log, log[-3000:]
