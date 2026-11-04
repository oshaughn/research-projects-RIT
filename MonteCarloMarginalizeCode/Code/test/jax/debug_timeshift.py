"""Is the ~1% lnL(truth) deficit a fractional-sample time misalignment between the
fslib injection and the RIFT precompute t=0?

Applies a pure time shift exp(2*pi*i*f*dt_shift) to the injected data (per detector,
in FD) over a fine grid of dt_shift within +-1 sample, rebuilds the freqresponse
recovery, and evaluates lnL(truth).  If the deficit half<d|d>-lnL(truth) is
minimized (->~0) at some dt_shift != 0, the injection<->precompute time reference
is misaligned by that fraction of a sample (a fixable convention), and that offset
is the self-consistent injection shift.  If the minimum sits at dt_shift=0 with the
deficit intact, the ~1% is a genuine response-model gap, not a time reference.

ANSWERED, and by neither of those two: the ~1% was almost entirely under-sampled time
interpolation.  The rholm used to sit at exactly the Nyquist rate of its own analysis
band, leaving a 4-point stencil no headroom.  Oversampling the rholm collapses the
deficit by ~99%, and what survives is a precompute floor (finite t_window, Qmax
truncation, PSD band) that is flat in oversampling -- not a time-reference convention
and not a response-model gap of the size this script was chasing.  The ladder and the
method are in analyses/slowrot_finite-size/DESIGN_sampling.md in the paper repo.  Note
they were measured on the library's own numpy likelihood, whereas this script drives the
JAX one -- the two agree on the sky offset to within the difference between them, but the
deficit ladder itself has not been reproduced on this path.

This script is therefore kept as a probe OF that regime rather than a question about it:
it pins oversample=1 (see main) so the scan still exhibits the effect.  Repointing it at
the surviving floor would be a different measurement and needs a different scan range.
"""
import inspect
import os, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

_FSLIB = os.environ.get("SLOWROT_FS_LIB_DIR",
                        os.path.expanduser("~/RIFT_roboto_paper/analyses/slowrot_finite-size"))
sys.path.insert(0, _FSLIB)
import slowrot_fs_lib as fslib
import RIFT.likelihood.factored_likelihood_freqresponse as flfr
from RIFT.likelihood.jax_ile.wrapper import (
    build_freqresponse_data_from_precompute, JAXDistanceMarginalizedLikelihood)

NET = os.environ.get("SLOWROT_NET", "CE+ET+K")
SNR = float(os.environ.get("SLOWROT_SNR_REP", "600"))
QMAX = int(os.environ.get("SLOWROT_QMAX", "4"))
INCL = float(os.environ.get("SLOWROT_INCL", "1.05"))
IWH, TBUF = 0.03, 0.12


def _shift_data(data_dict, dt_shift):
    """Return a copy of data_dict with each series multiplied by exp(2 pi i f dt)."""
    out = {}
    for det, d in data_dict.items():
        n = d.data.length
        fvals = flfr.evaluate_fvals_from_length(n, d.deltaF)
        nd = lal_copy(d)
        nd.data.data[:] = d.data.data * np.exp(2j * np.pi * fvals * dt_shift)
        out[det] = nd
    return out


def lal_copy(d):
    import lal
    nd = lal.CreateCOMPLEX16FrequencySeries(d.name, d.epoch, d.f0, d.deltaF,
                                            d.sampleUnits, d.data.length)
    nd.data.data[:] = d.data.data
    return nd


def main():
    # DEFAULT to oversample=1, rather than inheriting the library's.  This scan is
    # defined RELATIVE to deltaT -- dts = fr * deltaT over +-1 sample -- so it does not
    # merely get finer when the library samples the rholm more finely, it covers
    # proportionally less physical time and stops probing the near-Nyquist regime the
    # scan exists to characterise.  At the library's current default (4) the deficit it
    # reads is flat and near zero, which looks like "nothing to see" rather than "you are
    # no longer looking".  Overridable, so the surviving floor can be scanned on purpose.
    # Old libraries have no such knob and are already at 1: honour that silently, but
    # refuse an explicit request they cannot satisfy rather than quietly giving 1.
    _ovs = int(os.environ.get("SLOWROT_OVERSAMPLE", "1"))
    if "oversample" in inspect.signature(fslib.Source.__init__).parameters:
        _kw = {"oversample": _ovs}
    elif _ovs != 1:
        raise SystemExit(
            "SLOWROT_OVERSAMPLE=%d needs a slowrot_fs_lib with the oversample knob; this "
            "one sets deltaT = 1/(2*fmax) and is already at 1" % _ovs)
    else:
        _kw = {}
    src = fslib.Source(m1=1.6, m2=1.4, ra=1.2, dec=0.3, psi=0.5, incl=INCL,
                       phiref=0.0, fmin=50.0, fmax=1024.0, seglen=32.0, approx="IMRPhenomD",
                       **_kw)
    net = fslib.network(NET)
    dist = fslib.distance_for_snr(src, net, SNR)
    dd, pd, arm, meta = fslib.build_finite_size_data(src, net, dist)
    P0 = fslib._base_params(src, dist, meta["deltaT"], meta["deltaF"])
    half_dd = meta["half_dd"]; deltaT = meta["deltaT"]
    rt, dt = src.ra, src.dec
    print("=== %s SNR=%.0f incl=%.1f deg  half<d|d>=%.1f  deltaT=%.3e ===" %
          (NET, meta["snr"], np.degrees(INCL), half_dd, deltaT))

    def eval_truth(data_dict):
        data, _ = build_freqresponse_data_from_precompute(
            P0, data_dict, pd, fslib.EVENT_TIME, IWH, fslib.LMAX, src.fmax,
            t_window=TBUF, Qmax=QMAX, L_arm=arm, analyticPSD_Q=True, verbose=False)
        d_min = max(1.0, dist * 0.3); d_max = dist * 2.5
        like = JAXDistanceMarginalizedLikelihood(data, d_min, d_max, n_grid=256, interp="cubic")
        return float(np.asarray(like.log_likelihood(
            np.array([rt]), np.array([dt]), np.array([src.psi]),
            np.array([src.incl]), np.array([src.phiref]))[0])), data

    fracs = np.linspace(-1.0, 1.0, 21)
    best = (-np.inf, 0.0)
    for fr in fracs:
        dts = fr * deltaT
        lnL_t, _ = eval_truth(_shift_data(dd, dts) if fr != 0 else dd)
        print("  dt_shift=%+.3f samples (%+.3e s): lnL(truth)=%.1f  deficit=%.1f (%.3f%%)" %
              (fr, dts, lnL_t, half_dd - lnL_t, 100 * (half_dd - lnL_t) / half_dd))
        if lnL_t > best[0]:
            best = (lnL_t, fr)
    print("\nBEST dt_shift = %+.3f samples, lnL(truth)=%.1f deficit=%.1f (%.3f%%)" %
          (best[1], best[0], half_dd - best[0], 100 * (half_dd - best[0]) / half_dd))


if __name__ == "__main__":
    main()
