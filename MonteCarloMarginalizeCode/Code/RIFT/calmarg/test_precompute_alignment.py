"""
Regression test for the calibration-marginalization PRECOMPUTE time alignment.

This exercises the real PrecomputeLikelihoodTerms / ComputeModeIPTimeSeries path
(unlike backtest_calmarg.py / test_calmarg_reduction.py, which feed synthetic rholms
with a hand-set epoch and therefore cannot catch a precompute-alignment bug).

With the calibration factor set to 1 (identity), the calibration-marginalized rholm
series must, block by block, reproduce the non-calibration rholm series -- same data AND
the same epoch.  A wrong epoch on the concatenated series (the bug fixed in this branch)
shifts ifirst into the wrong realization block downstream, the signal is zeroed, and the
calmarg likelihood collapses.  The epoch check below is the one that fails on that bug.

Runs on CPU (no GPU needed).
"""
import numpy as np
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lalsimutils
import RIFT.likelihood.factored_likelihood as fl

fSample = 4096.0
fmin = 30.0
fmax = 1700.0
event_time = 1000000000.0
t_window = 0.1

# A short BBH so the test is fast.
Psig = lalsimutils.ChooseWaveformParams(
    fmin=fmin, radec=True, incl=0.0, phiref=0.0, theta=0.2, phi=0.0, psi=0.0,
    m1=30 * lal.MSUN_SI, m2=30 * lal.MSUN_SI,
    detector='H1', dist=200e6 * lal.PC_SI, deltaT=1. / fSample,
    tref=event_time, deltaF=1. / 4.)

data_dict = {}
for det in ("H1", "L1", "V1"):
    P = Psig.manual_copy(); P.detector = det
    data_dict[det] = lalsimutils.non_herm_hoff(P)
psd_dict = {det: lalsim.SimNoisePSDaLIGOZeroDetHighPower for det in data_dict}

Lmax = 2
n_cal = 5

# baseline (no calibration marginalization) -- DEFAULT 6-value API, unchanged
rholms_intp_b, ct_b, ctV_b, rholms_base, snr_b, _ = fl.PrecomputeLikelihoodTerms(
    event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
    analyticPSD_Q=True, verbose=False, quiet=True, skip_interpolation=True)

# calibration marginalization with the IDENTITY calibration (factor == 1);
# opt in to the two trailing self-term cross-term structures.
cal_real = {det: np.ones((data_dict[det].data.length, n_cal), dtype=complex)
            for det in data_dict}
rholms_intp_c, ct_c, ctV_c, rholms_cal, snr_c, _, ctcal_c, ctVcal_c = fl.PrecomputeLikelihoodTerms(
    event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
    analyticPSD_Q=True, verbose=False, quiet=True, skip_interpolation=True,
    calibration_realizations=cal_real, return_calibration_crossterms=True)

ok = True
for det in data_dict:
    for pair in rholms_base[det]:
        base = rholms_base[det][pair]
        cal = rholms_cal[det][pair]
        N_window = base.data.length
        # (1) concatenated length is n_cal blocks
        assert cal.data.length == N_window * n_cal, \
            "%s %s: cal length %d != %d*%d" % (det, pair, cal.data.length, N_window, n_cal)
        # (2) EPOCH must match the non-calibration series (the alignment bug)
        d_epoch = abs(float(cal.epoch) - float(base.epoch))
        # (3) every block must reproduce the baseline rholm (cal factor == 1)
        block_err = 0.0
        for c in range(n_cal):
            blk = cal.data.data[c * N_window:(c + 1) * N_window]
            block_err = max(block_err, float(np.max(np.abs(blk - base.data.data))))
        flag_e = "OK" if d_epoch < 1e-9 else "**EPOCH MISMATCH**"
        flag_b = "OK" if block_err < 1e-6 else "**BLOCK MISMATCH**"
        print("%s %s : |delta epoch|=%.3e %s   max|block-baseline|=%.3e %s" % (
            det, pair, d_epoch, flag_e, block_err, flag_b))
        if d_epoch >= 1e-9 or block_err >= 1e-6:
            ok = False

assert ok, "calmarg precompute alignment MISMATCH (epoch and/or block data)"

# (4) fused-calmarg self-term fix: with the IDENTITY calibration (|C_c|==1), the
# per-realization |C_c|^2-weighted cross terms must reproduce the baseline cross
# terms exactly (rho_sq_c == rho_sq).  This checks the ComputeModeCrossTermIPCal
# path and its packing/keying alignment.
assert ctcal_c is not None and ctVcal_c is not None, "cal cross terms not returned"
cal_err = 0.0
for det in data_dict:
    assert len(ctcal_c[det]) == n_cal and len(ctVcal_c[det]) == n_cal
    for c in range(n_cal):
        for pair in ct_b[det]:
            cal_err = max(cal_err, abs(complex(ctcal_c[det][c][pair]) - complex(ct_b[det][pair])))
            cal_err = max(cal_err, abs(complex(ctVcal_c[det][c][pair]) - complex(ctV_b[det][pair])))
flag_c = "OK" if cal_err < 1e-8 else "**CAL CROSSTERM MISMATCH**"
print("identity-cal cross terms vs baseline: max|delta|=%.3e %s" % (cal_err, flag_c))
assert cal_err < 1e-8, "identity-cal |C|^2-weighted cross terms != baseline cross terms"

print("\nPASS: calmarg precompute is time-aligned with the baseline (epoch + per-block data),\n      and identity-cal self-term cross terms reproduce the baseline.")
