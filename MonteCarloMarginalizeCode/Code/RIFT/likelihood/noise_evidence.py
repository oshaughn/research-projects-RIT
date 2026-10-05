"""Bilby-compatible fixed-PSD noise-evidence utilities.

For the Gaussian transient likelihood used by Bilby and RIFT, the noise
hypothesis has no sampled parameters.  Its log evidence is therefore the
zero-signal log likelihood, ``-0.5 * sum_k (d_k | d_k)``.  The Gaussian
determinant term is omitted, matching Bilby's GW likelihood convention.

ILE's own lnL is already measured from that hypothesis, so ILE's lnZ is a log
Bayes factor: adding this number to it gives the signal-hypothesis log evidence
Bilby reports.  The two agree only if both sides used the same conditioned data
and PSD, which is why the value is computed inside ILE rather than offline.
"""

import math

import numpy as np


def compute_network_log_noise_evidence(
        data_dict, psd_dict, fmin, fmax, fnyq,
        inv_spec_trunc_Q=False, T_spec=0.0, inner_product_factory=None):
    """Compute ``-0.5 * sum_k (d_k | d_k)`` for a detector network.

    Parameters use the same conditioned data, PSDs, and frequency bounds as
    ILE.  ``inner_product_factory`` is injectable to keep the bookkeeping
    independently testable; production calls use :class:`RIFT.lalsimutils.ComplexIP`.

    Returns
    -------
    total : float
        Network log noise evidence.
    per_detector : dict
        Per-detector ``d_inner_d`` and ``log_noise_evidence`` values.
    """
    data_detectors = set(data_dict)
    psd_detectors = set(psd_dict)
    if not data_detectors:
        raise ValueError("Cannot compute noise evidence without detector data")
    if data_detectors != psd_detectors:
        raise ValueError(
            "Detector mismatch between data ({}) and PSDs ({})".format(
                sorted(data_detectors), sorted(psd_detectors)))

    if inner_product_factory is None:
        from RIFT.lalsimutils import ComplexIP
        inner_product_factory = ComplexIP

    per_detector = {}
    total = 0.0
    for detector in sorted(data_detectors):
        data = data_dict[detector]
        inner_product = inner_product_factory(
            fLow=fmin, fMax=fmax, fNyq=fnyq, deltaF=data.deltaF,
            psd=psd_dict[detector], analyticPSD_Q=False,
            inv_spec_trunc_Q=inv_spec_trunc_Q, T_spec=T_spec)
        # An empty integration band -- fmin above where the PSD is nonzero, or a
        # per-detector --fmin-ifo cut that zeroed all of it -- returns (d|d)=0 and
        # a log evidence of 0, which is a plausible-looking number that nothing
        # else would flag.  Zero-strain data also gives 0 and is not an error, so
        # check the weights rather than the result.  Stubs need not expose them.
        weights = getattr(inner_product, "weights2side", None)
        if weights is not None and not np.any(np.asarray(weights) > 0):
            raise ValueError(
                "No nonzero PSD weight in [{}, {}) Hz for detector {}: the "
                "noise evidence would be reported as 0".format(
                    fmin, fmax, detector))

        d_inner_d = float(inner_product.ip(data, data).real)
        if not math.isfinite(d_inner_d) or d_inner_d < 0:
            raise ValueError(
                "Invalid (d|d)={} for detector {}".format(
                    d_inner_d, detector))
        detector_log_evidence = -0.5 * d_inner_d
        per_detector[detector] = {
            "d_inner_d": d_inner_d,
            "log_noise_evidence": detector_log_evidence,
        }
        total += detector_log_evidence

    return total, per_detector
