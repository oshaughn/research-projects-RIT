"""Direct Gaussian likelihood for response-applied, independent A/E/T channels.

Inputs are continuous Fourier amplitudes (dt * rfft(time_series)), ordered A,E,T.
PSDs are one-sided and use the same channel observable/units as the amplitudes.
The caller supplies a common uniform frequency grid and an explicit bin mask.
Exclude DC and any real-valued Nyquist endpoint: this module uses weight 4*df
for every selected bin. It does not apply a LISA response, transform channels,
or establish that a supplied covariance is diagonal.
"""

import numpy as np


def aet_inner_product(a, b, psd, delta_f, *, mask):
    """Return 4*df*Re sum(conj(a)*b/PSD) on explicitly selected bins.

    Arrays have shape (3, n_frequency); mask has shape (n_frequency,).
    Correlated channels require a covariance likelihood and are unsupported here.
    Nonfinite values and nonpositive PSDs are refused on selected bins. Excluded
    bins may contain singular PSDs or nonfinite data and are never evaluated.
    """
    a, b, psd = np.asarray(a), np.asarray(b), np.asarray(psd)
    if a.ndim != 2 or a.shape[0] != 3 or b.shape != a.shape or psd.shape != a.shape:
        raise ValueError('a, b and psd must have matching shape (3, n_frequency)')
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != (a.shape[1],) or not np.any(mask):
        raise ValueError('mask must be a nonempty boolean frequency selection')
    if np.ndim(delta_f) != 0 or not np.isrealobj(delta_f):
        raise ValueError('delta_f must be a finite positive real scalar')
    delta_f = float(delta_f)
    if not np.isfinite(delta_f) or delta_f <= 0:
        raise ValueError('delta_f must be a finite positive real scalar')
    a, b, psd = a[:, mask], b[:, mask], psd[:, mask]
    if not np.isrealobj(psd) or not np.all(np.isfinite(psd)) or np.any(psd <= 0):
        raise ValueError('selected PSD values must be finite, real and positive')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('selected Fourier amplitudes must be finite')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result = float(4 * delta_f * np.real(np.sum(np.conj(a) * b / psd)))
    if not np.isfinite(result):
        raise ValueError('inner product is not finite')
    return result


def aet_log_likelihood(data, model, psd, delta_f, *, mask):
    """Return -0.5*(data-model|data-model), omitting a fixed noise constant.

    Data, PSD and mask must be fixed across model evaluations. Parameter-dependent
    noise would also require its normalization term. No parameters are marginalized.
    """
    data, model = np.asarray(data), np.asarray(model)
    if data.shape != model.shape:
        raise ValueError('data and model shapes must match; broadcasting is refused')
    residual = data - model
    return -0.5 * aet_inner_product(residual, residual, psd, delta_f, mask=mask)
