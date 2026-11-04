"""Small, deliberately slow NumPy oracle for GPU precompute tests.

The frequency axis uses RIFT's centered, descending two-sided layout
[+f_Nyq, ..., +df, 0, ..., -f_Nyq+df].  LAL's reverse complex FFT is
N*df*ifft(ifftshift(X)), not bare numpy.ifft(X).
"""

import numpy as np


def q_oracle(basis_fd, data_fd, weights2side, delta_f, n_shift, n_window):
    basis_fd = np.asarray(basis_fd, dtype=np.complex128)
    data_fd = np.asarray(data_fd, dtype=np.complex128)
    weights2side = np.asarray(weights2side, dtype=np.float64)
    n = basis_fd.shape[-1]
    integrand = 2.0 * np.conj(basis_fd) * data_fd * weights2side
    full = n * delta_f * np.fft.ifft(np.fft.ifftshift(integrand, axes=-1), axis=-1)
    return np.roll(full, -int(n_shift), axis=-1)[..., : int(n_window)]


def uv_oracle(basis_fd, basis_conj_fd, weights2side, delta_f):
    basis_fd = np.asarray(basis_fd, dtype=np.complex128)
    basis_conj_fd = np.asarray(basis_conj_fd, dtype=np.complex128)
    weights2side = np.asarray(weights2side, dtype=np.float64)
    # a,i,f ; b,j,f -> a,b,i,j
    u = 2.0 * delta_f * np.einsum(
        "aif,bjf,f->abij", np.conj(basis_fd), basis_fd, weights2side,
        optimize=False,
    )
    v = 2.0 * delta_f * np.einsum(
        "aif,bjf,f->abij", np.conj(basis_conj_fd), basis_fd, weights2side,
        optimize=False,
    )
    return u, v


def direct_log_likelihood(basis_fd, data_fd, weights2side, delta_f, coeff):
    """Direct Re<h|d> - <h|h>/2 for h=sum_{a,m} coeff[a,m] chi[a,m]."""
    h = np.einsum("am,amf->f", coeff, basis_fd, optimize=False)
    hd = 2.0 * delta_f * np.sum(np.conj(h) * data_fd * weights2side)
    hh = 2.0 * delta_f * np.sum(np.conj(h) * h * weights2side)
    return float(np.real(hd - 0.5 * hh))
