#include <cupy/complex.cuh>

/*
  Fused calibration-marginalized factored log-likelihood (Option C).

  One thread per extrinsic sample.  For each sample j we loop over the n_cal
  calibration realizations (selected by shifting the rholm window offset by
  c*N_window), the integration time window, the detectors and the (l,m) modes,
  forming the data term kappa, applying the default (distance-unmarginalized)
  factored-likelihood helper

        lnL_t(j,c,t) = invDist[j] * Re(kappa) - 0.5 * rho_sq[j,t]

  and accumulating a streaming, Simpson-weighted log-sum-exp over (c,t).  The
  result is the calibration-marginalized log likelihood

        out[j] = log( sum_c sum_t w_t * exp(lnL_t(j,c,t) + log_w[c]) ) - log_w_norm,

  where log_w[c] are per-realization importance log-weights (log_w_norm =
  logsumexp(log_w)); for uniform weights log_w[c]=0 and log_w_norm=log(n_cal), giving
  the plain (1/n_cal) average.  This supports adaptive / importance cal sampling.

  rho_sq is calibration-independent and is passed in pre-summed over detectors.
  w_t are the composite-Simpson quadrature weights (including dx=deltaT), so the
  time integration matches Option B's simps() exactly.

  Array layouts (all C-contiguous):
    Q       : (n_det, npts_full, n_lms)   complex128   rholm timeseries.T per det
    A       : (n_det, n_ext,     n_lms)   complex128   conj(F*Ylm) per det
    ifirst  : (n_det, n_ext)              int32        within-block window offset
    invDist : (n_ext,)                    float64      distMpcRef/distMpc
    rho_sq  : (n_ext, npts)               float64      template (U,V) term, summed over det
    w_t     : (npts,)                     float64      Simpson weights * deltaT
    out     : (n_ext,)                    float64
*/

extern "C" {

  __global__ void Q_fused_calmarg(
    const complex<double> * Q,
    const complex<double> * A,
    const int * ifirst,
    const double * invDist,
    const double * rho_sq,
    const double * w_t,
    const double * log_w,
    double log_w_norm,
    int phase_marg,
    int n_det,
    int n_cal,
    int N_window,
    int npts,
    int n_lms,
    int n_ext,
    int npts_full,
    double * out
  ){
    size_t j = threadIdx.x + (size_t)blockDim.x * blockIdx.x;
    if (j >= (size_t)n_ext) return;

    const double inv = invDist[j];

    /* streaming weighted log-sum-exp accumulators (first-iteration flag avoids
       needing an explicit -infinity, which is awkward under NVRTC) */
    double m = 0.0;         /* running max of (lnL_t + log_w[c]) */
    double S = 0.0;         /* running sum_  w_t * exp(lnL_t + log_w[c] - m) */
    bool first = true;

    for (int c = 0; c < n_cal; ++c) {
      const double lw = log_w[c];   /* per-realization importance log-weight */
      for (int t = 0; t < npts; ++t) {
        complex<double> kappa = complex<double>(0.0, 0.0);
        for (int d = 0; d < n_det; ++d) {
          long within = (long)ifirst[(size_t)d * n_ext + j] + (long)t;
          /* The window must stay inside THIS detector's realization block c, i.e.
             the within-block offset must lie in [0, N_window).  An out-of-range
             offset (window extends past the rholm buffer for an extreme sky
             position, or a pathological/NaN draw) means this detector contributes
             zero at this (c,t) -- matching the per-block n_cal==1 behavior.  This
             also prevents reading a neighbouring block (a non-last-block overflow
             would otherwise bleed into block c+1) or out of bounds entirely. */
          if (within < 0 || within >= (long)N_window) continue;
          long idx = within + (long)c * N_window;
          const complex<double> * Qd = Q + (size_t)d * npts_full * n_lms;
          const complex<double> * Ad = A + ((size_t)d * n_ext + j) * n_lms;
          long qrow = idx * (long)n_lms;
          for (int lm = 0; lm < n_lms; ++lm) {
            kappa += Ad[lm] * Qd[qrow + lm];
          }
        }

        /* phase marginalization: use |kappa| (the (2,-2) conjugation is already baked
           into Q/A by the caller), else Re(kappa). */
        double kre = phase_marg ? sqrt(kappa.real()*kappa.real() + kappa.imag()*kappa.imag()) : kappa.real();
        double lnLt = inv * kre - 0.5 * rho_sq[(size_t)j * npts + t] + lw;
        double wt = w_t[t];

        if (first) {
          m = lnLt;
          S = wt;
          first = false;
        } else if (lnLt > m) {
          S = S * exp(m - lnLt) + wt;
          m = lnLt;
        } else {
          S += wt * exp(lnLt - m);
        }
      } /* t */
    } /* c */

    out[j] = m + log(S) - log_w_norm;
  } /* Q_fused_calmarg */

} /* extern "C" */
