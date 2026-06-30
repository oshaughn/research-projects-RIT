#include <cupy/complex.cuh>

/*
  Fused calibration-marginalized factored log-likelihood with DISTANCE
  marginalization (Option C, stage 2).  Separate from cuda_Q_fused_calmarg.cu by
  design: it keeps the simpler default-helper kernel untouched as a review baseline
  and fallback, and the Python 'loop' path (Option B) remains a full fallback for
  distmarg too.

  This reproduces the distance-marginalization loglikelihood used at the ILE
  distmarg call sites (and EvenBivariateLinearInterpolator):

      x0   = kappa_sq / rho_sq                          (kappa_sq = invDist*Re(kappa))
      s    = asinh(sqrt_bmax*(x0 - xmin)) - asinh(sqrt_bmax*(xmax - x0))
      t    = asinh(rho_sq / bref)
      lnI  = bilinear interp of lnI_array at (s,t)   if smin<s<smax and t<tmax
           = -inf (i.e. contributes 0)               otherwise
      x0c  = clip(x0, xmin, xmax)
      lnL_t(j,c,t) = rho_sq*x0c*(x0 - 0.5*x0c) + lnI

  then a streaming, Simpson-weighted log-sum-exp over (c,t):
      out[j] = log( (1/n_cal) sum_c sum_t w_t * exp(lnL_t(j,c,t)) ).

  The in-bounds mask guarantees the bilinear stencil indices are valid; an extra
  index guard avoids any illegal access from floating-point edge cases.

  Array layouts match cuda_Q_fused_calmarg.cu; lnI_array is (ns, nt) row-major
  (axis 0 = s, axis 1 = t).
*/

extern "C" {

  __device__ double _asinh_stable(double x) {
    /* stable for all x (avoids cancellation for x<0); avoids needing <math.h> */
    if (x >= 0.0) return log(x + sqrt(x * x + 1.0));
    else          return -log(-x + sqrt(x * x + 1.0));
  }

  __global__ void Q_fused_calmarg_distmarg(
    const complex<double> * Q,
    const complex<double> * A,
    const int * ifirst,
    const double * invDist,
    const double * rho_sq,
    const double * w_t,
    const double * log_w,
    double log_w_norm,
    int phase_marg,
    const double * lnI_array,
    double s0, double ds, double smin, double smax, int ns,
    double t0, double dt, double tmax, int nt,
    double xmin, double xmax, double sqrt_bmax, double bref,
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

    double m = 0.0;     /* running max of lnL_t */
    double S = 0.0;     /* running sum_ w_t * exp(lnL_t - m) */
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

        /* phase marginalization: |kappa| (conjugation baked into Q/A), else Re(kappa) */
        double kre = phase_marg ? sqrt(kappa.real()*kappa.real() + kappa.imag()*kappa.imag()) : kappa.real();
        double kappa_sq = inv * kre;
        double rsq = rho_sq[(size_t)j * npts + t];
        double x0 = kappa_sq / rsq;

        double s = _asinh_stable(sqrt_bmax * (x0 - xmin))
                 - _asinh_stable(sqrt_bmax * (xmax - x0));
        double tt = _asinh_stable(rsq / bref);

        /* in-bounds test matches distmarg_loglikelihood's mask */
        if (!(s > smin && s < smax && tt < tmax)) continue;

        double i_mid = (s - s0) / ds;
        double j_mid = (tt - t0) / dt;
        int i_lo = (int)floor(i_mid);
        int i_hi = (int)ceil(i_mid);
        int j_lo = (int)floor(j_mid);
        int j_hi = (int)ceil(j_mid);
        /* defensive guard against floating-point edge cases */
        if (i_lo < 0 || i_hi >= ns || j_lo < 0 || j_hi >= nt) continue;

        double p = i_mid - i_lo;
        double q = j_mid - j_lo;
        double p_ = 1.0 - p;
        double q_ = 1.0 - q;
        double lnI = p_ * q_ * lnI_array[(size_t)i_lo * nt + j_lo]
                   + p  * q_ * lnI_array[(size_t)i_hi * nt + j_lo]
                   + p_ * q  * lnI_array[(size_t)i_lo * nt + j_hi]
                   + p  * q  * lnI_array[(size_t)i_hi * nt + j_hi];

        double x0c = x0;
        if (x0c < xmin) x0c = xmin;
        if (x0c > xmax) x0c = xmax;
        double lnLt = rsq * x0c * (x0 - 0.5 * x0c) + lnI + lw;

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
  } /* Q_fused_calmarg_distmarg */

} /* extern "C" */
