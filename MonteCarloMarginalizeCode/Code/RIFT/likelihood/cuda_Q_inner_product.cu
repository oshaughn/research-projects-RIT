#include <cupy/complex.cuh>

extern "C" {

  __global__ void Q_inner(
    const complex<double> * Q, const complex<double> * A,
    const int * index_start,
    int window_size,
    int num_time_points,
    int num_extrinsic_samples,
    int num_lms,
    complex<double> * out
  ){
    extern __shared__ complex<double> A_sample[];

    /* Figure out which extrinsic sample number we're on. */
    size_t sample_idx = threadIdx.x + blockDim.x*blockIdx.x;
    
    // time index in the window for each sample
    size_t t_idx = threadIdx.y + blockDim.y * blockIdx.y;

    /* Only do something if we're not out of bounds. */
    if (sample_idx < num_extrinsic_samples) {
      for (size_t i = 0; i<num_lms; ++i) {
        A_sample[threadIdx.x*num_lms+i] = A[sample_idx*num_lms+i];
      }
      __syncthreads();

      /* Determine the time index we need to use. */
      size_t i_first_time = index_start[sample_idx];

      /* Iterate over the time window. */
      for (size_t i_time = t_idx; i_time < window_size; i_time+=blockDim.y) {
        /* Determine the index we're going to output to. */
        size_t i_output = sample_idx*window_size + i_time;

        complex<double> out_tmp = 0.;

        /* Guard against out-of-range time offsets: for some sky positions the
           window can extend a sample past the precomputed rholm buffer, and with
           calibration marginalization the buffer is n_cal blocks long so an
           over-read in the last block hits unmapped memory (CUDA illegal access).
           Out-of-range time samples contribute zero rather than reading OOB.
           (i_first_time is size_t, so a negative int index wraps to a large value
           and is also caught here.) */
        size_t q_time = i_first_time + i_time;
        if (q_time < (size_t)num_time_points) {
          /* Take the outer product over the lm axis. */
          for (size_t i_lm = 0; i_lm < num_lms; ++i_lm) {
            out_tmp +=
              A_sample[threadIdx.x*num_lms + i_lm] *
              Q[q_time*num_lms + i_lm];
          }
        }

        out[i_output] = out_tmp;
      }
    } // if
  } // Q_inner

  __global__ void Q_inner_cubic(
    const double2 * Q, const double2 * A,
    const int * index_start,
    const double * fractional_offset,
    int window_size,
    int num_time_points,
    int num_extrinsic_samples,
    int num_lms,
    double2 * out
  ){
    size_t sample_idx = threadIdx.x + blockDim.x*blockIdx.x;
    size_t t_idx = threadIdx.y + blockDim.y * blockIdx.y;

    if (sample_idx < num_extrinsic_samples) {
      int i_first_time = index_start[sample_idx];
      double u = fractional_offset[sample_idx];
      double w_m1 = -u * (u - 1.0) * (u - 2.0) / 6.0;
      double w_0 = (u + 1.0) * (u - 1.0) * (u - 2.0) / 2.0;
      double w_p1 = -(u + 1.0) * u * (u - 2.0) / 2.0;
      double w_p2 = (u + 1.0) * u * (u - 1.0) / 6.0;

      for (size_t i_time = t_idx; i_time < window_size; i_time+=blockDim.y) {
        size_t i_output = sample_idx*window_size + i_time;
        int q_time = i_first_time + (int)i_time;
        double out_re = 0.0;
        double out_im = 0.0;

        for (size_t i_lm = 0; i_lm < num_lms; ++i_lm) {
          double q_re = 0.0;
          double q_im = 0.0;
          int q_m1 = q_time - 1;
          int q_0 = q_time;
          int q_p1 = q_time + 1;
          int q_p2 = q_time + 2;
          if (q_m1 >= 0 && q_m1 < num_time_points) {
            double2 q = Q[((size_t)q_m1)*num_lms + i_lm];
            q_re += w_m1 * q.x;
            q_im += w_m1 * q.y;
          }
          if (q_0 >= 0 && q_0 < num_time_points) {
            double2 q = Q[((size_t)q_0)*num_lms + i_lm];
            q_re += w_0 * q.x;
            q_im += w_0 * q.y;
          }
          if (q_p1 >= 0 && q_p1 < num_time_points) {
            double2 q = Q[((size_t)q_p1)*num_lms + i_lm];
            q_re += w_p1 * q.x;
            q_im += w_p1 * q.y;
          }
          if (q_p2 >= 0 && q_p2 < num_time_points) {
            double2 q = Q[((size_t)q_p2)*num_lms + i_lm];
            q_re += w_p2 * q.x;
            q_im += w_p2 * q.y;
          }
          double2 a = A[sample_idx*num_lms + i_lm];
          out_re += a.x*q_re - a.y*q_im;
          out_im += a.x*q_im + a.y*q_re;
        }

        out[i_output] = make_double2(out_re, out_im);
      }
    }
  } // Q_inner_cubic
} // extern
