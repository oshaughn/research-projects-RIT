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
} // extern
