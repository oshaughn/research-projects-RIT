# Frozen pre-port O4c 28b1f71d9 numerical reference.
# Loaded with factored_likelihood globals by the adversarial tests.
def _DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopUnchunked(tvals, P_vec, lookupNKDict, rholmsArrayDict, ctUArrayDict,ctVArrayDict,epochDict,Lmax=2,array_output=False,xpy=np, loglikelihood=_factored_lnL_helper,return_lnLt=False,phase_marginalization=False,time_interp='nearest',time_quadrature='simpson',return_time_draw=False,time_draw_uniforms=None,time_draw_minimum_srate=None):
    """
    DiscreteFactoredLogLikelihoodViaArray uses the array-ized data structures to compute the log likelihood,
    either as an array vs time *or* marginalized in time.
    This generally is marginally faster, particularly if Lmax is large.
    The timeseries quantities are computed via discrete shifts of an existing grid
    Note 'P' must have the *sampling rate* set to correctly interpret the event time.
     Note arguments passed are NOW ARRAYS, in contrast to similar function which does not have 'Vector' postfix

    time_quadrature : {'simpson', 'bandlimited'}
        Opt-in reflected finite-window FFT reconstruction and adaptive dense
        trapezoid quadrature, as on rift_O4d. Reflection mitigates endpoint
        ringing; it is not exact recovery of the unavailable full correlation.
        The callback is evaluated on the dense grid, so cost grows with SNR.

    return_time_draw : bool
        With bandlimited quadrature, return (continuous time offset, lnL at
        draw), using the validated refined conditional posterior. Mutually
        exclusive with return_lnLt.
    time_draw_minimum_srate : float or None
        Minimum knot rate for continuous export; never changes the integral.

    time_interp : {'nearest', 'cubic'}
        Detector-time sampling convention for the data term.  'nearest'
        preserves the historical NoLoop integer-bin gather.  'cubic' evaluates
        the precomputed Q_lm time series at the fractional detector arrival time
        using a four-sample cubic Lagrange stencil, with zero extension outside
        the precomputed buffer.
    """
    global distMpcRef

    time_quad.validate_time_quadrature(time_quadrature)
    if return_time_draw and return_lnLt:
        raise ValueError('return_time_draw and return_lnLt are mutually exclusive')
    if return_time_draw and time_quadrature != 'bandlimited':
        raise ValueError("return_time_draw requires time_quadrature='bandlimited'")
    if time_interp not in ('nearest', 'cubic'):
        raise ValueError("time_interp must be 'nearest' or 'cubic'")

    detectors = rholmsArrayDict.keys()
    npts = len(tvals)
    npts_extrinsic = len(P_vec.phi)
    # npts_full = len(rholmsArrayDict[detectors[0]][0]) # all have same size
    # print " npts :", npts
    # print " npts_full:", npts_full

    # All arrays of length `npts_extrinsic`, except for `tref` which is a scalar
    RA = P_vec.phi
    DEC = P_vec.theta

    # geocenter time, stored as a scalar
    tref = P_vec.tref
    phiref = P_vec.phiref
    incl = P_vec.incl
    psi = P_vec.psi
    dist = P_vec.dist
    distMpc = dist/(lal.PC_SI*1e6)
    invDistMpc = distMpcRef/distMpc


    deltaT = float(P_vec.deltaT) # this is stored as a scalar


    # Convert tref to greenwich mean sidereal time
    greenwich_mean_sidereal_time_tref = xpy.asarray(
        lal.GreenwichMeanSiderealTime(tref)
    )

    # this is stored as a scalar
    deltaT = P_vec.deltaT

    # Array to accumulate lnL(t) summed across all detectors.
#    lnL_t_accum = xpy.zeros((npts_extrinsic, npts), dtype=np.float64)

    # Used to accumulate kappa^2 and rho^2 over all detectors.  They are just
    # the sum in quadrature of the individual detector contributions.
    kappa_sq = xpy.zeros((npts_extrinsic, npts), dtype=np.complex128)
    rho_sq = xpy.zeros((npts_extrinsic, npts), dtype=np.float64)

    if (xpy is np) or (optimized_gpu_tools is None):
        simps = my_simps
    elif not (xpy is np):
        simps = optimized_gpu_tools.simps
    else:
        raise NotImplementedError("Backend not supported: {}".format(xpy))

    # strings right now - need to change to make ufunc-able
    for det in detectors:
        # Compute the detector's location and response matrix
        detector = lalsim.DetectorPrefixToLALDetector(det)
        detector_location = xpy.asarray(detector.location)
        detector_response = xpy.asarray(detector.response)

        # These do not depend on extrinsic params.
        # Arrays of shape (n_lms, n_lms).
        # Axis 0 corresponds to (l,m), and axis 1 corresponds to (l',m').
        U = ctUArrayDict[det]
        V = ctVArrayDict[det]

        lms = lookupNKDict[det]
        n_lms = len(lms)

        # These do depend on extrinsic params
        # Array of shape (npts_extrinsic, n_lms,)
        Ylms_vec = SphericalHarmonicsVectorized(
            lms, incl, -phiref,
            xpy=xpy,
            l_max=Lmax,
        )

        # Array of shape (npts_extrinsic,)
#        F_vec_old = xpy.asarray(lalF(det, RA, DEC, psi, tref))
        F_vec = ComputeDetAMResponse(
            detector_response,
            RA, DEC, psi,
            greenwich_mean_sidereal_time_tref,
            xpy=xpy
        )

        # Scalar -- is constant for each IFO
        t_ref = epochDict[det]

        # This is the GPS time at the detector,
        # Note that to save on precision compared to ...NoLoopOrig, we CHANGE the t_det definition to be relative to the IFO statt time t_ref
        #    ... this means we don't keep a 1e9 out in front, so we have more significant digits in the event time (and can if needed reduce precision in GPU ops)
        # an array of shape (npts_extrinsic,)
        t_det = float(tref - float(t_ref)) + TimeDelayFromEarthCenter(
            detector_location, RA, DEC,
            float(greenwich_mean_sidereal_time_tref),
            xpy=xpy
        )
        tfirst = t_det + tvals[0]

        sample_first = tfirst / deltaT
        if time_interp == 'nearest':
            ifirst = (xpy.rint(sample_first) + 0.5).astype(np.int32)  # C uses 32 bit integers : be careful
            frac_first = None
        else:
            ifirst = xpy.floor(sample_first).astype(np.int32)
            frac_first = (sample_first - xpy.floor(sample_first)).astype(np.float64)
#        ilast = ifirst + npts


        # Has shape (npts_extrinsic,)
        rho_sq_det = (
            (F_vec*xpy.conj(F_vec)).real *
            xpy.einsum(
                "...i,...j,ij",
                xpy.conj(Ylms_vec), Ylms_vec, U,
            ).real
        )

        rho_sq_det += (
            xpy.square(F_vec) *
            xpy.einsum(
                "...i,...j,ij",
                Ylms_vec, Ylms_vec, V,
            )
        ).real
        # NOTE: Double-check this 0.5 is correct.  It used to be -0.25, but
        # we're slightly reorganizing the expressions from term2 = -0.5 rho_sq
        # into using rho_sq directly
        rho_sq_det *= 0.5 * xpy.square(distMpcRef / distMpc)

        # If phase_marginalization is turned on, the (2, -2) term should be
        # replaced by its complex conjugate before the absolute value of
        # kappa_sq is calculated
        if phase_marginalization:
            Ylms_vec[:, 1] = xpy.conj(Ylms_vec[:, 1])
            Q = xpy.ascontiguousarray(
                xpy.concatenate(
                    [
                        rholmsArrayDict[det][0][..., np.newaxis],
                        xpy.conj(rholmsArrayDict[det][1][..., np.newaxis])
                    ],
                    axis=1
                )
            )
            F_vec_dummy_lm = xpy.concatenate(
                [F_vec[..., np.newaxis], xpy.conj(F_vec[..., np.newaxis])],
                axis=1
            )
        else:
            Q = xpy.ascontiguousarray(rholmsArrayDict[det].T)
            # # Note: Very inefficient, need to avoid making `Qlms` by doing the
            # # inner product in a CUDA kernel.
            # det_rholms = xpy.asarray(rholmsArrayDict[det])
            # Qlms = xpy.empty((npts_extrinsic, npts, n_lms), dtype=complex)
            # for i in range(npts_extrinsic):
            #     Qlms[i] = det_rholms[...,ifirst[i]:ilast[i]].T

            # Has shape (npts_extrinsic, npts).
            # Starts as term1, and accumulates term2 after.

            # View into F with shape (npts_extrinsic, n_lms)
            F_vec_dummy_lm = F_vec[..., np.newaxis]
            # # View into F * Ylm with shape (npts_extrinsic, npts, n_lms)
            # FY_dummy_t = xpy.broadcast_to(
            #     (F_vec_dummy_lm * Ylms_vec)[:, np.newaxis],
            #     Qlms.shape,
            # )

            # lnL_t_accum += xpy.einsum(
            #     "...i,...i",
            #     xpy.conj(FY_dummy_t), Qlms,
            # ).real * (distMpcRef/distMpc)[...,None]

        if not (xpy is np):
          FY_conj = xpy.conj(F_vec_dummy_lm * Ylms_vec)
          # Shape Q = (npts_time_full, nlms)
          # Shape A=FY_conj = (npts_extrinsic, nlms)
          # shape result = (npts_extrinsic, npts_time_*window* = npts)
          if time_interp == 'nearest':
            Q_prod_result = Q_inner_product.Q_inner_product_cupy(
              Q, FY_conj,
              ifirst, npts,
              )
          else:
            Q_prod_result = Q_inner_product.Q_inner_product_cubic_cupy(
              Q, FY_conj,
              ifirst, frac_first, npts,
              )
        else:
          Q_block = rholmsArrayDict[det].T
          if time_interp == 'nearest':
              Qlms = _nearest_Q_window_numpy(Q_block, ifirst, npts, xpy=xpy)
          else:
              Qlms = _cubic_Q_window_numpy(Q_block, ifirst, frac_first, npts)
          if phase_marginalization:
              Qlms[:, :, 1] = xpy.conj(Qlms[:, :, 1])

          FY_dummy_t = np.broadcast_to(
            (F_vec_dummy_lm * Ylms_vec)[:, np.newaxis],
            Qlms.shape,
            )

          Q_prod_result =  np.einsum(
            "...i,...i",
            np.conj(FY_dummy_t), Qlms,
            )

        kappa_sq += Q_prod_result * (distMpcRef/distMpc)[..., np.newaxis]
        # lnL_t_accum += Q_prod_result * (distMpcRef/distMpc)[...,None]

        # lnL_t_accum += Q_inner_product.Q_inner_product_cupy(
        #     FY_conj, Q,
        #     ifirst, npts,
        # ).real * (distMpcRef/distMpc)[...,None]


        # Accumulate term2 into the time-dependent log likelihood.
        # Have to create a view with an extra axis so they broadcast.
        rho_sq += rho_sq_det[..., np.newaxis]
        # lnL_t_accum += term2[..., np.newaxis]

#        print lnL_t_accum.shape, lnL_t.shape

#        lnL_t_accum += lnL_t


    if phase_marginalization:
        lnL_t = loglikelihood(xpy.abs(kappa_sq), rho_sq)
    else:
        lnL_t = loglikelihood(kappa_sq.real, rho_sq)

    # Take exponential of the log likelihood in-place.
    # The offset must be PER EXTRINSIC SAMPLE (axis=-1 is time).  A batch-wide
    # offset underflows exp() to zero across the whole time axis for any row
    # peaking more than ~745 nats below the loudest row in the batch, returning
    # -inf where the likelihood is finite (onset near max lnL ~ 745, rho ~ 40).
    # keepdims=True is load-bearing: a (npts_extrinsic,) offset would broadcast
    # along the time axis instead.  See issue oshaughnessy-junior/
    # research-projects-RIT#232.
    lnLmax  = xpy.max(lnL_t, axis=-1, keepdims=True)
    if return_lnLt:
      return lnL_t  #- lnLmax    # we want the verbatim lnL_t values, no shift

    if time_quadrature == 'bandlimited':
        # Match the current O4d numerical boundary construction and per-row
        # offsets. The gather steps at deltaT, not the linspace spacing in tvals.
        # Pass the caller's Simpson implementation for unrefined rows.
        result = time_quad.time_marginalize_bandlimited(
            kappa_sq, rho_sq, deltaT, loglikelihood,
            phase_marginalization=phase_marginalization, lnL_coarse=lnL_t,
            simps=simps, return_time_draw=return_time_draw,
            draw_uniforms=time_draw_uniforms, t0=float(tvals[0]),
            time_draw_minimum_srate=time_draw_minimum_srate, xpy=xpy)
        if return_time_draw:
            _, drawn_t, drawn_lnL = result
            return drawn_t, drawn_lnL
        return result

    L_t = xpy.exp(lnL_t - lnLmax, out=lnL_t)

    L = simps(L_t, dx=deltaT, axis=-1)

    # Compute log likelihood in-place.
    lnL = lnLmax[...,0] + xpy.log(L, out=L)

    return lnL
