import numpy
try:
    import cupy
    xpy_default=cupy
    junk_to_check_installed = cupy.array(5)  # this will fail if GPU not installed correctly
except:
    xpy_default=numpy



def TimeDelayFromEarthCenter(
        detector_earthfixed_xyz_metres,
        source_right_ascension_radians,
        source_declination_radians,
        greenwich_mean_sidereal_time,
        xpy=xpy_default, dtype=numpy.float64,
    ):
    """

    Parameters
    ----------
    detector_earthfixed_xyz_metres : array_like, shape = det_shape + (3,)
      Location of detector(s) relative to Earth's center in meters. May provide
      multiple detectors, last axis must be (x,y,z) but other axes can take
      whatever form is desired.
    source_right_ascension_radians : array_like, shape = sample_shape
      Right ascension of source in radians, can be an arbitrary dimensional
      array.
    source_declination_radians : array_like, shape = sample_shape
      Declination of source in radians, can be an arbitrary dimensional array.
    greenwich_mean_sidereal_time : float
      Should be equivalent to XLALGreenwichMeanSiderealTime(gpstime).

    Returns
    -------
    time_delay_from_earth_center : array_like, shape = det_shape + sample_shape
    """
    negative_speed_of_light = xpy.asarray(-299792458.0)

    det_shape = detector_earthfixed_xyz_metres.shape[:-1]
    sample_shape = source_right_ascension_radians.shape

    cos_dec = xpy.cos(source_declination_radians)

    greenwich_hour_angle = (
        greenwich_mean_sidereal_time - source_right_ascension_radians
    )

    ehat_src = xpy.empty(sample_shape + (3,), dtype=dtype)

    ehat_src[...,0] = cos_dec * xpy.cos(greenwich_hour_angle)
    ehat_src[...,1] = -cos_dec * xpy.sin(greenwich_hour_angle)
    ehat_src[...,2] = xpy.sin(source_declination_radians)

    neg_separation = xpy.inner(detector_earthfixed_xyz_metres, ehat_src)
    return xpy.divide(
        neg_separation, negative_speed_of_light,
        out=neg_separation,
    )


def ComputeDetAMResponse(
        detector_response_matrix,
        source_right_ascension_radians,
        source_declination_radians,
        source_polarization_radians,
        greenwich_mean_sidereal_time,
        xpy=xpy_default, dtype_real=numpy.float64, dtype_complex=numpy.complex128,
    ):
    """
    Parameters
    ----------
    detector_response_matrix : array_like, shape = det_shape + (3, 3)
      Detector response matrix, or matrices for multiple detectors.  Last two
      axes must be 3-by-3 response matrix, and may include arbitrary axes before
      that for various detectors.
    source_right_ascension_radians : array_like, shape = sample_shape
      Right ascension of source in radians, can be an arbitrary dimensional
      array.
    source_declination_radians : array_like, shape = sample_shape
      Declination of source in radians, can be an arbitrary dimensional array.
    source_polarization_radians : array_like, shape = sample_shape
      Polarization angle of source in radians, can be an arbitrary dimensional
      array.
    greenwich_mean_sidereal_time : float
      Should be equivalent to XLALGreenwichMeanSiderealTime(gpstime).

    Returns
    -------
    F : array_like, shape = det_shape + sample_shape
    """
    det_shape = detector_response_matrix.shape[:-1]
    sample_shape = source_right_ascension_radians.shape
    matrix_shape = 3, 3

    # Initialize trig matrices.
    X = xpy.empty(sample_shape+(3,), dtype=dtype_real)
    Y = xpy.empty(sample_shape+(3,), dtype=dtype_real)

    # Greenwich hour angle of source in radians.
    source_greenwich_radians = (
        greenwich_mean_sidereal_time - source_right_ascension_radians
    )

    # Pre-compute trig functions
    cos_gha = xpy.cos(source_greenwich_radians)
    sin_gha = xpy.sin(source_greenwich_radians)
    cos_dec = xpy.cos(source_declination_radians)
    sin_dec = xpy.sin(source_declination_radians)
    cos_psi = xpy.cos(source_polarization_radians)
    sin_psi = xpy.sin(source_polarization_radians)

    # Populate trig matrices.
    X[...,0] = -cos_psi*sin_gha - sin_psi*cos_gha*sin_dec
    X[...,1] = -cos_psi*cos_gha + sin_psi*sin_gha*sin_dec
    X[...,2] =  sin_psi*cos_dec

    Y[...,0] =  sin_psi*sin_gha - cos_psi*cos_gha*sin_dec
    Y[...,1] =  sin_psi*cos_gha + cos_psi*sin_gha*sin_dec
    Y[...,2] =  cos_psi*cos_dec

    # Compute F for each polarization state.
    F_plus = (
        X*xpy.inner(X, detector_response_matrix) -
        Y*xpy.inner(Y, detector_response_matrix)
    ).sum(axis=-1)
    F_cross = (
        X*xpy.inner(Y, detector_response_matrix) +
        Y*xpy.inner(X, detector_response_matrix)
    ).sum(axis=-1)

    return F_plus + 1.0j*F_cross
