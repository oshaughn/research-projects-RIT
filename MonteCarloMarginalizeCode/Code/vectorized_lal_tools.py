import numpy
import cupy

def TimeDelayFromEarthCenter(
        detector_earthfixed_xyz_metres,
        source_right_ascension_radians,
        source_declination_radians,
        greenwich_mean_sidereal_time,
        xpy=cupy, dtype=numpy.float64,
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
    negative_speed_of_light = xpy.asarray(-2.99792e8)

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
