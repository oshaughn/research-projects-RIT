rotations
==========

The ``RIFT.misc.sky_rotations`` module provides utilities for rotating sky coordinates between the global celestial frame and a coordinate system defined by a pair of detectors (typically Hanford H1 and Livingston L1). This alignment is crucial for computing the detector response for a given gravitational-wave source location.

The Rotation Framework
----------------------

To avoid the overhead of recalculating rotation matrices during every iteration of a likelihood sampler, the module uses a global state to store the current rotation frame and its inverse.

The ``assign_sky_frame`` function initializes this state by:
1. Calculating the separation vector between two specified detectors.
2. Determining the Greenwich Mean Sidereal Time for a given fiducial epoch.
3. Constructing an orthonormal rotation matrix (``frm``) and its inverse (``frmInverse``) that maps the global z-axis to the detector separation axis.

Coordinate Transformations
--------------------------

The module provides functions to transform polar angles ($\theta, \phi$) between frames:

1. **Forward Rotation**: The ``rotate_sky_forwards_scalar`` function transforms coordinates from the global frame to the detector frame. This is used when the source location is given in celestial coordinates, but the detector response must be computed in the local frame.
2. **Backward Rotation**: The ``rotate_sky_backwards_scalar`` function performs the inverse transformation, mapping coordinates from the detector frame back to the global frame.

These scalars are wrapped by ``rotate_sky_forwards`` and ``rotate_sky_backwards`` to handle arrays of coordinates.

Implementation Notes
--------------------

- **Efficiency**: The module relies on ``RIFT.lalsimutils.polar_angles_in_frame_alt`` for the underlying spherical trigonometry.
- **Performance**: The current implementation is non-vectorized, performing transformations in a loop. For large-scale coordinate transformations, using a fully vectorized library like ``astropy`` is recommended.

API Reference
-------------

.. automodule:: RIFT.misc.sky_rotations
    :members:
    :undoc-members:
    :show-inheritance:
