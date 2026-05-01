Monotonic Spline Interpolation
=============================

This module provides monotonic cubic spline interpolation, based on the algorithm from [Rizzo, 2014](https://uglyduckling.nl/library_files/PRE-PRINT-UD-2014-03.pdf).

Overview
--------

Unlike standard cubic splines, monotonic interpolating splines **preserve the monotonicity** of the data. This means the interpolant will not introduce artificial oscillations between data points, which is crucial for physical quantities like:

- **Tidal deformabilities**: Must be non-negative and monotonically related to EOS parameters
- **Phase functions**: Must be monotonically increasing (or decreasing) in frequency
- **Cumulative distributions**: Must be non-decreasing

Standard cubic splines can wiggle between data points even when the underlying function is monotonic, making them unsuitable for these applications.

The Algorithm
-------------

The implementation uses a **monotonicity-preserving cubic interpolation** scheme:

1. Compute slopes $m_i$ between adjacent points
2. At points where $m_{i-1} \cdot m_i \leq 0$, set the derivative to zero (prevents local extrema)
3. For points with same-sign slopes, use a weighted average based on the harmonic mean
4. Construct the cubic polynomial coefficients from the derivatives

This ensures the resulting spline has no interior extrema, maintaining monotonicity throughout.

API Reference
-------------

interpolate(x_values, y_values)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Computes the spline coefficients for a monotonic cubic interpolant.

**Parameters:**
- ``x_values`` (array): X-coordinates of data points
- ``y_values`` (array): Y-coordinates of data points

**Returns:**
- ``consts`` (ndarray): Array of shape (n, 3) containing cubic coefficients for each interval

**Example:**

.. code-block:: python

    import numpy as np
    from RIFT.physics import MonotonicSpline as ms

    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    y = np.array([0.0, 0.6, 1.0, 1.3, 1.5])  # monotonically increasing

    consts = ms.interpolate(x, y)

lin_extrapolate(x_values, y_values)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Computes linear extrapolation coefficients for data outside the original range.

**Parameters:**
- ``x_values`` (array): X-coordinates of data points
- ``y_values`` (array): Y-coordinates of data points

**Returns:**
- ``line_consts`` (ndarray): Array of shape (2, 2) with [[m1, b1], [m2, b2]] where the first row is the fit to the start and the second row is the fit to the end

**Example:**

.. code-block:: python

    line_consts = ms.lin_extrapolate(x, y)
    # line_consts[0] = [slope_at_start, intercept_at_start]
    # line_consts[1] = [slope_at_end, intercept_at_end]

interp_func(x, x_table, y_table, consts, line_consts=None, verbose=False)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluates the interpolant (and optional extrapolant) at a given point.

**Parameters:**
- ``x`` (float): Point at which to evaluate
- ``x_table`` (array): X-coordinates passed to ``interpolate``
- ``y_table`` (array): Y-coordinates passed to ``interpolate``
- ``consts`` (ndarray): Coefficients returned by ``interpolate``
- ``line_consts`` (ndarray, optional): Extrapolation coefficients from ``lin_extrapolate``
- ``verbose`` (bool): If True, prints input and output values

**Returns:**
- ``float``: Interpolated (or extrapolated) value

**Example:**

.. code-block:: python

    consts = ms.interpolate(x, y)
    line_consts = ms.lin_extrapolate(x, y)

    # Interpolate within range
    y_at_0_7 = ms.interp_func(0.7, x, y, consts)

    # Extrapolate outside range
    y_at_neg_0_5 = ms.interp_func(-0.5, x, y, consts, line_consts)
    y_at_2_5 = ms.interp_func(2.5, x, y, consts, line_consts)

Usage Example: Tidal EOS Interpolation
-------------------------------------

Monotonic splines are particularly useful for interpolating equation of state (EOS) data:

.. code-block:: python

    import numpy as np
    from RIFT.physics import MonotonicSpline as ms

    # Example: Pressure as a function of density (must be monotonic)
    density = np.array([1e14, 2e14, 5e14, 1e15, 2e15])  # g/cm^3
    pressure = np.array([1e32, 5e33, 3e34, 1e35, 4e35])  # dyn/cm^2

    # Create interpolant
    consts = ms.interpolate(density, pressure)
    line_consts = ms.lin_extrapolate(density, pressure)

    # Evaluate at intermediate densities
    p_at_3e14 = ms.interp_func(3e14, density, pressure, consts)

Comparison with Alternatives
--------------------------

+----------------------+---------------+------------------------+
| Feature              | Monotonic     | Standard Cubic Spline   |
|                      | Spline        |                        |
+======================+===============+========================+
| Preserves monotonicity| Yes          | No (can oscillate)     |
| Preserves sign       | Yes          | No                     |
| Computation cost     | Similar      | Similar                |
| Requires sorted data | Yes          | Yes                    |
+----------------------+---------------+------------------------+

For RIFT applications where physical monotonicity must be preserved (tidal parameters, mass functions, etc.), use this module rather than standard scipy interpolation.
