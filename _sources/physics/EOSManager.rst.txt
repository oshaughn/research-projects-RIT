EOS Manager
------------

The ``EOSManager.py`` module provides tools for handling Neutron Star Equations of State (EOS). It allows RIFT to incorporate various EOS models, from standard LALSimulation tables to parameterized spectral decompositions and tabular data files.

Overview
--------

The EOS manager is designed to provide a consistent interface for querying critical neutron star properties, such as the tidal deformability $\Lambda$ and the mass-radius relationship, regardless of the underlying EOS representation.

Core Classes
-------------

**EOSConcrete**
The base class for all EOS implementations. It provides common utility methods:
- ``lambda_from_m(m)``: Computes the dimensionless tidal deformability $\Lambda$ for a given gravitational mass.
- ``pressure_density_on_grid(logrho_grid)``: Extracts the pressure-density relationship on a specified grid.
- ``test_speed_of_sound_causal()``: Verifies that the sound speed does not exceed the speed of light.

**EOSLALSimulation**
Implements EOS using the standard LALSimulation library. It loads EOS by name and creates the associated neutron star family.

**EOSFromTabularData** / **EOSFromDataFile**
Handles EOS defined by tabular data (e.g., baryon density, pressure, energy density). It can load data from files and convert them into a format compatible with LALSimulation.

**EOSPiecewisePolytrope**
Implements a 4-parameter piecewise polytrope EOS, allowing for flexible parameterization of the neutron star interior.

**EOSLindblomSpectral**
Implements the spectral decomposition of the EOS, providing a mathematically rigorous way to parameterize the EOS while ensuring stability and causality.

**EOSReprimand**
A specialized interface to the ``RePrimAnd`` library, which provides robust recovery of primitive variables and high-precision TOV solver results.

Implementation Details
------------------------

- **Unit Conversions**: The module handles complex conversions between CGS, SI, and geometerized units.
- **Monotonicity Checks**: Includes utilities to ensure that the EOS is monotonic in pressure and density, which is a requirement for physical stability.
- **Baryon Mass Estimation**: Provides methods to estimate the baryon mass from the gravitational mass using established approximations.
