.. _rom_waveform_manager:

ROMWaveformManager
===================

.. module:: ROMWaveformManager
.. automodule:: ROMWaveformManager
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``ROMWaveformManager`` module provides an interface for managing Reduced Order Model (ROM) gravitational waveform surrogates. It supports various surrogate models including NRSur4d, NRSur7dq4, NRHybSur, and others for generating inspiral-merger-ringdown gravitational wave signals.

The module handles:
- Loading and managing surrogate models
- Mode-by-mode basis truncation
- Reflection symmetry handling for computational efficiency
- Parameter conversion between different coordinate systems

Classes
-------

.. class:: ROMWaveformManager

   Main class for managing ROM waveform surrogates.

   .. method:: __init__(self, param, dirBaseFiles='/Users/rossma/catalog/public/NR/Surrogates', group='NRSur7dq2', lmax=4, mode_list_to_load=None, max_nbasis_per_mode=None, reflection_symmetric=True, rosDebug=False)

      Initialize the ROM waveform manager with a specified surrogate model.

      :param param: Surrogate model name (e.g., 'NRSur4d', 'NRSur7dq4', 'NRHybSur3dq8', 'NRHybSur3dq8Tidal')
      :type param: str
      :param dirBaseFiles: Base directory containing surrogate model files
      :type dirBaseFiles: str
      :param group: Surrogate group name
      :type group: str
      :param lmax: Maximum spherical harmonic mode order
      :type lmax: int
      :param mode_list_to_load: Specific modes to load (overrides default mode set)
      :type mode_list_to_load: list of tuples or None
      :param max_nbasis_per_mode: Maximum number of basis functions per mode (for truncation)
      :type max_nbasis_per_mode: int or None
      :param reflection_symmetric: Enable reflection symmetry for mode loading
      :type reflection_symmetric: bool
      :param rosDebug: Enable debug output
      :type rosDebug: bool

      The constructor loads the specified surrogate model and sets up mode dictionaries. Supported surrogate models include:

      - ``NRSur4d``: Precessing spin surrogate
      - ``NRSur7dq4``: Full precessing spin surrogate
      - ``NRHybSur3dq8``: Hybrid surrogate for aligned spins
      - ``NRHybSur3dq8Tidal``: Hybrid surrogate with tidal extensions

   .. method:: print_params(self)

      Print available modes and their associated basis sizes.

      Prints to stdout the list of available spherical harmonic modes and the number of basis functions for each mode in the loaded surrogate model.

   .. method:: complex_hoft(self, P, force_T=False, deltaT=1./16384, time_over_M_zero=0., sgn=-1)

      Generate a complex-valued time-domain waveform h(t) summed over available modes.

      :param P: Waveform parameters containing mass ratio, spins, etc.
      :type P: ChooseWaveformParams
      :param force_T: If float, forces a specific time length; if True, uses default duration
      :type force_T: float or bool
      :param deltaT: Time step for the waveform
      :type deltaT: float
      :param time_over_M_zero: Reference time offset in units of total mass
      :type time_over_M_zero: float
      :param sgn: Sign convention for the waveform (+1 or -1)
      :type sgn: int
      :returns: Complex strain waveform h(t) = h_plus - i*h_cross
      :rtype: complex array

      This method evaluates the surrogate model at the specified parameters and returns the complex strain. The waveform is constructed by summing over all available (l, m) modes.

Internal Data Structures
------------------------

The ROMWaveformManager uses several internal dictionaries to manage mode data:

- ``self.sur_dict``: Dictionary mapping (l, m) modes to surrogate objects
- ``self.post_dict``: Post-processing functions for each mode
- ``self.post_dict_complex``: Complex conjugation functions
- ``self.parameter_convert``: Parameter conversion functions for each mode
- ``self.nbasis_per_mode``: Number of basis functions per mode
- ``self.modes_available``: List of available (l, m) modes

Parameter Conversion
--------------------

The module includes several parameter conversion functions:

- ``ConvertWPtoSurrogateParams``: Standard parameter conversion
- ``ConvertWPtoSurrogateParamsPrecessing``: For precessing spin models
- ``ConvertWPtoSurrogateParamsAligned``: For aligned-spin models

These convert from internal parameter representations to the coordinate system expected by the surrogate models.

Examples
--------

Basic usage to generate a waveform::

    from ROMWaveformManager import ROMWaveformManager
    import lalsimutils

    # Initialize ROM with NRSur7dq4 surrogate
    rom = ROMWaveformManager('NRSur7dq4', lmax=4)

    # Set waveform parameters
    P = lalsimutils.ChooseWaveformParams()
    P.m1 = 30.0
    P.m2 = 30.0
    P.s1z = 0.5
    P.s2z = -0.3

    # Generate waveform
    h = rom.complex_hoft(P)

Loading specific modes::

    # Load only specific modes (l=2, m=±2)
    rom = ROMWaveformManager('NRSur7dq4', 
                            mode_list_to_load=[(2, 2), (2, -2)],
                            lmax=2)

Truncating basis for efficiency::

    # Limit basis size for faster evaluation
    rom = ROMWaveformManager('NRSur7dq4', 
                            max_nbasis_per_mode=50)

Notes
-----

- The module requires the ``gwsurrogate`` package for loading and evaluating surrogate models
- Reflection symmetry is enabled by default, which automatically generates negative-m modes from positive-m modes
- The time array is normalized such that t=0 corresponds to the peak of the (2,2) mode
- For precessing models, the waveform is returned in the inertial frame

See Also
--------

- :ref:`gwsignal` - Alternative waveform generation interface
- :ref:`lalsimutils` - Parameter management utilities