GWSignal Interface
===================

The ``GWSignal.py`` module acts as a wrapper around the ``gwsignal`` library. It is designed to avoid some of the stdout noise and slow import times associated with the base library while providing a clean interface for generating both time-domain waveforms and their harmonic decompositions.

Overview
--------

This interface allows RIFT to leverage the wide array of waveform models available in the ``gwsignal`` ecosystem. It handles the conversion between RIFT's internal parameter structures (``ChooseWaveformParams``) and the Python dictionaries required by the ``gwsignal`` generators.

Core Functions
---------------

**Harmonic Decomposition**

- ``hlmoft(P, Lmax=2, ...)``: The core function for generating harmonic modes. It takes a parameter set ``P`` and returns a dictionary of LAL ``COMPLEX16TimeSeries`` objects for each $(l, m)$ mode up to ``Lmax``.
    - It handles the necessary coordinate conversions (e.g., masses to solar masses).
    - It applies required tapering and phase shifts.
    - It can optionally adjust the waveform epoch based on the peak of the signal.
- ``hlmoff(P, ...)``: A convenience wrapper around ``hlmoft`` that returns Fourier-transformed harmonic modes.
- ``std_and_conj_hlmoff(P, ...)``: Returns both the Fourier-transformed modes and their complex conjugates.

**Time-Domain Waveforms**

- ``hoft(P, ...)``: Generates a real-valued time-domain waveform. It handles the projection of the $h_+$ and $h_\times$ polarizations onto a specific detector strain based on the sky location and orientation.
- ``complex_hoft(P, ...)``: Generates a complex-valued time-domain waveform ($h_+ + i h_\times$), which is useful for certain internal RIFT calculations.

Implementation Details
------------------------

- **Coordinate Handling**: The module carefully converts RIFT's internal units to the units expected by ``gwsignal`` (e.g., using ``astropy.units``).
- **Tapering**: Since ``gwsignal`` output is often not tapered, the module manually applies a cosine taper to the start of the segments to avoid spectral leakage.
- **Epoch Alignment**: The module supports an ``align_method='peak'`` option, which shifts the waveform epoch to coincide with the peak of the total power $\sum |h_{lm}|^2$.

Backend options
---------------

With ``--use-gwsignal``, ILE passes the complete dictionary supplied to
``--internal-waveform-extra-kwargs`` to the selected waveform generator. For
example, the production SEOBNRv5PHM settings use::

    --use-gwsignal --approx SEOBNRv5PHM \
    --internal-waveform-extra-kwargs "{'antisymmetric_modes_hm': True, 'enable_antisymmetric_modes': True, 'lmax_nyquist': 1}"

No list of approved backend options is maintained in RIFT. Nested dictionaries,
false values and zero values are preserved, and errors from the generator are
allowed to propagate. A generator can still reject unsupported options or
ignore them according to its own interface; RIFT does not validate a plugin's
option schema.

The explicit dictionary is kept separate from RIFT alignment and conditioning
controls, including ``--internal-waveform-fd-L-frame``. It takes precedence over
``--use-gwsignal-lmax-nyquist``. Direct Python callers can use
``gwsignal_extra_kwargs={...}`` with ``hlmoft``, ``hoft`` or ``complex_hoft`` for
the same behavior. Legacy direct keyword arguments are forwarded too, except
for known RIFT wrapper controls.

Masses, spins, sampling and other physical parameters must be set on
``ChooseWaveformParams``. Backend options that override those translated fields
raise ``ValueError`` to prevent disagreement with likelihood parameters or
series metadata. ``condition`` may override the taper-derived backend
conditioning value. RIFT's own tapering and alignment of returned modes still
apply. Plain nested dictionaries, lists and tuples are copied before generation;
opaque backend objects are passed through unchanged.

``--internal-waveform-extra-lalsuite-args`` targets the native LAL-dictionary
route, not GWSignal. ILE rejects its combination with ``--use-gwsignal`` before
loading detector data, including an explicitly supplied empty dictionary.
Use ``--internal-waveform-extra-kwargs`` for GWSignal backend options. Direct
GWSignal callers receive an error for a nonempty legacy ``extra_waveform_args``
payload; an explicitly namespaced backend option with that name remains allowed.

The same dictionary separation is preserved when pseudo-pipeline arguments are
forwarded to calibration reweighting. ``--extra-waveform-kwargs`` in that
executable is a wrapper-options dictionary; its ``gwsignal_extra_kwargs`` member
holds explicit backend options. The pseudo-pipeline constructs this member from
``--internal-waveform-extra-kwargs`` when GWSignal is selected.

Asimov's waveform dictionaries are quoted as shell arguments. BasicIteration
preserves their nested list syntax and backslashes, and the ILE submit writer
translates complete shell argument boundaries into HTCondor's new argument
syntax. Multiple dictionaries and quoted string values are supported. Literal
line breaks inside an individual submit argument are rejected.
