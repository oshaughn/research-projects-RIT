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

In Asimov metadata, ``waveform.gwsignal arguments`` becomes
``--internal-waveform-extra-kwargs`` and requires
``scheduler.pipeline.use-gwsignal: true`` to select GWSignal.
``waveform.arguments`` becomes ``--internal-waveform-extra-lalsuite-args``.
If both are supplied, only ``waveform.gwsignal arguments`` is emitted, as in
earlier O4c releases. ``sampler.ile.manual extra args`` accepts a list or a single string.

Native SEOB mode conditioning
-----------------------------

The optional ``--gwsignal-mode-conditioning native`` selector uses native
SEOBNRv5HM/PHM conditioning on shared modes before padding and projection.
``legacy`` remains the default. Native conditioning requires GWSignal, a
positive explicit reference frequency, enabled tapering and backend
conditioning 2. It refuses ``--internal-waveform-fd-no-condition`` and
projection-based epoch realignment. The selector is a RIFT wrapper control;
it is not a member of ``gwsignal arguments``.

An Asimov configuration can select it without additional backend switches::

    waveform:
      approximant: SEOBNRv5PHM
      reference frequency: 20
      gwsignal mode conditioning: native
    scheduler:
      pipeline:
        use-gwsignal: true

Asimov forwards the same selection through ordinary ILE, extrinsic jobs and
calibration reweighting. Omitting it preserves existing defaults. The public
selector and default match O4d batch and JAX ILE; O4c has no JAX ILE executable.
