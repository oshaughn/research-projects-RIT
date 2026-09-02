
RIFT asimov interface, attempting plugin form.
Based on 
* https://git.ligo.org/deanna.fernando/asimov/-/blob/review/asimov/configs/rift.ini
* https://git.ligo.org/deanna.fernando/asimov/-/blob/review/asimov/pipelines/rift.py?ref_type=heads

See related documentation and examples in 
* https://asimov.docs.ligo.org/asimov/master/pipelines-dev.html
* https://git.ligo.org/asimov/pipelines/gwdata/-/blob/master/datafind/asimov.py

Rimsky integration
------------------

The ``rift-rimsky-analysis`` command generates a RIFT follow-up document for
Rimsky's ``sample_sink.asimov_configuration`` hook. It bootstraps from the
PESummary metafile produced by Rimsky's online Bilby analysis and normalizes
Rimsky's underscore-separated prior names for the RIFT template. See
``RIFT/rimsky/README.md`` for configuration and operational details.
