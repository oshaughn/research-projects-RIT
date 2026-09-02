# Rimsky integration

Rimsky performs online Bilby parameter estimation and can launch follow-up
analyses through its Asimov hook.  RIFT supplies a bridge for that hook:

1. `rift-rimsky-analysis rimsky.yaml rift-followup.yaml` reads the Rimsky
   configuration and writes both a RIFT Asimov analysis document and a runnable
   `rimsky-rift.yaml`.
2. Initialize an Asimov 0.7 project named by `asimovdir` once, install RIFT in its
   environment, and run `rimsky rimsky-rift.yaml`.

The generated Rimsky configuration defaults `event_sink.bilby_pipe_format` to
`full-submit`, points `sample_sink.asimov_configuration` at the generated RIFT
analysis, and makes relative output paths absolute.  Thus the first online
Bilby result is written as a PESummary metafile and Rimsky immediately adds the
ready RIFT follow-up to Asimov.  A running Asimov manager then builds and submits
that production.  Existing explicit Bilby run modes and Asimov project paths
are preserved.  Use `--configured-rimsky PATH` to choose a different filename.

For example:

```yaml
output_dir: ./output
asimovdir: ./asimov
detectors: [H1, L1, V1]

sample_sink:
  # Written automatically in rimsky-rift.yaml.
  asimov_configuration: /absolute/path/to/rift-followup.yaml

# Optional. Rimsky ignores this extra section; the RIFT generator consumes it.
rift:
  name: rift-online
  waveform:
    approximant: IMRPhenomXPHM
  scheduler:
    accounting group: ligo.dev.o4.cbc.pe.rift
    osg: false
```

Rimsky writes a PESummary metafile before applying the follow-up file.  The
generated analysis uses an absolute `output_dir/*/*/{event}/...` glob to find
that event's metafile, sets its dataset to `bilby-online`, and bootstraps RIFT
and its coincidence XML from the online posterior. Exactly one metafile must match; RIFT fails closed
if the path is missing or ambiguous.

Rimsky 0.1 event documents use Bilby-style prior names (`chirp_mass`,
`mass_ratio`, `a_1`, and so on).  The RIFT pipeline retains those keys and adds
the space-separated aliases expected by its Asimov template.  This makes the
same event usable by both Bilby and RIFT analyses.

The bridge itself consumes plain YAML mappings and does not import Rimsky.  Its
unit tests remain isolated from streaming, GraceDB, and HTCondor.  Dedicated
end-to-end lanes install Rimsky `0.1.0rc1` on Python 3.12 and pinned current main
on Python 3.14, both forced onto Asimov 0.7 and the merged bilby_pipe 0.7 adapter.
They load the generated configuration through Rimsky, invoke its real post-PE
Asimov hook, discover the RIFT pipeline, and resolve the first metafile as the
bootstrap input. External scheduler submission is the only mocked boundary.
The current-main pin is commit `2621d15` (2026-09-01); the bilby_pipe adapter pin
is `be6c770` pending its next release.
