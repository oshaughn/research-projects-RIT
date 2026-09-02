# Rimsky integration

Rimsky performs online Bilby parameter estimation and can launch follow-up
analyses through its Asimov hook.  RIFT supplies a bridge for that hook:

1. `rift-rimsky-analysis rimsky.yaml rift-followup.yaml` reads the Rimsky
   configuration and writes a RIFT Asimov analysis document.
2. Set `sample_sink.asimov_configuration` in `rimsky.yaml` to the absolute path
   of `rift-followup.yaml`.
3. Set `asimovdir` to an initialized Asimov project in which the RIFT package is
   installed and its pipeline is configured.

For example:

```yaml
output_dir: ./output
asimovdir: ./asimov
detectors: [H1, L1, V1]

sample_sink:
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

The bridge consumes plain YAML mappings and does not import Rimsky.  It is
therefore lightweight to test and isolated from Rimsky's streaming, GraceDB,
and HTCondor dependencies.  The contract targets Rimsky `0.1.0rc1` and current
main as of 2026-09-02.
