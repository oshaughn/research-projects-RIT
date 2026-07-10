# demo/reconstruct : whitened strain reconstruction with a confidence band

Reconstruct the whitened gravitational-wave strain in each detector, with a 90%
credible band, and overlay it on the data -- the "waveform reconstruction"
figure (cf. Fig. 2 of arXiv:2009.05461).  Two self-contained walkthroughs:

* **`nr/`**   -- reconstruct a **fixed numerical-relativity simulation** against
  an event (example: GW190521, RIT-Five eBBH-1794).
* **`model/`** -- a fully self-contained **waveform-model** example: download
  GW150914 from GWOSC, run RIFT end-to-end (IMRPhenomD, aligned spin), and
  reconstruct from the resulting posterior.

Shared tools (used by both):

| file | role |
|---|---|
| `reconstruct_strain.py`       | pool fair-draw samples → generate whitened waveforms at each sample's own (time, phase) → 90% band vs data.  Back-ends: `--approx MODEL` or `--group/--nr-param` (NR). |
| `extract_ile_samples.py`      | fast RIFT ILE `sim_inspiral` XML → compact `.npz` (keeps the per-sample `time`). |
| `make_reconstruct_subfile.sh` | turn a pipeline `ILE.sub` into a reconstruction submit file (condor). |

## The one thing you must get right

The reconstruction is only tight and physical if every posterior sample carries
its **own coalescence time**, coherent with its phase.  The ILE extrinsic run
that produces the samples **must** be invoked with

```
--fairdraw-extrinsic-output --resample-time-marginalization
```

(keep `--time-marginalization`; do **not** use `--maximize-only`).  With these,
RIFT draws a geocenter time per output sample from that sample's own lnL(t),
coherent with `coa_phase`; the exported `geocent_end_time` then varies per row.
`reconstruct_strain.py` places each realization at that (time, phase), so the
band coheres **with no post-hoc alignment**.  Without these flags the coalescence
time is marginalized away, the phase is unconstrained, and the band smears to
the full waveform amplitude.

Whitening is done in the frequency domain against the same PSD the analysis
used (do not rely on `gwpy.whiten()`; the analysis PSD often only reaches
`srate/2`).

## Running under HTCondor (recommended for real events)

Most events are analyzed with the standard pipeline
(`util_RIFT_pseudo_pipe.py` / `create_event_parameter_pipeline_BasicIteration`),
which produces the posterior and an `ILE.sub`.  To get reconstruction-ready
samples without hand-running anything, copy that submit file and inject the two
required flags, keeping the pipeline's container / GPU / accounting settings:

```
./make_reconstruct_subfile.sh /path/to/run/ILE.sub ILE_extr.sub
condor_submit ILE_extr.sub          # emits <prefix>_0_.xml.gz (fair-draw, per-sample time)
./extract_ile_samples.py <prefix>_0_.xml.gz samples.npz
./reconstruct_strain.py --samples samples.npz --fair-draw --approx IMRPhenomD \
    --psd-file H1=H1-psd.xml.gz --psd-file L1=L1-psd.xml.gz \
    --event-time <GPS> --out reconstruction.png
```

See `nr/README.md` and `model/README.md` for the two complete examples.

## Requirements

RIFT (`RIFT.lalsimutils`, the ILE executable), `lalsimulation`, `gwpy`,
`pesummary`, numpy/scipy/matplotlib; for NR mode also `NRWaveformCatalogManager3`
and the NR data files (git-annex).  Do not set `RIFT_LOWLATENCY=1` (it breaks NR
git-annex lookup).
