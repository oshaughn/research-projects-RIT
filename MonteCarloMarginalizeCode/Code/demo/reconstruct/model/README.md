# model : self-contained GW150914 waveform-MODEL strain reconstruction

A fully self-contained, end-to-end example of the waveform-reconstruction figure
using a **waveform model** (IMRPhenomD) rather than a fixed NR simulation:

1. download **GW150914** from GWOSC (O1, H1 + L1 only),
2. estimate an off-source PSD,
3. run a **full RIFT parameter-estimation DAG** (IMRPhenomD, aligned spin pinned
   to zero) on HTCondor to a posterior, and
4. **reconstruct** the whitened detector strain, with a 90% credible band, from
   that posterior — each posterior sample drawn as an IMRPhenomD waveform at its
   own coalescence time.

Nothing is faked and nothing is imported from another event: the frames, PSD,
coinc, DAG, and posterior are all produced here.

## Quick start

```bash
make data         # download O1 frames + build event.cache + detect channel
make psd          # estimate off-source PSDs (gwpy) -> H1/L1-psd.xml.gz
make coinc        # build coinc.xml (event time + IFOs)
make dag          # generate the RIFT DAG (rundir_gw150914_D)
make submit       # apply CIT-local fixes + condor_submit_dag
make status       # condor_q for this run
# ... the DAG runs on condor over hours ...
make reconstruct  # once extrinsic_posterior_samples.dat exists -> GW150914_reconstruction.png
```

`make all` runs `data psd coinc dag submit`.  All configuration (container path,
event GPS, sizes, paths) is in **`config.sh`**.

## The one thing that makes the reconstruction cohere

The band is only tight and physical if every posterior sample carries its **own
coalescence time**, coherent with its phase.  In the pipeline that is arranged by
building the DAG with

```
--add-extrinsic --batch-extrinsic --add-extrinsic-time-resampling \
--internal-ile-srate-time-resampling 4096
```

**These flags make the pipeline's final `ILE_extr` stage emit, into
`ILE_extr.sub`,**

```
--fairdraw-extrinsic-output --resample-time-marginalization
```

(kept alongside `--time-marginalization`).  With them, RIFT draws a geocenter time
per output sample from that sample's own `lnL(t)`, coherent with `coa_phase`, and
the exported `time` column in `extrinsic_posterior_samples.dat` **varies per row**.
`build_dag.sh` verifies both flags landed in `ILE_extr.sub`.

**The reconstruction reads `rundir_gw150914_D/extrinsic_posterior_samples.dat`**
(via `../dat_to_compact.py` -> compact `.npz`), then `../reconstruct_strain.py`
places each IMRPhenomD realization at its own `(time, phase)` — so the band
coheres with **no post-hoc alignment**.

## GW150914 specifics (verified here)

| item | value |
|---|---|
| GPS event time | `1126259462.4` |
| detectors | H1, L1 (no Virgo in O1) |
| GWOSC data | O1 4 kHz open data, 4096-s block starting `1126256640` |
| in-frame channel | `<IFO>:LOSC-STRAIN` — **not** `GWOSC-4KHZ_R1_STRAIN` |
| approximant | IMRPhenomD (aligned spin), `--assume-nospin`, `--l-max 2` |
| seglen / srate / fmax | 8 s / 4096 Hz / 1024 Hz |
| chirp-mass window | `[25,35]` (Mc ~ 28) |

The **channel name is the key adaptation** for O1: the O1 open-data frames
(`H-H1_LOSC_4_V1-...gwf`) carry `H1:LOSC-STRAIN`, the older LOSC convention, not
the O4-era `GWOSC-4KHZ_R1_STRAIN`.  `fetch_gwosc_data.py` reads the real channel
from the frame table rather than assuming it.

The PSD off-source window deliberately starts 1060 s into the block
(`1126257700`): the first ~500 s of the L1 O1 block contains a data-quality gap
(NaNs).

## What each file does

| file | role |
|---|---|
| `config.sh`             | container path, event GPS, channels, sizes, paths, `rift_env` |
| `fetch_gwosc_data.py`   | resolve GWOSC O1 .gwf URLs, curl frames, build `event.cache`, detect channel |
| `estimate_psd.py`       | off-source median-Welch PSD (gwpy) -> ascii -> `convert_psd_ascii2xml` xml.gz, sanity-check load |
| `make_coinc.sh`         | `util_WriteInjectionFile.py` + `util_SimInspiralToCoinc.py` -> `coinc.xml` |
| `GW150914_D.ini`        | data geometry the pipeline has no CLI flag for (seglen/srate/fmin/fmax, channels, ifos) |
| `build_dag.sh`          | `util_RIFT_pseudo_pipe.py` -> `rundir_gw150914_D`; stages PSDs; verifies the fair-draw flags |
| `submit.sh`             | CIT-local sub fixes (container universe / local .sif transfer / getenv / GPU floor / local pin) + `condor_submit_dag` |
| `reconstruct_gw150914.sh` | posterior -> compact `.npz` -> whitened strain band PNG |
| `Makefile`              | `data psd coinc dag submit status reconstruct clean` |

Shared, read-only tools live one level up: `../reconstruct_strain.py`,
`../dat_to_compact.py`, `../extract_ile_samples.py`,
`../make_reconstruct_subfile.sh`.

## Environment

The pipeline generator runs on the submit node.  `config.sh`'s `rift_env`
activates an igwn conda env (deps: lal, gwpy, igwn_ligolw, numpy/scipy) and
prepends the RIFT **source** tree (`$RIFT_SRC/MonteCarloMarginalizeCode/Code`,
`bin/`) so the exact RIFT version is used.  The condor jobs themselves run inside
the RIFT singularity image `$CONTAINER_SIF` (container universe, local .sif file
transfer for the CIT pool).

## Finishing the reconstruction once the posterior lands

```bash
../dat_to_compact.py rundir_gw150914_D/extrinsic_posterior_samples.dat gw150914_samples.npz
../reconstruct_strain.py --samples gw150914_samples.npz --fair-draw --approx IMRPhenomD \
    --psd-file H1=H1-psd.xml.gz --psd-file L1=L1-psd.xml.gz \
    --event-time 1126259462.4 --event-name GW150914 --sim-id IMRPhenomD --srate 4096 \
    --out GW150914_reconstruction.png
```

`make reconstruct` runs exactly this.  Whitening uses the same PSDs the analysis
used (`H1/L1-psd.xml.gz`), in the frequency domain.
```
