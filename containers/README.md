# RIFT containers

This directory holds the multi-architecture container build and the "container
family" deployment mechanism. It has two related but independent pieces:

1. **Multi-target build** — build a *family* of RIFT containers (different base
   image + cupy/CUDA variant, targeting different GPU compute capabilities) from
   one template.
2. **Family deployment** — let `SINGULARITY_RIFT_IMAGE` point at a YAML
   *manifest* describing that family, so each Condor job picks the right image
   for the machine it lands on.

The top-level [`rift_container.def`](../rift_container.def) is unchanged and
remains the default single-image build.

---

## 1. Building a family

```
containers/build_family.sh [--render-only] [OUTPUT_DIR]
```

- [`rift_container.def.in`](rift_container.def.in) is a template with
  `@@BASE_IMAGE@@` / `@@CUPY_PKG@@` placeholders (apptainer `.def` files take no
  build args, so we render then build).
- [`build_family.sh`](build_family.sh) holds the build `MATRIX`. The **first**
  entry is the default and uses the current production base image, so the family
  always includes a broadly-compatible image for older machines. Add rows to
  target more architectures.
- `--render-only` writes the per-entry `.def` files without invoking apptainer
  (useful in CI or on a machine without apptainer).
- Each build also emits a `rift_container_family.generated.yaml` stub — fill in
  each `image:` with where you published the `.sif` (a CVMFS path or `osdf://`
  URL), and you have a deployable manifest.

All matrix entries share the pip set in
[`requirements-container.txt`](requirements-container.txt) (the cupy wheel is the
only per-entry difference). That file is the **single source of truth** also
consumed by the CI dependency canary (below). `build_family.sh` stages it into
each image via the `.def`'s `%files` section, so the build does **not** depend on
the cloned RIFT branch shipping the file.

### Build troubleshooting

**`proot error: ptrace(TRACEME): Operation not permitted` /
`mksquashfs command failed`** (seen on shared clusters such as CIT). Apptainer
has no usable user namespaces or setuid install, so it falls back to its
unprivileged `proot` build engine — which cannot run the `mksquashfs` helper.
**Setting `PROOT_NO_SECCOMP=1` is not sufficient** (it silences the seccomp
message but proot still fails to exec mksquashfs). Avoid the proot path instead:

1. **Build with `--fakeroot`** (recommended; the IGWN/CIT path):

   ```console
   containers/build_family.sh --fakeroot ./container_family
   ```

   Requires `/etc/subuid` + `/etc/subgid` entries for your user and unprivileged
   user namespaces enabled (check: `grep $USER /etc/subuid` and
   `apptainer build --fakeroot` on a tiny def). This produces a real `.sif`
   without proot.

2. **If even `--fakeroot` is unavailable, build a `--sandbox`** (a directory).
   This skips `mksquashfs` entirely, so it sidesteps the failing step:

   ```console
   containers/build_family.sh --sandbox ./container_family
   # later, on a host where apptainer can make a SIF:
   apptainer build rift_container_default.sif ./container_family/rift_container_default/
   ```

3. **Or build elsewhere** — on a node/registry with proper apptainer (or build
   the OCI image with Docker/podman, push to a registry, then
   `apptainer pull`/`build` the `.sif` on a capable host).

`build_family.sh` still exports `PROOT_NO_SECCOMP=1` as a harmless best-effort,
and passes any extra `--flag` you give it straight through to `apptainer build`.
If a build runs out of space mid-way, point `APPTAINER_TMPDIR` at a large local
disk.

---

## 2. Deploying a family via a manifest

Set `SINGULARITY_RIFT_IMAGE` to a `.yaml`/`.yml` manifest instead of a single
`.sif`. Everything else (pseudo_pipe, `--use-singularity`, etc.) is unchanged —
the manifest is detected by file extension. A plain `.sif` path or single
`osdf://` URL keeps the **exact** legacy single-image behavior; the manifest path
is never consulted in that case.

See [`rift_container_family.yaml`](rift_container_family.yaml) for a worked
example. Schema:

| field             | meaning |
|-------------------|---------|
| `version`         | manifest schema version (currently `1`) |
| `capability_attr` | machine ClassAd attribute the selection expression tests (default `GPUs_Capability`) |
| `fallback`        | label of the catch-all image (innermost `else`); **must be CPU-safe** |
| `containers[]`    | the family |
| ↳ `label`         | human id; also referenced by `fallback` |
| ↳ `image`         | a CVMFS/local path (referenced in place, lazy-fetched) **or** an `osdf://` URL (selectively transferred) |
| ↳ `cuda_capability_min` | inclusive lower capability bound for this image |
| ↳ `cuda_capability_max` | informational upper bound (`null` = open-ended) |
| ↳ `note`          | free-text |

> **Keep the family consistent.** A *single* `SINGULARITY_BASE_EXE_DIR` is
> applied to **every** image in the family — the ILE/CIP jobs locate the
> executable as `SINGULARITY_BASE_EXE_DIR + <exe name>`, with no per-image
> override. So all images in a manifest **must install RIFT's executables at the
> same in-container path** (and share a common layout/Python/entrypoints). Build
> them from the same `rift_container.def.in` template (`build_family.sh` does
> this) and do **not** hand-mix images with different internal layouts. The same
> applies to `SINGULARITY_BASE_EXE_DIR_HYPERPIPE` if you use hyperpipe.

### What the pipeline generates

For the ILE (and CIP) Condor submit, a manifest produces:

- **`MY.SingularityImage`** — an *unquoted* `ifThenElse(...)` expression that
  selects the highest-capability image the matched machine can run, defaulting to
  the `fallback` image (also used when the capability attribute is `undefined`,
  e.g. on a CPU-only CIP slot — hence the fallback must be CPU-safe):

  ```
  ifThenElse(TARGET.GPUs_Capability >= 8.0, "./rift_container_modern.sif", "/cvmfs/.../rift_container_default.sif")
  ```

- **Selective transfer** — only `osdf://` images get fetched, and only on the
  machine that selected them, via one HTCondor `$$()` match-time token appended
  to `transfer_input_files` (CVMFS/local images are referenced in place and never
  transferred, so the *whole family is never pulled*):

  ```
  $$([ (TARGET.GPUs_Capability >= 8.0 ? "osdf:///.../rift_container_modern.sif" : "") ])
  ```

  `request_disk` is **not** auto-sized (image sizes are unknown at submit time) —
  size it to your largest single transferred image.

- **`require_gpus` floor** — `Capability >= <lowest min across the family>`,
  composed (`&&`) with any user-supplied `RIFT_REQUIRE_GPUS` (which today you use
  to block incompatible hosts by `DeviceName`). Both apply; neither is dropped.

### HTCondor GPU attribute names — important

Two different namespaces are in play and are kept separate:

- The **image-selection `ifThenElse`** reads the *machine* ClassAd. Default
  `GPUs_Capability` (advertised on the OSG; some pools differ). Override per-run
  with `RIFT_GPU_CAPABILITY_ATTR`, or per-manifest with `capability_attr`. Verify
  on your pool:

  ```
  condor_status -constraint 'TotalGPUs > 0' -autoformat GPUs_DeviceName GPUs_Capability GPUs_GlobalMemoryMb
  ```

  Not every GPU host advertises this; on such hosts the expression collapses to
  the fallback image and the `require_gpus` floor does the steering.

- The **`require_gpus` floor** uses the require_gpus sub-ad attribute
  `Capability` (unprefixed — *not* `TARGET.`, *not* `GPUs_`).

### Requirements

- `PyYAML` must be importable wherever the pipeline is built (only when a
  manifest is actually used). Single-`.sif` runs never require it.

### Validation status

Validated on a real HTCondor pool + GPU (a cap-3.0 machine):

- The advertised attributes are `GPUs_Capability` (machine ad) and `Capability`
  (require_gpus sub-ad) — matching the defaults above.
- The `require_gpus` capability floor matches a compatible GPU and correctly
  *excludes* an incompatible one (`Capability >= 7.0` did not match a cap-3.0
  GPU), so the floor steers GPU selection as intended.
- The `$$([ ifThenElse(TARGET.GPUs_Capability >= …, …) ])` transfer token is
  honored at match time: only the matched image's URL is selected/transferred.
- The empty-result case — when a manifest *mixes* CVMFS and osdf entries and a
  CVMFS branch is selected, the `$$()` token expands to `""` — is **tolerated**:
  the empty entry is skipped and the job runs clean. (So mixed manifests are
  safe; you do *not* need uniform all-osdf / all-cvmfs retrieval.)

One item still needs a real **OSG/GWMS** pilot (a local pool has no singularity
wrapper to exercise it): that the pilot evaluates the expression-valued
`MY.SingularityImage` and honors a relative `./name.sif` produced by it.

---

## 3. CI dependency-resolution canary

The default container build uses *unpinned* deps, so a fresh upstream release
(e.g. `swig>=4.4.0`, see issue #136) can silently break RIFT and we only find out
when a container rebuild fails. The `container-dep-canary` job in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) installs the unpinned
[`requirements-container.txt`](requirements-container.txt) set (minus the
GPU-only cupy wheel) and the pixi `swig-post44` lane, then runs the import check —
on every push/PR **and weekly** — to flag such breakage early. It is
non-blocking (advisory): it tracks upstream changes outside any PR author's
control.
