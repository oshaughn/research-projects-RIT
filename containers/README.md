# RIFT containers

This directory holds the multi-architecture container build and the "container
family" deployment mechanism. It has three related pieces:

1. **Multi-target build** — build a *family* of RIFT containers (different base
   image + cupy/CUDA variant, targeting different GPU compute capabilities) from
   one template.
2. **Family deployment** — let `SINGULARITY_RIFT_IMAGE` point at a YAML
   *manifest* describing that family, so each Condor job picks the right image
   for the machine it lands on.
3. **Survey + warmup scans** — survey a target Condor GPU pool and emit
   representative CuPy/JAX warmup jobs for the image bands that pool actually
   uses.

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
  selects the highest-capability image the matched machine can run, with the
  `fallback` image as the innermost `else` (used when the machine's capability is
  below every threshold):

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

- **A capability-defined `Requirements` clause** — `TARGET.GPUs_Capability =!=
  undefined`. The selection is deliberately *not* undefined-guarded: an
  undefined-capability slot could be anything (including a Blackwell that
  hard-fails on the older fallback), so the safe action is to not match it rather
  than guess. Measured on CIT, a large fraction of GPU slots satisfy the per-GPU
  `require_gpus` floor yet do not advertise the machine-level rollup attribute;
  without this clause those jobs hold with "Cannot expand $$ expression".

**CIP is different.** It requests no GPU, so its matched slot advertises no
capability and a capability-keyed selection cannot resolve (it would hold the
job). CIP therefore collapses to a **single fixed container** — the manifest
`fallback` image as a quoted literal, with no `$$()` token and no capability
`Requirements` clause. This is why the fallback must be CPU-safe.

### OSG: pick a delivery mode

The expression-valued `MY.SingularityImage` is evaluated *execute-side*. OSPool
glidein pilots read `SingularityImage` as a **literal string**, so an
`ifThenElse` lands verbatim and the job holds. Two opt-in modes fix this,
selected by an environment variable at DAG-build time:

| env var | behaviour |
|---|---|
| *(unset)* | legacy `universe = vanilla` + expression-valued `MY.SingularityImage`. Correct on a local/CIT pool; **not OSG-safe**. |
| `RIFT_CONTAINER_UNIVERSE=1` | **recommended for OSG.** `universe = container` + `container_image = $$([ ifThenElse(...) ])`. `$$()` is HTCondor's match-time (schedd-side) machine-ad substitution, so the pilot only ever sees a literal URL. No `MY.SingularityImage`, no `MY.SingularityBindCVMFS`, no `$$()` transfer token — the image arrives via `container_image`. GPU access is automatic under `request_gpus`. Works on CIT-local too. |
| `RIFT_CONTAINER_RUNTIME_SELECT=1` | older ILE-only fallback: Condor runs a generated `rift_container_select.sh` on the bare node, which reads the real capability from `nvidia-smi`, fetches only the matching image (`stashcp`/`pelican`) and re-execs under `apptainer exec --nv`. |

Under asimov set it from the blueprint, not the shell:

```yaml
scheduler:
  singularity image: /path/to/rift_container_family.yaml
  singularity base exe directory: /usr/local/bin/
  environment variables:
    RIFT_CONTAINER_UNIVERSE: 1
```

With `osdf://` images inside a manifest, the pipeline also enables the matching
transfer credential automatically (`use_oauth_services = scitokens`, or `igwn`
for `igwn+osdf:`) by inspecting the manifest's image URLs — the single-image path
keys off the `SINGULARITY_RIFT_IMAGE` string, which for a family is only a
`.yaml` path.

> **Why the container-universe selector names basenames, not URLs.**
> `condor_submit` parses `container_image` *before* any `$$` expansion and derives
> the job ad's `ContainerImage` -- the name the image gets in the job scratch dir --
> as the text after the **last** `/`. A selector containing full paths is cut in
> half, and the fragment that survives is not a valid image name. Submitting that
> form to the IGWN pool holds the job at the execute point:
> `PREPARE_JOB (prepare-hook) failed: Unable to download or build singularity image
> cutest_busybox_...sif") ])`.
>
> So the selector emits **basenames only** (no `/`); the whole `$$` token survives
> into `ContainerImage` and the schedd expands it at match time
> (`MATCH_EXP_ContainerImage = "rift_container_modern.sif"`). The image itself
> arrives via the comma-free `$$()` transfer token, and `MY.TransferInput` is pinned
> so `condor_submit` does not append the basename selector to `TransferInput` as a
> bogus extra input file. Verified end to end on an OSPool glidein against
> `$CondorVersion: 25.11.1`.
>
> Consequence: **every image in a family used with container universe must be a
> transferable URL.** An in-place (CVMFS/local) image can only be named by its full
> path, which reintroduces the truncation, so `build_container_image_select()`
> raises `ContainerManifestError` for such a family. Stage those images at a URL, or
> use `RIFT_CONTAINER_RUNTIME_SELECT=1`.


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

The expression-valued `MY.SingularityImage` is *not* OSPool-safe — a GWMS pilot
reads it as a literal string. Use `RIFT_CONTAINER_UNIVERSE=1` on the OSG (see
"OSG: pick a delivery mode" above).

---

## 3. Not included on this branch

Two adjacent pieces live on ``rift_O4d`` and were deliberately left out of this
port, which is scoped to the container *family* deployment path:

- `containers/survey_scan/` — an operator workflow that inventories a target
  GPU pool and emits per-image container-cache warmup jobs.
- the `container-dep-canary` GitHub Actions job, which tracks unpinned
  dependency drift in `requirements-container.txt`. This branch has no
  `.github/` workflows (CI is GitLab-based here).
