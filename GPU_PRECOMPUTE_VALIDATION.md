# GPU compound precompute: implementation and validation

Status: PR #325 open and draft, 2026-09-12. The target is `rift_O4d`;
the branch includes the 2026-09-12 base merge after waveform PR #328 landed.

### Current JAX cost and readiness update (2026-09-12)

The earlier bounded-loop reduction removed most of the cold compile cost but
regressed warm execution. The replacement gathers small row/sample tiles under
an explicit forward scratch estimate. It retains the compact JAX loop graph
and reverse-mode differentiation without materializing the full production
`(A, K, S, npts)` gather. Fifteen focused CPU tests pass: independent value
oracles for nearest/linear/cubic/sinc, reverse-mode derivatives, non-divisible
row/sample tiles and padded tails, scratch-budget rejection, an empty-batch
compatibility case, and graph size. The actual JAX CI harness collects 860
tests from 49 files against its floor of 855, and its isolated new-test shard
passes all 15 (pinned JAX 0.9.0).

Paired Condor job 60769878 ran frozen expanded-loop baseline snapshot15
(`492aa421...`) and chunked candidate snapshot17 (`8996e237...`) on the same
NVIDIA RTX PRO 4000 Blackwell SFF worker. Both used the same captured 40-element,
five-sample H1/L1 bank, JAX/jaxlib 0.9.0, x64, pinned runtime image
`898a1261...`, and separate cold caches. Compilation fell from 196.691 to
17.855 s; warm median execution improved from 1.109 to 0.900 ms. First
compiled execution remained about 15.4--15.8 s. Maximum absolute error
against the independent fixed-point likelihood oracle was at most 9.10e-10.
These are one-worker observations for a captured bank, not a full ILE or BNS
throughput estimate. Raw products remain in scratch `jax_chunked_ab/`.

The exact post-base-merge source snapshot18 (`97f9acda...`) passed the
mandatory real-GPU regression job 60769879: 92 tests passed in 311.36 s,
exit 0. The subsequent zero-sample compatibility branch is covered by the
focused CPU test; the full GPU gate predates only that branch. Fresh GitHub CI
is running; keep the PR draft until those checks pass. A bounded full-length C1/E1/K1 probe uses the same
pushed code and frozen input record. Its first attempt (60769880) stopped
before RIFT execution because `/usr/bin/time` was absent from the container;
the corrected wrapper was verified inside that image and resubmitted as
60769881. Neither attempt is a posterior result. Per-intrinsic JAX closure
compilation reuse remains a separate cost issue.

### Historical readiness checkpoint before quota expiry (2026-09-12)

Fresh independent review found missing CI registration and two cache-lifetime
issues. The CPU CI gate now explicitly runs all 13 added non-JAX-directory
test files in separate processes and rejects all-skipped files. Roster and
shell checks pass. Contexts now bind to CUDA devices; changing devices rejects
an explicit old context and selects a distinct default context. Stable cache
roles replace old cutoff/response-order versions. Both new regressions passed
on CPU. The full per-file gate subsequently produced all 13 expected reports:
62 passed, 30 optional tests skipped, no failures/errors, and at least one
passing test in every file (temporary reports `/tmp/tmp.ZAmOyN5C0a/`).

Completed GPU job 60769877 compares frozen source 7dc4058c4 with b99bca825 on
one allocation, with separate cold caches and the same five-point captured
bank. Compile time fell from 293.469 s to 24.057 s, but warm median increased
from 0.001204 s to 0.009285 s. Both matched the independent likelihood oracle
to 9.064e-10 absolute. This single-host tradeoff is NOT a general speedup claim.
Raw outputs are in scratch `jax_compact_ab/`. PR325 remains draft pending
resolution of warm throughput and a final GPU gate. That earlier chunked-gather experiment was subsequently validated and
committed, as described in the current update above. A backup of its initial
untested form is `/tmp/jax_chunked_gather_UNTESTED_20260912.patch`.

PR328's fail-closed waveform helpers and corrected tests are synchronized here
to avoid conflicting alternative versions of the two added files. Independent
review identified a physical-strain absolute-tolerance bug in its LAL test;
amplitude-normalized checks now reject deliberate zero/sign mutations. All
11 waveform tests passed. This certifies helpers, not real Ripple/LAL carrier
parity or completed production native-GPU waveform conditioning.

Current correction: the pre-bounds-fix JAX AV smoke evidences are INVALID for
their requested boxes. The JAX adapter omitted AV's `enforce_bounds=True`;
every saved snapshot10 sample lay outside the requested sky/distance bounds.
Independent repair PR327 matches classic's existing bounds flag. Its outward-
rising synthetic regression fails before the fix and passes after it; all27
JAX AV tests pass, including on the isolated clean branch. Prior normalization
is unchanged. GPU Q/U/V and handoff parity results survive this correction;
the corrected short integration below removes the large discrepancy but does
not establish precise cross-driver evidence agreement.
The smoke harness now also rejects exported samples outside its declared box.

### Isolated profiling checkpoint (2026-09-12)

These are completed single-host diagnostic observations, not replicated
production speedup estimates. Job 60769876 first passed 21 GPU correctness
tests, then compared old source 35db72f60 and new source 99e6ff998 using the
same new harness, separate processes and cold caches on the same GPU. Both
profiles used five identical 128-second XPHM banks (8+7 Msun, generic spins,
K=21, A=40, H1/L1); no long NumPy oracle was run. Warm medians over calls 2--5
were 14.648 s before and 14.222 s after the duplication fixes (about 2.9%
lower total time). V fell from 4.478 to 4.059 s and primary basis construction
from 0.286 to 0.147 s. These are overlapping stage totals, not independent
terms to sum. New warm waveform generation was 5.956 s; packing and upload
were 0.035 s. Warm calls recorded zero storage read bytes and zero major
faults. This observation does not rule out cold I/O or longer-signal effects.
Raw profiles, gate output and hashes are outside the repository in
`/scratch/richard.oshaughnessy/rift_gpu_precompute_20260912/dupfix_ab_128s/`.
The new archive SHA256 is
`2d9812ac31ba1be07e7d120c2d9e39c2d13c2fd481eda077895ab89bdd750314`.

Separately, job 60769875 profiled a fixed captured short bank with JAX 0.9.0,
x64, on an RTX PRO 4000 Blackwell SFF Edition: K=2, A=40, two detectors,
five extrinsic points and 153 time bins. Backend initialization took 16.853 s,
device handoff 7.103 s, wrapper setup 0.019 s, lowering 13.397 s, compilation
297.930 s, and first execution 18.029 s. Seven warm calls had median 0.010997 s
(range 0.010987--0.011972 s). The maximum absolute discrepancy against the
independent fixed-bank NumPy oracle was 9.064e-10 in log likelihood. This is
consumer-only timing: imports, capture I/O, precompute, initial CuPy upload,
and the oracle are excluded; no AV integral or long waveform was timed.
Compilation dominates this measured cold consumer; the responsible graph
structure and cost of rebuilding a wrapper for another intrinsic remain to
be isolated. Raw output is `profile_jax_consumer.out` in the same scratch
root; source archive snapshot10 SHA256 is
`a4a3cd5d33c35851aca7003fed224a64c0fc6b4097ffd3c99535ef8eea928fa2`,
and the capture hash is
`d798f29fb59f51e2ad080ae2afb95dba384fc19bff984fd8c8ff9dda21835c32`.
No end-to-end or independently replicated performance claim follows.

Code inspection identifies a separate reuse limitation: each
`JAXExtrinsicLikelihood` constructor defines a fresh jitted closure over its
`JAXLikelihoodData`, including Q/U/V. Same-shaped intrinsic banks therefore
do not use a shared dynamic-bank kernel. This is distinct from the expanded
A-by-K data-term graph. A future reuse change must pass bank arrays as dynamic
arguments and explicitly test two distinct same-shape banks for both compile
reuse and different correct outputs; caching the first closure would silently
reuse stale physics. No wrapper-cache repair is claimed here.

The first compile-cost candidate replaces the A-by-K Python expansion in the
banded data term with statically bounded JAX loops. Twelve short synthetic CPU
tests passed in 82.38 s with both JAX platform selectors explicitly set to CPU:
nearest/linear/cubic/sinc, with and without post-phase, eager/JIT value parity,
linear/cubic position and coefficient-phase derivative parity, bounded graph
size from A=2 to A=40, and rejection of incomplete phase arguments. The oracle
retains the literal previous contraction. Full captured-bank GPU parity and
cold/warm performance remain pending; no speedup is claimed for this candidate.

### Latest completed validity checkpoint (2026-09-12)

Snapshot11 GPU job 60769868 passed 70 tests in 387.32 s. This includes short
SEOBNRv5PHM through GWSignal (21 modes through l=4), ordinary and conjugate
legacy-mode uploads, NumPy/CuPy Q/U/V parity, unequal detector arm lengths,
and nearest/cubic classic GPU consumers. The deliberately small SEOBNR test
disables the model's per-mode Nyquist veto; it tests transport compatibility,
not the physical accuracy of high modes on that grid.

Actual captured banks replayed through NumPy, CuPy, and JAX in job 60769867
agree to at most 1.06e-9 in pointwise log likelihood and 9.17e-10 after time
marginalization (five fixed extrinsics, 153 time bins, both driver-origin banks).
This tests identical banks, not waveform equivalence across differing inputs.

Corrected snapshot12 AV job 60769869 (source SHA256
`1e55243f1191fee15e57c785395b65ccc6e896dce73e87aeac745acd6a5c1a0d`)
passed the two-intrinsic short PhenomD integration and all exported sample-bound
guards. Both drivers now explicitly use reference frequency 100 Hz.

| Intrinsic | Corrected JAX lnZ | Reported sigma | neff | Evaluations | Fairdraw rows | Earlier classic lnZ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 50.98011 | 0.16864 | 20.44349 | 242372 | 107 | 51.50609 |
| 1 | 50.78177 | 0.15037 | 20.00309 | 323314 | 188 | 51.27185 |

Worker host runtime was 465.84 s for both points, excluding queue/container
transfer. This is an end-to-end execution smoke, not isolated precompute timing.
The roughly 235-nat discrepancy disappears after enforcing AV bounds. The
remaining roughly 0.5-nat differences require separate investigation before
claiming precise evidence agreement: low ESS, edge contact, and no independent
seed replication preclude such a claim. These fairdraw clouds are not usable
scientific posteriors. Native GPU waveform conditioning remains fail-closed.
Raw logs and samples remain under
`/scratch/richard.oshaughnessy/rift_gpu_precompute_20260912/` in
`snapshot11_tests.*`, `replay2_snapshot10_*_bank.json`, and
`snapshot12_jaxav_products/`; no raw products are committed.

### Short generic-mode timing checkpoint (2026-09-12)

Job 60769870 exited zero using committed source 35db72f60, snapshot SHA256
`f34adf33aed302c4bd542b18de733ee0684c15974b2f987d1601dda03ca94338`.
Short XPHM, 30+25 Msun with generic spins, H1/L1, N=2048, K=21,
pmax=Qmax=1 (A=40), 153 retained time bins; one RTX PRO 4000 Blackwell SFF
Edition and one CPU thread. This compares the same batched algorithm on NumPy
and CuPy, not the scalar legacy implementation. The CPU oracle runs first.

| Intrinsic | Batched CPU seconds | GPU seconds | Observed ratio CPU/GPU |
| --- | ---: | ---: | ---: |
| 0 (first call) | 85.381 | 39.702 | 2.15 |
| 1 | 9.068 | 1.122 | 8.08 |
| 2 | 7.202 | 0.746 | 9.65 |

Both paths include waveform generation and stop at resident-bank return;
queue, container transfer, and later oracle copies are excluded. The first
CPU call has about 80 s outside existing stage timers, so its ratio is not a
fair isolated hardware comparison. Initialization timing is being added to
locate this cost. GPU first-call basis and Q/U stages cost 13.94 and 24.98 s;
these observations do not alone identify kernel compilation versus other setup.

For intrinsic 2, GPU Q FFTs total 0.357 s, U Gram reductions 0.050 s,
V total 0.176 s, main basis 0.059 s, and legacy waveform plus upload 0.090 s.
Q currently batches only four rows, so a controlled larger-batch test is next;
production defaults remain unchanged. CPU U/V reductions dominate its warm cost.
Six cached detector arrays (294912 bytes) were uploaded initially, with no
additional uploads at either later intrinsic. Retained Q/U/V uses 49271040
bytes; primary basis per detector uses 27525120 bytes. All three numerical
parity checks passed, with maximum downstream absolute lnL error 2.17e-9.

These are one-worker profiling observations, not replicated speedup estimates
or a long-BNS runtime projection. Raw output: scratch `snapshot13_xphm_short.out`
and its adjacent scheduler log/error files. No samples or sampler were involved.

## Question and failure criteria

Can a GPU construct the compound response bank with the same numerical
likelihood as the LAL reference, while reusing detector inputs across intrinsic
points? Per Richard's correction, all executable tests use tiny synthetic
inputs or short BBH waveforms. No full BNS benchmark is authorized for this
validation stage; the long-grid memory bound is analytic only.

Fail if frequency ordering, Fourier normalization, complex conjugation, epoch,
retained-time window, detector weights, or waveform conditioning changes the
likelihood; if mutated inputs reuse stale cache entries; if unsupported physics
silently falls back; or if the long case exceeds device memory.

## Fixed diagnostics

- D1: Q, U, V maximum absolute and scale-relative differences against LAL.
- D2: pointwise downstream log-likelihood differences, including shifted times
  and nontrivial phases; both near the signal and across the prior.
- D3: sequential intrinsic points and changed detector/PSD/grid inputs, compared
  to fresh contexts; count data uploads and record retained bytes.
- D4: finite inputs/output and explicit rejection of unsupported configurations.
  Review (do not benchmark) the memory bound for the eventual long grid.
- D5: synchronized process runtime for initial setup, waveform, basis, Q, U/V,
  transfer of compact results, and subsequent intrinsic points. Container
  transfer and queue turnaround are excluded.
- D6: if AV integration is run, log weights, n-eff=20 per Richard's updated
  smoke target, n-max=800000, n-chunk=20000 classic/8000 JAX;
  output only bounded fairdraw (200). Report achieved
  ESS, collapse state, prior-edge contact, and seed variation. A failed
  convergence check is not a posterior result.

## Method

Independent adversarial tests use small synthetic arrays and LAL as oracle;
short physical-waveform runs use frozen snapshots outside the repository. First validate overlap
operations using identical waveform modes. Then validate the Ripple waveform
adapter including the conditioning needed by the production RIFT convention.
Use multiple intrinsic evaluations in one worker. Every measured snapshot is
identified by a source hash. CPU reference and GPU use matching physical inputs.

## Results and verdict

- Clean container, user-site disabled: 19 CPU tests passed, 18 GPU cases
  deselected. Includes independent LAL FFT normalization, full maintained
  CPU precompute/packing/downstream likelihood, and streamed V reduction.
- Short physical harness (30+25 Msun, df=0.5 Hz, N=2048, H1/L1/V1,
  A=40, K=2), NumPy backend, two nearby intrinsic points: pointwise maximum
  absolute lnL differences 1.2333e-9 and 7.8353e-10. Nine cached input arrays
  copied initially; second point added zero copies (nine cache hits).
  Reference times 9.2616/2.1810 s; candidate NumPy times 0.2374/0.2250 s.
  These are harness observations, not replicated performance estimates and
  emphatically not GPU speedups. First reference includes lazy startup.
- GPU snapshot01 c1a6cd17db7fda9459187d3565e5aa597f6e86888fdfcff80f768e4b75cc49a5:
  infrastructure failure before tests (worker lacked pytest).
- GPU snapshot02 c00aac0cd29bc0ed8d96e2df33d7de5ee240d705573db979e31e21407799dcf5:
  35 tests passed, two failed writing the default CuPy kernel cache (ENOSPC).
  The benchmark failed at the same cache-write boundary, not an array mismatch.
  Dependencies archived separately, SHA256
  83e8c50e5f683380d953af38ed40456aa31e0247ab4124040e31072a84932248.
- Snapshot03 AV submission failed before ILE because /scratch was not bound
  inside the container. Corrected launch uses an explicit /results bind and
  independent writable per-job CuPy/Numba/CUDA/temp directories.
- GPU snapshot04 49a3382f54bc0b9cf41099d034e8114be15bd277d9f8bfcd913d8926cb523326:
  42 tests passed on a real GPU (60769855.0, exit 0). Five-point short BBH
  benchmark also exited 0 (60769855.1), NVIDIA RTX PRO 4000 Blackwell SFF,
  3 detectors, A40/K2/N2048. Warm reference CPU times were
  [4.91756, 5.02197, 4.96536, 4.86751] s; warm GPU times were
  [0.57125, 1.03418, 0.57905, 0.58016] s. Medians 4.94146/0.57960 s.
  These are old-reference-vs-new-algorithm timings, not an isolated GPU
  hardware gain. First-use reference/GPU calls were 84.61696/45.28720 s.
  All five downstream lnL comparisons had max absolute difference <=1.188e-9.
  Nine input-array uploads initially (442368 bytes); none on later points,
  36 total cache hits, nine retained input arrays. This is NOT a BNS scaling
  measurement. Timers cover the precompute calls and synchronize device work;
  queue and container transfer are excluded.
- Final short CPU suite: 29 passed, 18 CUDA cases deselected, using explicit
  JAX_PLATFORM_NAME=cpu/JAX_PLATFORMS=cpu in the GPU-oriented image.
- Snapshot05 42d883faf574cc763f3e4492820df8417a6a12de50c2dcc2f5550f4d1c0bf7d0:
  adds finer FFT/Gram timing, final native-provider guards, same-worker batched
  NumPy comparison, and hardened AV output parsing; final GPU run 60769857
  completed with five matched intrinsic points and 47 passing tests (60769857).
  Same-worker warm medians were 4.60593 s reference CPU, 0.47379 s batched
  NumPy, 0.53046 s CuPy. Warm CuPy range 0.52167--0.93523 s. Thus the
  short-grid speedup over the original loop is predominantly algorithmic;
  there is no demonstrated incremental GPU gain at this N. Representative
  last-point per-detector device stages: primary basis 0.037 s, Q FFT 0.022 s,
  U Gram 0.0008 s, conjugate basis 0.072 s, V Gram 0.012 s. Fine-grained
  timings are nested inside the coarse stages; do not add both sets.
- Snapshot04 bounded AV run 60769856 FAILED convergence, correctly rejected
  by the runner: neff=1.2349/1.0055, collapsed=true at both intrinsic points.
  This failure is retained and is not a posterior result. The fixture was at
  200 Mpc with broad inherited bounds. Its software path executed, but that
  does not validate its integral. Snapshot06 uses 400 Mpc, sky truth +/-0.1
  rad, inclination/psi +/-0.2 rad, distance [300,500] Mpc and cubic time
  interpolation; convergence requirements unchanged.
- Snapshot06 d66150cafff35c584adfac571e04bf9cd1c98175ef94c7919e91cd41f3d15f43:
  explicit legacy-generator option forwarding, both mode-bank transfers,
  per-mode grid/epoch rejection and real TaylorF2 compatibility tests added;
  GPU test 60769858 passed all 52 tests. Corrected short AV 60769859
  returned neff=22.637 and 12.640, both finite and collapse=false. The old
  runner exited on its 300 threshold. The user subsequently set the test-run
  target to 20: event 0 meets it, event 1 remains below it. This is a smoke
  test, not a production posterior certification.
- Native Ripple adapter remains excluded from the production environment
  switch pending conditioning/epoch validation against the actual PhenomD
  LAL path, not the distinct ChooseFDModes conditioning path.

No full-BNS run or converged posterior claim has been made.

## Response-order budget and generic-mode validity

Read `paper/research_notes_paper1.tex` in full and the modulation, FD-precompute,
and generalized-response validation appendices of `paper/paper1_scaling3g.tex`
in the paper repository before selecting further response-order tests.
The exact compound response count is
`A = (P+1)*[(Q+2)*(P+5) + Q*(Q+1)]`, from the sum over `(b,p,n)`
with `w_0=2`, `w_(1+q)=q+2`, and `|n| <= w_b+p`.
For `(P,Q)=(0,0),(1,1),(2,2),(3,6)`, A is 10,40,102,424.
With K waveform modes and N frequency samples, the primary complex128 bank
alone uses `16*A*K*N` bytes; Q FFT work scales as `A*K*N*log(N)` and
U/V Gram work as `(A*K)^2*N`. Scratch space and other resident arrays are
additional. Streaming the conjugate bank does not remove the quadratic Gram
work. Thus increasing both response orders and Lmax blindly is prohibited in
the validation plan; budget using actual A, K and N first.

Response-order selection for production must use coherent omitted-waveform
U/V norms over the intended extrinsic support and a declared error budget,
not SNR alone, the highest frequency, or data-overlap Q arrays. The notes'
finite angular scans are estimates, not rigorous prior-wide certificates.
Any optional higher-reference-order selector also incurs that reference bank's
precompute cost, even when it selects a cheap production truncation.

Snapshot07 SHA256 `5412158f150081dec5127dbbf299b0349011456c3265da25953703820bcba373`
adds mode-label alignment for reordered conjugate dictionaries and a real short
precessing IMRPhenomXPHM test: 30+25 solar masses, fmin40 Hz, dt1/512 s,
df0.25 Hz, all 21 modes through l=4, p=Q=0, one H1 detector. The NumPy
comparison passes Q/U/V, epochs and downstream likelihood checks near and
away from the source. GPU correctness job 60769860 passed all 55 tests. This is not a
performance test or a certification of response truncation for long BNS.

Raw logs and snapshots: /scratch/richard.oshaughnessy/rift_gpu_precompute_20260912.

## Internal use (not merged)

### Merged template-angle correction follow-up

PR326 was merged by the user as 8eb362ef. The GPU branch incorporates that
development change without merging PR325. Snapshot10
`a4a3cd5d33c35851aca7003fed224a64c0fc6b4097ffd3c99535ef8eea928fa2`
reruns only short JAX AV, job60769863, to test whether the corrected XML/grid
template finalization explains the smoke evidence/peak discrepancy. The target
remains neff20, two intrinsic points, and fairdraw capped at200. The run completed
successfully in 413 s of host worker runtime: lnZ=286.4796/284.4658,
neff=23.81/26.44. Thus the merged correction does NOT explain this fixture's
large evidence discrepancy. For this (2,+/-2)-only model with full orbital-phase
support, the baked polarization phase can be absorbed by the sampled phase.
Actual per-driver bank construction and callback inputs remain under investigation;
no additional integration or performance run is justified until they agree.
`test_cross_driver_parity.py` pins same-bank p1/q1 cubic pointwise and time-
marginalized classic/JAX equality independently of the two sampling runs.
After integrating the merged fix, six focused CPU tests passed: XML/grid
finalization, same-bank cubic cross-driver parity, and handoff/order guards.

The follow-up found a separate classic-driver compatibility omission: its
compound calls did not receive the waveform controls used by ordinary precompute.
The three calls now share one waveform-control dictionary, including alternate
generators and conditioning. A sentinel-based caller-boundary regression passes.
For the current PhenomD fixture, a bounded mode-generation comparison with and
without the omitted conditioning arguments gave exactly identical ordinary and
conjugate modes; this omission is not an explanation of the evidence gap.
Frozen snapshot10 jobs60769864.0/.1 capture the first actual classic/JAX compound
bank, input data/PSD arrays, and parameters, then stop before sampling.

The short timing harness now reports cropped retained Q bytes separately from
full-frequency primary-basis bytes. Candidate NumPy and GPU timings end at the
resident-bank return; host copies used only for numerical comparison occur after
that timing. Approximant, Lmax, response orders, and short-grid parameters are
selectable. No new performance run has been launched with this harness.

An additional short synthetic regression passes for H1/L1 with distinct
40 km/20 km arm lengths, p1/q1, and cubic interpolation: direct JAX handoff,
host adapter, and classic NumPy agree for each detector and the network.
The test explicitly checks network additivity and a nonzero Q contribution.
This does not yet test classic CuPy's separate cubic Q-contraction kernel;
the real-bank replay must cover that path before declaring cross-driver parity.

The first boundary-capture writer failed to serialize a LIGOTimeGPS object after
writing the array archive; this was a diagnostic-output failure, not a numerical
failure. Corrected captures60769865.0/.1 completed. They show identical modes,
compound labels, cadence, Q epochs, and metadata. Data differ by only 1.37e-14
relative scale between independently regenerated fixtures; PSDs are identical.
The actual template parameters reveal a previously missed executable-default
difference: reference frequency100 Hz in classic versus30 Hz in JAX. Both have
zero phiref/psi/incl. Q banks differ by opposite constant mode phases; such a
phase can be absorbed by full orbital-phase sampling for this two-mode fixture,
so this is not yet an explanation of the evidence discrepancy. Both smoke
harnesses now explicitly request100 Hz, pinned by a caller-contract test.
Replay job60769866 failed before computation because its script was outside
the container bind. Corrected job60769867 replays each captured bank at five
fixed extrinsic points through
classic NumPy, classic CuPy, and direct JAX, including their actual terminal
time-marginalization routines. No extrinsic sampler is called.

Short SEOBNRv5PHM through GWSignal: NumPy-backend compatibility test passed
(29.04 s test runtime), with 21 modes through l4, N2048, df0.25 Hz, and common
epoch -0.204755017. It compares reference and candidate Q/U/V and epochs, and
asserts that both actually invoke the GWSignal provider. This is not a CuPy
result or a performance measurement. The corresponding GPU test and an expanded
nearest/cubic classic-GPU consumer regression are prepared for the next frozen
snapshot. The paired reference-frequency and waveform-forwarding caller tests
both pass.

### Historical device handoff checkpoint (superseded by latest checkpoint above)

Snapshot08 `c0353bd4a65c62641793979924fc41e7a8a450b86b713fcf37c52abe8db1b7c0`
adds direct compound-bank routing for both conventional GPU ILE and ILE-JAX.
Conventional ILE uses Q row views and the original dense U/V; JAX shares arrays
through DLPack, with a device-local Q layout conversion. The legacy host-return
API remains available. Native waveform conditioning is still gated.

Preregistered checks: no CuPy-to-host calls during handoff; GPU array residence;
Q/U/V and downstream likelihood parity; nonzero Q contribution inside the stored
time support; source-buffer deletion/allocator churn without corruption; reject
host inputs and unsupported Q pregrid factors. Short conventional and JAX AV
smokes use two intrinsic points, neff20, and fairdraw capped at 200. Queue and
container transfer are excluded from runtime. Jobs 60769861.0/.1/.2 respectively
run the GPU suite, conventional AV and JAX AV. Conventional AV returned
neff=20.14955 and 18.09552, finite and non-collapsed for both points; independent
XML inspection found 20 fairdraw rows each, 1673/1695 bytes. The strict runner
exited on event 1's below-20 value, not a likelihood/device failure. The user
requested a reasonable approximately-20 smoke target, not repeated runs to clear
a sharp threshold; these values are recorded without rerunning for convergence.
The snapshot08 GPU suite passed all 61 tests in 386.40 s. JAX AV completed both
points with neff=27.5798/22.4957 and 160/114 fairdraw rows. This certifies the
execution smoke only: lnZ=285.6272/284.4376 differs substantially from the classic
smoke's lnZ=51.5061/51.2719. The known ln(2) convention offset cannot explain
that gap; input/driver differences are under investigation. No cross-driver
evidence agreement or production posterior claim is made.

Snapshot09 `7279a60119bb6a58985b90f45524cc7cfd921874a0468c7f669fb6bc5f5ec6f2`
adds the full high-level precompute-to-classic-consumer no-bulk-host-transfer
test, physical-device consistency guards, and pre-import allocator setup in
the JAX executable/harness. GPU regression job 60769862 passed all 62 tests
in 342.34 s, including full precompute-to-classic-consumer no-bulk-transfer
and final physical-device/lifetime guards. Focused CPU
handoff/dispatch tests passed 5 tests before the final device guards; the final
structural/device-guard suite passed 3 tests. No long-waveform performance run.

Integrated development base 3ee682fe into the draft branch without merging a PR
or rewriting published history. Conflicts were confined to the new response-order
control and device-routing blocks. The host order controls are preserved; the
device-resident path rejects explicit check/choose controls until a device-native
selector is implemented. Upstream response-order tests: 8 passed. Fresh-process
handoff/dispatch/order-guard tests: 3 passed. These must run separately because
the upstream response-order test module installs stub RIFT modules globally.

Set `RIFT_GPU_PRECOMPUTE=1` in the worker to replace compound
rotation-plus-frequency-response precompute in conventional ILE or ILE-JAX.
CuPy is mandatory on this opt-in path; failure does not silently fall back.
The default still generates the conditioned base modes with LAL, then performs
compound basis FFTs and Q/U/V reduction on the GPU. The ordinary ILE driver
also computes its pre-existing ordinary bank; that separate overhead has not
been removed in this change.

Compatibility is load-bearing: the default calls the existing
`factored_likelihood.internal_hlm_generator(P, Lmax, **hlm_kwargs)` for every
intrinsic point and uploads BOTH its ordinary and conjugate mode dictionaries.
It does not restrict the default path to IMRPhenomD and does not require JAX or
Ripple. Existing waveform configuration is passed through. The numerical GPU
bank requires modes to share a frequency grid and epoch; it explicitly rejects
a mismatched legacy bank rather than silently shifting its modes. Native
generation is an optional, separate provider hook, not an automatic replacement.

`GPUPrecomputeContext` reuses detector data, response weights, and inverse PSD
arrays across intrinsic points. Content hashes invalidate modified input arrays;
old versions of a cache role are replaced. An optional timing callback reports
synchronized waveform, input preparation, basis, Q/U, V, and export durations.
The legacy return structure copies only compact Q windows and U/V to the host.
The direct API also offers device returns and a native-provider hook, but the
production environment switch refuses unvalidated native-waveform selection.

For integration tests use AV with internal log weights, multiple intrinsic
points, bounded n-max/n-eff, and save-samples only with fairdraw capped at 200.
All generated frames, grids, outputs, containers, dependency bundles, and logs
remain outside this source repository. The user subsequently authorized a draft
PR, and later requested landing it. Draft PR325 tracks the connected handoff;
the completed final-device-guard and corrected JAX AV results are recorded in
the latest checkpoint above.
