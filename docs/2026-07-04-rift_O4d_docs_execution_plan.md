# RIFT O4d Documentation Execution Plan — 2026-07-04

## Goal
Build the new O4d documentation work on a clean branch/worktree synced to `oshaughn/rift_O4d`, beginning with a Sphinx demos catalog that turns existing repository demos into discoverable documentation.

## Workspace decision
- Active documentation workspace: `sandbox_no_git/research-projects-RIT/research-projects-RIT-rift_O4d_docs`
- Branch: `rift_O4d_docs`
- Base: `oshaughn/rift_O4d` at `027cc21d` (`0.0.18.0rc1`)
- Reason: avoid conflicts with ongoing `rift_O4d_junior` work and keep docs changes isolated.

## Steps
1. Record this execution plan in the RIFT repo.
2. Create `docs/source/demos.rst` cataloging existing demos, with clear audience/use-case labels and links to source files.
3. Add the demos catalog to the main Sphinx toctree.
4. Run a minimal verification gate: inspect diff and attempt docs build if the local environment supports it.
5. Next planned chunks after catalog: HyperPipe/tracer split, NoLoop interpolation notes, calmarg production guide, multi-GPU ILE guide, LISA docs, population/EOS docs, executable reference refresh.

## Definition of done for this chunk
- `docs/source/demos.rst` exists and is linked from `docs/source/index.rst`.
- The page references only existing demo paths or explicitly marks entries as internal/advanced.
- Diff is reviewed.
- Build/verification result is recorded.
