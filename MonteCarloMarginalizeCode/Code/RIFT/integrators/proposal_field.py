"""
proposal_field.py  --  L3 substrate: an intrinsic-space field of extrinsic proposals.

Scaffolding for iteration-to-iteration warm-start reuse (the "breadcrumb" strategy).
After an ILE iteration, each converged point contributes a compact extrinsic proposal
(its high-likelihood samples, or an AV live-volume state).  Those are aggregated, keyed
by the intrinsic parameters lambda, into a ProposalField.  The next iteration's ILE
workers query the field at their own lambda and warm-start from the nearest proposal.

Design constraints (see DESIGN_warmstart_threading.md):
  * A warm start only shapes p_s, never the estimator -> a stale/mismatched field entry
    can only cost efficiency, never bias.  Callers should still pass cover_frac>0 to
    bootstrap_from_samples for cross-problem reuse as a belt-and-suspenders floor.
  * Entries are small (samples ~KB, AV state ~8KB) -> the whole field is a small file,
    fine for Condor transfer_input_files.
  * Nearest-neighbour lookup uses a whitened intrinsic metric so "nearby" respects the
    very different scales of chirp mass, mass ratio, and spins.

This module intentionally does NOT touch the DAG.  Pipeline wiring (build the field as a
post-iteration node, pass it forward, query it in the ILE driver via a new
--extrinsic-proposal-field hook) is the next step and mirrors the existing calmarg
extrinsic-breadcrumb plumbing.
"""
import numpy as np


class ProposalField(object):
    """A set of (lambda, extrinsic-proposal) entries with nearest-lambda lookup.

    lambda vectors are intrinsic-parameter coordinates (e.g. [mc, q, chi1z, chi2z, ...]);
    proposals are (M_k, d_extrinsic) arrays of high-likelihood extrinsic samples in the
    sampler's coordinate convention.  `extrinsic_params` names the proposal columns."""

    def __init__(self, intrinsic_params=None, extrinsic_params=None):
        self.intrinsic_params = list(intrinsic_params) if intrinsic_params else None
        self.extrinsic_params = list(extrinsic_params) if extrinsic_params else None
        self._lambdas = []        # list of (d_intrinsic,) arrays
        self._proposals = []      # list of (M_k, d_extrinsic) arrays
        self._scale = None        # per-intrinsic-dim scale for the whitened metric

    def add(self, lam, proposal):
        """Add one point's proposal.  lam: (d_intrinsic,); proposal: (M, d_extrinsic)."""
        lam = np.asarray(lam, dtype=float).ravel()
        proposal = np.atleast_2d(np.asarray(proposal, dtype=float))
        if proposal.shape[0] < 2:
            return  # too few points to seed a live volume
        self._lambdas.append(lam)
        self._proposals.append(proposal)
        self._scale = None  # invalidate cached metric

    def _metric_scale(self):
        if self._scale is None and self._lambdas:
            L = np.vstack(self._lambdas)
            s = np.std(L, axis=0)
            s[s <= 0] = 1.0
            self._scale = s
        return self._scale

    def nearest(self, lam, k=1):
        """Return the proposal(s) from the k intrinsic points nearest `lam` (whitened
        Euclidean).  For k=1 returns a single (M, d_extrinsic) array; for k>1 the
        vertically-stacked union (a broader, safer seed)."""
        if not self._lambdas:
            return None
        lam = np.asarray(lam, dtype=float).ravel()
        s = self._metric_scale()
        d = np.array([np.sum(((lam - lj) / s) ** 2) for lj in self._lambdas])
        order = np.argsort(d)[:max(1, int(k))]
        if len(order) == 1:
            return self._proposals[order[0]]
        return np.vstack([self._proposals[i] for i in order])

    def warm_seed_for(self, lam, k=1):
        """Convenience: the seed array to hand to AV.bootstrap_from_samples(...,
        params=self.extrinsic_params, cover_frac=...) for intrinsic point `lam`."""
        return self.nearest(lam, k=k)

    # --- serialization (small npz; fine for Condor transfer) ---
    def save(self, path):
        if not self._lambdas:
            raise ValueError("ProposalField is empty")
        arr = {}
        arr['lambdas'] = np.vstack(self._lambdas)
        arr['sizes'] = np.array([p.shape[0] for p in self._proposals])
        arr['proposals'] = np.vstack(self._proposals)   # concatenated; split by sizes
        arr['intrinsic_params'] = np.array(self.intrinsic_params or [])
        arr['extrinsic_params'] = np.array(self.extrinsic_params or [])
        np.savez_compressed(path, **arr)
        return path

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        pf = cls(intrinsic_params=[str(x) for x in d['intrinsic_params']] or None,
                 extrinsic_params=[str(x) for x in d['extrinsic_params']] or None)
        L = d['lambdas']; sizes = d['sizes']; P = d['proposals']
        off = 0
        for i in range(len(L)):
            n = int(sizes[i])
            pf._lambdas.append(np.asarray(L[i], dtype=float))
            pf._proposals.append(np.asarray(P[off:off + n], dtype=float))
            off += n
        return pf

    def __len__(self):
        return len(self._lambdas)


def build_field_from_run_outputs(entries, intrinsic_params=None, extrinsic_params=None):
    """Aggregate a list of (lam, proposal) pairs (one per converged ILE point in an
    iteration) into a ProposalField.  Intended to be called by a small post-iteration
    node that scans the iteration's ILE outputs."""
    pf = ProposalField(intrinsic_params=intrinsic_params, extrinsic_params=extrinsic_params)
    for lam, proposal in entries:
        pf.add(lam, proposal)
    return pf
