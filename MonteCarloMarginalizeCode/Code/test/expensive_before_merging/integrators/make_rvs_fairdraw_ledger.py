"""Regenerate rvs_fairdraw_verdicts.json from the classification RULES.

Run after moving or editing a post-rebind _rvs consumer:

    python3 make_rvs_fairdraw_ledger.py && python3 audit_rvs_fairdraw.py --check

The rules are the audit's reasoning in executable form -- ship the thing that
regenerates the list, not just the list.

Every rule below corresponds to a code path that was READ during the audit; the rule is
how the verdict is applied to each of that path's sites, not a substitute for having
looked.  Sites that match nothing stay TODO and fail --check, which is the intended
behaviour for anything this pass did not actually reach.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_rvs_fairdraw as A  # noqa: E402

hits = [h for t in A.TARGETS for h in A.scan_file(t)
        if h.get("phase") in A.POST_REBIND_PHASES]


def verdict(h):
    f, fn, src, line = h["file"], h["function"], h["source"], h["line"]
    s = " ".join(src.split())

    # --- the integrators themselves ------------------------------------------------
    # --- option A: the record (DESIGN_rvs_naming.md) ---------------------------
    if "RvsRecord.fair_draw(" in s or "RvsRecord.retained(" in s \
            or "n_retained=self._rvs_record.n_retained()" in s \
            or "reserve=getattr(self, '_warm_seed_reserve', None))" in s:
        return ("PER_ROW",
                "Hands the columns to RvsRecord as a VIEW, with the pre-draw count taken from "
                "the previous record's PROVENANCE (eager) rather than from len() (lazy, and "
                "would read the already-rebound dict). Reads no statistic of the rows: it "
                "records WHAT THEY ARE at the moment that changes.")
    if "_rebound_record(sampler, dict(sampler._rvs)" in s:
        return ("PER_ROW",
                "Snapshots the columns for a possible restore and rebinds the record to that "
                "copy, so the restored record describes what is actually put back rather than "
                "the original dict (which would fail every identity check and be inert). A "
                "dict copy; reads no statistic of the rows.")
    if "_rvs_record_for(sampler, sampler._rvs)" in s:
        return ("PER_ROW",
                "Looks up the record describing these columns, declining it if _rvs has been "
                "replaced since. The consumer then asks a NAMED question "
                "(rows_are_resampled / blocks_were_flattened) instead of combining flags; the "
                "flags remain the fallback until every sampler is converted.")
    if f.startswith("RIFT/integrators/"):
        if "n_retained=_n_retained_before_draw" in s or "RvsRecord.fair_draw" in s \
                or "reserve=getattr(self, '_warm_seed_reserve', None))" in s:
            return ("PER_ROW",
                    "(DESIGN_rvs_naming.md) hands the just-rebound columns to RvsRecord "
                    "as a VIEW, together with the pre-draw row count. Reads no statistic of "
                    "them -- it records that they ARE the export resample, at the moment that "
                    "becomes true, which is the whole point of the record. Nothing consumes it "
                    "yet.")
        if "indx_list" in s:
            return ("PER_ROW",
                    "The rebind's own right-hand side: this IS the fair draw, gathering each "
                    "key at the drawn indices. Read before the write it feeds.")
        if "identity_convert(self._rvs[name])" in s or "for name in self._rvs" in s \
                or "isinstance(self._rvs[name]" in s:
            return ("PER_ROW",
                    "Element-wise dtype/backend conversion (cupy -> numpy) over whatever rows "
                    "are present. Touches no population statistic; correct on any row set.")
        if "self._rvs['integrand'] = self._rvs['log_integrand']" in s.replace('"', "'"):
            return ("PER_ROW",
                    "Aliases one per-row column onto another so raw-field consumers find it. "
                    "Row-for-row; unaffected by which rows survived.")

    # --- distance exporters: the live defect ----------------------------------------
    if "rvs=(dict(sampler._rvs) if rvs is None else rvs)" in s:
        return ("PER_ROW",
                "_snapshot_pass_state takes a dict copy of whatever rows are present so a "
                "rejected warm pass can be undone. It makes no claim about their statistics, "
                "and it snapshots the fair-draw MARKER alongside them so the restored record "
                "and the marker describing it cannot disagree.")
    if s in ("rvs = sampler._rvs", "_rvs = sampler._rvs") or \
            'sampler._rvs["distance"]' in s:
        return ("PER_ROW",
                "Binds the record, or reads one COLUMN of it, for the distance exporters. The "
                "resample changes which rows are present, never the value a row carries, and "
                "the weighting decision is made separately by ln_weights_for_posterior (.dgrid) "
                "or by forcing the all-fresh path (.dslice). Correct on either row set.")
    if "ln_weights_for_posterior" in s:
        return ("FIXED",
                "Asks for POSTERIOR weights, which are uniform when the record is the "
                "fair-draw export and the derived importance weights otherwise. The predicate "
                "is set by the sampler at the rebind, so it means 'the draw fired' rather than "
                "'the flag was passed'. Pinned by test_fairdraw_double_weighting.py.")
    if f == "RIFT/misc/distance_slices.py":
        return ("FIXED",
                "The .dslice reweight core cannot be corrected after the fact -- it "
                "double-counts pi_Omega/q_Omega and takes N from the resample, and the "
                "pre-draw record is gone by then. The ILE now forces the exact all-fresh path "
                "(K independent fixed-d integrations) when the record is fair-drawn, and says "
                "so, rather than reporting a plausible wrong number.")

    # --- the L0 rescue and the sequential warm start ---------------------------------
    # BOTH ILE drivers.  The LISA driver now carries a ported copy of the L0 rescue, and the
    # `_rvs` reads inside it are byte-identical to the ones here -- these rules match on
    # source TEXT, so the same text earns the same verdict.  (It lives in a module-level
    # _maybe_l0_rescue there rather than inlined in analyze_event, because that driver has TWO
    # analyze_event variants; the enclosing function name is not part of the match.)  Rules in
    # this block naming things the LISA driver does not have -- _rep_rvs, extrinsic_handoff,
    # the sequential-warm-start seeds -- simply never match for it.
    if f in ("bin/integrate_likelihood_extrinsic_batchmode",
             "bin/integrate_likelihood_extrinsic_batchmode_lisa"):
        if "_lnZ_of_reserve_or_rvs" in s:
            return ("FIXED",
                    "PR #79, re-landed as #86. sampler._rvs is passed as the FALLBACK "
                    "argument only: the helper prefers the retained reserve via "
                    "lnZ_from_reserve, and the gate refuses to compare across sources "
                    "(_cold_src != _warm_src forces BOTH back to the fair-draw reading, which "
                    "is at least self-consistent). Measured before #79: two passes with "
                    "identical true lnZ at n_eff 1.8 vs 53 produced a +3.48 nat gap and "
                    "rejected the good warm pass 100% of the time at the 0.5 default.")
        if "_lnZ_of_rvs" in s:
            return ("BROKEN",
                    "PR #79's CROSS-SOURCE FALLBACK: fires only when the cold and warm passes "
                    "produced different reading sources (one had a reserve, the other did "
                    "not), and then re-reads BOTH sides from the fair-draw record. That is "
                    "self-consistent, which is what #79 claims for it, but it is not "
                    "unbiased: the two passes sit at different n_eff, so the log(n/n_eff) "
                    "artifact does not cancel and this branch is back in the regime measured "
                    "at +3.48 nats / 100% rejection at the 0.5 default. A known, documented, "
                    "BOUNDED residual -- not a defect anyone introduced. Closing it needs a "
                    "retained-set reading on both sides, i.e. a reserve for the samplers that "
                    "keep none. Follow-up, not a regression.")
        if "_kish_neff_of_rvs" in s or "list(sampler._rvs.values())[0]" in s:
            return ("FIXED",
                    "Kish of the pooled record is only used when the export is NOT fair-drawn. "
                    "When it is, _pool_replica_rvs has flattened each block, and the Kish of "
                    "piecewise-constant weights is just the row count (5K at the default "
                    "--fairdraw-extrinsic-output-n-max 5). The ILE now computes the same "
                    "quantity one level up -- (sum Z_k)^2 / sum(Z_k^2/neff_k), Kish over the "
                    "BLOCKS -- which reduces to sum(neff_k) when replicas agree and falls "
                    "below it when they do not. The row count beside it is a deliberate "
                    "report of the export size.")
        if "_warm_seed_reserve_for" in s or "_res_l0" in s or "_seed_ws" in s \
                or "_res_ws" in s or "_SEQ_WS_PENDING = _seed_ws" in s:
            return ("FIXED",
                    "Seeds from the retained-sample reserve, falling back to _rvs only for a "
                    "sampler that keeps none, and judges the seed by affine RANK via "
                    "build_warm_seed. PR #78 for the L0 rescue; this change for the sequential "
                    "warm start. Pinned by test_l0_rescue_seed.py and test_seq_warmstart_seed.py.")
        if "_cold_rvs" in s:
            return ("PER_ROW",
                    "Snapshots the cold record so the reject path can restore it. A dict copy "
                    "of whatever rows exist; makes no claim about their statistics.")
        if "_rep_rvs" in s:
            return ("PER_ROW",
                    "Collects each replica's record for pooling. _pool_replica_rvs is told "
                    "already_resampled=_rep_fairdraw -- the PER-REPLICA sequence, captured "
                    "beside each record from that pass's own _rvs_is_fairdraw marker -- and "
                    "forces flat within-block weights for the blocks that were resampled, so "
                    "the resampling is accounted for THERE rather than here. (This text used "
                    "to say bool(opts.fairdraw_extrinsic_output); that is the CLI flag, which "
                    "is precisely the Finding-6 defect the sequence exists to avoid. A verdict "
                    "whose reason describes a mechanism the code does not use certifies "
                    "nothing.)")

        if "extrinsic_handoff" in s or (s == "_rvs = sampler._rvs"):
            return ("FIXED",
                    "Extrinsic-proposal breadcrumb: fits a GMM to _rvs rows with "
                    "log_weights=_lw, where _lw is rebuilt from those same rows. Under "
                    "--fairdraw-extrinsic-output the rows are ALREADY a w-proportional draw, so "
                    "the proposal is fitted to w^2 and comes out over-concentrated -- and it is "
                    "then handed to the NEXT iteration via --extrinsic-proposal-breadcrumb, "
                    "which is the truncated-support failure mode this whole line of work is "
                    "about. Same class as .dgrid/.dslice; propagates forward, so arguably the "
                    "worst of the three. FIXED: it now takes ln_weights_for_posterior, "
                    "which is uniform on a fair-drawn record.")
        if "_lnkey else np.array" in s or "_lnv = (np.asarray" in s or \
                ("_cols = (np.vstack" in s):
            return ("BENIGN",
                    "The DELIBERATE no-reserve fallback for a sampler that keeps none: reads "
                    "_rvs so the feature degrades to its previous behaviour rather than to no "
                    "seed at all. The rank hazard is still handled -- build_warm_seed puffs the "
                    "result to full rank -- so what remains is only 'fewer points', which "
                    "cannot be improved without a reserve.")
        if "(fair draw left" in s or ("len(np.asarray(sampler.identity_convert(sampler._rvs['log_integrand']))" in s):
            return ("BENIGN",
                    "Reports how many rows the fair draw left, for the log line that contrasts "
                    "it with the retained count. Reading the resample's size is the POINT here.")

    # --- MAP seed / coordinate transform / export, in all three ILE scripts ----------
    if f.startswith("bin/integrate_likelihood_extrinsic"):
        if "indx_guess" in s:
            return ("BENIGN",
                    "MAP-seed read: argmax over the record, then the SAME row's coordinates. "
                    "The fair draw picks rows proportional to weight, so the retained argmax is "
                    "a genuinely-drawn high-likelihood point; it seeds a local search and a "
                    "printed diagnostic, and no downstream number is a statistic of it. "
                    "Degraded (it may not be the global MAP), not wrong.")
        if "numpy.arccos" in s or "numpy.pi/2 -" in s or 'sampler._rvs[("declination"' in s:
            return ("PER_ROW",
                    "In-place coordinate transform, row for row.")
        if "copy.deepcopy(sampler._rvs)" in s or s.startswith("samples = sampler._rvs") \
                or "for key in sampler._rvs.keys()" in s:
            return ("BENIGN",
                    "THE export itself. Under --fairdraw-extrinsic-output the fair draw is "
                    "precisely what these rows are supposed to be, so the resample is the "
                    "correct input here rather than a hazard.")
        if 'len(sampler._rvs["psi"])' in s:
            return ("PER_ROW",
                    "Builds a constant t_ref column conformal with the exported rows. Uses the "
                    "length only to match shape, and makes no claim about it.")
        if "lnL at start" in s:
            return ("BENIGN",
                    "Printed diagnostic of the best retained sample. Explicitly labelled as "
                    "what ILE reports including weights; not an input to any result.")
        if '"distance" not in sampler._rvs' in s or "in sampler._rvs" in s \
                or 'sampler._rvs["psi"],' in s:
            return ("PER_ROW",
                    "Key-presence / column reference, not a population statistic.")

    # --- CIP: no fair draw happens on these samplers ---------------------------------
    if f.startswith("bin/util_Construct"):
        return ("NO_FAIRDRAW",
                "Verified by grep: neither util_ConstructIntrinsicPosterior_GenericCoordinates "
                "nor util_ConstructEOSPosterior passes igrand_fairdraw_samples, so integrate() "
                "never runs the rebind and _rvs is still the retained set. (The known CIP "
                "posterior-export defect was a different mechanism -- replace=False successive "
                "sampling, PR #44 lineage -- not this one.)")

    return (None, None)


ledger, todo = {}, []
for h in hits:
    k = A.site_key(h)
    v, why = verdict(h)
    if v is None:
        todo.append((k, h["file"], h["line"], h["source"]))
        continue
    ledger[k] = {"verdict": v, "why": why, "source": h.get("source", "")}

path = A.LEDGER_PATH
with open(path, "w") as f:
    json.dump(ledger, f, indent=2, sort_keys=True)
    f.write("\n")

from collections import Counter
print("wrote", len(ledger), "verdicts:", dict(Counter(v["verdict"] for v in ledger.values())))
if todo:
    print("\nUNMATCHED (left TODO, will fail --check):")
    for k, f_, ln, src in todo:
        print("  {}:{}  {}".format(f_, ln, " ".join(src.split())[:110]))
