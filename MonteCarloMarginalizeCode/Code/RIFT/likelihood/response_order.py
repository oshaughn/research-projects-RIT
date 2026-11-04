"""Physics-based response-series order estimates from the precomputed U,V bank.

This module is deliberately not imported by either ILE driver unless one of the
response-order check/choose command-line options is active.  The diagnostic is
data independent: it contracts only model self-overlaps, never Q=<h|d>.

The estimate uses a deterministic, full-prior angular design.  It is therefore
an estimate, not a mathematical supremum over the prior.  Increase
``n_samples`` and the reference order for a production convergence check.
"""
from __future__ import print_function, division

import math
import warnings

import numpy as np


def _angular_design(n_samples):
    """Deterministic five-dimensional Halton design (no RNG state)."""
    n = max(8, int(n_samples))
    def radical_inverse(base):
        out = np.zeros(n, dtype=float)
        denominator = 1.0
        integer = np.arange(1, n + 1, dtype=np.int64)
        while np.any(integer):
            integer, digit = np.divmod(integer, base)
            denominator *= base
            out += digit / denominator
        return out
    u_ra, u_dec, u_inc, u_psi, u_phase = [
        radical_inverse(base) for base in (2, 3, 5, 7, 11)]
    ra = 2.0 * np.pi * u_ra
    dec = np.arcsin(1.0 - 2.0 * u_dec)
    incl = np.arccos(1.0 - 2.0 * u_inc)
    psi = np.pi * u_psi
    phiref = 2.0 * np.pi * u_phase
    return ra, dec, incl, psi, phiref


def reference_bank_size(feature, p_max, q_max, lmax, n_detectors):
    """Dense-U,V planning estimate using every mode through lmax."""
    p_max, q_max = int(p_max), int(q_max)
    if feature == 'rotation':
        basis = (p_max + 1) * (2 * (2 + p_max) + 1)
    elif feature == 'finite':
        basis = q_max + 2
    elif feature == 'combined':
        basis = sum(
            2 * ((2 if b == 0 else b + 1) + p) + 1
            for b in range(q_max + 2) for p in range(p_max + 1))
    else:
        raise ValueError("unknown response feature %r" % feature)
    modes = sum(2 * ell + 1 for ell in range(2, int(lmax) + 1))
    uv_bytes = 2 * basis ** 2 * modes ** 2 * 16 * int(n_detectors)
    return dict(basis=basis, modes=modes, uv_gib=uv_bytes / 2.0 ** 30)


def guard_reference_bank(feature, p_max, q_max, lmax, n_detectors,
                         max_bank_gib=4.0):
    """Refuse an oversized diagnostic before building its response bank."""
    if not np.isfinite(max_bank_gib) or float(max_bank_gib) <= 0:
        raise ValueError("response-order max-bank-gib must be finite and positive")
    size = reference_bank_size(feature, p_max, q_max, lmax, n_detectors)
    if size['uv_gib'] > float(max_bank_gib):
        raise ValueError(
            "response-order dense-U,V planning estimate is {:.3g} GiB "
            "(basis={}, all modes through lmax={}, detectors={}); this estimate "
            "excludes dictionary overhead and Q banks. Lower the diagnostic reference "
            "orders or raise --response-order-max-bank-gib explicitly".format(
                size['uv_gib'], size['basis'], size['modes'], int(n_detectors)))
    return size


def _feature(meta):
    if meta.get('feature') == 'rotation_freqresponse':
        return 'combined'
    if 'p_list' in meta:
        return 'finite'
    if 'a_list' in meta:
        return 'rotation'
    raise ValueError("unrecognized response bank metadata")


def _indices(meta):
    return list(meta['p_list'] if _feature(meta) == 'finite' else meta['a_list'])


def _matrix(bank, det, a, ap, ia, iap, modes):
    item = bank[det]
    block = item[(a, ap)] if isinstance(item, dict) else item[ia, iap]
    if isinstance(block, dict):
        return np.asarray([[block[(m1, m2)] for m2 in modes] for m1 in modes],
                          dtype=complex)
    return np.asarray(block)


def _arm_for_detector(meta, det):
    arm = meta.get('L_arm', None)
    return arm.get(det, None) if isinstance(arm, dict) else arm


def _coefficients(meta, det, ra, dec, psi):
    """Return physical coefficients C and reflected coefficients C_R."""
    feature = _feature(meta)
    tref = float(meta.get('event_time_geo', meta.get('tref', 0.0)))
    if feature == 'finite':
        from . import factored_likelihood_freqresponse as fr
        rows = [fr.response_coefficients(
            det, float(r), float(d), float(p), tref, int(meta['Qmax']),
            L_arm=_arm_for_detector(meta, det))
            for r, d, p in zip(ra, dec, psi)]
        idx = _indices(meta)
        c = np.column_stack([[row.get(a, 0j) for row in rows] for a in idx])
        return c, c

    if feature == 'combined':
        from . import factored_likelihood_rotating_freqresponse as rf
        from . import factored_likelihood_with_rotation as rot
        coeff = rf.combined_response_coefficients_vector(
            det, ra, dec, psi, tref, int(meta['p_max']),
            Qmax=int(meta['Qmax']), L_arm=_arm_for_detector(meta, det))
        reflect = lambda a: (a[0], a[1], -a[2])
    else:
        from . import factored_likelihood_with_rotation as rot
        coeff = rot.rotation_coefficients_vector(
            det, ra, dec, psi, tref, int(meta['p_max']))
        reflect = lambda a: (a[0], -a[1])

    # Evaluate the norm at geocentric coalescence time.  The U,V model norm
    # owes the same arrival-time post-phase as the likelihood evaluator.
    import lal
    from . import factored_likelihood as fl
    location = fl.lalsim.DetectorPrefixToLALDetector(det).location
    delta = np.asarray(fl.TimeDelayFromEarthCenter(
        np.asarray(location), ra, dec,
        float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(tref))), xpy=np))
    omega = 2.0 * np.pi * float(meta['f_sidereal'])
    idx = _indices(meta)
    zero = np.zeros(len(ra), dtype=complex)
    c = np.column_stack([
        coeff.get(a, zero) * np.exp(1j * a[-1] * omega * delta) for a in idx])
    cr = np.column_stack([
        coeff.get(reflect(a), zero) *
        np.exp(1j * reflect(a)[-1] * omega * delta) for a in idx])
    return c, cr


def _retained(meta, p_max, q_max):
    feature = _feature(meta)
    if feature == 'finite':
        return np.asarray([a <= int(q_max) + 1 for a in _indices(meta)])
    if feature == 'rotation':
        # A production p bank uses one common harmonic width 2+p_max.
        # Remove the wider reference bank's identically-zero outer bands too,
        # otherwise auto-selection would be physically right but not minimal.
        return np.asarray([a[0] <= int(p_max) and abs(a[1]) <= 2 + int(p_max)
                           for a in _indices(meta)])
    return np.asarray([a[0] <= int(q_max) + 1 and a[1] <= int(p_max)
                       for a in _indices(meta)])


def basis_size(meta, p_max=None, q_max=None):
    """Number of response basis elements retained by an order pair."""
    feature = _feature(meta)
    if p_max is None:
        p_max = int(meta.get('p_max', 0))
    if q_max is None:
        q_max = int(meta.get('Qmax', 0))
    return int(np.sum(_retained(meta, p_max, q_max)))


def pack_uv_from_raw(meta, cross, cross_v):
    """Pack only model moments, deliberately skipping the large Q=<h|d> bank."""
    indices = _indices(meta)
    modes = list(meta['modes'])

    def one(source):
        out = {}
        for det, pairs in source.items():
            out[det] = {}
            for a in indices:
                for ap in indices:
                    block = pairs[(a, ap)]
                    if isinstance(block, dict):
                        out[det][(a, ap)] = np.asarray(
                            [[block[(m1, m2)] for m2 in modes] for m1 in modes],
                            dtype=complex)
                    else:
                        out[det][(a, ap)] = np.asarray(block)
        return out
    return one(cross), one(cross_v)


def estimate_response_orders(meta, U, V, target_snr, lnL_tolerance=0.1,
                             n_samples=128, selected_p=None, selected_q=None,
                             vary_p=True, vary_q=True):
    """Scan truncations of one reference bank and return an order report.

    A candidate passes when max_theta ||h_ref-h_candidate||^2/||h_ref||^2
    is at most ``2*lnL_tolerance/target_snr**2``.  ``U`` and ``V`` are the
    raw or packed response-bank model-overlap dictionaries (or dense compound arrays).
    """
    if not np.isfinite(target_snr) or target_snr <= 0:
        raise ValueError("target_snr must be positive")
    if not np.isfinite(lnL_tolerance) or lnL_tolerance <= 0:
        raise ValueError("lnL_tolerance must be positive")

    from . import factored_likelihood as fl
    feature = _feature(meta)
    idx = _indices(meta)
    modes = list(meta['modes'])
    lookup = np.asarray(modes, dtype=int)
    ra, dec, incl, psi, phiref = _angular_design(n_samples)
    Y = np.asarray(fl.ComputeYlmsArrayVector(lookup, incl, -phiref)).T
    ns = len(ra)

    detector_terms = []
    for det in U:
        c, cr = _coefficients(meta, det, ra, dec, psi)
        # Precontract the small mode matrices once.  The response-order scan is
        # then only a quadratic contraction over basis indices.
        um = np.empty((len(idx), len(idx), ns), dtype=complex)
        vm = np.empty_like(um)
        for ia, a in enumerate(idx):
            for iap, ap in enumerate(idx):
                um[ia, iap] = np.einsum(
                    'si,ij,sj->s', np.conj(Y),
                    _matrix(U, det, a, ap, ia, iap, modes), Y)
                vm[ia, iap] = np.einsum(
                    'si,ij,sj->s', Y,
                    _matrix(V, det, a, ap, ia, iap, modes), Y)
        detector_terms.append((c, cr, um, vm))

    def norm_for_mask(mask):
        out = np.zeros(ns, dtype=float)
        for c, cr, um, vm in detector_terms:
            cm = c * mask[None, :]
            crm = cr * mask[None, :]
            val = np.einsum('sa,sb,abs->s', np.conj(cm), cm, um)
            val += np.einsum('sa,sb,abs->s', crm, cm, vm)
            out += 0.5 * np.real(val)
        # Roundoff can produce tiny negative self norms.
        return np.maximum(out, 0.0)

    full = norm_for_mask(np.ones(len(idx), dtype=bool))
    good = full > max(np.max(full), 1.0) * 1e-13
    if not np.any(good):
        raise ValueError("reference response has zero norm on the angular design")
    threshold = 2.0 * float(lnL_tolerance) / float(target_snr) ** 2
    p_ref = int(meta.get('p_max', 0))
    q_ref = int(meta.get('Qmax', 0))
    if selected_p is None:
        selected_p = p_ref
    if selected_q is None:
        selected_q = q_ref
    p_values = range(p_ref + 1) if vary_p and feature != 'finite' else [int(selected_p)]
    q_values = range(q_ref + 1) if vary_q and feature != 'rotation' else [int(selected_q)]

    rows = []
    for p in p_values:
        for q in q_values:
            keep = _retained(meta, p, q)
            residual = norm_for_mask(~keep)
            mu = float(np.max(residual[good] / full[good]))
            nb = int(np.sum(keep))
            rows.append(dict(p_max=int(p), Qmax=int(q), max_mu=mu,
                             max_delta_lnL=0.5 * target_snr ** 2 * mu,
                             basis=nb, uv_pairs=nb * nb,
                             passes=bool(mu <= threshold)))
    selected_keep = _retained(meta, selected_p, selected_q)
    selected_mu = float(np.max(norm_for_mask(~selected_keep)[good] / full[good]))

    # A finite-reference resolution diagnostic.  It is deliberately not called
    # an analytic tail bound: require two decreasing shells, extrapolate their
    # norm ratio geometrically, and reserve one quarter of the error budget.
    shell_rows = []
    def add_shell(axis, final_mask, previous_mask):
        final_mu = float(np.max(norm_for_mask(final_mask)[good] / full[good]))
        previous_mu = float(np.max(norm_for_mask(previous_mask)[good] / full[good]))
        if previous_mu <= 0.0:
            ratio = 0.0 if final_mu <= 0.0 else float('inf')
        else:
            ratio = math.sqrt(final_mu / previous_mu)
        tail_mu = ((math.sqrt(final_mu) * ratio / (1.0 - ratio)) ** 2
                   if np.isfinite(ratio) and ratio < 1.0 else float('inf'))
        shell_rows.append(dict(axis=axis, final_mu=final_mu,
                               previous_mu=previous_mu, norm_ratio=ratio,
                               extrapolated_tail_mu=tail_mu))
    if vary_p and feature != 'finite' and p_ref >= 2:
        add_shell('p',
                  _retained(meta, p_ref, q_ref) & ~_retained(meta, p_ref - 1, q_ref),
                  _retained(meta, p_ref - 1, q_ref) & ~_retained(meta, p_ref - 2, q_ref))
    elif vary_p and feature != 'finite':
        shell_rows.append(dict(axis='p', final_mu=float('inf'),
                               previous_mu=float('inf'), norm_ratio=float('inf'),
                               extrapolated_tail_mu=float('inf')))
    if vary_q and feature != 'rotation' and q_ref >= 2:
        add_shell('q',
                  _retained(meta, p_ref, q_ref) & ~_retained(meta, p_ref, q_ref - 1),
                  _retained(meta, p_ref, q_ref - 1) & ~_retained(meta, p_ref, q_ref - 2))
    elif vary_q and feature != 'rotation':
        shell_rows.append(dict(axis='q', final_mu=float('inf'),
                               previous_mu=float('inf'), norm_ratio=float('inf'),
                               extrapolated_tail_mu=float('inf')))
    edge_mu = max((row['final_mu'] for row in shell_rows), default=0.0)
    tail_mu = sum(math.sqrt(row['extrapolated_tail_mu']) for row in shell_rows) ** 2
    reference_resolved = bool(shell_rows) and all(
        row['norm_ratio'] <= 0.5 and row['final_mu'] <= 0.25 * threshold
        for row in shell_rows) and tail_mu <= 0.25 * threshold

    # Charge the unresolved part of the response series to every candidate by
    # the triangle inequality.  The finite-reference residual and extrapolated
    # tail can add coherently, so their powers must not be added directly.
    for row in rows:
        row['finite_reference_mu'] = row['max_mu']
        row['max_mu'] = (
            math.sqrt(row['finite_reference_mu']) + math.sqrt(tail_mu)) ** 2
        row['max_delta_lnL'] = 0.5 * target_snr ** 2 * row['max_mu']
        row['passes'] = bool(row['max_mu'] <= threshold)
    selected_finite_mu = selected_mu
    selected_mu = (math.sqrt(selected_finite_mu) + math.sqrt(tail_mu)) ** 2
    passing = [row for row in rows if row['passes']]
    chosen = min(passing, key=lambda x: (x['uv_pairs'], x['basis'],
                                         x['p_max'] + x['Qmax'])) if passing else None

    mode_block_sensitivity = []
    blocks = {}
    for im, mode in enumerate(modes):
        blocks.setdefault((int(mode[0]), abs(int(mode[1]))), []).append(im)
    for block, members in sorted(blocks.items()):
        # Pair +/-m modes before taking the norm; this is diagnostic only and
        # never prunes the production bank.
        out = np.zeros(ns, dtype=float)
        Ym = np.zeros_like(Y)
        Ym[:, members] = Y[:, members]
        for det in U:
            c, cr = _coefficients(meta, det, ra, dec, psi)
            for ia, a in enumerate(idx):
                for iap, ap in enumerate(idx):
                    uu = np.einsum('si,ij,sj->s', np.conj(Ym),
                                   _matrix(U, det, a, ap, ia, iap, modes), Ym)
                    vv = np.einsum('si,ij,sj->s', Ym,
                                   _matrix(V, det, a, ap, ia, iap, modes), Ym)
                    out += 0.5 * np.real(np.conj(c[:, ia]) * c[:, iap] * uu
                                         + cr[:, ia] * c[:, iap] * vv)
        mu_mode = float(np.max(np.maximum(out, 0.0)[good] / full[good]))
        mode_block_sensitivity.append((block, mu_mode))

    return dict(feature=feature, target_snr=float(target_snr),
                lnL_tolerance=float(lnL_tolerance), threshold_mu=threshold,
                reference_p=p_ref, reference_Q=q_ref,
                selected_p=int(selected_p), selected_Q=int(selected_q),
                selected_mu=selected_mu,
                selected_finite_reference_mu=selected_finite_mu,
                selected_delta_lnL=0.5 * target_snr ** 2 * selected_mu,
                selected_passes=bool(selected_mu <= threshold), chosen=chosen,
                rows=rows, reference_edge_mu=edge_mu,
                reference_tail_mu=tail_mu, reference_shells=shell_rows,
                reference_resolved=bool(reference_resolved),
                n_samples=ns, n_valid_samples=int(np.sum(good)),
                mode_block_sensitivity=sorted(
                    mode_block_sensitivity, key=lambda item: item[1], reverse=True))


def print_order_report(report, prefix='response-order'):
    """Stable one-line diagnostics suitable for scheduler stdout."""
    chosen = report['chosen']
    choice = 'NONE' if chosen is None else 'p_max={p_max} Qmax={Qmax}'.format(**chosen)
    print('[{}] SNR={:.6g} eps_lnL={:.3g} samples={}/{} choice={} '
          'selected_DeltaLnL={:.3g} reference_edge_DeltaLnL={:.3g}'.format(
              prefix, report['target_snr'], report['lnL_tolerance'],
              report['n_valid_samples'], report['n_samples'], choice,
              report['selected_delta_lnL'],
              0.5 * report['target_snr'] ** 2 * report['reference_edge_mu']))
    if not report['reference_resolved']:
        warnings.warn(
            '{} finite reference is unresolved: two-shell decrease and '
            'extrapolated-tail tests do not fit inside the reserved likelihood-'
            'error budget; raise the diagnostic reference order'.format(prefix),
            RuntimeWarning)


def truncate_precompute_products(products, p_max=None, q_max=None):
    """Return a response precompute tuple restricted to the selected rectangle."""
    rint, cross, cross_v, rho, meta0 = products
    meta = dict(meta0)
    feature = _feature(meta)
    if p_max is None:
        p_max = int(meta.get('p_max', 0))
    if q_max is None:
        q_max = int(meta.get('Qmax', 0))
    keep = _retained(meta, p_max, q_max)
    old = _indices(meta)
    new = [a for a, yes in zip(old, keep) if yes]
    if feature == 'finite':
        meta['p_list'] = new
        meta['Qmax'] = int(q_max)
    else:
        meta['a_list'] = new
        meta['p_max'] = int(p_max)
        if feature == 'combined':
            meta['Qmax'] = int(q_max)
        else:
            meta['harmonics'] = tuple(sorted(set(a[1] for a in new)))

    def restrict_primary(bank):
        return {det: {a: values[a] for a in new} for det, values in bank.items()}

    def restrict_cross(bank):
        return {det: {(a, ap): values[(a, ap)] for a in new for ap in new}
                for det, values in bank.items()}

    return (restrict_primary(rint), restrict_cross(cross),
            restrict_cross(cross_v), restrict_primary(rho), meta)
