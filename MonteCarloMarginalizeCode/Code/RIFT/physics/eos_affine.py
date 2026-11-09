"""Affine parameter along a fixed EOS's neutron-star sequence.

``mass`` (the default) is the historical fixed-EOS path: each component mass
maps to one tidal deformability through ``eos.lambda_from_m(m)``.  Nothing in
this module is used in that mode.

``central`` admits every stable branch.  The prior is dM per stable branch:
a mass with n(m) stable branches counts n(m) times.  The integrand is
Monte-Carlo summed over branch labels with one auxiliary uniform variate per
component, u_i in [0, 1):

    B_i = eos.branches_for_m(m_i),  n_i = len(B_i)
    b_i = B_i[floor(u_i n_i)],      Lambda_i = eos.lambda_from_m(m_i, branch_id=b_i)
    integrand *= n_1 n_2

so that  int du_i  n_i f(B_i[floor(u_i n_i)]) = sum_{b in B_i} f(b).

A mass on no stable branch, or at or above the all-branch maximum mass, is
rejected exactly as the mass path rejects m >= eos.mMaxMsun.
"""

import inspect
import math

import numpy as np

AFFINE_CHOICES = ("mass", "central")
CENTRAL_AUX_NAMES = ("eos_branch_u1", "eos_branch_u2")
NO_BRANCH = -1   # branch label recorded for a component without matter


def validate_eos_affine_request(affine, using_eos=None, using_eos_branch=None,
                                tabular_eos_file=None, using_eos_for_prior=False,
                                assume_eos_but_primary_bh=False, no_matter1=False,
                                oracle_reference_sample_file=None, sampler_oracle=None):
    """Raise ValueError for option combinations that central mode cannot honor."""
    if affine not in AFFINE_CHOICES:
        raise ValueError("--eos-affine-parameter must be one of {}".format(AFFINE_CHOICES))
    if affine == "mass":
        return
    if using_eos is None:
        raise ValueError("--eos-affine-parameter central requires --using-eos")
    if using_eos_branch is not None:
        raise ValueError(
            "--eos-affine-parameter central sums over every stable branch; "
            "it cannot be combined with --using-eos-branch")
    if tabular_eos_file is not None:
        raise ValueError(
            "--eos-affine-parameter central is incompatible with --tabular-eos-file: "
            "the fit over the ordering coordinate cannot represent branch labels")
    if using_eos_for_prior:
        raise ValueError(
            "--eos-affine-parameter central is not supported with --using-eos-for-prior; "
            "the EOS hyperprior plugin contract does not carry branch identity")
    if assume_eos_but_primary_bh and not no_matter1:
        raise ValueError(
            "--eos-affine-parameter central with --assume-eos-but-primary-bh also "
            "requires --no-matter1, so the integrand and the exported samples agree")
    if oracle_reference_sample_file or sampler_oracle:
        raise ValueError(
            "--eos-affine-parameter central does not support sampler oracles: "
            "reference samples carry no branch variates {}".format(CENTRAL_AUX_NAMES))


def _accepts_branch_id(fn):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "branch_id" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def all_branch_max_mass_msun(eos):
    """Largest stable mass over every branch, in Msun.

    ``getattr(eos, 'mMaxMsun_all', eos.mMaxMsun)``: NMB sequences expose
    ``mMaxMsun_all`` beside a primary-branch ``mMaxMsun``; LALSimulation
    families have no ``mMaxMsun_all`` because their ``mMaxMsun`` is already
    the all-branch maximum.
    """
    value = getattr(eos, "mMaxMsun_all", None)
    if value is None:
        value = getattr(eos, "mMaxMsun", None)
    if value is None:
        raise TypeError(
            "--eos-affine-parameter central requires the EOS to expose its "
            "all-branch maximum mass (mMaxMsun_all, or mMaxMsun)")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("EOS all-branch maximum mass is invalid: {!r}".format(value))
    return value


def require_branch_surface(eos):
    """Configuration check: central mode needs branch enumeration and selection."""
    if eos is None:
        raise TypeError("--eos-affine-parameter central requires an EOS object")
    if not callable(getattr(eos, "branches_for_m", None)):
        raise TypeError(
            "--eos-affine-parameter central requires an EOS with branches_for_m(m); "
            "{} has none".format(type(eos).__name__))
    if not _accepts_branch_id(getattr(eos, "lambda_from_m", None)):
        raise TypeError(
            "--eos-affine-parameter central requires lambda_from_m(m, branch_id=...); "
            "{} does not accept branch_id".format(type(eos).__name__))
    return all_branch_max_mass_msun(eos)


def select_branch(eos, m_msun, u, m_max_all):
    """Return (Lambda, branch_id, n) for one component; Lambda=-inf rejects.

    Every failure (exception, nonfinite value, no branch) is a rejected
    sample, never an error, following the sample evaluation failure policy.
    """
    from RIFT.physics.lalsim_eos_compat import evaluate_eos_lambda
    import lal
    if not (math.isfinite(m_msun) and m_msun > 0 and m_msun < m_max_all):
        return -np.inf, NO_BRANCH, 0
    try:
        branches = list(eos.branches_for_m(m_msun))
    except Exception:
        return -np.inf, NO_BRANCH, 0
    n = len(branches)
    if n == 0:
        return -np.inf, NO_BRANCH, 0
    if not (math.isfinite(u) and 0 <= u <= 1):
        return -np.inf, NO_BRANCH, n
    branch = int(branches[min(int(math.floor(u * n)), n - 1)])
    lam = evaluate_eos_lambda(eos, m_msun * lal.MSUN_SI, use_vector=False, branch_id=branch)
    return float(lam), branch, n


def convert_waveform_coordinates_with_eos_central(
        x_in, u_in, coord_names=('mc', 'eta'), low_level_coord_names=('m1', 'm2'),
        enforce_kerr=False, eos_class=None, m_max_all=None, no_matter1=False,
        no_matter2=False, source_redshift=0, fref=None):
    """Central-mode twin of ``lalsimutils.convert_waveform_coordinates_with_eos``.

    Returns ``(x_out, log_weight, branch_ids)``.  ``x_out`` rows are -inf when
    rejected (same convention as the mass path); ``log_weight`` is
    ln n_1 + ln n_2 (0 on rejected rows and for components without matter);
    ``branch_ids`` has shape (N, 2), ``NO_BRANCH`` where none applies.
    """
    from RIFT import lalsimutils
    assert eos_class is not None
    if m_max_all is None:
        m_max_all = all_branch_max_mass_msun(eos_class)
    x_in = np.atleast_2d(np.asarray(x_in, dtype=float))
    u_in = np.atleast_2d(np.asarray(u_in, dtype=float))
    n_rows = len(x_in)
    x_out = np.zeros((n_rows, len(coord_names)))
    log_weight = np.zeros(n_rows)
    branch_ids = np.full((n_rows, 2), NO_BRANCH, dtype=int)
    for row in range(n_rows):
        P = lalsimutils.ChooseWaveformParams()
        if fref is not None:
            P.fref = fref
        # As in the mass path: P.m1, P.m2 are in Msun here.
        for indx, name in enumerate(low_level_coord_names):
            P.assign_param(name, x_in[row, indx])
        log_w = 0.
        ok = True
        for comp, (no_matter, mass) in enumerate(((no_matter1, P.m1), (no_matter2, P.m2))):
            if no_matter:
                lam = 0.
            else:
                lam, branch, n = select_branch(eos_class, float(mass), float(u_in[row, comp]), m_max_all)
                branch_ids[row, comp] = branch
                if not np.isfinite(lam):
                    ok = False
                else:
                    log_w += math.log(n)
            if comp == 0:
                P.lambda1 = lam
            else:
                P.lambda2 = lam
        if not ok:
            x_out[row] = -np.inf
            branch_ids[row] = NO_BRANCH
            continue
        log_weight[row] = log_w
        P.m1 = P.m1 * (1 + source_redshift)
        P.m2 = P.m2 * (1 + source_redshift)
        for indx, name in enumerate(coord_names):
            x_out[row, indx] = P.extract_param(name)
        if enforce_kerr and (P.extract_param('chi1') > 1 or P.extract_param('chi2') > 1):
            x_out[row] = -np.inf
            log_weight[row] = 0.
    return x_out, log_weight, branch_ids


def make_central_log_likelihood(fit, convert_central, log_prior_scale, n_native):
    """Integrand ln L for central mode, as CIP passes it to every sampler.

    Arguments are the ``n_native`` sampling coordinates followed by the two
    branch variates.  ``convert_central(x_native, u)`` returns
    ``(x_fit, log_weight)``; ``fit`` must return -inf for nonfinite rows.
    The per-branch prior weight n_1 n_2 enters here, beside the existing
    ``log_prior_scale`` correction, so it multiplies both the evidence and
    the posterior weights for every sampler.
    """
    def log_likelihood_function(*args):
        if len(args) != n_native + len(CENTRAL_AUX_NAMES):
            raise TypeError("central EOS integrand expects {} arguments, got {}".format(
                n_native + len(CENTRAL_AUX_NAMES), len(args)))
        cols = [np.atleast_1d(np.asarray(x, dtype=float)) for x in args]
        x_native = np.column_stack(cols[:n_native])
        u = np.column_stack(cols[n_native:])
        x_fit, log_weight = convert_central(x_native, u)
        result = np.asarray(fit(x_fit), dtype=float).reshape(-1) + log_weight + log_prior_scale(x_native)
        if all(np.ndim(x) == 0 for x in args):
            return result[0]
        return result

    def likelihood_function(*args):
        return np.exp(log_likelihood_function(*args))

    return log_likelihood_function, likelihood_function


def unit_uniform_prior(x):
    """Normalized U[0, 1) prior for the branch variates."""
    return np.ones_like(x, dtype=float)
