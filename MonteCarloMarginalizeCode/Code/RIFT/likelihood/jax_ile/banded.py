"""
Builders for the multi-band (slow-rotation / finite-size) JAX likelihood data.

These take the SAME packed precompute banks the cupy NoLoop path uses -- the
outputs of ``factored_likelihood_with_rotation.pack_rotation_arrays`` (Path A/B)
and ``factored_likelihood_freqresponse.pack_freqresponse_arrays`` (Path D) -- and
wrap them into a :class:`~RIFT.likelihood.jax_ile.core.JAXLikelihoodData` tagged
with a ``feature`` so that :func:`core._accumulate_unit` routes through the
multi-band accumulator.  Because that accumulator returns the identical
``(kappa_unit, rho_sq_unit)`` contract as the baseline, every marginalization
variant (distance / phi_ref / psi) works with the feature unchanged.

The heavy, data-touching precompute (frame reading, ``<h_lm|d>``, U/V, packing)
is reused verbatim -- only the cheap extrinsic->lnL contraction is JAX.
"""

import numpy as np
import jax.numpy as jnp

import lal
import lalsimulation as lalsim

from .core import build_likelihood_data, DIST_MPC_REF
from . import response_slowrot as _rs
from . import response_freqresponse as _rf


def _stack_bank(by_key, keys, det):
    """Stack ``by_key[det][k]`` over the ordered ``keys`` -> leading axis = band."""
    return np.stack([np.asarray(by_key[det][k], dtype=np.complex128) for k in keys],
                    axis=0)


def _base_data(packed_scalar, deltaT, tref, tvals, distMpcRef):
    """Build the JAXLikelihoodData scaffold (scalars + minimal detector dict).

    We reuse the baseline container for the time grid / Simpson weights / gmst /
    epoch bookkeeping, then attach the banded arrays and geometry below.
    """
    return build_likelihood_data(packed_scalar, deltaT, tref, tvals, distMpcRef)


def build_rotation_data(meta, lookupNKDict, rho_by_a, U_by_aa, V_by_aa, epochDict,
                        deltaT, tvals, distMpcRef=DIST_MPC_REF):
    """Banded JAXLikelihoodData for the slow-rotation (Path A/B) likelihood.

    Parameters mirror ``pack_rotation_arrays`` outputs plus the time grid.
    ``meta`` carries ``a_list`` (ordered ``(p,n)``), ``harmonics``, ``p_max`` and
    ``event_time_geo`` (the fiducial epoch / sidereal reference ``tref``).
    """
    a_list = [(int(p), int(n)) for (p, n) in meta["a_list"]]
    tref = float(meta["event_time_geo"])
    detectors = list(rho_by_a.keys())

    # The bank convention: Q^a = <chi_a(.-t)|d> against UNTOUCHED data, so the evaluator
    # owes the arrival-time post-phase C~_a = C_a exp(i n_a Omega (t - tref)) on BOTH the
    # data term and the model norm (rotation_post_phase).  core._accumulate_unit_banded
    # implements exactly that convention and nothing else, so refuse a bank that does not
    # declare it rather than silently evaluating the wrong likelihood.
    if not bool(meta.get("post_phase_required", False)):
        raise ValueError(
            "build_rotation_data requires meta['post_phase_required'] == True: the JAX "
            "rotation evaluator applies the arrival-time post-phase (rotation_post_phase) "
            "to both the data term and the model norm, which is only correct for a bank "
            "built in that convention.  Got meta['post_phase_required']=%r.\n"
            "That key is set by PrecomputeLikelihoodTermsWithRotation as of PR #117, which "
            "is the REQUIRED PARENT of this code -- if you are seeing this, the tree most "
            "likely does not carry #117, in which case its precompute still uses the old "
            "convention and the JAX rotation path must not be used on it at all (merge or "
            "cherry-pick #117 first).  If the tree does carry #117, regenerate the bank "
            "with PrecomputeLikelihoodTermsWithRotation rather than hand-assembling meta."
            % (meta.get("post_phase_required"),))

    # Minimal baseline-shaped packed dict (rholmArray of the FIRST band as a
    # stand-in) so build_likelihood_data can set up lms/epoch/location/response.
    a0 = a_list[0]
    packed_scalar = {}
    for det in detectors:
        packed_scalar[det] = dict(
            lms=np.asarray(lookupNKDict[det]),
            rholmArray=np.asarray(rho_by_a[det][a0], dtype=np.complex128),
            U=np.asarray(U_by_aa[det][(a0, a0)], dtype=np.complex128),
            V=np.asarray(V_by_aa[det][(a0, a0)], dtype=np.complex128),
            epoch=float(epochDict[det]))
    data = _base_data(packed_scalar, deltaT, tref, tvals, distMpcRef)

    # Attach the full band banks + geometry.
    pairs = [(a, ap) for a in a_list for ap in a_list]  # unused; kept for clarity
    for det in detectors:
        dd = data.detectors[det]
        Q_bank = _stack_bank(rho_by_a, a_list, det)             # (A, K, npts_full)
        dd["Q_bank"] = jnp.asarray(np.ascontiguousarray(
            np.transpose(Q_bank, (0, 2, 1))))                   # (A, npts_full, K)
        A = len(a_list)
        K = len(dd["lms"])
        U = np.empty((A, A, K, K), dtype=np.complex128)
        V = np.empty((A, A, K, K), dtype=np.complex128)
        for i, a in enumerate(a_list):
            for j, ap in enumerate(a_list):
                U[i, j] = np.asarray(U_by_aa[det][(a, ap)], dtype=np.complex128)
                V[i, j] = np.asarray(V_by_aa[det][(a, ap)], dtype=np.complex128)
        dd["U_bank"] = jnp.asarray(U)
        dd["V_bank"] = jnp.asarray(V)

    m_values, pp_term1_idx, pp_term2_idx = _rs.post_phase_bucketing(a_list)

    data.feature = "rotation"
    data.band = dict(
        a_list=a_list,
        p_max=int(meta["p_max"]),
        harmonics=tuple(int(h) for h in meta["harmonics"]),
        refl_idx=np.asarray(_rs.reflection_index(a_list), dtype=np.int64),
        # Arrival-time post-phase (see _rs.post_phase_bucketing): omega and the static
        # m-bucket maps the accumulator needs to build exp(i m omega (t - tref)).
        f_sidereal=float(meta["f_sidereal"]),
        post_phase_required=True,
        pp_m_values=np.asarray(m_values, dtype=np.int64),
        pp_term1_idx=np.asarray(pp_term1_idx, dtype=np.int64),
        pp_term2_idx=np.asarray(pp_term2_idx, dtype=np.int64),
    )
    return data


def build_freqresponse_data(meta, lookupNKDict, rho_by_p, U_by_pp, V_by_pp,
                            epochDict, deltaT, tvals, det_geom,
                            distMpcRef=DIST_MPC_REF):
    """Banded JAXLikelihoodData for the finite-size (Path D) likelihood.

    Parameters mirror ``pack_freqresponse_arrays`` outputs plus the time grid.
    ``meta`` carries ``p_list`` (0..Qmax+1), ``Qmax`` and ``event_time_geo``.
    ``det_geom`` maps ``det -> (response, x_arm, y_arm, L)`` (from
    ``slowrot_freqresponse.detector_geometry``); the arm unit vectors enter the
    finite-size coefficients ``beta_q`` -- they are NOT recoverable from the LAL
    response tensor alone.
    """
    p_list = [int(p) for p in meta["p_list"]]
    tref = float(meta["event_time_geo"])
    detectors = list(rho_by_p.keys())

    p0 = p_list[0]
    packed_scalar = {}
    for det in detectors:
        packed_scalar[det] = dict(
            lms=np.asarray(lookupNKDict[det]),
            rholmArray=np.asarray(rho_by_p[det][p0], dtype=np.complex128),
            U=np.asarray(U_by_pp[det][(p0, p0)], dtype=np.complex128),
            V=np.asarray(V_by_pp[det][(p0, p0)], dtype=np.complex128),
            epoch=float(epochDict[det]))
    data = _base_data(packed_scalar, deltaT, tref, tvals, distMpcRef)

    for det in detectors:
        dd = data.detectors[det]
        Q_bank = _stack_bank(rho_by_p, p_list, det)             # (A, K, npts_full)
        dd["Q_bank"] = jnp.asarray(np.ascontiguousarray(
            np.transpose(Q_bank, (0, 2, 1))))                   # (A, npts_full, K)
        A = len(p_list)
        K = len(dd["lms"])
        U = np.empty((A, A, K, K), dtype=np.complex128)
        V = np.empty((A, A, K, K), dtype=np.complex128)
        for i, p in enumerate(p_list):
            for j, pp in enumerate(p_list):
                U[i, j] = np.asarray(U_by_pp[det][(p, pp)], dtype=np.complex128)
                V[i, j] = np.asarray(V_by_pp[det][(p, pp)], dtype=np.complex128)
        dd["U_bank"] = jnp.asarray(U)
        dd["V_bank"] = jnp.asarray(V)
        resp, x_arm, y_arm, L = det_geom[det]
        dd["x_arm"] = jnp.asarray(np.asarray(x_arm, dtype=np.float64))
        dd["y_arm"] = jnp.asarray(np.asarray(y_arm, dtype=np.float64))
        dd["L_arm"] = float(L)

    data.feature = "freqresponse"
    data.band = dict(
        p_list=p_list,
        Qmax=int(meta["Qmax"]),
        refl_idx=np.asarray(_rf.reflection_index(p_list), dtype=np.int64),
    )
    return data
