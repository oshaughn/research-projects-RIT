"""Analytic supplementary likelihood factors with CLOSED-FORM marginals.

Used by test_e2e_analytic_pipeline.py, and importable on its own so the same factor can be
pointed at a hand-run pipeline:

    --supplementary-likelihood-factor-code analytic_supplement_for_e2e \\
    --supplementary-likelihood-factor-function ln_analytic_factor

ILE calls the factor as f(ra, dec, phi_orb, inclination, psi, distance) and ADDS it to lnL,
with xpy=xpy_default at three of its five call sites and without it at the other two, so the
signature below takes xpy with a default.
Under --zero-likelihood the signal term is exactly 0, so the marginal likelihood ILE reports
for every intrinsic grid point is ln E_prior[exp(f)], which the factors here make exact.

THE FACTOR

    f = A cos(phi_orb) + B cos(iota)

with A from E2E_A_COEFF and B from E2E_B_COEFF (B = 0 drops the second term).  phi_orb's prior
is uniform on [0, 2pi) and inclination's is (1/2) sin(iota) on [0, pi], they are independent,
and the two terms depend on nothing else, so

    ln Z = ln I0(A) + ln( sinh(B)/B )

exactly -- I0 the modified Bessel function of the first kind.  A full pipeline run then has an
expected answer with no fit error and no MC scatter in the TARGET: the constant-integrand
technique of the sampler unit tests, lifted to the pipeline.

WHY TWO TERMS.  A cos(phi_orb) alone CANNOT detect a wiring error, which is the defect class
the code it guards actually has.  phi_orb, psi and right_ascension are all sampled uniformly
on [0, 2pi), so

    (1/2pi) int_0^2pi exp(A cos t) dt  =  I0(A)

comes out the same whichever of the three the factor is handed: feeding it kwargs['psi']
instead of kwargs['phi_orb'] passes every lane.  The B term is evaluated against inclination's
sin prior, whose marginal sinh(B)/B is a different function, so mis-routing inclination into
any of the three circular slots (or a circular angle into inclination's) changes ln Z by far
more than the gate's tolerance.

WHAT REMAINS UNDETECTABLE, and provably so: phi_orb, psi and right_ascension are independent
and identically distributed, so NO factor's marginal can distinguish a permutation among those
three.  Only a per-sample check could, and that is not what this file is for.

INCLINATION'S COORDINATE.  ILE hands a supplementary factor the RAW SAMPLED inputs (see the
contract where the factor is imported: "called with identical raw inputs (including
cosines/etc)").  Without --inclination-cosine-sampler the sampled variable is iota in radians,
so cos(iota) is computed here; with it, the sampled variable IS cos(iota) already and the
caller must set E2E_INCL_IS_COSINE=1.  The closed form is the SAME either way, so a lane run
with the flag and the env var set is a direct check that the raw contract is honoured: a
caller that applied the arccos itself would move ln Z off sinh(B)/B.
"""
import os

import numpy as np

A_COEFF = float(os.environ.get("E2E_A_COEFF", "0.75"))
B_COEFF = float(os.environ.get("E2E_B_COEFF", "0.0"))
# Set when --inclination-cosine-sampler is in use, i.e. when the raw sampled 'inclination' is
# already cos(iota) rather than iota.
INCL_IS_COSINE = os.environ.get("E2E_INCL_IS_COSINE", "0") not in ("", "0", "false", "False")


def ln_analytic_factor(right_ascension, declination, phi_orb, inclination, psi, distance,
                       xpy=np):
    # xpy HAS to be accepted, not merely ignored: three of ILE's five call sites pass
    # xpy=xpy_default and two do not, so a factor declared with the six positional arguments
    # alone raises TypeError on the vectorized paths.
    #
    # And it has to be USED, not just accepted.  The three sites that pass it do
    # `lnL += factor(...)` with lnL on the device, so a factor that always returns numpy raises
    # there on a GPU host.  Computing through xpy is what makes this example portable, and it IS
    # checked now: test_zero_likelihood_standin.py section 4 runs this function with xpy=cupy,
    # and the e2e gate's section 4 runs the whole pipeline with a device visible.  Both skip
    # where there is no usable GPU.
    #
    # The CAST is not decoration, and the EXPLICIT dtype is the whole of it.  mcsampler
    # (--sampler-method adaptive_cartesian) hands its integrand object-dtype draws, on which
    # np.cos raises "loop of ufunc does not support argument 0 of type float"; the driver's own
    # non-vectorized likelihood casts for the same reason ("get rid of 'object'").
    #
    # ONE line does BOTH jobs -- object dtype to float, and host to device -- and it does so
    # only because `dtype=` is passed.  Measured on cupy 12.0.0, 2026-09-17, on ldas-pcdev2
    # (A100, cc 8.0) and ldas-pcdev13 (RTX 2080 Ti, cc 7.5):
    #     cupy.asarray(obj_arr)                 -> ValueError: Unsupported dtype object
    #     cupy.asarray(obj_arr, dtype=float)    -> float64 cupy.ndarray, values preserved
    #     cupy.asarray(device_arr, dtype=float) -> float64 cupy.ndarray
    # A review read the first of those three and concluded this line could not be doing both.
    # Do NOT act on that by rewriting it as xpy.asarray(np.asarray(x, dtype=float)): np.asarray
    # of a cupy array raises TypeError ("Implicit conversion to a NumPy array is not allowed"),
    # so that form breaks every GPU run -- the exact path the line exists for.  Re-check by
    # running those three calls; it is a three-line probe, not an argument.
    phi_orb = xpy.asarray(phi_orb, dtype=float)
    out = A_COEFF * xpy.cos(phi_orb)
    if B_COEFF:
        inclination = xpy.asarray(inclination, dtype=float)
        cos_iota = inclination if INCL_IS_COSINE else xpy.cos(inclination)
        out = out + B_COEFF * cos_iota
    return out


def exact_ln_Z(a=None, b=None):
    """ln I0(A) + ln(sinh(B)/B): the exact marginal of the factor above against ILE's priors."""
    from scipy.special import i0
    a = A_COEFF if a is None else a
    b = B_COEFF if b is None else b
    out = float(np.log(i0(a)))
    if b:
        # sinh(b)/b, written so it does not overflow for the |b| this gate uses
        out += float(np.log(np.sinh(b) / b))
    return out
