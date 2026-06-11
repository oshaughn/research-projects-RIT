"""
Validation of RIFT.likelihood.jax_ile against the production numpy reference
``factored_likelihood.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop``.

Strategy: build synthetic but structurally-faithful precomputed arrays
(rholm timeseries, U/V cross terms, epoch) for a couple of detectors, then
compare:

  * JAX interp="nearest"  vs  numpy reference   (should agree to ~1e-6)
  * JAX interp="linear"   gradients             vs  finite differences
  * jit / vmap / grad all execute

Run:
  PYTHONPATH=<...>/Code  python test/jax/test_jax_likelihood.py
"""

import types
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import lal

import RIFT.likelihood.factored_likelihood as FL
from RIFT.likelihood.jax_ile import build_likelihood_data, fused_log_likelihood


def make_synthetic(detectors=("H1", "L1"), modes=((2, 2), (2, -2)),
                   deltaT=1.0 / 4096, npts_full=4096, tw=0.075, seed=1):
    """Build synthetic packed arrays + window, in-bounds for the reference slice."""
    rng = np.random.default_rng(seed)
    npts = int(2 * tw / deltaT)
    tvals = np.linspace(-tw, tw, npts)
    tref = 1126259462.413
    K = len(modes)

    packed = {}
    rholmsArrayDict = {}
    lookupNKDict = {}
    ctUArrayDict = {}
    ctVArrayDict = {}
    epochDict = {}
    for det in detectors:
        # Smooth (bandlimited) complex timeseries so linear interpolation is
        # meaningful: low-pass white noise via a Gaussian smoothing kernel.
        white = (rng.standard_normal((K, npts_full))
                 + 1j * rng.standard_normal((K, npts_full)))
        kx = np.arange(-40, 41)
        kern = np.exp(-0.5 * (kx / 12.0) ** 2)
        kern = kern / kern.sum()
        rho = np.stack([
            np.convolve(white[k].real, kern, mode="same")
            + 1j * np.convolve(white[k].imag, kern, mode="same")
            for k in range(K)
        ]).astype(np.complex128) * np.sqrt(len(kx))
        U = (rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K)))
        V = (rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K)))
        # epoch places the analysis window comfortably mid-array
        epoch = tref - 0.5
        lms = np.array(modes, dtype=int)

        packed[det] = dict(lms=lms, rholmArray=rho, U=U, V=V, epoch=epoch)
        rholmsArrayDict[det] = rho
        lookupNKDict[det] = lms
        ctUArrayDict[det] = U
        ctVArrayDict[det] = V
        epochDict[det] = epoch

    ref_inputs = dict(rholmsArrayDict=rholmsArrayDict, lookupNKDict=lookupNKDict,
                      ctUArrayDict=ctUArrayDict, ctVArrayDict=ctVArrayDict,
                      epochDict=epochDict)
    return packed, ref_inputs, tvals, deltaT, tref


def make_Pvec(S, tref, deltaT, seed=2):
    rng = np.random.default_rng(seed)
    P = types.SimpleNamespace()
    P.phi = rng.uniform(0, 2 * np.pi, S)            # RA
    P.theta = rng.uniform(-np.pi / 2, np.pi / 2, S)  # DEC
    P.psi = rng.uniform(0, np.pi, S)
    P.incl = rng.uniform(0, np.pi, S)
    P.phiref = rng.uniform(0, 2 * np.pi, S)
    distMpc = rng.uniform(100.0, 2000.0, S)
    P.dist = distMpc * lal.PC_SI * 1e6              # SI metres
    P.tref = tref
    P.phiref = P.phiref
    P.deltaT = deltaT
    return P, distMpc


def test_nearest_matches_reference():
    packed, ref, tvals, deltaT, tref = make_synthetic()
    S = 23
    P, distMpc = make_Pvec(S, tref, deltaT)

    lnL_ref = FL.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, P, ref["lookupNKDict"], ref["rholmsArrayDict"],
        ref["ctUArrayDict"], ref["ctVArrayDict"], ref["epochDict"],
        Lmax=2, xpy=np)

    data = build_likelihood_data(packed, deltaT, tref, tvals)
    lnL_jax = np.asarray(fused_log_likelihood(
        data, P.phi, P.theta, P.psi, P.incl, P.phiref, distMpc,
        interp="nearest"))

    err = np.max(np.abs(lnL_ref - lnL_jax))
    rel = np.max(np.abs(lnL_ref - lnL_jax) / (1 + np.abs(lnL_ref)))
    print(f"[nearest vs reference]  max|abs|={err:.3e}  max|rel|={rel:.3e}")
    print("  sample lnL_ref[:4] =", np.round(lnL_ref[:4], 6))
    print("  sample lnL_jax[:4] =", np.round(lnL_jax[:4], 6))
    assert err < 1e-6, f"nearest-mode mismatch {err}"


def test_linear_gradients():
    packed, ref, tvals, deltaT, tref = make_synthetic(seed=7)
    P, distMpc = make_Pvec(5, tref, deltaT, seed=3)
    data = build_likelihood_data(packed, deltaT, tref, tvals)

    def scalar_lnL(ra, dec, psi, incl, phiref, dMpc):
        v = fused_log_likelihood(
            data,
            jnp.array([ra]), jnp.array([dec]), jnp.array([psi]),
            jnp.array([incl]), jnp.array([phiref]), jnp.array([dMpc]),
            interp="linear")
        return v[0]

    x0 = [float(P.phi[0]), float(P.theta[0]), float(P.psi[0]),
          float(P.incl[0]), float(P.phiref[0]), float(distMpc[0])]
    grad = jax.grad(scalar_lnL, argnums=(0, 1, 2, 3, 4, 5))(*x0)
    grad = np.array([float(g) for g in grad])

    # finite differences
    fd = np.zeros(6)
    steps = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-3]
    for i in range(6):
        xp = list(x0); xm = list(x0)
        xp[i] += steps[i]; xm[i] -= steps[i]
        fp = float(scalar_lnL(*xp)); fm = float(scalar_lnL(*xm))
        fd[i] = (fp - fm) / (2 * steps[i])

    names = ["RA", "DEC", "psi", "incl", "phiref", "dMpc"]
    print("[linear gradients vs finite differences]")
    ok = True
    for i, nm in enumerate(names):
        denom = max(1.0, abs(fd[i]))
        rel = abs(grad[i] - fd[i]) / denom
        flag = "" if rel < 1e-3 else "  <-- CHECK"
        if rel >= 1e-3:
            ok = False
        print(f"  d/d{nm:6s} AD={grad[i]:+.6e}  FD={fd[i]:+.6e}  rel={rel:.2e}{flag}")
    assert ok, "AD gradient disagrees with finite differences"


def test_jit_vmap():
    packed, ref, tvals, deltaT, tref = make_synthetic(seed=11)
    P, distMpc = make_Pvec(31, tref, deltaT, seed=5)
    data = build_likelihood_data(packed, deltaT, tref, tvals)

    f = jax.jit(lambda *a: fused_log_likelihood(data, *a, interp="linear"))
    out = f(P.phi, P.theta, P.psi, P.incl, P.phiref, distMpc)
    out.block_until_ready()
    assert out.shape == (31,)
    assert np.all(np.isfinite(np.asarray(out)))
    print(f"[jit] ran on S=31, lnL range [{float(out.min()):.3f},"
          f" {float(out.max()):.3f}]")

    # vmap over single-point evaluations (per-sample grad map)
    def one(ra, dec, psi, incl, phiref, dMpc):
        return fused_log_likelihood(
            data, ra[None], dec[None], psi[None], incl[None],
            phiref[None], dMpc[None], interp="linear")[0]
    g = jax.vmap(jax.grad(one, argnums=0))
    grads = g(P.phi, P.theta, P.psi, P.incl, P.phiref, distMpc)
    grads.block_until_ready()
    assert grads.shape == (31,)
    print(f"[vmap grad] dlnL/dRA finite: {np.all(np.isfinite(np.asarray(grads)))}")


if __name__ == "__main__":
    test_nearest_matches_reference()
    test_linear_gradients()
    test_jit_vmap()
    print("\nALL JAX LIKELIHOOD TESTS PASSED")
