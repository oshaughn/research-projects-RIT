"""Zero-host-copy handoff from GPU compound precompute to ILE-JAX.

The conventional banded builder first materializes LAL time series, repacks them
through NumPy, and finally uploads them to JAX.  This adapter consumes the packed
``return_device=True`` result from :mod:`gpu_precompute` and shares its CuPy
buffers with JAX through DLPack.  Detector geometry and small static index tables
remain host-built; Q/U/V never return to host.
"""
from __future__ import division, print_function

import os
import sys

import numpy as np


def _prepare_jax():
    # JAX's default large preallocation competes with a live CuPy compound bank.
    # This variable only has effect before JAX initializes; production launchers
    # should set it explicitly as well.
    if "jax" not in sys.modules:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.dlpack
    import jax.numpy as jnp
    if not bool(jax.config.x64_enabled):
        raise RuntimeError("device handoff requires JAX x64")
    return jax, jnp


def _jax_device_array(value, jax, jnp, name, require_gpu=True):
    """Share a JAX/CuPy device array with JAX; reject host-backed inputs."""
    module = type(value).__module__.split(".")[0]
    if module in ("jax", "jaxlib"):
        out = jnp.asarray(value)
        if require_gpu and any(d.platform != "gpu" for d in out.devices()):
            raise RuntimeError("%s is a JAX array on a non-GPU device" % name)
        return out
    if module != "cupy":
        raise TypeError("%s must be a CuPy or JAX device array, got %s" %
                        (name, type(value).__name__))
    import cupy as cp
    contiguous = cp.ascontiguousarray(value)
    try:
        # copy=False makes an allocator/device mismatch explicit rather than
        # silently defeating the purpose of this handoff.
        return jax.dlpack.from_dlpack(contiguous, copy=False)
    except TypeError:  # JAX before the copy= API; DLPack was zero-copy by contract.
        try:
            return jax.dlpack.from_dlpack(contiguous)
        except Exception as exc:
            raise RuntimeError("DLPack handoff failed for %s" % name) from exc
    except Exception as exc:
        raise RuntimeError("zero-copy DLPack handoff failed for %s" % name) from exc


def _validate_tvals(tvals, delta_t):
    t = np.asarray(tvals, dtype=float)
    if t.ndim != 1 or t.size < 1 or not np.all(np.isfinite(t)):
        raise ValueError("tvals must be a nonempty finite one-dimensional grid")
    if t.size > 1 and not np.allclose(np.diff(t), float(delta_t),
                                      rtol=5e-13, atol=1e-15):
        raise ValueError("tvals cadence differs from the precompute delta_t")
    return t


def build_jax_rotating_freqresponse_data_from_device(
        packed, meta, tvals, det_geom, distMpcRef=None,
        q_time_pregrid_factor=1, require_gpu=True):
    """Build ``JAXLikelihoodData`` while preserving Q/U/V device residency.

    Parameters
    ----------
    packed, meta
        The two values returned by
        ``PrecomputeLikelihoodTermsRotatingFreqResponseGPU(...,
        return_device=True)``.
    tvals
        The ordinary ILE integration grid, sampled at ``packed['delta_t']``.
    det_geom
        ``det -> (response, x_arm, y_arm, L)`` from
        ``slowrot_freqresponse.detector_geometry``.

    The reflected Q pregrid is intentionally limited to factor one.  Its current
    implementation is host-side; accepting a larger factor here would silently
    reintroduce a full Q device-to-host-to-device round trip.
    """
    try:
        factor = int(q_time_pregrid_factor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("q_time_pregrid_factor must be exactly 1") from exc
    if factor != 1 or q_time_pregrid_factor != factor:
        raise NotImplementedError(
            "device-resident JAX handoff currently requires q_time_pregrid_factor=1")
    if not bool(meta.get("gpu_precompute")) or not bool(meta.get("device_resident")):
        raise ValueError("meta does not describe a device-resident GPU precompute")
    if meta.get("feature") != "rotation_freqresponse":
        raise ValueError("only the compound rotating frequency-response bank is supported")
    if not bool(meta.get("post_phase_required")):
        raise ValueError("compound bank must require the arrival-time post-phase")

    required = {"q", "U", "V", "epoch", "delta_t", "modes", "a_list"}
    missing = required.difference(packed)
    if missing:
        raise ValueError("packed device result is missing %s" % sorted(missing))
    detectors = list(packed["q"])
    if not detectors or set(detectors) != set(packed["U"]) \
            or set(detectors) != set(packed["V"]) \
            or set(detectors) != set(packed["epoch"]) \
            or set(detectors) != set(det_geom):
        raise ValueError("Q/U/V/epoch/geometry detector sets differ")

    modes = [tuple(map(int, lm)) for lm in packed["modes"]]
    a_list = [tuple(map(int, a)) for a in packed["a_list"]]
    if modes != [tuple(map(int, lm)) for lm in meta["modes"]] \
            or a_list != [tuple(map(int, a)) for a in meta["a_list"]]:
        raise ValueError("packed mode or compound-index order differs from meta")
    if not modes or not a_list:
        raise ValueError("empty mode or compound index set")
    delta_t = float(packed["delta_t"])
    if not np.isfinite(delta_t) or delta_t <= 0:
        raise ValueError("packed delta_t must be finite and positive")
    tvals = _validate_tvals(tvals, delta_t)

    jax, jnp = _prepare_jax()
    import lal
    import lalsimulation as lalsim
    from .jax_ile.core import JAXLikelihoodData, DIST_MPC_REF
    from .jax_ile import response_slowrot as rs
    from .jax_ile import response_rotating_freqresponse as rrf
    from .gpu_precompute import _physical_device_key

    if distMpcRef is None:
        distMpcRef = DIST_MPC_REF
    A, K = len(a_list), len(modes)
    detector_data = {}
    source_device = None
    target_device = None
    for det in detectors:
        q_source = packed["q"][det]
        u_source = packed["U"][det]
        v_source = packed["V"][det]
        if tuple(q_source.shape[:2]) != (A, K) or q_source.ndim != 3:
            raise ValueError("%s Q must have shape (A,K,N)" % det)
        if tuple(u_source.shape) != (A, A, K, K) \
                or tuple(v_source.shape) != (A, A, K, K):
            raise ValueError("%s U/V must have shape (A,A,K,K)" % det)
        for label, array in (("Q", q_source), ("U", u_source), ("V", v_source)):
            here = _physical_device_key(array, "%s %s" % (det, label))
            if require_gpu and here[0] != "gpu":
                raise RuntimeError("%s %s is on a non-GPU device" % (det, label))
            if source_device is None:
                source_device = here
            elif here != source_device:
                raise ValueError("mixed physical devices in packed bank: %r and %r" %
                                 (source_device, here))

        # Q needs one device-local layout conversion from (A,K,N) to the
        # accumulator's contiguous (A,N,K).  U/V already have their final shape.
        q_module = type(q_source).__module__.split(".")[0]
        if q_module == "cupy":
            import cupy as cp
            q_source = cp.ascontiguousarray(q_source.transpose(0, 2, 1))
        elif q_module in ("jax", "jaxlib"):
            q_source = jnp.transpose(q_source, (0, 2, 1))
        else:
            raise TypeError("%s Q is not device-resident" % det)
        Q = _jax_device_array(q_source, jax, jnp, "%s Q" % det, require_gpu)
        U = _jax_device_array(u_source, jax, jnp, "%s U" % det, require_gpu)
        V = _jax_device_array(v_source, jax, jnp, "%s V" % det, require_gpu)
        for label, array in (("Q", Q), ("U", U), ("V", V)):
            devices = tuple(array.devices())
            if len(devices) != 1:
                raise RuntimeError("%s %s JAX result is not single-device" % (det, label))
            if target_device is None:
                target_device = devices[0]
            elif devices[0] != target_device:
                raise RuntimeError("DLPack results landed on different JAX devices")
        target_key = (str(target_device.platform), int(target_device.id))
        if target_key != source_device:
            raise RuntimeError("DLPack changed physical device from %r to %r" %
                               (source_device, target_key))

        response, x_arm, y_arm, length = det_geom[det]
        lald = lalsim.DetectorPrefixToLALDetector(det)
        epoch = float(packed["epoch"][det])
        if not np.isfinite(epoch):
            raise ValueError("%s Q epoch is non-finite" % det)
        with jax.default_device(target_device):
            detector_data[det] = {
                "lms": list(modes),
                # Baseline-shaped aliases keep generic inspection helpers working;
                # the banded accumulator consumes the full banks below.
                "Q": Q[0], "U": U[0, 0], "V": V[0, 0],
                "Q_bank": Q, "U_bank": U, "V_bank": V,
                "q_time_pregrid_factor": 1,
                "q_time_pregrid_report": {"factor": 1, "device_handoff": "dlpack"},
                "npts_full_coarse": int(Q.shape[1]),
                "npts_full": int(Q.shape[1]),
                "epoch": epoch,
                "location": jnp.asarray(np.asarray(lald.location, dtype=np.float64)),
                "response": jnp.asarray(np.asarray(response, dtype=np.float64)),
                "x_arm": jnp.asarray(np.asarray(x_arm, dtype=np.float64)),
                "y_arm": jnp.asarray(np.asarray(y_arm, dtype=np.float64)),
                "L_arm": float(length),
                "l_max": max(l for l, unused_m in modes),
            }

    tref = float(meta["event_time_geo"])
    gmst = float(lal.GreenwichMeanSiderealTime(tref))
    with jax.default_device(target_device):
        data = JAXLikelihoodData(detector_data, delta_t, gmst, tvals, tref,
                                 float(distMpcRef), q_time_pregrid_factor=1)
    m_values, term1_idx, term2_idx = rs.post_phase_bucketing(a_list)
    data.feature = "rotation_freqresponse"
    data.band = {
        "a_list": a_list,
        "Qmax": int(meta["Qmax"]),
        "p_max": int(meta["p_max"]),
        "refl_idx": np.asarray(rrf.reflection_index(a_list), dtype=np.int64),
        "f_sidereal": float(meta["f_sidereal"]),
        "post_phase_required": True,
        "pp_m_values": np.asarray(m_values, dtype=np.int64),
        "pp_term1_idx": np.asarray(term1_idx, dtype=np.int64),
        "pp_term2_idx": np.asarray(term2_idx, dtype=np.int64),
    }
    data.gpu_handoff = {
        "contract_Q_U_V_host_copies": 0,
        "transport": "DLPack" if any(
            type(packed["q"][d]).__module__.split(".")[0] == "cupy"
            for d in detectors) else "JAX identity",
        "q_layout_device_copy": True,
    }
    return data


__all__ = ["build_jax_rotating_freqresponse_data_from_device"]
