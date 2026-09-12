#!/usr/bin/env python3
"""Short physical BBH, matched CPU/GPU precompute and repeated worker timings.

No BNS input is opened. Synthetic signal/noise products exist only in memory.
This tests precompute and point likelihoods, not posterior convergence.
"""
import time
_PROCESS_START = time.perf_counter()

import argparse
import json
import os
import platform
import resource
import sys

import numpy as np
import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lsu
from RIFT.likelihood import gpu_precompute as gpu
from RIFT.likelihood import factored_likelihood_rotating_freqresponse as fr

_MODULE_IMPORT_SECONDS = time.perf_counter() - _PROCESS_START


def emit(**record):
    print(json.dumps(record, sort_keys=True), flush=True)


def sync(xp):
    if xp is not np:
        xp.cuda.Stream.null.synchronize()


_TOP_LEVEL_TIMING_STAGES = {
    'initialization', 'waveform', 'input_prep', 'basis', 'Q_U', 'V',
    'device_export', 'host_export',
}


def timing_recorder(records, intrinsic, algorithm, phase):
    """Return a callback that retains non-overlapping stage wall times."""
    def callback(stage, elapsed, details):
        elapsed = float(elapsed)
        if stage in _TOP_LEVEL_TIMING_STAGES:
            records[stage] = records.get(stage, 0.0) + elapsed
        emit(stage=stage, seconds=elapsed, intrinsic=intrinsic, phase=phase,
             algorithm=algorithm, **details)
    return callback


def process_snapshot(xp):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    io_values = {}
    with open('/proc/self/io') as stream:
        for line in stream:
            key, value = line.split(':', 1)
            io_values[key] = int(value)
    out = dict(user_seconds=float(usage.ru_utime), system_seconds=float(usage.ru_stime),
               max_rss_kib=int(usage.ru_maxrss), major_faults=int(usage.ru_majflt),
               minor_faults=int(usage.ru_minflt), proc_io=io_values)
    if xp is not np:
        free, total = xp.cuda.runtime.memGetInfo()
        pool = xp.get_default_memory_pool()
        out['gpu_memory'] = dict(free_bytes=int(free), total_bytes=int(total),
                                 used_bytes=int(total-free),
                                 pool_used_bytes=int(pool.used_bytes()),
                                 pool_free_bytes=int(pool.free_bytes()))
    return out


def snapshot_delta(before, after):
    keys = ('user_seconds', 'system_seconds', 'major_faults', 'minor_faults')
    out = {key: after[key] - before[key] for key in keys}
    out['max_rss_kib_change'] = after['max_rss_kib'] - before['max_rss_kib']
    out['proc_io'] = {key: after['proc_io'].get(key, 0) - before['proc_io'].get(key, 0)
                      for key in sorted(set(before['proc_io']) | set(after['proc_io']))}
    if 'gpu_memory' in after:
        out['gpu_memory'] = {key: after['gpu_memory'][key] - before['gpu_memory'][key]
                             for key in after['gpu_memory']}
    return out


def host_legacy_pack(packed, meta, xp, require_gpu):
    """Pack first, then copy device arrays only for the post-timing oracle."""
    lookup, rho, U, V, epoch = gpu.pack_device_precompute(
        packed, meta, require_gpu=require_gpu)

    def host(value):
        return np.asarray(value) if xp is np else xp.asnumpy(value)

    return (lookup,
            {det: {a: host(value) for a, value in rows.items()}
             for det, rows in rho.items()},
            {det: host(value) for det, value in U.items()},
            {det: host(value) for det, value in V.items()},
            epoch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intrinsics', type=int, default=5)
    parser.add_argument('--repeat-same-intrinsic', action='store_true',
                        help='repeat one identical intrinsic point; requires at least five calls')
    parser.add_argument('--detectors', default='H1,L1,V1')
    parser.add_argument('--qmax', type=int, default=1)
    parser.add_argument('--pmax', type=int, default=1)
    parser.add_argument('--fft-batch', type=int, default=4,
                        help='benchmark-only compound-basis FFT batch')
    parser.add_argument('--q-row-batch', type=int, default=4,
                        help='benchmark-only Q inverse-FFT row batch')
    parser.add_argument('--lmax', type=int, default=2)
    parser.add_argument('--approximant', default='IMRPhenomD')
    parser.add_argument('--delta-t', type=float, default=1/1024.)
    parser.add_argument('--delta-f', type=float, default=0.5)
    parser.add_argument('--fmin', type=float, default=30.)
    parser.add_argument('--fref', type=float, default=100.)
    parser.add_argument('--fmax', type=float, default=512.)
    parser.add_argument('--t-window', type=float, default=0.15)
    parser.add_argument('--mass1', type=float, default=30.)
    parser.add_argument('--mass2', type=float, default=25.)
    parser.add_argument('--spin1x', type=float, default=0.)
    parser.add_argument('--spin1y', type=float, default=0.)
    parser.add_argument('--spin1z', type=float, default=0.)
    parser.add_argument('--spin2x', type=float, default=0.)
    parser.add_argument('--spin2y', type=float, default=0.)
    parser.add_argument('--spin2z', type=float, default=0.)
    parser.add_argument('--arm-length', type=float, default=40000.)
    parser.add_argument('--backend', choices=['cupy','numpy'], default='cupy',
                        help='numpy is harness validation only, not GPU performance')
    parser.add_argument('--compare-numpy', action='store_true',
                        help='also time the batched NumPy algorithm on this same worker')
    parser.add_argument('--gpu-only-profile-after-validation', action='store_true',
                        help=('skip all CPU oracles for timing only after a separate '
                              'matched small-case validation; emits no parity claim'))
    parser.add_argument('--legacy-reference-intrinsics', type=int, default=None,
                        help=('number of initial intrinsic points to run through the very '
                              'slow scalar legacy oracle (default: all, preserving old behavior; '
                              '0 uses batched NumPy as the numerical oracle)'))
    args = parser.parse_args()
    if (args.intrinsics < 1 or args.lmax < 2 or args.qmax < 0 or args.pmax < 0
            or args.fft_batch < 1 or args.q_row_batch < 1):
        parser.error('intrinsics must be positive; lmax>=2 and qmax,pmax>=0')
    if min(args.delta_t, args.delta_f, args.fmin, args.fref,
           args.fmax, args.t_window, args.mass1, args.mass2,
           args.arm_length) <= 0:
        parser.error('waveform grid, frequencies, masses, window, and arm length must be positive')
    if args.fmax > 0.5/args.delta_t:
        parser.error('fmax exceeds the Nyquist frequency implied by delta-t')
    if args.repeat_same_intrinsic and args.intrinsics < 5:
        parser.error('--repeat-same-intrinsic requires --intrinsics >= 5')
    legacy_count = (args.intrinsics if args.legacy_reference_intrinsics is None
                    else args.legacy_reference_intrinsics)
    if legacy_count < 0 or legacy_count > args.intrinsics:
        parser.error('legacy-reference-intrinsics must lie in [0, intrinsics]')
    if args.gpu_only_profile_after_validation:
        if args.backend != 'cupy':
            parser.error('GPU-only profiling requires --backend cupy')
        if args.legacy_reference_intrinsics != 0:
            parser.error('GPU-only profiling requires explicit --legacy-reference-intrinsics 0')
        if args.compare_numpy:
            parser.error('GPU-only profiling cannot be combined with --compare-numpy')
    elif legacy_count < args.intrinsics and not args.compare_numpy:
        parser.error('skipping any legacy reference requires --compare-numpy')
    if os.environ.get('RIFT_GPU_PRECOMPUTE') == '1':
        parser.error('unset RIFT_GPU_PRECOMPUTE: it would silently route the CPU oracle to GPU')
    backend_import_started = time.perf_counter()
    if args.backend == 'cupy':
        import cupy as xp
    else:
        xp = np
    backend_import_seconds = time.perf_counter() - backend_import_started
    readiness_started = time.perf_counter()
    sync(xp)
    context = gpu.GPUPrecomputeContext(xp)
    sync(xp)
    backend_readiness_seconds = time.perf_counter() - readiness_started
    started = _PROCESS_START
    emit(stage='startup_timing', module_import_seconds=_MODULE_IMPORT_SECONDS,
         backend_import_seconds=backend_import_seconds,
         backend_readiness_seconds=backend_readiness_seconds,
         timing_scope='from first Python statement; excludes interpreter startup')
    detectors = [det.strip() for det in args.detectors.split(',') if det.strip()]
    if not detectors:
        parser.error('at least one detector is required')
    try:
        approximant = lalsim.GetApproximantFromString(args.approximant)
    except Exception as exc:
        parser.error('unknown approximant %r: %s' % (args.approximant, exc))
    P0 = lsu.ChooseWaveformParams(
        m1=args.mass1*lal.MSUN_SI, m2=args.mass2*lal.MSUN_SI,
        s1x=args.spin1x, s1y=args.spin1y, s1z=args.spin1z,
        s2x=args.spin2x, s2y=args.spin2y, s2z=args.spin2z,
        fmin=args.fmin, fref=args.fref,
        deltaT=args.delta_t, deltaF=args.delta_f, approx=approximant,
        radec=True, phi=1.2, theta=0.3, incl=0.7, psi=0.5, phiref=0.4,
        tref=1e9, dist=200e6*lal.PC_SI, detector=detectors[0])
    data, psds = {}, {}
    for det in detectors:
        Pd = P0.manual_copy()
        Pd.detector = det
        data[det] = lsu.non_herm_hoff(Pd)
        n = data[det].data.length
        psd = lal.CreateREAL8FrequencySeries(
            det, lal.LIGOTimeGPS(0), 0., P0.deltaF, lal.SecondUnit, n//2+1)
        psd.data.data[:] = [lalsim.SimNoisePSDaLIGOZeroDetHighPower(max(10.,f))
                           for f in np.arange(n//2+1)*P0.deltaF]
        psds[det] = psd
    # This is deliberately a conservative upper bound based on every mode up
    # to Lmax.  It protects tuning jobs from a mistyped batch size without
    # changing the production allocator or its automatic fallback.
    a_count_bound = len(fr.compound_index_set(args.qmax, args.pmax))
    mode_count_bound = sum(2*l + 1 for l in range(2, args.lmax + 1))
    row_count_bound = a_count_bound * mode_count_bound
    if args.fft_batch > a_count_bound:
        parser.error('fft-batch exceeds the compound count')
    if args.q_row_batch > row_count_bound:
        parser.error('q-row-batch exceeds the maximum possible basis rows')
    complex_bytes = np.dtype(np.complex128).itemsize
    n_window_bound = int(2.0 * args.t_window / args.delta_t)
    if n_window_bound < 1 or n_window_bound > n:
        parser.error('t-window produces an empty or overlong Q window')
    primary_bound = a_count_bound * mode_count_bound * n * complex_bytes
    retained_q_bound = (len(detectors) * row_count_bound * n_window_bound *
                        complex_bytes)
    retained_uv_bound = (len(detectors) * 2 * row_count_bound**2 * complex_bytes)
    fft_scratch_bound = (5 * args.fft_batch * mode_count_bound * n *
                         complex_bytes)
    q_scratch_bound = 3 * args.q_row_batch * n * complex_bytes
    v_scratch_bound = 6 * min(args.fft_batch, 2) * mode_count_bound * n * complex_bytes
    peak_bound = (primary_bound + retained_q_bound + retained_uv_bound +
                  max(fft_scratch_bound, q_scratch_bound, v_scratch_bound))
    free_bytes = ((1 << 62) if xp is np
                  else int(xp.cuda.runtime.memGetInfo()[0]))
    if peak_bound > 0.80 * free_bytes:
        parser.error('requested benchmark batches exceed the 80% free-memory guard')
    emit(stage='batch_preflight', fft_batch=args.fft_batch,
         q_row_batch=args.q_row_batch, compound_count_bound=a_count_bound,
         mode_count_bound=mode_count_bound, row_count_bound=row_count_bound,
         estimated_peak_bytes=peak_bound,
         available_device_bytes=None if xp is np else free_bytes)
    # Injection angles belong only to the synthetic detector data.  The mode
    # bank follows the ILE convention and carries no extrinsic angles; this is
    # essential for XPHM as well as aligned-spin approximants.
    P_template = P0.manual_copy()
    P_template.phiref = P_template.psi = P_template.incl = 0.0
    sync(xp)
    numpy_context = gpu.GPUPrecomputeContext(np) if args.compare_numpy else None
    emit(stage='worker_setup', seconds=time.perf_counter()-started,
         bins=n, masses_msun=[args.mass1,args.mass2],
         spins=[[args.spin1x,args.spin1y,args.spin1z],
                [args.spin2x,args.spin2y,args.spin2z]],
         approximant=args.approximant, lmax=args.lmax,
         delta_t=P0.deltaT, delta_f=P0.deltaF, fmin=args.fmin,
         fmax=args.fmax, t_window=args.t_window,
         qmax=args.qmax, pmax=args.pmax,
         fft_batch=args.fft_batch, q_row_batch=args.q_row_batch,
         legacy_reference_intrinsics=legacy_count,
         gpu_only_profile_after_validation=args.gpu_only_profile_after_validation,
         backend=args.backend,
         gpu=None if xp is np else xp.cuda.runtime.getDeviceProperties(0)['name'].decode())
    environment = dict(python=platform.python_version(), numpy=np.__version__,
                       lal=getattr(lal, '__version__', None),
                       lalsimulation=getattr(lalsim, '__version__', None),
                       repeat_same_intrinsic=args.repeat_same_intrinsic,
                       timing_instrumentation=('synchronous stage callbacks; synchronization '
                                               'perturbs asynchronous GPU scheduling'))
    if xp is not np:
        props = xp.cuda.runtime.getDeviceProperties(0)
        environment.update(
            cupy=xp.__version__, cuda_runtime=int(xp.cuda.runtime.runtimeGetVersion()),
            cuda_driver=int(xp.cuda.runtime.driverGetVersion()),
            device=dict(name=props['name'].decode(),
                        compute_capability='%d.%d' % (props['major'], props['minor']),
                        total_global_memory_bytes=int(props['totalGlobalMem']),
                        multiprocessor_count=int(props['multiProcessorCount'])))
    emit(stage='runtime_environment', **environment)
    for index in range(args.intrinsics):
        phase = 'cold' if index == 0 else 'warm'
        P = P_template.manual_copy()
        if not args.repeat_same_intrinsic:
            P.m1 += index*0.01*lal.MSUN_SI
        common = dict(event_time_geo=1e9, t_window=args.t_window, P=P, data_dict=data,
                      psd_dict=psds, Lmax=args.lmax, fMax=args.fmax, Qmax=args.qmax,
                      p_max=args.pmax, L_arm=args.arm_length, skip_interpolation=True,
                      quiet=True, verbose=False)
        cpu = cpu_pack = None
        if index < legacy_count:
            t0 = time.perf_counter()
            cpu = fr.PrecomputeLikelihoodTermsRotatingFreqResponse(
                **dict(common, P=P.manual_copy()))
            emit(stage='cpu_precompute', intrinsic=index, seconds=time.perf_counter()-t0,
                 timing_role='numerical_reference_only')
            cpu_pack = fr.pack_rotating_freqresponse_arrays(
                cpu[4], cpu[3], cpu[1], cpu[2])
        numpy_seconds = None
        numpy_pack = numpy_meta = None
        if numpy_context is not None:
            numpy_timings = {}
            t0 = time.perf_counter()
            numpy_bank = gpu.PrecomputeLikelihoodTermsRotatingFreqResponseGPU(
                **dict(common, P=P.manual_copy()), backend=np,
                context=numpy_context,
                fft_batch=args.fft_batch, q_row_batch=args.q_row_batch,
                return_device=True,
                timing_callback=timing_recorder(
                    numpy_timings, index, 'batched_numpy', phase))
            numpy_seconds = time.perf_counter()-t0
            emit(stage='batched_numpy_precompute',intrinsic=index,
                 seconds=numpy_seconds,
                 stage_seconds=numpy_timings,
                 context_uploads=numpy_context.uploads,
                 context_hits=numpy_context.cache_hits,
                 timing_scope='host wall through resident bank return')
            # Exercise the exact classic handoff contract after the timed region.
            numpy_pack = host_legacy_pack(
                numpy_bank[0], numpy_bank[1], np, require_gpu=False)
            numpy_meta = numpy_bank[1]
            del numpy_bank
        sync(xp)
        resources_before = process_snapshot(xp)
        candidate_timings = {}
        t0 = time.perf_counter()
        bank = gpu.PrecomputeLikelihoodTermsRotatingFreqResponseGPU(
            **dict(common, P=P.manual_copy()), context=context, backend=xp,
            fft_batch=args.fft_batch, q_row_batch=args.q_row_batch,
            return_device=True,
            timing_callback=timing_recorder(candidate_timings, index, args.backend, phase))
        sync(xp)
        candidate_seconds = time.perf_counter()-t0
        resources_after = process_snapshot(xp)
        emit(stage='candidate_precompute',backend=args.backend, intrinsic=index,
             phase=phase,
             seconds=candidate_seconds, stage_seconds=candidate_timings,
             context_uploads=context.uploads, context_hits=context.cache_hits,
             context_entries=len(context._arrays),
             device_pool_used_bytes=0 if xp is np else xp.get_default_memory_pool().used_bytes(),
             timing_scope='host wall through resident bank return')
        emit(stage='candidate_resources', backend=args.backend, intrinsic=index,
             phase=phase, before=resources_before, after=resources_after,
             delta=snapshot_delta(resources_before, resources_after))
        actual_modes = list(bank[1]['modes'])
        actual_a = list(bank[1]['a_list'])
        mode_count, a_count = len(actual_modes), len(actual_a)
        q_bytes = sum(int(value.nbytes) for value in bank[0]['q'].values())
        basis_bytes_per_detector = (a_count*mode_count*n*
                                    np.dtype(np.complex128).itemsize)
        uv_bytes = sum(int(value.nbytes) for family in ('U', 'V')
                       for value in bank[0][family].values())
        emit(stage='bank_geometry', intrinsic=index, actual_mode_count=mode_count,
             actual_modes=actual_modes, exact_compound_count=a_count,
             exact_q_bytes=q_bytes, exact_uv_bytes=uv_bytes,
             exact_retained_quv_bytes=q_bytes+uv_bytes,
             primary_basis_bytes_per_detector=basis_bytes_per_detector,
             full_fft_bins=n)
        if numpy_seconds is not None and xp is not np:
            emit(stage='batched_algorithm_comparison', intrinsic=index,
                 numpy_seconds=numpy_seconds, gpu_seconds=candidate_seconds,
                 end_to_end_speedup=numpy_seconds/candidate_seconds,
                 numpy_stage_seconds=numpy_timings,
                 gpu_stage_seconds=candidate_timings,
                 timing_scope=('same-process host wall through resident bank return; '
                               'container staging, scheduler queue, and later oracle copies excluded'))
        if args.gpu_only_profile_after_validation:
            if a_count != a_count_bound or not (0 < mode_count <= mode_count_bound):
                raise RuntimeError('unexpected compound-bank geometry in GPU-only profile')
            expected_shapes = {
                'q': (a_count, mode_count, n_window_bound),
                'U': (a_count, a_count, mode_count, mode_count),
                'V': (a_count, a_count, mode_count, mode_count),
            }
            for family, shape in expected_shapes.items():
                for det in detectors:
                    value = bank[0][family][det]
                    if value.shape != shape:
                        raise RuntimeError('%s %s has shape %r, expected %r' %
                                           (det, family, value.shape, shape))
                    if not bool(xp.asnumpy(xp.all(xp.isfinite(value)))):
                        raise RuntimeError('%s %s is nonfinite' % (det, family))
            expected_uploads = 3*len(detectors)
            if context.uploads != expected_uploads:
                raise RuntimeError('detector inputs were unexpectedly re-uploaded')
            expected_hits = index * expected_uploads
            if context.cache_hits != expected_hits:
                raise RuntimeError('detector input cache-hit count is inconsistent')
            emit(stage='gpu_profile_validation', intrinsic=index,
                 arrays_finite=True, geometry_valid=True,
                 cache_uploads=context.uploads, cache_hits=context.cache_hits,
                 parity_measured=False,
                 status='timing_only_after_separate_small_case_validation')
            del bank
            continue
        # This is deliberately after candidate_seconds: production hands these
        # buffers directly to JAX/classic GPU ILE.  Host copies exist only so
        # the small numerical oracle below can use its NumPy implementation.
        got_pack = host_legacy_pack(
            bank[0], bank[1], xp, require_gpu=(xp is not np))
        oracle_pack = cpu_pack if cpu_pack is not None else numpy_pack
        oracle_meta = cpu[4] if cpu is not None else numpy_meta
        oracle_name = 'legacy_cpu' if cpu is not None else 'batched_numpy'
        errors = dict(Q=0., U=0., V=0.)
        for det in detectors:
            for ai in bank[1]['a_list']:
                a, b = got_pack[1][det][ai], oracle_pack[1][det][ai]
                np.testing.assert_allclose(a, b, rtol=2e-9, atol=1e-8)
                errors['Q'] = max(errors['Q'],float(np.max(np.abs(a-b))))
            for name, slot in [('U',2),('V',3)]:
                a, b = got_pack[slot][det], oracle_pack[slot][det]
                np.testing.assert_allclose(a, b, rtol=2e-9, atol=1e-7)
                errors[name] = max(errors[name],float(np.max(np.abs(a-b))))
            assert got_pack[4][det] == oracle_pack[4][det]
        Pv = P0.manual_copy()
        for name in ['phi','theta','incl','psi','phiref','dist']:
            setattr(Pv,name,np.full(4,getattr(P0,name)))
        Pv.phi += np.array([0.,1e-4,-1e-4,0.01])
        tvals = np.array([-P0.deltaT,0.,P0.deltaT])
        oracle_lnl = fr.DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
            tvals,Pv,oracle_meta,*oracle_pack,Lmax=args.lmax,time_interp='nearest',xpy=np,array_output=True)
        got_lnl = fr.DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
            tvals,Pv,bank[1],*got_pack,Lmax=args.lmax,time_interp='nearest',xpy=np,array_output=True)
        np.testing.assert_allclose(got_lnl,oracle_lnl,rtol=2e-9,atol=1e-8)
        if not np.all(np.isfinite(got_lnl)):
            raise RuntimeError('nonfinite downstream likelihood')
        emit(stage='oracle_gpu_parity', intrinsic=index, oracle=oracle_name,
             max_abs_error=errors,
             max_abs_lnl_error=float(np.max(np.abs(oracle_lnl-got_lnl))))
        del bank,got_pack
        if cpu is not None:
            del cpu,cpu_pack
    assert context.uploads == 3*len(detectors), 'detector inputs were re-uploaded'
    emit(stage='worker_total',seconds=time.perf_counter()-started,
         benchmark_mode=('gpu_only_profile_after_validation'
                         if args.gpu_only_profile_after_validation else 'matched_oracle'),
         timing_scope=('first Python statement through completed benchmark; interpreter startup, '
                       'container launch, and queue excluded; cross-check with /usr/bin/time -v'))


if __name__ == '__main__':
    main()
