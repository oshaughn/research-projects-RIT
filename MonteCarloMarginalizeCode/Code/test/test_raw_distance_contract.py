"""Known-answer raw-likelihood contracts, independent of fitted catalog models."""
import copy

import numpy as np
import pytest

from RIFT.compression.distance import DistanceLikelihood, FEATURES, SCHEMA, intrinsic_coordinates


class Core:
    def __init__(self, floor):
        self.floor = floor
        self.rows = 0

    def predict(self, x):
        self.rows += len(x)
        return self.floor + (2+x[:, 4])*x[:, 8] - .4*x[:, 8]**2


class Tail:
    def __init__(self):
        self.calls = 0
        self.rows = 0

    def predict(self, x):
        self.calls += 1
        self.rows += len(x)
        return np.column_stack((2+x[:, 4], np.full(len(x), -.4)))


@pytest.fixture
def contract():
    return dict(schema=SCHEMA, quantity="raw_log_likelihood", features=list(FEATURES),
                spin_frame="test-frame", reference_frequency_hz=20.,
                source_provenance={"fixture": "known polynomial"}, validation_status="experimental",
                intrinsic_bounds=[[-3, 0, -1, -1, -1, -1, -1, -1],
                                  [8, .99, 1, 1, 1, 1, 1, 1]],
                spin_caps=[.99, .99], distance_support_mpc=[100., 20000.],
                distance_scale_mpc=10000.,
                normalization=dict(kind="constant", raw_zero_signal_log_likelihood=np.log(.1484375),
                                   overflow_offset_restored=True, measure={"time": "test Simpson"},
                                   provenance={"fixture": "known quadrature volume"}),
                tail=dict(overlap_mpc=[8000., 12000.], distance_scale_mpc=10000., degree=2,
                          validation_provenance="synthetic only"))


def states():
    return np.array([[3., .2, 0., 0., .1, 0., 0., 0.],
                     [3.2, .3, 0., 0., -.2, 0., 0., 0.],
                     [3., .2, 0., 0., .1, 0., 0., 0.]])


def test_raw_ratio_overlap_scalar_batch_and_reusable_coefficients(contract):
    floor = contract['normalization']['raw_zero_signal_log_likelihood']
    core, tail = Core(floor), Tail()
    model = DistanceLikelihood(contract, core, tail=tail, enable_tail=True)
    cached = model.prepare(states())
    assert tail.calls == 1 and tail.rows == 2  # nonadjacent duplicate states
    d = np.array([4000., 8000., 9000., 10000., 12000., 20000.])
    u = 10000/d
    truth = floor + (2+states()[:, 4, None])*u - .4*u**2
    np.testing.assert_allclose(cached.raw_log_likelihood(d), truth, atol=1e-14)
    np.testing.assert_allclose(cached.log_noise_ratio(d), truth-floor, atol=1e-14)
    for i, value in enumerate(d):
        np.testing.assert_allclose(cached.raw_log_likelihood(value)[:, 0], truth[:, i], atol=1e-14)
    assert tail.calls == 1
    np.testing.assert_array_equal(cached.raw_zero_signal_log_likelihood(), np.full(3, floor))
    cached.raw_zero_signal_log_likelihood()[:] = 0.
    np.testing.assert_allclose(cached.log_noise_ratio(d), truth-floor, atol=1e-14)
    with pytest.raises(ValueError, match="finite"):
        cached.raw_log_likelihood(np.inf)  # boundary contract does not extend support
    with pytest.raises(ValueError, match="support"):
        cached.raw_log_likelihood(20001.)


def test_rf_default_does_not_use_optional_tail(contract):
    class BadTail:
        def predict(self, x):
            raise AssertionError("Tail must be opt-in")
    model = DistanceLikelihood(contract, Core(7.), tail=BadTail())
    np.testing.assert_allclose(model.prepare(states()).raw_log_likelihood(20000.)[:, 0],
                               7.+(2+states()[:, 4])*.5-.4*.25)


def test_distinct_core_tail_blend_and_core_call_mask(contract):
    class DifferentTail:
        def predict(self, x):
            return np.zeros((len(x), 2))
    floor = contract['normalization']['raw_zero_signal_log_likelihood']
    core = Core(floor)
    cached = DistanceLikelihood(contract, core, tail=DifferentTail(), enable_tail=True).prepare(states()[:1])
    got = cached.raw_log_likelihood([8000., 10000., 12000.])[0]
    expected_core = floor+2.1*np.array([1.25, 1., 10000/12000])-.4*np.array([1.25, 1., 10000/12000])**2
    np.testing.assert_allclose(got, [expected_core[0], .5*(expected_core[1]+floor), floor])
    assert core.rows == 2


def test_state_dependent_baseline_and_grid_shape(contract):
    class Baseline:
        def predict(self, x):
            return x[:, 0]-4
    contract['normalization'].update(kind="state_evaluator", evaluator_id="audited-v1")
    with pytest.raises(ValueError, match="baseline evaluator"):
        DistanceLikelihood(contract, Core(0.))
    model = DistanceLikelihood(contract, Core(0.), baseline=Baseline(), baseline_id="audited-v1")
    prepared = model.prepare(states())
    distances = np.array([[5000., 10000.], [6000., 12000.], [7000., 14000.]])
    raw = prepared.raw_log_likelihood(distances)
    np.testing.assert_allclose(raw-prepared.log_noise_ratio(distances),
                               np.broadcast_to((states()[:, 0]-4)[:, None], raw.shape))


def test_contract_roundtrip_and_no_shape_relabeling(tmp_path, contract):
    core = Core(0.)
    model = DistanceLikelihood(contract, core)
    path = tmp_path/'metadata.json'
    model.save_contract(path)
    with pytest.raises(ValueError, match="Experimental"):
        DistanceLikelihood.load_contract(path, core)
    restored = DistanceLikelihood.load_contract(path, core, allow_experimental=True)
    np.testing.assert_array_equal(model.prepare(states()).raw_log_likelihood([1000., 20000.]),
                                  restored.prepare(states()).raw_log_likelihood([1000., 20000.]))
    shape = copy.deepcopy(contract)
    shape['quantity'] = 'preserved_intrinsic_marginal_shape'
    with pytest.raises(ValueError, match="raw likelihood"):
        DistanceLikelihood(shape, core)


@pytest.mark.parametrize('change', [
    lambda c: c['normalization'].pop('raw_zero_signal_log_likelihood'),
    lambda c: c['normalization'].update(overflow_offset_restored=False),
    lambda c: c['normalization'].update(raw_zero_signal_log_likelihood=float('nan')),
    lambda c: c['normalization'].update(measure={}),
    lambda c: c.update(distance_scale_mpc=0),
    lambda c: c.update(reference_frequency_hz=float('inf')),
    lambda c: c['tail'].update(overlap_mpc=[0., 5000.]),
    lambda c: c.update(features=list(reversed(FEATURES))),
    lambda c: c.pop('normalization'),
    lambda c: c['normalization'].update(raw_zero_signal_log_likelihood="-1.0"),
    lambda c: c.update(reference_frequency_hz="20"),
    lambda c: c.update(distance_support_mpc=[0., 20000.]),
    lambda c: c['tail'].update(overlap_mpc=[8000., 30000.]),
    lambda c: c['tail'].update(degree=0),
    lambda c: c['tail'].update(degree=True),
    lambda c: c['tail'].update(distance_scale_mpc=-1.),
    lambda c: c.update(spin_frame=""),
    lambda c: c.update(reference_frequency_hz=0.),
    lambda c: c.update(spin_caps=[.99, 1.01]),
    lambda c: c['tail'].update(validation_provenance=""),
])
def test_missing_or_incompatible_normalization_rejected(contract, change):
    change(contract)
    with pytest.raises(ValueError):
        DistanceLikelihood(contract, Core(0.))


def test_baseline_identity_and_constant_contract_guards(contract):
    class Baseline:
        def predict(self, x):
            return np.zeros(len(x))
    with pytest.raises(ValueError, match="constant contract"):
        DistanceLikelihood(contract, Core(0.), baseline=Baseline())
    contract['normalization'].update(kind="state_evaluator", evaluator_id="audited-v1")
    with pytest.raises(ValueError, match="baseline evaluator"):
        DistanceLikelihood(contract, Core(0.), baseline=Baseline(), baseline_id="audited-v2")
    class ShortBaseline:
        def predict(self, x):
            return np.zeros(len(x)+1)
    model = DistanceLikelihood(contract, Core(0.), baseline=ShortBaseline(), baseline_id="audited-v1")
    with pytest.raises(ValueError, match="one value per state"):
        model.prepare(states())


def test_numpy_metadata_is_normalized_for_json(tmp_path, contract):
    contract['intrinsic_bounds'] = np.asarray(contract['intrinsic_bounds'])
    contract['tail']['degree'] = np.int64(2)
    model = DistanceLikelihood(contract, Core(0.), tail=Tail(), enable_tail=True)
    model.save_contract(tmp_path/'metadata.json')
    restored = DistanceLikelihood.load_contract(tmp_path/'metadata.json', Core(0.), tail=Tail(),
                                                enable_tail=True, allow_experimental=True)
    assert restored.contract == model.contract


@pytest.mark.parametrize('edit', [
    lambda x: x.__setitem__((0, 0), 9.),     # above ln chirp-mass bound
    lambda x: x.__setitem__((0, 0), -4.),    # below ln chirp-mass bound
    lambda x: x.__setitem__((0, 1), -.1),    # negative delta
    lambda x: x.__setitem__((0, 1), .995),   # delta above declared bound
])
def test_intrinsic_support_rejected(contract, edit):
    x = states()
    edit(x)
    with pytest.raises(ValueError, match="support"):
        DistanceLikelihood(contract, Core(0.)).prepare(x)


@pytest.mark.parametrize('delta', [-.1, 1.])
def test_delta_chart_limits_hold_inside_wide_bounds(contract, delta):
    contract['intrinsic_bounds'][0][1], contract['intrinsic_bounds'][1][1] = -1., 2.
    x = states()
    x[0, 1] = delta
    with pytest.raises(ValueError, match="support"):
        DistanceLikelihood(contract, Core(0.)).prepare(x)


def test_mass_order_and_redshift_rejected():
    with pytest.raises(ValueError, match="m1 >= m2"):
        intrinsic_coordinates(20., 30., np.zeros(6), mass_frame="detector")
    with pytest.raises(ValueError, match="redshift"):
        intrinsic_coordinates(30., 20., np.zeros(6), mass_frame="source", redshift=-.1)


def test_distance_grid_shape_and_lower_support(contract):
    prepared = DistanceLikelihood(contract, Core(0.)).prepare(states())
    with pytest.raises(ValueError, match="support"):
        prepared.raw_log_likelihood(99.)
    with pytest.raises(ValueError, match="grid"):
        prepared.raw_log_likelihood(np.full((2, 4), 5000.))
    # A 1-D input is a shared grid even when its length equals the state count.
    shared = prepared.raw_log_likelihood([1000., 2000., 3000.])
    np.testing.assert_array_equal(shared, prepared.raw_log_likelihood(np.tile([1000., 2000., 3000.], (3, 1))))
    paired = prepared.raw_log_likelihood(np.array([[1000.], [2000.], [3000.]]))[:, 0]
    np.testing.assert_array_equal(paired, np.diag(shared))
    class WrongShape:
        def predict(self, x):
            return np.zeros((len(x), 1))
    with pytest.raises(ValueError, match="one raw log likelihood per row"):
        DistanceLikelihood(contract, WrongShape()).prepare(states()).raw_log_likelihood(5000.)


def test_overlap_weight_is_smoothstep(contract):
    class ZeroCore:
        def predict(self, x):
            return np.zeros(len(x))
    class ConstantTail:
        def predict(self, x):
            return np.zeros((len(x), 2))
    floor = contract['normalization']['raw_zero_signal_log_likelihood']
    cached = DistanceLikelihood(contract, ZeroCore(), tail=ConstantTail(), enable_tail=True).prepare(states()[:1])
    t = np.array([.1, .25, .75, .9])
    got = cached.raw_log_likelihood(8000.+4000.*t)[0]
    np.testing.assert_allclose(got, floor*t*t*(3-2*t), atol=1e-14)


def test_input_and_prediction_failures_cannot_publish_a_number(contract):
    model = DistanceLikelihood(contract, Core(0.))
    bad = states()
    bad[0, 2:5] = [.8, .8, 0.]
    with pytest.raises(ValueError, match="Spin"):
        model.prepare(bad)
    class Nonfinite:
        def predict(self, x):
            return np.full(len(x), np.nan)
    with pytest.raises(ValueError, match="finite"):
        DistanceLikelihood(contract, Nonfinite()).prepare(states()).raw_log_likelihood(10000.)


def test_source_mass_chart_and_all_six_spin_components(contract):
    spin = np.array([[.1, .2, .3, -.1, -.2, -.3]])
    source = intrinsic_coordinates([30.], [20.], spin, mass_frame="source", redshift=[.7])
    detector = intrinsic_coordinates([51.], [34.], spin, mass_frame="detector")
    np.testing.assert_allclose(source, detector)
    np.testing.assert_array_equal(source[:, 2:], spin)
    with pytest.raises(ValueError, match="redshift"):
        intrinsic_coordinates(30., 20., spin[0], mass_frame="source")
    class SpinProbe:
        def predict(self, x):
            return x[:, 2:8] @ np.arange(1, 7)
    probe = DistanceLikelihood(contract, SpinProbe())
    expected = (spin @ np.arange(1, 7)).item()
    assert probe.prepare(source).raw_log_likelihood(5000.)[0, 0] == pytest.approx(expected)


def test_prepared_metadata_and_backend_reference_are_snapshotted(contract):
    floor = contract['normalization']['raw_zero_signal_log_likelihood']
    model = DistanceLikelihood(contract, Core(floor), tail=Tail(), enable_tail=True)
    prepared = model.prepare(states())
    before = prepared.raw_log_likelihood([4000., 10000., 20000.])
    model.contract['distance_scale_mpc'] *= 2
    model.contract['normalization']['raw_zero_signal_log_likelihood'] += 50
    model.core = Core(100.)
    np.testing.assert_array_equal(prepared.raw_log_likelihood([4000., 10000., 20000.]), before)


def test_finite_parameters_cannot_publish_overflow(contract):
    class Huge:
        def predict(self, x):
            return np.full((len(x), 3), 1e308)
    contract['distance_scale_mpc'] = 20000.
    contract['tail']['distance_scale_mpc'] = 20000.
    contract['tail']['degree'] = 3
    prepared = DistanceLikelihood(contract, Core(0.), tail=Huge(), enable_tail=True).prepare(states())
    with np.errstate(over='ignore'), pytest.raises(ValueError, match="finite"):
        prepared.raw_log_likelihood(12000.)
    x = intrinsic_coordinates(1e308, 1e308, np.zeros(6), mass_frame='detector')
    assert np.isfinite(x).all()
    assert x[0] == pytest.approx(np.log(1e308)-.2*np.log(2))
    with np.errstate(over='ignore'), pytest.raises(ValueError, match="finite"):
        intrinsic_coordinates(1e308, 1e308, np.zeros(6), mass_frame='source', redshift=2.)


def test_complex_predictions_and_backend_input_mutation(contract):
    class ComplexCore:
        def predict(self, x):
            return np.full(len(x), 1+2j)
    with pytest.raises(ValueError, match='real'):
        DistanceLikelihood(contract, ComplexCore()).prepare(states()).raw_log_likelihood(10000.)
    class MutatingTail:
        def predict(self, x):
            x[:] = 100.
            return np.zeros((len(x), 2))
    cached = DistanceLikelihood(contract, Core(0.), tail=MutatingTail(), enable_tail=True).prepare(states())
    np.testing.assert_array_equal(cached.intrinsic[cached.inverse], states())
    np.testing.assert_allclose(cached.raw_log_likelihood(4000.)[:, 0],
                               (2+states()[:, 4])*2.5-.4*2.5**2)


@pytest.mark.parametrize('enable_tail', [False, True])
def test_explicit_prior_integral_against_independent_quadrature(contract, enable_tail):
    from scipy.integrate import quad
    lo, hi = 4000., 20000.
    contract['distance_support_mpc'] = [lo, hi]
    floor = contract['normalization']['raw_zero_signal_log_likelihood']
    cached = DistanceLikelihood(contract, Core(floor), tail=Tail(), enable_tail=enable_tail).prepare(states()[:1])
    nodes, weights = np.polynomial.legendre.leggauss(96)
    d = lo+(nodes+1)*(hi-lo)/2
    prior = 3*d**2/(hi**3-lo**3)  # explicitly normalized dL density
    numerical = np.sum(np.exp(cached.raw_log_likelihood(d)[0])*prior*weights*(hi-lo)/2)
    reference = quad(lambda d: np.exp(floor+2.1*10000/d-.4*(10000/d)**2)*3*d**2/(hi**3-lo**3),
                     lo, hi, epsabs=1e-11, epsrel=1e-11)[0]
    assert numerical == pytest.approx(reference, rel=1e-11)


def test_tail_polynomial_and_core_have_independent_distance_scales(contract):
    class RescaledTail(Tail):
        def predict(self, x):
            return super().predict(x)*np.array([10000/8000, (10000/8000)**2])
    contract['tail']['distance_scale_mpc'] = 8000.
    floor = contract['normalization']['raw_zero_signal_log_likelihood']
    cached = DistanceLikelihood(contract, Core(floor), tail=RescaledTail(), enable_tail=True).prepare(states())
    d = np.array([4000., 10000., 20000.])
    u = 10000/d
    np.testing.assert_allclose(cached.raw_log_likelihood(d),
                               floor+(2+states()[:, 4, None])*u-.4*u*u, atol=1e-14)
