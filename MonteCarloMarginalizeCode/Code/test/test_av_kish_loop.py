"""Run the real AV loop on a 2-D Gaussian: kish is opt-in, the default is unchanged."""
import numpy as np
import pytest

import RIFT.integrators.mcsamplerAdaptiveVolume as av

pytestmark = pytest.mark.skipif(av.xpy_default is not np, reason="pinned values are for the numpy backend")

# Default-path result of the merge-base (rift_O4d 76ead4ce), same seed and settings.
BASE_LNZ = -5.0105830977664745
BASE_ROWS = 4506


def _run(**kwargs):
    np.random.seed(20261004)
    s = av.MCSampler(n_chunk=4000)
    for name in ('x', 'y'):
        s.add_parameter(name, pdf=None, left_limit=-5., right_limit=5.,
                        prior_pdf=lambda x: np.ones(np.shape(x))/10., adaptive_sampling=True)
    fn = lambda x, y: -0.5*((x-0.3)**2/0.2**2 + (y+0.1)**2/0.5**2)
    res = s.integrate_log(fn, 'x', 'y', nmax=400000, neff=300, n=4000,
                          no_protect_names=True, verbose=False, **kwargs)
    return res, np.array(s._rvs['log_integrand']), s.last_stopping_statistics


def test_default_matches_base_and_explicit_max_weight():
    res, lw, stats = _run()
    assert res[0] == pytest.approx(BASE_LNZ, rel=1e-12, abs=0)
    assert len(lw) == BASE_ROWS
    res2, lw2, _ = _run(av_stop_metric='max-weight')
    assert res2[0] == res[0] and res2[2] == res[2]
    np.testing.assert_array_equal(lw, lw2)
    assert stats['metric'] == 'max-weight' and stats['total_draws'] == 12054


def test_kish_stops_at_different_draw_count():
    _, _, base = _run()
    res, lw, kish = _run(av_stop_metric='kish')
    assert kish['metric'] == 'kish' and kish['selected'] == kish['kish'] >= 300
    assert kish['total_draws'] < base['total_draws']
    assert res[0] == pytest.approx(-5.0699, abs=0.2)  # ln(2 pi 0.2 0.5 / 100)
