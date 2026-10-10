import numpy as np
import pytest

from grid_forecast.reconciliation import SUMMING, fit_projection, fit_bands, predict_bands, reconcile


def test_projection_is_identity_on_coherent_vectors_and_repairs_incoherence():
    rng = np.random.default_rng(7)
    state = fit_projection(rng.normal(size=(100, 6))*np.arange(1, 7))
    p = np.asarray(state["projection"])
    np.testing.assert_allclose(p@SUMMING, SUMMING, atol=1e-10)
    np.testing.assert_allclose(p@p, p, atol=1e-10)
    raw = np.array([[100, 200, 300, 400, 500, 1600], [-100, 2, 3, 4, 5, -90]])
    result = reconcile(raw, state)
    assert (result >= 0).all()
    np.testing.assert_allclose(result[:, -1], result[:, :5].sum(axis=1), atol=1e-10)
    assert not np.allclose(raw, result)


def test_conformal_rank_and_coherent_simultaneous_envelope():
    center = np.tile(np.array([100, 100, 100, 100, 100, 500]), (20, 1))
    raw = np.stack([center-1, center, center+1], axis=-1)
    actual = center.copy(); actual[:, :5] += np.arange(1, 21)[:, None]
    actual[:, -1] = actual[:, :5].sum(axis=1)
    projection = {"projection": (SUMMING@np.linalg.pinv(SUMMING)).tolist()}
    state = fit_bands(actual, raw, projection, np.ones(6), .8)
    assert state["rank"] == 17
    assert state["multiplier"] == pytest.approx(17)
    bands = predict_bands(raw, state)
    np.testing.assert_allclose(bands[:, -1], bands[:, :5].sum(axis=1), atol=1e-10)
    covered = ((actual>=bands[..., 0]) & (actual<=bands[..., 2]))
    assert covered[:, :5].all(axis=1).mean() == pytest.approx(.85)
    assert np.all(covered[:, -1] >= covered[:, :5].all(axis=1))
    assert (bands[..., 0] <= bands[..., 1]).all() and (bands[..., 1] <= bands[..., 2]).all()


def test_invalid_calibration_and_covariance_fail():
    with pytest.raises(ValueError): fit_projection(np.zeros((9, 6)))
    with pytest.raises(ValueError): fit_projection(np.full((20, 6), np.nan))
    q = np.ones((10, 6, 3))
    with pytest.raises(ValueError): fit_bands(q[..., 1], q, {}, np.ones(6), 1)
