import numpy as np
import pytest

from grid_forecast.robustness import causal_history, observed_scenario, shift_summary


def test_causal_missing_repair_ignores_future_and_stale_aggregate(synthetic):
    _, values = synthetic
    observed = values.copy(); observed[985:1001, 2] = np.nan
    repaired, counts = causal_history(observed, 1000)
    np.testing.assert_array_equal(repaired[985:1001, 2], values[817:833, 2])
    assert counts["SA"] == 16
    changed = observed.copy(); changed[1001:] *= 100; changed[:, 5] = -9999
    np.testing.assert_array_equal(causal_history(changed, 1000)[0], repaired)
    np.testing.assert_array_equal(repaired[:, 5], repaired[:, :5].sum(axis=1))
    assert np.isnan(observed[1000, 2])


@pytest.mark.parametrize("bad", ["gap", "leading", "negative", "infinite"])
def test_history_rejects_unrecoverable_observations(synthetic, bad):
    _, values = synthetic; observed = values.copy()
    if bad == "gap": observed[976:1001, 0] = np.nan
    if bad == "leading": observed[0, 0] = np.nan
    if bad == "negative": observed[1000, 0] = -1
    if bad == "infinite": observed[1000, 0] = np.inf
    with pytest.raises(ValueError): causal_history(observed, 1000)


def test_observation_scenarios_are_repeatable_and_do_not_change_labels(synthetic):
    _, values = synthetic; before = values.copy()
    for scenario, count in [("missing_6h", 30), ("missing_24h", 120), ("bias_up_10pct_24h", 0)]:
        observed, counts = observed_scenario(values, 1000, scenario)
        assert sum(counts.values()) == count and np.isfinite(observed).all()
    a, _ = observed_scenario(values, 1000, "missing_10pct")
    np.testing.assert_array_equal(a, observed_scenario(values, 1000, "missing_10pct")[0])
    np.testing.assert_array_equal(values, before)
    with pytest.raises(ValueError): observed_scenario(values, 1000, "unknown")


def test_shift_bins_and_peak_threshold_use_training_only():
    train = np.arange(1, 101, dtype=float)
    unchanged = shift_summary(train, train)
    shifted = shift_summary(train, train*2)
    assert unchanged["population_stability_index"] == 0
    assert shifted["population_stability_index"] > 0
    assert shifted["mean_change_fraction"] == 1
    assert shifted["training_peak_threshold_mw"] == unchanged["training_peak_threshold_mw"]
