import numpy as np
import pandas as pd
import pytest

from grid_forecast.data import STATES, assemble, clean_state, matrix_from_frame


def frame():
    return pd.DataFrame({"Date": np.repeat([40909, 40910], 48), "Period": np.tile(np.arange(1, 49), 2),
                         "OperationalLessIndustrial": np.arange(96)+100.0})


def test_hourly_means_interval_end_and_hierarchy():
    rows, audit = assemble({s: frame() for s in STATES}, "2012-01-01", "2012-01-03")
    index, values = matrix_from_frame(rows)
    assert str(index[0]) == "2012-01-01 01:00:00" and str(index[-1]) == "2012-01-03 00:00:00"
    assert values.shape == (48, 6) and values[0, 0] == 100.5
    np.testing.assert_allclose(values[:, -1], values[:, :-1].sum(axis=1))
    assert audit["NSW"]["selected_half_hours"] == 96


@pytest.mark.parametrize("failure", ["gap", "duplicate", "nan", "negative", "period", "fractional_day"])
def test_rejects_corrupt_source(failure):
    data = frame()
    if failure == "gap": data = data.iloc[1:]
    elif failure == "duplicate": data = pd.concat([data, data.iloc[[0]].assign(OperationalLessIndustrial=999)])
    elif failure == "nan": data.loc[0, "OperationalLessIndustrial"] = np.nan
    elif failure == "negative": data.loc[0, "OperationalLessIndustrial"] = -1
    elif failure == "period": data.loc[0, "Period"] = 49
    else: data["Date"] = data.Date.astype(float); data.loc[0, "Date"] += .5
    with pytest.raises(ValueError): clean_state(data, "NSW", "2012-01-01", "2012-01-03")


def test_identical_duplicates_are_audited():
    data = pd.concat([frame(), frame().iloc[[0]]])
    result, audit = clean_state(data, "NSW", "2012-01-01", "2012-01-03")
    assert len(result) == 48 and audit["identical_duplicates_removed"] == 1


@pytest.mark.parametrize("failure", ["aggregate", "gap", "missing_node", "duplicate"])
def test_rejects_bad_prepared_hierarchy(failure):
    rows, _ = assemble({s: frame() for s in STATES}, "2012-01-01", "2012-01-03")
    if failure == "aggregate": rows.loc[rows.node == "ALL_FIVE", "demand_mw"] += 1
    elif failure == "gap": rows = rows[rows.timestamp != rows.timestamp.drop_duplicates().iloc[1]]
    elif failure == "missing_node": rows = rows[rows.node != "SA"]
    else: rows = pd.concat([rows, rows.iloc[[0]]])
    with pytest.raises(ValueError): matrix_from_frame(rows)
