import numpy as np
import pandas as pd
import pytest

from grid_forecast.data import STATES


@pytest.fixture
def synthetic():
    index = pd.date_range("2012-01-01T01:00", periods=3000, freq="h")
    time = np.arange(len(index))
    leaves = np.column_stack([(100+20*i)*(1+.12*np.sin(time/24*2*np.pi)+.06*np.sin(time/168*2*np.pi))
                              +np.random.default_rng(i).normal(0, 2, len(time)) for i in range(len(STATES))])
    return index, np.column_stack([leaves, leaves.sum(axis=1)])


@pytest.fixture
def config():
    return {"folds": ["2012-03-01T00:00:00"], "test_days": 2, "validation_days": 10,
            "training_hours": 720, "training_stride": 6, "evaluation_stride": 6,
            "horizons": [1, 24], "quantiles": [.1, .5, .9],
            "candidates": [{"max_depth": 2, "n_estimators": 8}, {"max_depth": 3, "n_estimators": 8}],
            "learning_rate": .07, "seed": 42, "threads": 1}
