import numpy as np
import pandas as pd

MAX_LAG = 336
QUANTILES = np.array([0.1, 0.5, 0.9])


def feature_matrix(index, values, origins, horizon, scale):
    origins = np.asarray(origins, dtype=int)
    if not 1 <= horizon <= 168 or len(origins) == 0 or origins.min() < MAX_LAG or origins.max() >= len(values):
        raise ValueError("Origins need 336 hours of history; horizons must be 1..168")
    if np.any(np.asarray(scale) <= 0):
        raise ValueError("Training scales must be positive")
    normalized = values / scale
    # Every lag refers to an observed interval, including target-aligned weekly lags.
    parts = [normalized[origins - lag] for lag in (0, 1, 2, 23, 24, 25, 167, 168, 169)]
    parts.extend(normalized[origins + horizon - lag] for lag in (168, 336))
    for window in (24, 168):
        past = np.stack([normalized[origins - lag] for lag in range(window)], axis=0)
        parts.extend((past.mean(axis=0), past.std(axis=0)))
    history = np.stack(parts, axis=-1)
    target = pd.DatetimeIndex(index[origins]) + pd.Timedelta(hours=horizon)
    calendar = np.column_stack([np.sin(2*np.pi*target.hour/24), np.cos(2*np.pi*target.hour/24),
                                np.sin(2*np.pi*target.dayofweek/7), np.cos(2*np.pi*target.dayofweek/7),
                                np.sin(2*np.pi*target.dayofyear/365.25), np.cos(2*np.pi*target.dayofyear/365.25)])
    count, nodes = len(origins), values.shape[1]
    return np.concatenate([history.reshape(count*nodes, -1), np.repeat(calendar, nodes, axis=0),
                           np.tile(np.eye(nodes), (count, 1))], axis=1).astype(np.float32)


def training_origins(boundary, horizon, hours, stride):
    origins = np.arange(max(MAX_LAG, boundary-hours), boundary-horizon, stride, dtype=int)
    if len(origins) < 20:
        raise ValueError("Insufficient training history")
    return origins


def training_statistics(values, boundary, hours):
    prefix = values[max(0, boundary-hours):boundary]
    if len(prefix) <= 168 or not np.isfinite(prefix).all():
        raise ValueError("Invalid training prefix")
    scale = prefix.mean(axis=0)
    mase = np.abs(prefix[168:] - prefix[:-168]).mean(axis=0)
    if np.any(scale <= 0) or np.any(mase <= 0):
        raise ValueError("Undefined scaling for constant or zero series")
    residual = np.quantile(prefix[168:] - prefix[:-168], QUANTILES, axis=0).T
    return scale, mase, residual


def pinball(actual, prediction):
    error = np.asarray(actual)[..., None] - prediction
    return np.maximum(QUANTILES*error, (QUANTILES-1)*error)


def ordered_quantiles(raw):
    return np.sort(np.maximum(raw, 0), axis=-1)
