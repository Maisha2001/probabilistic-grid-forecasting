import math

import numpy as np
from sklearn.covariance import LedoitWolf

SUMMING = np.vstack([np.eye(5), np.ones(5)])


def fit_projection(residuals):
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2 or residuals.shape[1] != 6 or len(residuals) < 10 or not np.isfinite(residuals).all():
        raise ValueError("Reconciliation needs at least ten finite six-node residual vectors")
    estimate = LedoitWolf().fit(residuals)
    covariance = estimate.covariance_
    covariance += np.eye(6) * max(np.trace(covariance)/6, 1.0)*1e-6
    weighted = np.linalg.solve(covariance, SUMMING)
    projection = SUMMING @ np.linalg.solve(SUMMING.T @ weighted, weighted.T)
    if not np.allclose(projection @ SUMMING, SUMMING, atol=1e-8):
        raise ValueError("Unstable reconciliation matrix")
    return {"projection": projection.tolist(), "covariance": covariance.tolist(),
            "shrinkage": float(estimate.shrinkage_), "rows": len(residuals)}


def reconcile(point, state):
    point = np.asarray(point, dtype=float)
    projected = point @ np.asarray(state["projection"]).T
    leaves = np.maximum(projected[..., :5], 0)
    return leaves @ SUMMING.T


def width_scale(quantiles, floor):
    return np.maximum((quantiles[..., 2]-quantiles[..., 0])/2, floor)


def fit_bands(actual, raw_quantiles, state, training_scale, coverage=0.8):
    if not 0 < coverage < 1 or len(actual) < 10:
        raise ValueError("Invalid calibration coverage or sample size")
    floor = np.maximum(np.asarray(training_scale)*.001, 1e-6)
    point = reconcile(raw_quantiles[..., 1], state)
    widths = width_scale(raw_quantiles, floor)
    scores = np.max(np.abs(actual[:, :5]-point[:, :5])/widths[:, :5], axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("Invalid calibration scores")
    rank = math.ceil((len(scores)+1)*coverage)
    if rank > len(scores):
        raise ValueError("Too few calibration samples for requested finite band")
    return {**state, "multiplier": float(np.sort(scores)[rank-1]), "width_floor": floor.tolist(),
            "coverage": coverage, "calibration_rows": len(scores), "rank": rank,
            "calibration_scores": scores.tolist(), "interval_semantics": "simultaneous_leaf_bands_with_aggregate_envelope"}


def predict_bands(raw_quantiles, state):
    raw_quantiles = np.asarray(raw_quantiles, dtype=float)
    point = reconcile(raw_quantiles[..., 1], state)
    radius = width_scale(raw_quantiles, np.asarray(state["width_floor"]))[:, :5]*state["multiplier"]
    lower = np.maximum(point[:, :5]-radius, 0) @ SUMMING.T
    upper = (point[:, :5]+radius) @ SUMMING.T
    return np.stack([lower, point, upper], axis=-1)
