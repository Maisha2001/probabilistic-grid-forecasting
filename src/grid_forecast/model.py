from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .features import QUANTILES, feature_matrix, ordered_quantiles, pinball, training_origins, training_statistics
from .io import file_hash, read_json, write_json


def fit_model(index, values, boundary, horizon, candidate, config):
    scale, mase, residual = training_statistics(values, boundary, config["training_hours"])
    origins = training_origins(boundary, horizon, config["training_hours"], config["training_stride"])
    features = feature_matrix(index, values, origins, horizon, scale)
    target = (values[origins+horizon] / scale).reshape(-1)
    model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=QUANTILES,
                         tree_method="hist", learning_rate=config["learning_rate"],
                         random_state=config["seed"], n_jobs=config["threads"],
                         subsample=1.0, colsample_bytree=1.0, **candidate)
    model.fit(features, target, verbose=False)
    return model, {"scale": scale.tolist(), "mase_scale": mase.tolist(),
                   "seasonal_residual_quantiles": residual.tolist(),
                   "boundary": int(boundary), "last_training_target": int(origins[-1]+horizon),
                   "training_origins": len(origins), "training_rows": len(target), "horizon": int(horizon)}


def predict_model(model, metadata, index, values, origins):
    scale = np.asarray(metadata["scale"])
    features = feature_matrix(index, values, origins, metadata["horizon"], scale)
    raw = model.predict(features).reshape(len(origins), values.shape[1], 3) * scale[None, :, None]
    crossings = int(np.any(np.diff(raw, axis=-1) < 0, axis=-1).sum())
    return ordered_quantiles(raw), crossings


def select_and_fit(index, values, boundary, horizon, config, output):
    validation_start = boundary - config["validation_days"]*24
    validation_origins = np.arange(validation_start, boundary-horizon, config["training_stride"])
    if len(validation_origins) == 0:
        raise ValueError("Validation interval is shorter than the horizon")
    choices = []
    for candidate in config["candidates"]:
        model, metadata = fit_model(index, values, validation_start, horizon, candidate, config)
        predicted, crossings = predict_model(model, metadata, index, values, validation_origins)
        scale = np.asarray(metadata["scale"])
        loss = pinball(values[validation_origins+horizon]/scale, predicted/scale[None, :, None]).mean()
        choices.append({"parameters": candidate, "normalized_pinball": float(loss), "crossings": crossings,
                        "last_training_target": metadata["last_training_target"],
                        "last_validation_target": int(validation_origins[-1]+horizon)})
    winner = min(range(len(choices)), key=lambda i: (choices[i]["normalized_pinball"], i))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "selection.json", {"validation_start": int(validation_start), "boundary": int(boundary),
                                          "candidates": choices, "selected_index": winner})
    model, metadata = fit_model(index, values, boundary, horizon, config["candidates"][winner], config)
    model.save_model(output / "model.ubj")
    metadata.update(model_sha256=file_hash(output/"model.ubj"), selected_parameters=config["candidates"][winner],
                    feature_schema="lags-calendar-node-v1", forecast_origin_not_before=str(index[boundary]),
                    last_training_timestamp=str(index[metadata["last_training_target"]]))
    write_json(output / "model.json", metadata)
    return model, metadata, choices


def load_model(path):
    path = Path(path)
    metadata = read_json(path / "model.json")
    if metadata["feature_schema"] != "lags-calendar-node-v1" or file_hash(path/"model.ubj") != metadata["model_sha256"]:
        raise ValueError("Model identity or feature schema mismatch")
    model = XGBRegressor()
    model.load_model(path / "model.ubj")
    return model, metadata
