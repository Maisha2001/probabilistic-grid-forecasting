import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .features import training_origins, training_statistics
from .io import file_hash, read_json, write_json


def inputs(index, values, origins, horizons, scale, settings):
    origins = np.asarray(origins, dtype=int)
    history = settings["history_hours"]
    if len(origins) == 0 or origins.min() < max(336, history-1) or origins.max() >= len(values):
        raise ValueError("Not enough observed history")
    if any(not 1 <= h <= 168 for h in horizons):
        raise ValueError("Unsupported horizon")
    normalized = values / np.asarray(scale)
    past = normalized[origins[:, None] - np.arange(history-1, -1, -1)]
    patches = past.reshape(len(origins), history//settings["patch_hours"], -1)
    calendar, weekly = [], []
    for horizon in horizons:
        dates = pd.DatetimeIndex(index[origins])+pd.Timedelta(hours=horizon)
        calendar.append(np.column_stack([np.sin(2*np.pi*dates.hour/24), np.cos(2*np.pi*dates.hour/24),
                                         np.sin(2*np.pi*dates.dayofweek/7), np.cos(2*np.pi*dates.dayofweek/7),
                                         np.sin(2*np.pi*dates.dayofyear/365.25), np.cos(2*np.pi*dates.dayofyear/365.25)]))
        weekly.append(normalized[origins+horizon-168])
    return tuple(torch.from_numpy(x.astype(np.float32)) for x in
                 (patches, np.stack(calendar, axis=1), np.stack(weekly, axis=1)))


class PatchForecaster(nn.Module):
    def __init__(self, settings):
        super().__init__()
        d = settings["d_model"]
        patches = settings["history_hours"]//settings["patch_hours"]
        self.embedding = nn.Linear(settings["patch_hours"]*6, d)
        self.position = nn.Parameter(torch.empty(1, patches, d))
        layer = nn.TransformerEncoderLayer(d, settings["heads"], d*2, settings["dropout"], batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, settings["layers"], enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d*2+12, d*2), nn.GELU(), nn.Linear(d*2, 18))
        nn.init.normal_(self.position, std=.02)
        # Encoder layers start with independent weights, rather than cloned initial values.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)

    def forward(self, patches, calendar, weekly):
        encoded = self.encoder(self.embedding(patches-1)+self.position)
        pooled = torch.cat([encoded.mean(dim=1), encoded[:, -1]], dim=-1)
        repeated = pooled[:, None].expand(-1, calendar.shape[1], -1)
        raw = self.head(torch.cat([repeated, calendar, weekly], dim=-1)).reshape(len(patches), -1, 6, 3)
        center = weekly + .2*raw[..., 1]
        lower = center-.1*F.softplus(raw[..., 0])
        upper = center+.1*F.softplus(raw[..., 2])
        return torch.stack([lower, center, upper], dim=-1).clamp(min=0)


def quantile_loss(target, prediction):
    quantiles = prediction.new_tensor([.1, .5, .9])
    error = target[..., None]-prediction
    return torch.maximum(quantiles*error, (quantiles-1)*error).mean()


def configure(seed, threads):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def predict_temporal(model, metadata, index, values, origins, device="cpu"):
    model.to(device).eval()
    data = inputs(index, values, origins, metadata["horizons"], metadata["scale"], metadata["settings"])
    batches = []
    with torch.no_grad():
        for start in range(0, len(origins), metadata["settings"]["batch_size"]):
            batch = [x[start:start+metadata["settings"]["batch_size"]].to(device) for x in data]
            batches.append(model(*batch).cpu().numpy())
    return np.concatenate(batches)*np.asarray(metadata["scale"])[None, None, :, None]


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit_temporal(index, values, cutoff, selection_origins, config, seed, output, device="cpu", stop_after=None):
    configure(seed, config["threads"])
    settings = config["neural"]
    scale, _, _ = training_statistics(values, cutoff, config["training_hours"])
    origins = training_origins(cutoff, max(config["horizons"]), config["training_hours"], config["training_stride"])
    origins = origins[origins >= settings["history_hours"]-1]
    train = inputs(index, values, origins, config["horizons"], scale, settings)
    valid = inputs(index, values, selection_origins, config["horizons"], scale, settings)
    targets = lambda positions: torch.tensor(np.stack([values[positions+h]/scale for h in config["horizons"]], axis=1), dtype=torch.float32)
    target, validation_target = targets(origins), targets(selection_origins)
    digest = hashlib.sha256(json.dumps({"config": config, "seed": seed}, sort_keys=True).encode())
    digest.update(Path(__file__).read_bytes())
    for tensor in (*train, target, *valid, validation_target):
        digest.update(tensor.numpy().tobytes())
    fingerprint = digest.hexdigest()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output/"checkpoint.pt"
    model = PatchForecaster(settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    history, best_loss, best_weights, first = [], float("inf"), None, 1
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["fingerprint"] != fingerprint or state["device"] != device:
            raise ValueError("Checkpoint inputs, settings, source or device changed; choose a new run")
        model.load_state_dict(state["weights"]); optimizer.load_state_dict(state["optimizer"])
        history, best_loss, best_weights, first = state["history"], state["best_loss"], state["best_weights"], state["epoch"]+1
        torch.set_rng_state(state["cpu_rng"])
        if device == "cuda": torch.cuda.set_rng_state_all(state["cuda_rng"])
    for epoch in range(first, settings["epochs"]+1):
        model.train(); permutation = torch.randperm(len(origins)); total = 0.0
        for start in range(0, len(origins), settings["batch_size"]):
            ids = permutation[start:start+settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = quantile_loss(target[ids].to(device), model(*(x[ids].to(device) for x in train)))
            if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach())*len(ids)
        model.eval()
        with torch.no_grad():
            valid_loss = float(quantile_loss(validation_target.to(device), model(*(x.to(device) for x in valid))))
        history.append({"epoch": epoch, "training_pinball": total/len(origins), "selection_pinball": valid_loss})
        if valid_loss < best_loss:
            best_loss, best_weights = valid_loss, cpu_state(model)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"fingerprint": fingerprint, "device": device, "epoch": epoch, "weights": cpu_state(model),
                    "optimizer": optimizer.state_dict(), "best_weights": best_weights, "best_loss": best_loss,
                    "history": history, "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else []}, temporary)
        temporary.replace(checkpoint)
        if stop_after is not None and epoch >= stop_after:
            return None, {"interrupted_after_epoch": epoch}
    model.load_state_dict(best_weights)
    torch.save(best_weights, output/"weights.pt")
    metadata = {"settings": settings, "horizons": config["horizons"], "scale": scale.tolist(), "seed": seed,
                "weights_sha256": file_hash(output/"weights.pt"), "training_cutoff": int(cutoff),
                "last_training_target": int(origins[-1]+max(config["horizons"])),
                "last_selection_target": int(selection_origins[-1]+max(config["horizons"])),
                "training_origins": len(origins), "selected_epoch": min(history, key=lambda h: h["selection_pinball"])["epoch"],
                "history": history, "fingerprint": fingerprint, "parameters": sum(p.numel() for p in model.parameters())}
    write_json(output/"model.json", metadata)
    return model, metadata


def load_temporal(path, device="cpu"):
    path = Path(path); metadata = read_json(path/"model.json")
    if file_hash(path/"weights.pt") != metadata["weights_sha256"]:
        raise ValueError("Temporal model checksum mismatch")
    model = PatchForecaster(metadata["settings"])
    model.load_state_dict(torch.load(path/"weights.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval(), metadata
