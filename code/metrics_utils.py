from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def runtime_context() -> dict[str, Any]:
    context: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }
    try:
        import torch

        context["torch_version"] = torch.__version__
        context["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            devices = []
            for idx in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(idx)
                devices.append(
                    {
                        "index": idx,
                        "name": prop.name,
                        "total_vram_gb": round(prop.total_memory / (1024**3), 2),
                    }
                )
            context["cuda_devices"] = devices
    except Exception as exc:
        context["torch_error"] = str(exc)

    context["virtual_env"] = os.getenv("VIRTUAL_ENV", "")
    return context


def stratified_train_test_split(labels: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)

    idx_fact = np.where(labels == 0)[0]
    idx_hall = np.where(labels == 1)[0]

    rng.shuffle(idx_fact)
    rng.shuffle(idx_hall)

    n_fact_test = max(1, int(round(len(idx_fact) * test_size))) if len(idx_fact) > 1 else len(idx_fact)
    n_hall_test = max(1, int(round(len(idx_hall) * test_size))) if len(idx_hall) > 1 else len(idx_hall)

    test_idx = np.concatenate([idx_fact[:n_fact_test], idx_hall[:n_hall_test]])
    train_idx = np.concatenate([idx_fact[n_fact_test:], idx_hall[n_hall_test:]])

    if train_idx.size == 0:
        # Fallback for tiny datasets.
        train_idx = test_idx.copy()

    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return train_idx, test_idx


def _oriented_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, str]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return float("nan"), "undefined"

    auc = float(roc_auc_score(labels, scores))
    if math.isnan(auc):
        return auc, "undefined"
    if auc < 0.5:
        return float(1.0 - auc), "flipped"
    return auc, "as-is"


def bootstrap_auc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if labels.size < 2 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = labels.size

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_b = labels[idx]
        if len(np.unique(y_b)) < 2:
            continue
        s_b = scores[idx]
        auc_b, _ = _oriented_auc(y_b, s_b)
        if not math.isnan(auc_b):
            vals.append(float(auc_b))

    if not vals:
        return float("nan"), float("nan")

    lo = float(np.quantile(vals, 0.025))
    hi = float(np.quantile(vals, 0.975))
    return lo, hi


def centroid_scores_only(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    eps = 1e-8

    x_fact = x_train[y_train == 0]
    x_hall = x_train[y_train == 1]

    mu_fact = x_fact.mean(axis=0)
    mu_hall = x_hall.mean(axis=0)

    d_fact = np.linalg.norm(x_test - mu_fact, axis=1)
    d_hall = np.linalg.norm(x_test - mu_hall, axis=1)
    return d_fact / (d_hall + eps)


def mahalanobis_scores_only(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    max_dim: int = 256,
) -> np.ndarray:
    eps = 1e-8
    x_fact = x_train[y_train == 0]
    x_hall = x_train[y_train == 1]

    if x_train.shape[1] > max_dim:
        # Uniform feature subsampling keeps runtime manageable on high-dimensional layers.
        keep = np.linspace(0, x_train.shape[1] - 1, max_dim, dtype=np.int64)
        x_train = x_train[:, keep]
        x_test = x_test[:, keep]
        x_fact = x_fact[:, keep]
        x_hall = x_hall[:, keep]

    mu_fact = x_fact.mean(axis=0)
    mu_hall = x_hall.mean(axis=0)

    cov_fact = LedoitWolf().fit(x_fact).covariance_
    cov_hall = LedoitWolf().fit(x_hall).covariance_
    inv_fact = np.linalg.pinv(cov_fact)
    inv_hall = np.linalg.pinv(cov_hall)

    m_fact = np.sqrt(np.einsum("bi,ij,bj->b", x_test - mu_fact, inv_fact, x_test - mu_fact))
    m_hall = np.sqrt(np.einsum("bi,ij,bj->b", x_test - mu_hall, inv_hall, x_test - mu_hall))
    return m_fact / (m_hall + eps)


def evaluate_binary_layer(
    layer_states: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    test_size: float = 0.3,
    n_bootstrap: int = 1000,
    compute_mahalanobis: bool = True,
    mahalanobis_max_dim: int = 256,
) -> dict[str, Any]:
    train_idx, test_idx = stratified_train_test_split(labels, test_size=test_size, seed=seed)

    x_train = layer_states[train_idx]
    y_train = labels[train_idx]
    x_test = layer_states[test_idx]
    y_test = labels[test_idx]

    centroid_scores = centroid_scores_only(x_train, y_train, x_test)

    centroid_auc, centroid_direction = _oriented_auc(y_test, centroid_scores)

    c_lo, c_hi = bootstrap_auc_ci(y_test, centroid_scores, n_bootstrap=n_bootstrap, seed=seed)

    if compute_mahalanobis:
        try:
            maha_scores = mahalanobis_scores_only(
                x_train,
                y_train,
                x_test,
                max_dim=mahalanobis_max_dim,
            )
            maha_auc, maha_direction = _oriented_auc(y_test, maha_scores)
            m_lo, m_hi = bootstrap_auc_ci(y_test, maha_scores, n_bootstrap=n_bootstrap, seed=seed)
        except Exception:
            maha_auc = float("nan")
            maha_direction = "undefined"
            m_lo, m_hi = float("nan"), float("nan")
    else:
        maha_auc = float("nan")
        maha_direction = "skipped"
        m_lo, m_hi = float("nan"), float("nan")

    mu_fact = x_train[y_train == 0].mean(axis=0)
    mu_hall = x_train[y_train == 1].mean(axis=0)
    var_fact = float(np.mean(np.linalg.norm(x_train[y_train == 0] - mu_fact, axis=1) ** 2))
    var_hall = float(np.mean(np.linalg.norm(x_train[y_train == 1] - mu_hall, axis=1) ** 2))
    rho_var = float(var_fact / (var_hall + 1e-8))

    return {
        "centroid_auroc": float(centroid_auc),
        "centroid_ci": [float(c_lo), float(c_hi)],
        "centroid_direction": centroid_direction,
        "mahalanobis_auroc": float(maha_auc),
        "mahalanobis_ci": [float(m_lo), float(m_hi)],
        "mahalanobis_direction": maha_direction,
        "mahalanobis_max_dim": int(mahalanobis_max_dim),
        "variance_ratio": float(rho_var),
        "basin_separation": float(np.linalg.norm(mu_fact - mu_hall)),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "test_indices": test_idx.tolist(),
    }


def choose_best_layer(layer_metrics: dict[int, dict[str, Any]], metric_key: str = "centroid_auroc") -> int:
    best_layer = None
    best_val = -1.0

    for layer, metrics in layer_metrics.items():
        val = float(metrics.get(metric_key, float("nan")))
        if math.isnan(val):
            continue
        if val > best_val:
            best_val = val
            best_layer = layer

    if best_layer is None:
        raise ValueError("No valid layers found while choosing best layer")
    return int(best_layer)


def load_teacher_forced_hidden_states(npz_path: Path) -> tuple[dict[int, np.ndarray], np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    labels = np.asarray(data["labels"]).astype(np.int64)

    layer_keys = sorted(
        [k for k in data.files if k.startswith("layer_")],
        key=lambda k: int(k.split("_")[1]),
    )
    if not layer_keys:
        raise ValueError(f"No layer_* arrays found in {npz_path}")

    layers: dict[int, np.ndarray] = {}
    for key in layer_keys:
        layer = int(key.split("_")[1])
        layers[layer] = np.asarray(data[key], dtype=np.float32)

    return layers, labels


def load_autoregressive_last_token_hidden_states(npz_path: Path) -> tuple[dict[int, np.ndarray], np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    labels = np.asarray(data["labels"]).astype(np.int64)
    hidden_states = data["hidden_states"]

    per_sample: list[np.ndarray] = []
    valid_indices: list[int] = []

    for idx, sample in enumerate(hidden_states):
        arr = np.asarray(sample)
        if arr.size == 0 or arr.ndim != 3:
            continue
        # Shape expected: (generated_tokens, n_layers, hidden_dim)
        last_token = arr[-1]
        if last_token.ndim != 2:
            continue
        per_sample.append(last_token.astype(np.float32))
        valid_indices.append(idx)

    if not per_sample:
        raise ValueError(f"No valid autoregressive hidden states found in {npz_path}")

    stacked = np.stack(per_sample, axis=0)  # (n_samples, n_layers, hidden_dim)
    labels_valid = labels[np.asarray(valid_indices, dtype=np.int64)]

    layers: dict[int, np.ndarray] = {}
    for layer in range(stacked.shape[1]):
        layers[layer] = stacked[:, layer, :]

    return layers, labels_valid


def sanitize_json_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_json_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_json_obj(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_json_obj(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = sanitize_json_obj(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_payload, f, indent=2)
