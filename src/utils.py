"""Data loading, evaluation metrics, and visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

AZURE_BLOB_BASE: Final[str] = "https://azuremlsampleexperiments.blob.core.windows.net/datasets"

DATASET_FILES: dict[str, str] = {
    "telemetry": f"{AZURE_BLOB_BASE}/PdM_telemetry.csv",
    "errors": f"{AZURE_BLOB_BASE}/PdM_errors.csv",
    "maint": f"{AZURE_BLOB_BASE}/PdM_maint.csv",
    "failures": f"{AZURE_BLOB_BASE}/PdM_failures.csv",
    "machines": f"{AZURE_BLOB_BASE}/PdM_machines.csv",
}


def download_datasets(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Download all 5 Azure PdM CSVs, caching locally."""
    data_dir.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, pd.DataFrame] = {}

    for name, url in DATASET_FILES.items():
        filepath: Path = data_dir / f"PdM_{name}.csv"
        if not filepath.exists():
            print(f"Downloading {name}...")
            df: pd.DataFrame = pd.read_csv(url, parse_dates=["datetime"])
            df.to_csv(filepath, index=False)
        else:
            df = pd.read_csv(filepath, parse_dates=["datetime"])
        datasets[name] = df

    return datasets


def bootstrap_ci(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    metric_fn: Any,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (point_estimate, ci_lower, ci_upper)."""
    rng: np.random.Generator = np.random.default_rng(seed)
    point_estimate: float = float(metric_fn(y_true, y_pred))
    scores: list[float] = []

    n: int = len(y_true)
    for _ in range(n_bootstrap):
        idx: npt.NDArray[np.float64] = rng.integers(0, n, size=n)
        scores.append(float(metric_fn(y_true[idx], y_pred[idx])))

    alpha: float = (1.0 - confidence) / 2.0
    ci_lower: float = float(np.percentile(scores, 100 * alpha))
    ci_upper: float = float(np.percentile(scores, 100 * (1 - alpha)))

    return point_estimate, ci_lower, ci_upper


def event_level_recall(
    y_true: npt.NDArray[np.float64],
    y_pred_proba: npt.NDArray[np.float64],
    failure_events: pd.DataFrame,
    telemetry_index: pd.DataFrame,
    threshold: float,
) -> float:
    """Fraction of failure events caught (any prediction >= threshold in [F-24h, F) window)."""
    caught: int = 0
    total: int = len(failure_events)

    predictions_df: pd.DataFrame = telemetry_index.copy()
    predictions_df["pred_proba"] = y_pred_proba

    for _, event in failure_events.iterrows():
        window_start: pd.Timestamp = event["datetime"] - pd.Timedelta(hours=24)
        # [fail_time - 24h, fail_time) -- excludes failure timestamp
        window_preds: pd.DataFrame = predictions_df[
            (predictions_df["machineID"] == event["machineID"])
            & (predictions_df["datetime"] >= window_start)
            & (predictions_df["datetime"] < event["datetime"])
        ]
        if (window_preds["pred_proba"] >= threshold).any():
            caught += 1

    return caught / total if total > 0 else 0.0


def compute_pearson_residuals(
    y_true: npt.NDArray[np.float64],
    y_pred_proba: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Pearson residuals for binary classification: (y - p) / sqrt(p * (1 - p))."""
    p: npt.NDArray[np.float64] = np.clip(y_pred_proba, 1e-8, 1 - 1e-8)
    return (y_true - p) / np.sqrt(p * (1 - p))


def compute_all_metrics(
    y_true: npt.NDArray[np.float64],
    y_pred_proba: npt.NDArray[np.float64],
    threshold: float,
) -> dict[str, float]:
    """All evaluation metrics at a given threshold."""
    y_pred: npt.NDArray[np.float64] = (y_pred_proba >= threshold).astype(int)

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_pred_proba)),
        "threshold": threshold,
    }


def set_plot_style() -> None:
    """Consistent matplotlib style for all plots."""
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_calibration_curve(
    y_true: npt.NDArray[np.float64],
    y_pred_proba: npt.NDArray[np.float64],
    n_bins: int = 10,
    title: str = "Calibration Curve",
) -> tuple[plt.Figure, plt.Axes]:
    """Reliability diagram with Brier score annotation."""
    fraction_of_positives: npt.NDArray[np.float64]
    mean_predicted_value: npt.NDArray[np.float64]
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins
    )

    brier: float = float(brier_score_loss(y_true, y_pred_proba))

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(mean_predicted_value, fraction_of_positives, "o-",
            label=f"Model (Brier={brier:.4f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()

    return fig, ax
