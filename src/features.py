"""Feature engineering for predictive maintenance."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

SENSORS: list[str] = ["volt", "rotate", "pressure", "vibration"]


def compute_telemetry_features(
    telemetry: pd.DataFrame,
    window_sizes: list[int],
) -> pd.DataFrame:
    """Rolling statistics per sensor per window. Window sizes from ACF/PACF analysis."""
    df: pd.DataFrame = telemetry.sort_values(["machineID", "datetime"]).copy()
    feature_cols: list[str] = []

    for sensor in SENSORS:
        for window in window_sizes:
            prefix: str = f"{sensor}_w{window}"
            grouped_rolling: pd.core.window.Rolling = df.groupby("machineID")[sensor].rolling(
                window=window, min_periods=1
            )
            df[f"{prefix}_mean"] = grouped_rolling.mean().droplevel(0)
            df[f"{prefix}_std"] = grouped_rolling.std().droplevel(0)
            df[f"{prefix}_min"] = grouped_rolling.min().droplevel(0)
            df[f"{prefix}_max"] = grouped_rolling.max().droplevel(0)
            feature_cols.extend([
                f"{prefix}_mean", f"{prefix}_std", f"{prefix}_min", f"{prefix}_max",
            ])

        for window in window_sizes:
            prefix = f"{sensor}_w{window}"
            df[f"{prefix}_range"] = df[f"{prefix}_max"] - df[f"{prefix}_min"]
            feature_cols.append(f"{prefix}_range")

        df[f"{sensor}_diff"] = df.groupby("machineID")[sensor].diff()
        feature_cols.append(f"{sensor}_diff")

    return df[["datetime", "machineID"] + feature_cols]


def compute_cross_sensor_features(
    telemetry: pd.DataFrame,
    window_size: int = 24,
) -> pd.DataFrame:
    """Pairwise rolling correlations between sensors.

    Sensors are uncorrelated during normal operation but become correlated
    before failures -- this captures that shift.
    """
    df: pd.DataFrame = telemetry.sort_values(["machineID", "datetime"]).copy()
    corr_cols: list[str] = []

    for i in range(len(SENSORS)):
        for j in range(i + 1, len(SENSORS)):
            col_name: str = f"corr_{SENSORS[i]}_{SENSORS[j]}_w{window_size}"
            df[col_name] = df.groupby("machineID").apply(
                lambda g: g[SENSORS[i]].rolling(window=window_size, min_periods=4).corr(
                    g[SENSORS[j]]
                ),
                include_groups=False,
            ).droplevel(0)
            corr_cols.append(col_name)

    return df[["datetime", "machineID"] + corr_cols]


def compute_error_features(
    errors: pd.DataFrame,
    telemetry_index: pd.DataFrame,
    window_sizes_hours: list[int],
) -> pd.DataFrame:
    """Error count and time-since features per error type.

    Leakage guard: only errors up to time T are used (backward-looking).
    Uses np.searchsorted for efficient counting.
    """
    error_types: list[str] = sorted(errors["errorID"].unique().tolist())
    base: pd.DataFrame = telemetry_index[["datetime", "machineID"]].copy()
    base = base.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    feature_cols: list[str] = []

    for et in error_types:
        et_errors: pd.DataFrame = errors[errors["errorID"] == et].copy()

        for machine_id in base["machineID"].unique():
            machine_mask: pd.Series = base["machineID"] == machine_id
            machine_times: pd.Series = base.loc[machine_mask, "datetime"]
            machine_errors: npt.NDArray[np.float64] = et_errors[et_errors["machineID"] == machine_id][
                "datetime"
            ].sort_values().values

            for window in window_sizes_hours:
                col_name: str = f"{et}_count_w{window}h"
                if col_name not in base.columns:
                    base[col_name] = 0

                for idx in machine_times.index:
                    t: pd.Timestamp = base.at[idx, "datetime"]
                    cutoff: pd.Timestamp = t - pd.Timedelta(hours=window)
                    count: int = int(
                        np.searchsorted(machine_errors, t, side="right")
                        - np.searchsorted(machine_errors, cutoff, side="right")
                    )
                    base.at[idx, col_name] = count

            col_hours: str = f"{et}_hours_since"
            if col_hours not in base.columns:
                base[col_hours] = -1.0

            for idx in machine_times.index:
                t = base.at[idx, "datetime"]
                past: npt.NDArray[np.float64] = machine_errors[machine_errors <= t]
                if len(past) > 0:
                    delta: float = (t - pd.Timestamp(past[-1])).total_seconds() / 3600.0
                    base.at[idx, col_hours] = delta
                else:
                    base.at[idx, col_hours] = -1.0

        for window in window_sizes_hours:
            col_name = f"{et}_count_w{window}h"
            if col_name not in feature_cols:
                feature_cols.append(col_name)
        if f"{et}_hours_since" not in feature_cols:
            feature_cols.append(f"{et}_hours_since")

    return base[["datetime", "machineID"] + feature_cols]


def compute_maintenance_features(
    maint: pd.DataFrame,
    telemetry_index: pd.DataFrame,
) -> pd.DataFrame:
    """Days since last component replacement. Uses merge_asof per machine.

    Leakage guard: allow_exact_matches=False ensures only maintenance
    strictly BEFORE the prediction time is visible.
    """
    components: list[str] = sorted(maint["comp"].unique().tolist())
    base: pd.DataFrame = telemetry_index[["datetime", "machineID"]].copy()
    base = base.sort_values(["machineID", "datetime"])
    feature_cols: list[str] = []

    for comp in components:
        col_name: str = f"{comp}_days_since_replacement"
        comp_maint: pd.DataFrame = (
            maint[maint["comp"] == comp][["datetime", "machineID"]]
            .sort_values(["machineID", "datetime"])
            .rename(columns={"datetime": "maint_datetime"})
        )

        merged_parts: list[pd.DataFrame] = []
        for machine_id in base["machineID"].unique():
            machine_base: pd.DataFrame = base[base["machineID"] == machine_id].copy()
            machine_maint: pd.DataFrame = comp_maint[comp_maint["machineID"] == machine_id]

            if machine_maint.empty:
                machine_base[col_name] = -1.0
            else:
                merged: pd.DataFrame = pd.merge_asof(
                    machine_base.sort_values("datetime"),
                    machine_maint.sort_values("maint_datetime"),
                    left_on="datetime",
                    right_on="maint_datetime",
                    by="machineID",
                    direction="backward",
                    allow_exact_matches=False,
                )
                merged[col_name] = (
                    (merged["datetime"] - merged["maint_datetime"]).dt.total_seconds() / 86400.0
                )
                merged[col_name] = merged[col_name].fillna(-1.0)
                machine_base = merged.drop(columns=["maint_datetime"])

            merged_parts.append(machine_base)

        result: pd.DataFrame = pd.concat(merged_parts)
        base = base.merge(
            result[["datetime", "machineID", col_name]],
            on=["datetime", "machineID"],
            how="left",
        )
        feature_cols.append(col_name)

    return base[["datetime", "machineID"] + feature_cols]


def compute_machine_features(
    telemetry_index: pd.DataFrame,
    machines: pd.DataFrame,
) -> pd.DataFrame:
    """Machine metadata (model one-hot, age) and hour of day (cyclical encoding)."""
    base: pd.DataFrame = telemetry_index[["datetime", "machineID"]].merge(
        machines, on="machineID", how="left"
    )

    model_dummies: pd.DataFrame = pd.get_dummies(base["model"], prefix="model", dtype=int)
    base = pd.concat([base, model_dummies], axis=1).drop(columns=["model"])

    base["hour"] = base["datetime"].dt.hour
    base["hour_sin"] = np.sin(2 * np.pi * base["hour"] / 24)
    base["hour_cos"] = np.cos(2 * np.pi * base["hour"] / 24)

    return base


def compute_labels(
    failures: pd.DataFrame,
    telemetry_index: pd.DataFrame,
    prediction_horizon_hours: int = 24,
) -> pd.Series:
    """Binary labels: 1 if any failure in [T - horizon, T) for that machine.

    Window is [fail_time - horizon, fail_time): the record exactly at the
    failure timestamp is EXCLUDED (at that point, the failure is already
    happening -- not a forecast).
    """
    horizon: pd.Timedelta = pd.Timedelta(hours=prediction_horizon_hours)
    labels: pd.Series = pd.Series(0, index=telemetry_index.index, name="failure_within_24h")

    for machine_id in telemetry_index["machineID"].unique():
        machine_mask: pd.Series = telemetry_index["machineID"] == machine_id
        machine_failures: pd.DataFrame = failures[failures["machineID"] == machine_id]

        for _, fail in machine_failures.iterrows():
            window_mask: pd.Series = (
                machine_mask
                & (telemetry_index["datetime"] >= fail["datetime"] - horizon)
                & (telemetry_index["datetime"] < fail["datetime"])
            )
            labels.loc[window_mask] = 1

    return labels


def build_feature_matrix(
    telemetry: pd.DataFrame,
    errors: pd.DataFrame,
    maint: pd.DataFrame,
    failures: pd.DataFrame,
    machines: pd.DataFrame,
    telemetry_window_sizes: list[int],
    error_window_sizes_hours: list[int],
    cross_sensor_window: int = 24,
    prediction_horizon_hours: int = 24,
) -> tuple[pd.DataFrame, pd.Series]:
    """Orchestrate all feature computation and label construction."""
    telemetry_index: pd.DataFrame = telemetry[["datetime", "machineID"]].copy()

    print("Computing telemetry rolling features...")
    tel_features: pd.DataFrame = compute_telemetry_features(telemetry, telemetry_window_sizes)

    print("Computing cross-sensor correlation features...")
    cross_features: pd.DataFrame = compute_cross_sensor_features(telemetry, cross_sensor_window)

    print("Computing error features...")
    err_features: pd.DataFrame = compute_error_features(errors, telemetry_index, error_window_sizes_hours)

    print("Computing maintenance features...")
    maint_features: pd.DataFrame = compute_maintenance_features(maint, telemetry_index)

    print("Computing machine + temporal features...")
    machine_features: pd.DataFrame = compute_machine_features(telemetry_index, machines)

    print("Constructing labels...")
    labels: pd.Series = compute_labels(failures, telemetry_index, prediction_horizon_hours)

    print("Merging feature matrix...")
    features: pd.DataFrame = tel_features
    for df in [cross_features, err_features, maint_features, machine_features]:
        features = features.merge(df, on=["datetime", "machineID"], how="left")

    print(f"Feature matrix: {features.shape[0]:,} rows x {features.shape[1]} columns")
    print(f"Labels: {int(labels.sum()):,} positive / {len(labels):,} total ({labels.mean()*100:.2f}%)")

    return features, labels


def compute_vif_screening(
    features: pd.DataFrame,
    threshold: float = 10.0,
) -> list[str]:
    """Iteratively drop features with VIF > threshold for LR interpretability."""
    non_zero_var: list[str] = [
        col for col in features.columns if features[col].std() > 1e-10
    ]
    remaining: list[str] = list(non_zero_var)
    dropped: bool = True

    while dropped:
        dropped = False
        X: npt.NDArray[np.float64] = features[remaining].dropna().values

        if X.shape[1] <= 1:
            break

        vifs: list[float] = [
            float(variance_inflation_factor(X, i)) for i in range(X.shape[1])
        ]

        max_vif: float = max(vifs)
        if max_vif > threshold:
            drop_idx: int = vifs.index(max_vif)
            remaining.pop(drop_idx)
            dropped = True

    return remaining
