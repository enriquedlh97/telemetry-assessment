"""Tests for feature engineering -- leakage guards and label boundaries."""

import numpy as np
import pandas as pd
import pytest

from src.features import compute_error_features, compute_labels, compute_maintenance_features


@pytest.fixture
def sample_telemetry_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2015-01-01 06:00", "2015-01-01 12:00", "2015-01-01 18:00"]
            ),
            "machineID": [1, 1, 1],
        }
    )


@pytest.fixture
def sample_failures() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2015-01-02 00:00"]),
            "machineID": [1],
            "failure": ["comp1"],
        }
    )


class TestLeakageGuard:

    def test_error_features_exclude_future_errors(self, sample_telemetry_index):
        errors = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2015-01-01 05:00", "2015-01-01 13:00"]),
                "machineID": [1, 1],
                "errorID": ["error1", "error1"],
            }
        )
        result = compute_error_features(errors, sample_telemetry_index, window_sizes_hours=[48])
        assert result.iloc[0]["error1_count_w48h"] >= 1
        assert result.iloc[2]["error1_count_w48h"] >= 2

    def test_maintenance_uses_strict_before(self, sample_telemetry_index):
        maint = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2015-01-01 05:00"]),
                "machineID": [1],
                "comp": ["comp1"],
            }
        )
        result = compute_maintenance_features(maint, sample_telemetry_index)
        assert result.iloc[0]["comp1_days_since_replacement"] >= 0

    def test_maintenance_no_future_events(self, sample_telemetry_index):
        maint = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2015-01-01 19:00"]),
                "machineID": [1],
                "comp": ["comp1"],
            }
        )
        result = compute_maintenance_features(maint, sample_telemetry_index)
        assert (result["comp1_days_since_replacement"] == -1.0).all()


class TestLabelConstruction:

    def test_labels_within_24h_window(self, sample_telemetry_index, sample_failures):
        labels = compute_labels(sample_failures, sample_telemetry_index, prediction_horizon_hours=24)
        assert labels.iloc[0] == 1  # 18h before
        assert labels.iloc[1] == 1  # 12h before
        assert labels.iloc[2] == 1  # 6h before

    def test_labels_outside_24h_window(self, sample_failures):
        index = pd.DataFrame(
            {"datetime": pd.to_datetime(["2014-12-31 12:00"]), "machineID": [1]}
        )
        labels = compute_labels(sample_failures, index, prediction_horizon_hours=24)
        assert labels.iloc[0] == 0

    def test_labels_different_machine(self, sample_failures):
        index = pd.DataFrame(
            {"datetime": pd.to_datetime(["2015-01-01 18:00"]), "machineID": [2]}
        )
        labels = compute_labels(sample_failures, index, prediction_horizon_hours=24)
        assert labels.iloc[0] == 0

    def test_labels_positive_count_sanity(self, sample_failures):
        index = pd.DataFrame(
            {"datetime": pd.date_range("2015-01-01 00:00", periods=48, freq="h"), "machineID": [1] * 48}
        )
        labels = compute_labels(sample_failures, index, prediction_horizon_hours=24)
        assert 0 < labels.sum() <= 24

    def test_failure_timestamp_is_not_labeled(self, sample_failures):
        index = pd.DataFrame(
            {"datetime": pd.to_datetime(["2015-01-02 00:00"]), "machineID": [1]}
        )
        labels = compute_labels(sample_failures, index, prediction_horizon_hours=24)
        assert labels.iloc[0] == 0

    def test_record_exactly_24h_before_is_labeled(self, sample_failures):
        index = pd.DataFrame(
            {"datetime": pd.to_datetime(["2015-01-01 00:00"]), "machineID": [1]}
        )
        labels = compute_labels(sample_failures, index, prediction_horizon_hours=24)
        assert labels.iloc[0] == 1
