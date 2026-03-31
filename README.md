# Predictive Maintenance: 24h Failure Forecasting

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run jupyter lab
```

## Structure

```
notebooks/predictive_maintenance.ipynb   # main analysis
src/features.py                          # feature engineering (tested)
src/utils.py                             # evaluation metrics, helpers
tests/test_features.py                   # 9 tests (leakage guards, label boundaries)
data/                                    # raw CSVs (not tracked)
```
