"""Shared inference helpers used by the Streamlit app, the FastAPI service and the monitoring script."""
from functools import lru_cache
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from airline_pipeline import clean as airline_clean, engineer as airline_engineer  # noqa: E402
from taxi_pipeline import add_features as taxi_features  # noqa: E402


@lru_cache
def airline_bundle():
    return joblib.load(ROOT / "models" / "airline_model.joblib")


@lru_cache
def taxi_bundle():
    return joblib.load(ROOT / "models" / "taxi_model.joblib")


def predict_airline(records: list[dict]) -> list[dict]:
    """records use the original dataset column names (e.g. 'Inflight wifi service')."""
    b = airline_bundle()
    df = pd.DataFrame(records)
    df, _ = airline_clean(df, fit_stats=b["clean_stats"])
    df = airline_engineer(df)
    proba = b["model"].predict_proba(df[b["features"]])[:, 1]
    return [{"satisfied_probability": round(float(p), 4),
             "prediction": "satisfied" if p >= 0.5 else "neutral or dissatisfied"} for p in proba]


def predict_taxi(records: list[dict]) -> list[dict]:
    """records: pickup_datetime, pickup/dropoff lat/lon, passenger_count, vendor_id, store_and_fwd_flag."""
    b = taxi_bundle()
    df = pd.DataFrame(records)
    df["store_and_fwd_flag"] = df.get("store_and_fwd_flag", "N")
    df = taxi_features(df, b["kmeans"])
    secs = np.expm1(b["model"].predict(df[b["features"]]))
    return [{"trip_duration_seconds": round(float(s), 1), "trip_duration_minutes": round(float(s) / 60, 2),
             "distance_km": round(float(d), 2)} for s, d in zip(secs, df["haversine_km"])]
