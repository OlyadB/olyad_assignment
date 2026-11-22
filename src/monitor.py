"""
Post-deployment monitoring simulation.

1. Replays "production" traffic through the FastAPI service (in-process TestClient):
   - airline: 500 passengers from the held-out Kaggle test file
   - taxi:    500 trips from June 2016 only (latest month -> realistic "new data")
2. Measures response latency (p50 / p95 / p99) per endpoint.
3. Scores live performance against the known true labels (as delayed feedback would).
4. Computes data drift with the Population Stability Index (PSI) between the
   training reference sample stored with each model and the logged production inputs.
   PSI < 0.1 = stable, 0.1-0.25 = moderate drift, > 0.25 = significant drift -> retrain.

Run:  python src/monitor.py   (writes reports/monitoring.json and reports/figures/monitoring_*.png)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from api.main import app, LOG  # noqa: E402
from inference import airline_bundle, taxi_bundle  # noqa: E402
from airline_pipeline import clean, engineer  # noqa: E402
from taxi_pipeline import add_features  # noqa: E402

FIG = ROOT / "reports" / "figures"
SEED = 7


def psi(ref, cur, bins=10):
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.unique(np.concatenate([edges, [ref.min() - 1, ref.max() + 1]]))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


def main():
    if LOG.exists():
        LOG.unlink()
    client = TestClient(app)
    out = {}

    # ---------------- airline traffic
    air = pd.read_csv(ROOT / "data" / "airline" / "test.csv").drop(columns=["Unnamed: 0", "id"]).sample(500, random_state=SEED)
    y_true, y_pred = [], []
    for _, r in air.iterrows():
        rec = r.drop("satisfaction").to_dict()
        if pd.isna(rec["Arrival Delay in Minutes"]):
            rec["Arrival Delay in Minutes"] = None
        res = client.post("/predict/airline", json=rec).json()
        y_true.append(int(r["satisfaction"] == "satisfied")); y_pred.append(int(res["prediction"] == "satisfied"))
    out["airline_live"] = {"accuracy": round(accuracy_score(y_true, y_pred), 4), "f1": round(f1_score(y_true, y_pred), 4)}

    # ---------------- taxi traffic (June 2016 only)
    taxi = pd.read_csv(ROOT / "data" / "taxi" / "NYC.csv")
    june = taxi[(taxi.pickup_datetime >= "2016-06-01") & taxi.trip_duration.between(60, 14400)
                & taxi.pickup_latitude.between(40.49, 40.92) & taxi.dropoff_latitude.between(40.49, 40.92)
                & taxi.pickup_longitude.between(-74.27, -73.68) & taxi.dropoff_longitude.between(-74.27, -73.68)
                & taxi.passenger_count.between(1, 6)].sample(500, random_state=SEED)
    act, pred = [], []
    for _, r in june.iterrows():
        rec = {k: r[k] for k in ["pickup_datetime", "pickup_latitude", "pickup_longitude", "dropoff_latitude",
                                 "dropoff_longitude", "passenger_count", "vendor_id", "store_and_fwd_flag"]}
        rec = {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}
        res = client.post("/predict/taxi", json=rec).json()
        act.append(r["trip_duration"]); pred.append(res["trip_duration_seconds"])
    act, pred = np.array(act), np.array(pred)
    out["taxi_live"] = {"rmsle": round(float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(act)) ** 2))), 4),
                        "mae_seconds": round(float(np.mean(np.abs(pred - act))), 1)}

    # ---------------- latency from the request log
    logs = pd.DataFrame([json.loads(l) for l in LOG.read_text().splitlines()])
    out["latency_ms"] = {ep: {q: round(float(np.percentile(g.latency_ms, int(q[1:]))), 2) for q in ["p50", "p95", "p99"]}
                         for ep, g in logs.groupby("endpoint")}
    out["monitoring_endpoint"] = client.get("/monitoring").json()

    # ---------------- drift (PSI) on key features
    ab = airline_bundle()
    cur_air = engineer(clean(air.drop(columns="satisfaction"), fit_stats=ab["clean_stats"])[0])
    air_feats = ["Age", "Flight Distance", "Departure Delay in Minutes", "service_mean", "Online boarding", "Inflight wifi service"]
    tb = taxi_bundle()
    cur_taxi = add_features(june, tb["kmeans"])
    taxi_feats = ["haversine_km", "hour", "weekday", "month", "pickup_latitude", "passenger_count"]
    out["psi_airline"] = {f: round(psi(ab["reference"][f], cur_air[f]), 4) for f in air_feats}
    out["psi_taxi"] = {f: round(psi(tb["reference"][f], cur_taxi[f]), 4) for f in taxi_feats}
    flag = lambda v: "stable" if v < 0.1 else ("moderate" if v < 0.25 else "significant — retrain")
    out["psi_status"] = {k: flag(v) for d in (out["psi_airline"], out["psi_taxi"]) for k, v in d.items()}

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ep, g in logs.groupby("endpoint"):
        axes[0].hist(g.latency_ms.clip(upper=logs.latency_ms.quantile(0.99)), bins=40, alpha=0.6, label=ep)
    axes[0].set_xlabel("latency (ms, clipped at p99)"); axes[0].set_title("API response latency (500 requests each)"); axes[0].legend()
    for ax, d, title in [(axes[1], out["psi_airline"], "Airline drift (PSI)"), (axes[2], out["psi_taxi"], "Taxi drift, June vs train (PSI)")]:
        s = pd.Series(d).sort_values()
        ax.barh(s.index, s.values, color=["#2b6cb0" if v < 0.1 else "#dd6b20" if v < 0.25 else "#c53030" for v in s])
        ax.axvline(0.1, ls="--", color="grey"); ax.axvline(0.25, ls="--", color="#c53030"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(FIG / "monitoring_latency_drift.png", dpi=150); plt.close(fig)

    (ROOT / "reports" / "monitoring.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
