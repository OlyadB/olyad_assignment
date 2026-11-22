"""
NYC Taxi Trip Duration Prediction  (regression)
Author: Olyad Bizuneh

Runs the full ML process end to end:
  EDA -> cleaning -> feature engineering (incl. K-Means location clusters) ->
  train/val/test split -> baseline models -> hyperparameter tuning ->
  final test evaluation -> saved model.

Target is log1p(trip_duration); the headline metric is RMSLE (the Kaggle competition metric),
reported together with RMSE / MAE in seconds and R².

Usage:  python src/taxi_pipeline.py
Outputs: reports/figures/taxi_*.png, reports/taxi_metrics.json, models/taxi_model.joblib
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor, early_stopping
from scipy.stats import randint, uniform, loguniform
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "taxi" / "NYC.csv"
FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
FIG.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
SEED = 42
sns.set_theme(style="whitegrid", palette="muted")

NYC_BOX = {"lat": (40.49, 40.92), "lon": (-74.27, -73.68)}
JFK = (40.6413, -73.7781)
LGA = (40.7769, -73.8740)
FEATURES = ["vendor_id", "passenger_count", "store_and_fwd", "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude", "hour", "minute_of_day", "weekday", "month",
            "day_of_year", "is_weekend", "is_rush_hour", "haversine_km", "manhattan_km", "bearing",
            "delta_lat", "delta_lon", "center_lat", "center_lon", "pickup_jfk", "dropoff_jfk",
            "pickup_lga", "dropoff_lga", "pickup_cluster", "dropoff_cluster"]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.degrees(np.arctan2(y, x))


# ---------------------------------------------------------------- feature engineering (shared with the app/API)
def add_features(df, kmeans=None):
    df = df.copy()
    t = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = t.dt.hour
    df["minute_of_day"] = t.dt.hour * 60 + t.dt.minute
    df["weekday"] = t.dt.weekday
    df["month"] = t.dt.month
    df["day_of_year"] = t.dt.dayofyear
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_rush_hour"] = (df["weekday"].lt(5) & (df["hour"].between(7, 9) | df["hour"].between(16, 19))).astype(int)
    plat, plon, dlat, dlon = df["pickup_latitude"], df["pickup_longitude"], df["dropoff_latitude"], df["dropoff_longitude"]
    df["haversine_km"] = haversine(plat, plon, dlat, dlon)
    df["manhattan_km"] = haversine(plat, plon, plat, dlon) + haversine(plat, dlon, dlat, dlon)
    df["bearing"] = bearing(plat, plon, dlat, dlon)
    df["delta_lat"], df["delta_lon"] = dlat - plat, dlon - plon
    df["center_lat"], df["center_lon"] = (plat + dlat) / 2, (plon + dlon) / 2
    df["pickup_jfk"] = (haversine(plat, plon, *JFK) < 2).astype(int)
    df["dropoff_jfk"] = (haversine(dlat, dlon, *JFK) < 2).astype(int)
    df["pickup_lga"] = (haversine(plat, plon, *LGA) < 1.5).astype(int)
    df["dropoff_lga"] = (haversine(dlat, dlon, *LGA) < 1.5).astype(int)
    df["store_and_fwd"] = (df["store_and_fwd_flag"] == "Y").astype(int)
    if kmeans is not None:
        df["pickup_cluster"] = kmeans.predict(df[["pickup_latitude", "pickup_longitude"]].to_numpy())
        df["dropoff_cluster"] = kmeans.predict(df[["dropoff_latitude", "dropoff_longitude"]].to_numpy())
    return df


def reg_metrics(y_log, pred_log):
    y, p = np.expm1(y_log), np.expm1(pred_log)
    return {"rmsle": float(np.sqrt(mean_squared_error(y_log, pred_log))),
            "r2_log": float(r2_score(y_log, pred_log)),
            "rmse_seconds": float(np.sqrt(mean_squared_error(y, p))),
            "mae_seconds": float(mean_absolute_error(y, p)),
            "median_ae_seconds": float(np.median(np.abs(y - p))),
            "r2_seconds": float(r2_score(y, p))}


def r4(d):
    return {k: round(float(v), 4) for k, v in d.items()}


# ---------------------------------------------------------------- EDA
def eda(df, report):
    report["rows"], report["cols"] = int(df.shape[0]), int(df.shape[1])
    report["missing"] = int(df.isna().sum().sum())
    report["duplicate_ids"] = int(df["id"].duplicated().sum())
    report["duration_seconds_describe"] = df["trip_duration"].describe().round(1).to_dict()
    report["trips_over_24h"] = int((df["trip_duration"] > 86400).sum())
    report["trips_under_60s"] = int((df["trip_duration"] < 60).sum())
    report["passenger_count_counts"] = df["passenger_count"].value_counts().sort_index().to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.histplot(df.loc[df.trip_duration < 7200, "trip_duration"] / 60, bins=80, ax=axes[0], color="#9aa5b1")
    axes[0].set_title("Trip duration (raw, < 2 h shown) — right-skewed"); axes[0].set_xlabel("minutes")
    sns.histplot(np.log1p(df["trip_duration"]), bins=100, ax=axes[1], color="#2b6cb0")
    axes[1].set_title("log1p(trip duration) — near-normal, used as target"); axes[1].set_xlabel("log1p(seconds)")
    fig.tight_layout(); fig.savefig(FIG / "taxi_target_distribution.png", dpi=150); plt.close(fig)

    t = pd.to_datetime(df["pickup_datetime"])
    piv = df.assign(h=t.dt.hour, wd=t.dt.day_name().str[:3]).pivot_table(index="wd", columns="h", values="trip_duration", aggfunc="median") / 60
    piv = piv.loc[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(piv, cmap="Blues", ax=ax, cbar_kws={"label": "median minutes"})
    ax.set_title("Median trip duration by weekday and pickup hour"); ax.set_xlabel("hour of day"); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(FIG / "taxi_hour_weekday_heatmap.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    t.dt.date.value_counts().sort_index().plot(ax=ax, color="#2b6cb0")
    ax.set_title("Trips per day, Jan–Jun 2016 (note the Jan 23 blizzard dip)"); ax.set_ylabel("trips")
    fig.tight_layout(); fig.savefig(FIG / "taxi_trips_per_day.png", dpi=150); plt.close(fig)

    s = df.sample(200_000, random_state=SEED)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(s.pickup_longitude, s.pickup_latitude, s=0.3, alpha=0.3, color="#2b6cb0")
    ax.set_xlim(*NYC_BOX["lon"]); ax.set_ylim(*NYC_BOX["lat"]); ax.set_title("Pickup locations (200k sample)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    fig.tight_layout(); fig.savefig(FIG / "taxi_pickup_map.png", dpi=150); plt.close(fig)


def main():
    t0 = time.time()
    report = {"project": "NYC Taxi Trip Duration (regression)"}
    raw = pd.read_csv(DATA)
    eda(raw, report.setdefault("eda", {}))

    # ------------------------------------------------ cleaning
    df = raw.drop_duplicates(subset="id").copy()
    n0 = len(df)
    steps = {}
    inbox = (df.pickup_latitude.between(*NYC_BOX["lat"]) & df.dropoff_latitude.between(*NYC_BOX["lat"]) &
             df.pickup_longitude.between(*NYC_BOX["lon"]) & df.dropoff_longitude.between(*NYC_BOX["lon"]))
    steps["outside_nyc_bbox"] = int((~inbox).sum()); df = df[inbox]
    ok = df.trip_duration.between(60, 4 * 3600)
    steps["duration_outside_60s_4h"] = int((~ok).sum()); df = df[ok]
    ok = df.passenger_count.between(1, 6)
    steps["passenger_count_not_1_to_6"] = int((~ok).sum()); df = df[ok]
    d = haversine(df.pickup_latitude, df.pickup_longitude, df.dropoff_latitude, df.dropoff_longitude)
    ok = d > 0.05
    steps["zero_distance_trips"] = int((~ok).sum()); df = df[ok]
    # implausible average speed (> 120 km/h) = GPS or clock errors
    speed = d[ok] / (df.trip_duration / 3600)
    ok = speed < 120
    steps["speed_over_120kmh"] = int((~ok).sum()); df = df[ok]
    steps["rows_kept"] = len(df); steps["rows_removed_pct"] = round(100 * (1 - len(df) / n0), 2)
    report["cleaning"] = steps
    print("cleaning", steps)

    # ------------------------------------------------ split 70 / 15 / 15
    train, temp = train_test_split(df, test_size=0.30, random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, random_state=SEED)
    report["split"] = {"train": len(train), "validation": len(val), "test": len(test)}

    # K-Means (unsupervised) location clusters, fit on training coordinates only
    coords = np.vstack([train[["pickup_latitude", "pickup_longitude"]].to_numpy(),
                        train[["dropoff_latitude", "dropoff_longitude"]].to_numpy()])
    kmeans = MiniBatchKMeans(n_clusters=60, batch_size=10_000, random_state=SEED, n_init=3).fit(
        coords[np.random.default_rng(SEED).choice(len(coords), 500_000, replace=False)])
    train, val, test = (add_features(x, kmeans) for x in (train, val, test))
    Xtr, ytr = train[FEATURES], np.log1p(train.trip_duration)
    Xva, yva = val[FEATURES], np.log1p(val.trip_duration)
    Xte, yte = test[FEATURES], np.log1p(test.trip_duration)

    fig, ax = plt.subplots(figsize=(10, 4))
    s = train.sample(100_000, random_state=SEED)
    ax.scatter(s.haversine_km, s.trip_duration / 60, s=1, alpha=0.15, color="#2b6cb0")
    ax.set_xlim(0, 30); ax.set_ylim(0, 120); ax.set_xlabel("haversine distance (km)"); ax.set_ylabel("duration (min)")
    ax.set_title(f"Distance vs duration after cleaning (Spearman ρ = {s[['haversine_km','trip_duration']].corr('spearman').iloc[0,1]:.2f})")
    fig.tight_layout(); fig.savefig(FIG / "taxi_distance_vs_duration.png", dpi=150); plt.close(fig)

    corr = train[FEATURES].assign(log_duration=ytr).corr()["log_duration"].drop("log_duration").sort_values()
    report["feature_target_correlation"] = corr.round(3).to_dict()
    fig, ax = plt.subplots(figsize=(8, 7))
    corr.plot.barh(ax=ax, color=["#c05621" if v < 0 else "#2b6cb0" for v in corr]); ax.set_title("Pearson correlation with log duration")
    fig.tight_layout(); fig.savefig(FIG / "taxi_feature_correlation.png", dpi=150); plt.close(fig)

    # ------------------------------------------------ baselines
    sub = Xtr.sample(200_000, random_state=SEED); ysub = ytr.loc[sub.index]
    candidates = {
        "Mean baseline": None,
        "Ridge Regression": Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "Random Forest (200k sample)": RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=600, learning_rate=0.1, max_depth=8, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED, tree_method="hist"),
        "LightGBM": LGBMRegressor(n_estimators=600, learning_rate=0.1, num_leaves=127, random_state=SEED, verbose=-1),
    }
    baseline = {}
    for name, model in candidates.items():
        s0 = time.time()
        if model is None:
            ptr, pva = np.full(len(ytr), ytr.mean()), np.full(len(yva), ytr.mean())
        else:
            X_fit, y_fit = (sub, ysub) if "sample" in name else (Xtr, ytr)
            model.fit(X_fit, y_fit)
            ptr, pva = model.predict(Xtr.iloc[:200_000]), model.predict(Xva)
        tr = reg_metrics(ytr.iloc[:len(ptr)], ptr); va = reg_metrics(yva, pva)
        baseline[name] = {"train": r4(tr), "validation": r4(va), "fit_seconds": round(time.time() - s0, 1),
                          "train_val_gap_rmsle": round(va["rmsle"] - tr["rmsle"], 4)}
        print(f"{name:28s} val RMSLE={va['rmsle']:.4f} R2log={va['r2_log']:.4f} gap={va['rmsle']-tr['rmsle']:.4f}")
    report["baseline"] = baseline

    # ------------------------------------------------ tuning on a 300k sample (3-fold CV)
    tune = Xtr.sample(300_000, random_state=1); ytune = ytr.loc[tune.index]
    cv = KFold(3, shuffle=True, random_state=SEED)
    tuning = {}
    gs = GridSearchCV(candidates["Ridge Regression"], {"reg__alpha": [0.01, 0.1, 1, 10, 100]},
                      scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1).fit(tune, ytune)
    tuning["Ridge (Grid Search)"] = {"best_params": gs.best_params_, "cv_rmsle": round(-gs.best_score_, 4)}
    space = {"n_estimators": [800], "learning_rate": loguniform(0.03, 0.2), "num_leaves": randint(63, 511),
             "min_child_samples": randint(20, 200), "subsample": uniform(0.6, 0.4), "subsample_freq": [1],
             "colsample_bytree": uniform(0.5, 0.5), "reg_lambda": loguniform(1e-3, 10)}
    rs = RandomizedSearchCV(LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=2), space, n_iter=15,
                            scoring="neg_root_mean_squared_error", cv=cv, random_state=SEED, n_jobs=4).fit(tune, ytune)
    best_params = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rs.best_params_.items()}
    tuning["LightGBM (Random Search)"] = {"best_params": best_params, "cv_rmsle": round(-rs.best_score_, 4)}
    cvres = pd.DataFrame(rs.cv_results_)
    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(cvres["param_num_leaves"].astype(float), -cvres["mean_test_score"], c=cvres["param_learning_rate"].astype(float), cmap="Blues", edgecolor="k")
    plt.colorbar(sc, label="learning_rate"); ax.set_xlabel("num_leaves"); ax.set_ylabel("CV RMSLE (lower is better)")
    ax.set_title("LightGBM random search: 15 trials on 300k-row sample")
    fig.tight_layout(); fig.savefig(FIG / "taxi_random_search.png", dpi=150); plt.close(fig)

    # Refit tuned params on the FULL training set with more trees + early stopping on validation
    params = {**rs.best_params_, "n_estimators": 4000}
    params["n_jobs"] = -1
    final = LGBMRegressor(**params, random_state=SEED, verbose=-1)
    final.fit(Xtr, ytr, eval_set=[(Xtr.iloc[:200_000], ytr.iloc[:200_000]), (Xva, yva)], eval_names=["train", "validation"],
              eval_metric="rmse", callbacks=[early_stopping(100, verbose=False)])
    tuning["final_best_iteration"] = int(final.best_iteration_)
    va_best = reg_metrics(yva, final.predict(Xva)); tr_best = reg_metrics(ytr.iloc[:200_000], final.predict(Xtr.iloc[:200_000]))
    tuning["tuned_lightgbm_validation"] = r4(va_best); tuning["tuned_lightgbm_train"] = r4(tr_best)
    report["tuning"] = tuning
    print("tuned LightGBM val", r4(va_best), "iters", final.best_iteration_)

    ev = final.evals_result_
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ev["train"]["rmse"], label="train", color="#9aa5b1"); ax.plot(ev["validation"]["rmse"], label="validation", color="#2b6cb0")
    ax.set_ylim(0.15, 0.5)
    ax.set_xlabel("boosting round"); ax.set_ylabel("RMSLE"); ax.set_title(f"Learning curve: validation RMSLE {ev['validation']['rmse'][999]:.3f} at 1,000 trees, {ev['validation']['rmse'][-1]:.3f} at the end"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "taxi_learning_curve.png", dpi=150); plt.close(fig)

    comp = pd.Series({n: b["validation"]["rmsle"] for n, b in baseline.items()})
    comp["LightGBM (tuned)"] = va_best["rmsle"]
    fig, ax = plt.subplots(figsize=(9, 4))
    comp.sort_values(ascending=False).plot.barh(ax=ax, color=["#2b6cb0" if "tuned" in n else "#9aa5b1" for n in comp.sort_values(ascending=False).index])
    for i, v in enumerate(comp.sort_values(ascending=False)):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    ax.set_xlabel("validation RMSLE (lower is better)"); ax.set_title("Validation RMSLE by model")
    fig.tight_layout(); fig.savefig(FIG / "taxi_model_comparison.png", dpi=150); plt.close(fig)

    # ------------------------------------------------ test (once)
    pte = final.predict(Xte)
    report["test"] = r4(reg_metrics(yte, pte))
    print("TEST", report["test"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    idx = np.random.default_rng(SEED).choice(len(yte), 20_000, replace=False)
    axes[0].scatter(np.expm1(yte.iloc[idx]) / 60, np.expm1(pte[idx]) / 60, s=1, alpha=0.2, color="#2b6cb0")
    axes[0].plot([0, 120], [0, 120], "--", color="grey"); axes[0].set_xlim(0, 120); axes[0].set_ylim(0, 120)
    axes[0].set_xlabel("actual (min)"); axes[0].set_ylabel("predicted (min)"); axes[0].set_title("Test: predicted vs actual (20k sample)")
    sns.histplot(yte - pte, bins=100, ax=axes[1], color="#2b6cb0"); axes[1].set_xlim(-2, 2)
    axes[1].set_title("Test residuals (log scale)"); axes[1].set_xlabel("log actual − log predicted")
    fig.tight_layout(); fig.savefig(FIG / "taxi_test_pred_vs_actual.png", dpi=150); plt.close(fig)

    imp = pd.Series(final.booster_.feature_importance("gain"), index=FEATURES)
    imp = (imp / imp.sum()).sort_values(ascending=False).head(15)
    report["feature_importance_top15"] = imp.round(4).to_dict()
    fig, ax = plt.subplots(figsize=(8, 6))
    imp[::-1].plot.barh(ax=ax, color="#2b6cb0"); ax.set_title("Top 15 features (LightGBM gain share)"); ax.set_xlabel("Share of total gain")
    fig.tight_layout(); fig.savefig(FIG / "taxi_feature_importance.png", dpi=150); plt.close(fig)

    one = Xte.iloc[[0]]
    lat = []
    for _ in range(200):
        s0 = time.perf_counter(); final.predict(one); lat.append((time.perf_counter() - s0) * 1000)
    report["latency_ms_single_row"] = {"p50": round(float(np.percentile(lat, 50)), 2), "p95": round(float(np.percentile(lat, 95)), 2)}

    joblib.dump({"model": final, "kmeans": kmeans, "features": FEATURES,
                 "reference": Xtr.sample(5000, random_state=SEED)}, MODELS / "taxi_model.joblib", compress=3)
    report["runtime_seconds"] = round(time.time() - t0, 1)
    (ROOT / "reports" / "taxi_metrics.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"done in {report['runtime_seconds']}s")


if __name__ == "__main__":
    main()
