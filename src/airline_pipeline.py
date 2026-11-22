"""
Airline Passenger Satisfaction Prediction  (binary classification)
Author: Olyad Bizuneh

Runs the full ML process end to end:
  EDA -> cleaning -> feature engineering -> train/val/test split ->
  baseline models -> hyperparameter tuning -> final test evaluation -> saved model.

Usage:  python src/airline_pipeline.py
Outputs: reports/figures/airline_*.png, reports/airline_metrics.json, models/airline_model.joblib
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
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform, loguniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "airline"
FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
FIG.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
SEED = 42
sns.set_theme(style="whitegrid", palette="muted")

SERVICE_COLS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
                "Gate location", "Food and drink", "Online boarding", "Seat comfort",
                "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
                "Checkin service", "Inflight service", "Cleanliness"]
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUM_BASE = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
ENGINEERED = ["service_mean", "service_min", "low_scores", "total_delay", "delay_gained", "log_distance"]
NUM_COLS = NUM_BASE + SERVICE_COLS + ENGINEERED
TARGET = "satisfied"


# ---------------------------------------------------------------- 1. load
def load():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    for df in (train, test):
        df.drop(columns=[c for c in ["Unnamed: 0", "id"] if c in df.columns], inplace=True)
        df[TARGET] = (df.pop("satisfaction") == "satisfied").astype(int)
    return train, test


# ---------------------------------------------------------------- 2. EDA
def eda(df, report):
    report["rows"], report["cols"] = int(df.shape[0]), int(df.shape[1])
    report["missing"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}
    report["duplicates"] = int(df.duplicated().sum())
    report["class_balance"] = df[TARGET].value_counts(normalize=True).round(4).to_dict()

    fig, ax = plt.subplots(figsize=(6, 4))
    df[TARGET].map({0: "Neutral / dissatisfied", 1: "Satisfied"}).value_counts().plot.bar(ax=ax, color=["#9aa5b1", "#2b6cb0"])
    ax.set_title("Target class balance"); ax.set_ylabel("Passengers"); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(FIG / "airline_class_balance.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, col in zip(axes, CAT_COLS):
        (df.groupby(col)[TARGET].mean() * 100).sort_values().plot.barh(ax=ax, color="#2b6cb0")
        ax.set_title(f"% satisfied by {col}"); ax.set_xlabel("% satisfied"); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(FIG / "airline_categorical_vs_target.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, col in zip(axes, NUM_BASE):
        sns.histplot(data=df, x=col, hue=TARGET, bins=40, ax=ax, element="step", stat="density", common_norm=False)
        ax.set_title(col)
        if "Delay" in col:
            ax.set_xlim(0, 200)
    fig.tight_layout(); fig.savefig(FIG / "airline_numeric_distributions.png", dpi=150); plt.close(fig)

    corr = df[SERVICE_COLS + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation of service ratings with satisfaction")
    fig.tight_layout(); fig.savefig(FIG / "airline_correlation_heatmap.png", dpi=150); plt.close(fig)
    report["top_correlations"] = corr[TARGET].drop(TARGET).sort_values(ascending=False).round(3).head(6).to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=df, x="Flight Distance", ax=axes[0], color="#9aa5b1"); axes[0].set_title("Flight Distance (outliers)")
    sns.boxplot(data=df, x="Departure Delay in Minutes", ax=axes[1], color="#9aa5b1"); axes[1].set_title("Departure Delay (heavy right tail)")
    fig.tight_layout(); fig.savefig(FIG / "airline_outliers_boxplot.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["Departure Delay in Minutes"], df["Arrival Delay in Minutes"], s=2, alpha=0.2, color="#2b6cb0")
    ax.set_xlabel("Departure delay (min)"); ax.set_ylabel("Arrival delay (min)")
    ax.set_title(f"Departure vs arrival delay (r = {df[['Departure Delay in Minutes','Arrival Delay in Minutes']].corr().iloc[0,1]:.2f})")
    fig.tight_layout(); fig.savefig(FIG / "airline_delay_scatter.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 3. cleaning + feature engineering
def clean(df, report=None, fit_stats=None):
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    # Missing arrival delay (~0.3%) is imputed from departure delay (r ~ 0.96) rather than dropped.
    miss = df["Arrival Delay in Minutes"].isna()
    df.loc[miss, "Arrival Delay in Minutes"] = df.loc[miss, "Departure Delay in Minutes"]
    # Outliers: cap extreme delays at the training 99.5th percentile (keeps rows, limits leverage).
    if fit_stats is None:
        fit_stats = {c: float(df[c].quantile(0.995)) for c in ["Departure Delay in Minutes", "Arrival Delay in Minutes"]}
    for c, cap in fit_stats.items():
        df[c] = df[c].clip(upper=cap)
    if report is not None:
        report["cleaning"] = {"duplicates_removed": before - len(df), "arrival_delay_imputed": int(miss.sum()),
                              "delay_caps_p99_5": fit_stats}
    return df, fit_stats


def engineer(df):
    df = df.copy()
    ratings = df[SERVICE_COLS].replace(0, np.nan)  # 0 = "not applicable" in this survey
    df["service_mean"] = ratings.mean(axis=1)
    df["service_min"] = ratings.min(axis=1)
    df["low_scores"] = (ratings <= 2).sum(axis=1)
    df["total_delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["delay_gained"] = df["Arrival Delay in Minutes"] - df["Departure Delay in Minutes"]
    df["log_distance"] = np.log1p(df["Flight Distance"])
    return df


def make_preprocessor(scale=True):
    num = Pipeline([("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else []))
    return ColumnTransformer([
        ("num", num, NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CAT_COLS),
    ])


def metrics(y, pred, proba):
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba)}


def r4(d):
    return {k: round(float(v), 4) for k, v in d.items()}


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    report = {"project": "Airline Passenger Satisfaction (classification)"}
    train_full, test = load()
    eda(pd.concat([train_full, test], ignore_index=True), report.setdefault("eda", {}))

    train_full, stats = clean(train_full, report)
    test, _ = clean(test, fit_stats=stats)
    train_full, test = engineer(train_full), engineer(test)

    X, y = train_full[NUM_COLS + CAT_COLS], train_full[TARGET]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_te, y_te = test[NUM_COLS + CAT_COLS], test[TARGET]
    report["split"] = {"train": len(X_tr), "validation": len(X_val), "test": len(X_te)}

    # ------------------------------------------------ 4/5. candidate algorithms, trained on train
    candidates = {
        "Logistic Regression": Pipeline([("prep", make_preprocessor()), ("clf", LogisticRegression(max_iter=2000))]),
        "Random Forest": Pipeline([("prep", make_preprocessor(False)), ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED))]),
        "XGBoost": Pipeline([("prep", make_preprocessor(False)), ("clf", XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, eval_metric="logloss", n_jobs=-1, random_state=SEED))]),
        "LightGBM": Pipeline([("prep", make_preprocessor(False)), ("clf", LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=SEED, verbose=-1, n_jobs=1))]),
    }
    baseline = {}
    for name, pipe in candidates.items():
        s = time.time(); pipe.fit(X_tr, y_tr); fit_s = time.time() - s
        tr = metrics(y_tr, pipe.predict(X_tr), pipe.predict_proba(X_tr)[:, 1])
        va = metrics(y_val, pipe.predict(X_val), pipe.predict_proba(X_val)[:, 1])
        baseline[name] = {"train": r4(tr), "validation": r4(va), "fit_seconds": round(fit_s, 1),
                          "train_val_gap_accuracy": round(tr["accuracy"] - va["accuracy"], 4)}
        print(f"{name:20s} val acc={va['accuracy']:.4f} f1={va['f1']:.4f} auc={va['roc_auc']:.4f} gap={tr['accuracy']-va['accuracy']:.4f}")
    report["baseline"] = baseline

    # ------------------------------------------------ 6. hyperparameter tuning (5-fold CV on train)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    tuning = {}
    # Grid search on Logistic Regression (small, interpretable space)
    gs = GridSearchCV(candidates["Logistic Regression"], {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"]},
                      scoring="f1", cv=cv, n_jobs=-1)
    gs.fit(X_tr, y_tr)
    tuning["Logistic Regression (Grid Search)"] = {"best_params": gs.best_params_, "cv_f1": round(gs.best_score_, 4)}
    # Random search on LightGBM (larger continuous space)
    lgb_space = {"clf__n_estimators": randint(300, 1200), "clf__learning_rate": loguniform(0.01, 0.2),
                 "clf__num_leaves": randint(31, 255), "clf__min_child_samples": randint(10, 100),
                 "clf__subsample": uniform(0.6, 0.4), "clf__subsample_freq": [1],
                 "clf__colsample_bytree": uniform(0.5, 0.5), "clf__reg_lambda": loguniform(1e-3, 10)}
    rs = RandomizedSearchCV(candidates["LightGBM"], lgb_space, n_iter=25, scoring="f1", cv=cv,
                            random_state=SEED, n_jobs=-1, verbose=1)
    rs.fit(X_tr, y_tr)
    tuning["LightGBM (Random Search)"] = {"best_params": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rs.best_params_.items()},
                                          "cv_f1": round(rs.best_score_, 4)}
    cvres = pd.DataFrame(rs.cv_results_).sort_values("rank_test_score")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(cvres["param_clf__learning_rate"].astype(float), cvres["mean_test_score"], c=cvres["param_clf__num_leaves"].astype(float), cmap="Blues", edgecolor="k")
    ax.set_xscale("log"); ax.set_xlabel("learning_rate"); ax.set_ylabel("CV F1"); ax.set_title("LightGBM random search: 25 trials (colour = num_leaves)")
    fig.tight_layout(); fig.savefig(FIG / "airline_random_search.png", dpi=150); plt.close(fig)

    best = rs.best_estimator_
    va_best = metrics(y_val, best.predict(X_val), best.predict_proba(X_val)[:, 1])
    tr_best = metrics(y_tr, best.predict(X_tr), best.predict_proba(X_tr)[:, 1])
    tuning["tuned_lightgbm_validation"] = r4(va_best)
    tuning["tuned_lightgbm_train"] = r4(tr_best)
    report["tuning"] = tuning
    print("tuned LightGBM val", r4(va_best))

    # Validation comparison chart
    comp = pd.DataFrame({n: b["validation"] for n, b in baseline.items()}).T
    comp.loc["LightGBM (tuned)"] = va_best
    fig, ax = plt.subplots(figsize=(10, 4.5))
    comp[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot.bar(ax=ax, width=0.8)
    ax.set_ylim(0.8, 1.0); ax.set_title("Validation metrics by model"); ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="lower right", ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "airline_model_comparison.png", dpi=150); plt.close(fig)

    # ------------------------------------------------ 8. final model: refit on train+val, evaluate once on test
    final = rs.best_estimator_.set_params()  # same hyperparameters
    final.fit(pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val]))
    proba = final.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    report["test"] = r4(metrics(y_te, pred, proba))
    report["test_classification_report"] = classification_report(y_te, pred, target_names=["neutral/dissatisfied", "satisfied"], output_dict=True)
    print("TEST", report["test"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred), display_labels=["Neutral/Dissat.", "Satisfied"]).plot(ax=axes[0], cmap="Blues", values_format=",")
    axes[0].set_title("Test confusion matrix")
    fpr, tpr, _ = roc_curve(y_te, proba)
    axes[1].plot(fpr, tpr, color="#2b6cb0", lw=2, label=f"Tuned LightGBM (AUC = {report['test']['roc_auc']:.3f})")
    axes[1].plot([0, 1], [0, 1], "--", color="grey"); axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
    axes[1].set_title("Test ROC curve"); axes[1].legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "airline_test_confusion_roc.png", dpi=150); plt.close(fig)

    # Feature importance (gain) from final LightGBM
    names = final.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(final.named_steps["clf"].booster_.feature_importance("gain"), index=[n.split("__", 1)[1] for n in names])
    imp = (imp / imp.sum()).sort_values(ascending=False).head(15)
    report["feature_importance_top15"] = imp.round(4).to_dict()
    fig, ax = plt.subplots(figsize=(8, 6))
    imp[::-1].plot.barh(ax=ax, color="#2b6cb0"); ax.set_title("Top 15 features (LightGBM gain share)"); ax.set_xlabel("Share of total gain")
    fig.tight_layout(); fig.savefig(FIG / "airline_feature_importance.png", dpi=150); plt.close(fig)

    # Inference latency (single-row, as an API would call it)
    one = X_te.iloc[[0]]
    lat = []
    for _ in range(200):
        s = time.perf_counter(); final.predict_proba(one); lat.append((time.perf_counter() - s) * 1000)
    report["latency_ms_single_row"] = {"p50": round(float(np.percentile(lat, 50)), 2), "p95": round(float(np.percentile(lat, 95)), 2)}

    joblib.dump({"model": final, "features": NUM_COLS + CAT_COLS, "clean_stats": stats,
                 "service_cols": SERVICE_COLS, "reference": X_tr.sample(5000, random_state=SEED)},
                MODELS / "airline_model.joblib", compress=3)
    report["runtime_seconds"] = round(time.time() - t0, 1)
    (ROOT / "reports" / "airline_metrics.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"done in {report['runtime_seconds']}s")


if __name__ == "__main__":
    main()
