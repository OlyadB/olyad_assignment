"""Writes report/metrics.tex: LaTeX macros holding every number the report quotes,
read from reports/*.json so the PDF always matches the latest pipeline run.
Run: python report/make_metrics.py   (then: latexmk -pdf report.tex inside report/)"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
air = json.loads((R / "airline_metrics.json").read_text())
taxi = json.loads((R / "taxi_metrics.json").read_text())
mon = json.loads((R / "monitoring.json").read_text()) if (R / "monitoring.json").exists() else None

out = []


def m(name, value):
    out.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def f3(v):
    return f"{v:.3f}"


def pc(v):
    return f"{v * 100:.1f}\\%"


def n(v):
    return f"{int(v):,}".replace(",", "{,}")


def letters(s):  # macro names may only contain letters
    return "".join(ch for ch in s.title() if ch.isalpha())


# ---------------- airline
m("airRows", n(air["eda"]["rows"]))
m("airTrain", n(air["split"]["train"])); m("airVal", n(air["split"]["validation"])); m("airTest", n(air["split"]["test"]))
m("airMissing", n(air["eda"]["missing"]["Arrival Delay in Minutes"]))
m("airSatShare", pc(air["eda"]["class_balance"]["1"]))
m("airCapDep", air["cleaning"]["delay_caps_p99_5"]["Departure Delay in Minutes"])
m("airCapArr", air["cleaning"]["delay_caps_p99_5"]["Arrival Delay in Minutes"])
for model, b in air["baseline"].items():
    k = letters(model)
    for split in ("train", "validation"):
        for met, v in b[split].items():
            m(f"air{k}{split.title()[:3]}{letters(met)}", f3(v))
    m(f"air{k}Fit", b["fit_seconds"])
    m(f"air{k}Gap", f"{b['train_val_gap_accuracy'] * 100:.1f}")
t = air["tuning"]
bp = t["LightGBM (Random Search)"]["best_params"]
for k, v in bp.items():
    m("airBp" + letters(k.replace("clf__", "")), v)
m("airLgbCvFone", t["LightGBM (Random Search)"]["cv_f1"])
m("airLrBestC", t["Logistic Regression (Grid Search)"]["best_params"]["clf__C"])
m("airLrCvFone", t["Logistic Regression (Grid Search)"]["cv_f1"])
for split, key in (("Val", "tuned_lightgbm_validation"), ("Tr", "tuned_lightgbm_train")):
    for met, v in t[key].items():
        m(f"airTuned{split}{letters(met)}", f3(v))
for met, v in air["test"].items():
    m(f"airTest{letters(met)}", f3(v))
m("airTestAccPct", pc(air["test"]["accuracy"]))
cr = air["test_classification_report"]
m("airTestSupportPos", n(cr["satisfied"]["support"])); m("airTestSupportNeg", n(cr["neutral/dissatisfied"]["support"]))
m("airLatP", air["latency_ms_single_row"]["p50"]); m("airLatPn", air["latency_ms_single_row"]["p95"])
fi = list(air["feature_importance_top15"].items())
m("airTopShare", f"{(fi[0][1] + fi[1][1]) * 100:.0f}\\%")
m("airRuntime", f"{air['runtime_seconds'] / 60:.1f}")

# ---------------- taxi
e = taxi["eda"]
m("taxiRows", n(e["rows"]))
m("taxiMedianMin", f"{e['duration_seconds_describe']['50%'] / 60:.1f}")
m("taxiMeanMin", f"{e['duration_seconds_describe']['mean'] / 60:.1f}")
m("taxiMaxDays", f"{e['duration_seconds_describe']['max'] / 86400:.0f}")
m("taxiOverDay", n(e["trips_over_24h"])); m("taxiUnderMin", n(e["trips_under_60s"]))
c = taxi["cleaning"]
m("taxiOutBox", n(c["outside_nyc_bbox"])); m("taxiBadDur", n(c["duration_outside_60s_4h"]))
m("taxiBadPax", n(c["passenger_count_not_1_to_6"])); m("taxiZeroDist", n(c["zero_distance_trips"]))
m("taxiFast", n(c["speed_over_120kmh"])); m("taxiKept", n(c["rows_kept"])); m("taxiRemovedPct", f"{c['rows_removed_pct']}\\%")
m("taxiTrain", n(taxi["split"]["train"])); m("taxiVal", n(taxi["split"]["validation"])); m("taxiTest", n(taxi["split"]["test"]))
for model, b in taxi["baseline"].items():
    k = letters(model.split(" (")[0])
    for split in ("train", "validation"):
        for met in ("rmsle", "r2_log", "mae_seconds"):
            v = b[split][met]
            m(f"taxi{k}{split.title()[:3]}{letters(met)}", f3(v) if met != "mae_seconds" else f"{v / 60:.1f}")
    m(f"taxi{k}Fit", b["fit_seconds"])
t = taxi["tuning"]
m("taxiRidgeAlpha", t["Ridge (Grid Search)"]["best_params"]["reg__alpha"]); m("taxiRidgeCv", t["Ridge (Grid Search)"]["cv_rmsle"])
for k, v in t["LightGBM (Random Search)"]["best_params"].items():
    m("taxiBp" + letters(k), v)
m("taxiLgbCv", t["LightGBM (Random Search)"]["cv_rmsle"])
m("taxiBestIter", n(t["final_best_iteration"]))
for split, key in (("Val", "tuned_lightgbm_validation"), ("Tr", "tuned_lightgbm_train")):
    for met in ("rmsle", "r2_log"):
        m(f"taxiTuned{split}{letters(met)}", f3(t[key][met]))
te = taxi["test"]
m("taxiTestRmsle", f3(te["rmsle"])); m("taxiTestRtwoLog", f3(te["r2_log"])); m("taxiTestRtwoSec", f3(te["r2_seconds"]))
m("taxiTestMaeMin", f"{te['mae_seconds'] / 60:.1f}"); m("taxiTestRmseMin", f"{te['rmse_seconds'] / 60:.1f}")
m("taxiTestMedAeMin", f"{te['median_ae_seconds'] / 60:.1f}")
m("taxiLatP", taxi["latency_ms_single_row"]["p50"]); m("taxiLatPn", taxi["latency_ms_single_row"]["p95"])
m("taxiRuntime", f"{taxi['runtime_seconds'] / 60:.0f}")

# ---------------- monitoring
if mon:
    m("monAirAcc", pc(mon["airline_live"]["accuracy"])); m("monAirFone", f3(mon["airline_live"]["f1"]))
    m("monTaxiRmsle", f3(mon["taxi_live"]["rmsle"])); m("monTaxiMaeMin", f"{mon['taxi_live']['mae_seconds'] / 60:.1f}")
    for ep in ("airline", "taxi"):
        for q in ("p50", "p95", "p99"):
            m(f"mon{ep.title()}Lat{letters(q.replace('5', 'five').replace('9', 'nine').replace('0', 'zero'))}", mon["latency_ms"][ep][q])
    for k, v in {**mon["psi_airline"], **mon["psi_taxi"]}.items():
        m("psi" + letters(k.replace("_", " ")), f3(v))

(Path(__file__).parent / "metrics.tex").write_text("% generated by make_metrics.py -- do not edit\n" + "\n".join(out) + "\n")
print(f"wrote {len(out)} macros")
