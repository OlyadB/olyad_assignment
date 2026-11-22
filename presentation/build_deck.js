// Builds presentation/Olyad_Bizuneh_ML_Process.pptx from reports/*.json and reports/figures/*.png
// Run: node presentation/build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const FIG = (f) => path.join(ROOT, "reports", "figures", f);
const J = (f) => JSON.parse(fs.readFileSync(path.join(ROOT, "reports", f), "utf8"));
const air = J("airline_metrics.json");
const taxi = J("taxi_metrics.json");
const mon = fs.existsSync(path.join(ROOT, "reports", "monitoring.json")) ? J("monitoring.json") : null;

// PNG size from header -> keep aspect ratio
function pngSize(f) {
  const b = fs.readFileSync(f);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function fitImage(slide, file, x, y, maxW, maxH, alt) {
  const { w, h } = pngSize(file);
  const r = Math.min(maxW / w, maxH / h);
  const W = w * r, H = h * r;
  slide.addImage({ path: file, x: x + (maxW - W) / 2, y: y + (maxH - H) / 2, w: W, h: H, altText: alt });
}

const THEME = {
  name: "Navy and Taxi Yellow",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F2A", lt1: "FFFFFF", dk2: "14213D", lt2: "EEF1F6",
    accent1: "14213D", accent2: "FCA311", accent3: "5C6B8A", accent4: "9AA5B1",
    accent5: "2E7D32", accent6: "C62828", hlink: "1F5FAD", folHlink: "5C6B8A",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "Olyad Bizuneh";
pres.title = "The ML Process — Airline Satisfaction & NYC Taxi Trip Duration";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const FOOT = { text: "Olyad Bizuneh  ·  Machine Learning Assignment: The ML Process", options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent3 } };

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.6, fontSize: 44, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.0, w: 11.7, h: 1.4, fontSize: 20, color: C.accent2, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.2, fontSize: 40, bold: true, color: C.background1, fontFace: THEME.headFontFace }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.8, w: 11.7, h: 1.0, fontSize: 18, color: C.accent2 }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent4 },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 30, bold: true, color: C.text2, fontFace: THEME.headFontFace, valign: "middle" }, text: "" } },
    { text: FOOT },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent3 },
});

const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const n0 = (v) => Number(v).toLocaleString("en-US");
let oid = 0;
const name = (s) => `${s}-${++oid}`;

function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function card(slide, x, y, w, h, head, body, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: opts.fill || C.background2 }, line: { type: "none" }, objectName: name("card") });
  slide.addText(head, { x: x + 0.2, y: y + 0.15, w: w - 0.4, h: 0.45, fontSize: 16, bold: true, color: opts.headColor || C.text2, isTextBox: true, margin: 0, valign: "top" });
  slide.addText(body, { x: x + 0.2, y: y + 0.65, w: w - 0.4, h: h - 0.8, fontSize: opts.size || 14, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 4 });
}
function stat(slide, x, y, w, value, label, color) {
  slide.addText(value, { x, y, w, h: 0.9, fontSize: 40, bold: true, color: color || C.accent1, fontFace: THEME.headFontFace, isTextBox: true, margin: 0 });
  slide.addText(label, { x, y: y + 0.9, w, h: 0.5, fontSize: 13, color: C.accent3, isTextBox: true, margin: 0 });
}
const bullets = (arr) => arr.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < arr.length - 1 } }));
const CHART_TXT = { catAxisLabelColor: "5C6B8A", valAxisLabelColor: "5C6B8A", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 11, dataLabelColor: "1B1F2A" };

// ===================================================================== 1. title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("The Machine Learning Process, End to End", { placeholder: "title" });
s.addText("Airline Passenger Satisfaction (classification)  ·  NYC Taxi Trip Duration (regression)\nOlyad Bizuneh", { placeholder: "body" });
s.addNotes("Two complete projects following every step of the ML process from the assignment brief.");

// ===================================================================== 2. process flow
s = content("Both projects follow the same nine-step ML process", "Introduction");
const steps = ["Problem definition", "Data collection", "EDA & preparation", "Algorithm selection", "Training", "Evaluation & tuning", "Testing", "Deployment", "Monitoring"];
steps.forEach((t, i) => {
  const col = i % 5, row = Math.floor(i / 5);
  const x = 0.6 + col * 2.45, y = 1.7 + row * 2.3;
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.7, h: 0.7, fill: { color: i === 5 || i === 6 ? C.accent2 : C.accent1 }, line: { type: "none" }, objectName: name("step-dot") });
  s.addText(String(i + 1), { x, y, w: 0.7, h: 0.7, align: "center", valign: "middle", fontSize: 20, bold: true, color: i === 5 || i === 6 ? C.text1 : C.background1, isTextBox: true, margin: 0 });
  s.addText(t, { x: x - 0.1, y: y + 0.8, w: 2.2, h: 0.8, fontSize: 16, bold: true, color: C.text2, isTextBox: true, margin: 0, valign: "top" });
  if (col < 4 && i < steps.length - 1) s.addShape(pres.shapes.LINE, { x: x + 0.8, y: y + 0.35, w: 1.55, h: 0, line: { color: C.accent4, width: 1.5, endArrowType: "triangle" }, objectName: name("arrow") });
});
s.addText("Code: src/airline_pipeline.py and src/taxi_pipeline.py run every step and save metrics, figures and the final model.", { x: 0.6, y: 6.3, w: 12, h: 0.5, fontSize: 14, color: C.accent3, italic: true, isTextBox: true, margin: 0 });

// ===================================================================== Airline
pres.addSection({ title: "Project A — Airline" });
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Project A — Airline" });
s.addText("Project A: Airline Passenger Satisfaction", { placeholder: "title" });
s.addText("Binary classification — satisfied vs neutral or dissatisfied", { placeholder: "body" });

s = content("Goal: find dissatisfied passengers and the services that drive it", "Project A — Airline");
card(s, 0.6, 1.5, 3.9, 4.9, "Business problem", bullets(["Dissatisfied passengers switch airlines", "Management needs to know which services to fix first", "Predict satisfaction from a post-flight survey"]));
card(s, 4.75, 1.5, 3.9, 4.9, "ML task", bullets(["Supervised binary classification", "Target: satisfied (1) vs neutral or dissatisfied (0)", `${n0(air.eda.rows)} passengers, 22 input features`]));
card(s, 8.9, 1.5, 3.85, 4.9, "Success criteria (test set)", bullets(["Accuracy ≥ 95%", "F1-score ≥ 0.94", "ROC-AUC ≥ 0.98", "Train–validation gap < 2 points"]), { fill: "FEF3DC" });

s = content("Data: 129,880 survey responses, 44% satisfied", "Project A — Airline");
fitImage(s, FIG("airline_categorical_vs_target.png"), 0.6, 1.4, 12.1, 3.2, "Percent satisfied by gender, customer type, type of travel and class");
s.addText(bullets([
  "Source: Kaggle teejmahal20/airline-passenger-satisfaction (103,904 train + 25,976 test rows)",
  "Business travellers and Business class are far more satisfied; gender makes almost no difference",
  `Classes are balanced enough (${pct(air.eda.class_balance["1"])} satisfied) — no resampling needed`,
]), { x: 0.6, y: 4.9, w: 12.1, h: 1.8, fontSize: 15, color: C.text1, isTextBox: true, paraSpaceAfter: 6 });

s = content("Online boarding and wifi ratings correlate most with satisfaction", "Project A — Airline");
fitImage(s, FIG("airline_correlation_heatmap.png"), 0.6, 1.3, 6.4, 5.5, "Correlation heatmap of service ratings with satisfaction");
const tc = Object.entries(air.eda.top_correlations);
s.addText("Correlation with satisfied", { x: 7.4, y: 1.5, w: 5.3, h: 0.5, fontSize: 18, bold: true, color: C.text2, isTextBox: true, margin: 0 });
s.addChart(pres.charts.BAR, [{ name: "r", labels: tc.map((e) => e[0]), values: tc.map((e) => e[1]) }], {
  x: 7.3, y: 2.0, w: 5.4, h: 4.2, barDir: "bar", chartColors: [HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
  valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisOrientation: "maxMin", showLegend: false, ...CHART_TXT,
});

s = content("Cleaning and feature engineering", "Project A — Airline");
card(s, 0.6, 1.5, 3.9, 5.0, "Cleaning", bullets([
  `${air.eda.missing["Arrival Delay in Minutes"]} missing arrival delays → filled from departure delay (r ≈ 0.96)`,
  `${air.eda.duplicates} duplicate rows found`,
  `Delay outliers capped at 99.5th percentile (${air.cleaning.delay_caps_p99_5["Departure Delay in Minutes"]} / ${air.cleaning.delay_caps_p99_5["Arrival Delay in Minutes"]} min)`,
  "Rating 0 = 'not applicable' → treated as missing in derived features",
]));
card(s, 4.75, 1.5, 3.9, 5.0, "New features (6)", bullets(["service_mean — average of 14 ratings", "service_min — worst rating", "low_scores — count of ratings ≤ 2", "total_delay, delay_gained", "log_distance"]));
card(s, 8.9, 1.5, 3.85, 5.0, "Encoding, scaling, split", bullets([
  "One-hot encoding for Gender, Customer Type, Type of Travel, Class",
  "StandardScaler for the linear model",
  `Train ${n0(air.split.train)} · Validation ${n0(air.split.validation)} · Test ${n0(air.split.test)}`,
  "Stratified split; Kaggle test file kept untouched",
]));

s = content("Gradient boosting beats the linear baseline by 7 accuracy points", "Project A — Airline");
const am = Object.keys(air.baseline);
s.addChart(pres.charts.BAR, [
  { name: "Train accuracy", labels: am, values: am.map((m) => air.baseline[m].train.accuracy) },
  { name: "Validation accuracy", labels: am, values: am.map((m) => air.baseline[m].validation.accuracy) },
], {
  x: 0.6, y: 1.4, w: 7.6, h: 5.3, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent4, HEX.accent1],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0.85, valAxisMaxVal: 1.02, valAxisLabelFormatCode: "0.00",
  valGridLine: { color: "E3E7EE", size: 0.5 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 12, ...CHART_TXT,
});
card(s, 8.5, 1.4, 4.25, 5.3, "Reading the results", bullets([
  `Logistic Regression underfits: ${pct(air.baseline["Logistic Regression"].validation.accuracy)} validation accuracy`,
  `Random Forest overfits: 100% train vs ${pct(air.baseline["Random Forest"].validation.accuracy)} validation`,
  `LightGBM has the best validation score and a small gap (${(air.baseline.LightGBM.train_val_gap_accuracy * 100).toFixed(1)} points) → chosen for tuning`,
]));

s = content("Tuning: grid search and random search with 5-fold CV", "Project A — Airline");
fitImage(s, FIG("airline_random_search.png"), 0.6, 1.4, 6.6, 4.4, "LightGBM random search results");
const bp = air.tuning["LightGBM (Random Search)"].best_params;
card(s, 7.5, 1.4, 5.25, 5.3, "Best LightGBM (25 random trials)", bullets([
  `learning_rate ${bp.clf__learning_rate}, n_estimators ${bp.clf__n_estimators}`,
  `num_leaves ${bp.clf__num_leaves}, min_child_samples ${bp.clf__min_child_samples}`,
  `subsample ${bp.clf__subsample}, colsample ${bp.clf__colsample_bytree}`,
  `CV F1: ${air.tuning["LightGBM (Random Search)"].cv_f1}  →  validation F1: ${air.tuning.tuned_lightgbm_validation.f1}`,
  `Grid search on Logistic Regression C: best C = ${air.tuning["Logistic Regression (Grid Search)"].best_params.clf__C}, CV F1 ${air.tuning["Logistic Regression (Grid Search)"].cv_f1}`,
]), { size: 14 });
s.addText("Random search fits the large continuous LightGBM space; grid search fits the one-parameter linear model.", { x: 0.6, y: 6.0, w: 6.6, h: 0.7, fontSize: 13, color: C.accent3, italic: true, isTextBox: true, margin: 0 });

s = content("Test set: 96.5% accuracy, all success criteria met", "Project A — Airline");
stat(s, 0.6, 1.5, 2.4, pct(air.test.accuracy), "Accuracy (target ≥ 95%)");
stat(s, 0.6, 3.0, 2.4, air.test.f1.toFixed(3), "F1-score (target ≥ 0.94)");
stat(s, 0.6, 4.5, 2.4, air.test.roc_auc.toFixed(3), "ROC-AUC (target ≥ 0.98)");
s.addText(`Precision ${air.test.precision.toFixed(3)} · Recall ${air.test.recall.toFixed(3)}`, { x: 0.6, y: 6.0, w: 3, h: 0.5, fontSize: 13, color: C.text1, isTextBox: true, margin: 0 });
fitImage(s, FIG("airline_test_confusion_roc.png"), 3.3, 1.4, 9.4, 5.3, "Test confusion matrix and ROC curve");

s = content("What drives satisfaction: online boarding, wifi and class", "Project A — Airline");
fitImage(s, FIG("airline_feature_importance.png"), 0.6, 1.3, 6.6, 5.5, "Top 15 features by LightGBM gain");
card(s, 7.5, 1.4, 5.25, 5.3, "Business recommendations", bullets([
  "Fix the digital journey first: online boarding and inflight wifi carry ~42% of the model's gain",
  "Economy and personal-travel passengers are the least satisfied — target improvements there",
  "Disloyal customers are a risk group worth a retention offer",
  "Delays matter less than service quality once on board",
]));

// ===================================================================== Taxi
pres.addSection({ title: "Project B — Taxi" });
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Project B — Taxi" });
s.addText("Project B: NYC Taxi Trip Duration", { placeholder: "title" });
s.addText("Regression — predict how many seconds a trip will take, at pickup", { placeholder: "body" });

s = content("Goal: quote an accurate trip time before the ride starts", "Project B — Taxi");
card(s, 0.6, 1.5, 3.9, 4.9, "Business problem", bullets(["Dispatch, pricing and rider trust depend on accurate ETAs", "Only information known at pickup may be used", "Dropoff time is removed (it would leak the answer)"]));
card(s, 4.75, 1.5, 3.9, 4.9, "ML task", bullets(["Supervised regression", "Target: log1p(trip_duration in seconds)", `${n0(taxi.eda.rows)} trips, Jan–Jun 2016`, "Source: Kaggle NYC Taxi Trip Duration (NYC TLC data)"]));
card(s, 8.9, 1.5, 3.85, 4.9, "Success criteria (test set)", bullets(["RMSLE ≤ 0.40 (competition metric)", `Mean-prediction baseline: ${taxi.baseline["Mean baseline"].validation.rmsle.toFixed(3)}`, "R² on log duration ≥ 0.70", "Also report RMSE and MAE in seconds"]), { fill: "FEF3DC" });

s = content("Trip durations are heavily right-skewed, so the target is log-transformed", "Project B — Taxi");
fitImage(s, FIG("taxi_target_distribution.png"), 0.6, 1.3, 12.1, 3.7, "Raw and log-transformed trip duration distributions");
s.addText(bullets([
  `Median trip ${(taxi.eda.duration_seconds_describe["50%"] / 60).toFixed(1)} min, but the maximum is ${(taxi.eda.duration_seconds_describe.max / 86400).toFixed(0)} days — ${taxi.eda.trips_over_24h} trips over 24 h are meter errors`,
  "log1p(duration) is close to normal, and optimising RMSE on it is exactly the RMSLE metric",
]), { x: 0.6, y: 5.2, w: 12.1, h: 1.5, fontSize: 15, color: C.text1, isTextBox: true, paraSpaceAfter: 6 });

s = content("Time of day and weekday change trip length", "Project B — Taxi");
fitImage(s, FIG("taxi_hour_weekday_heatmap.png"), 0.6, 1.3, 12.1, 3.6, "Median trip duration by weekday and hour");
fitImage(s, FIG("taxi_pickup_map.png"), 0.6, 4.9, 2.0, 2.0, "Pickup locations map");
s.addText(bullets([
  "Weekday trips are longest 8 am–6 pm; weekend nights are busy but fast",
  "Pickups concentrate in Manhattan with clusters at JFK and LaGuardia — airport flags and K-Means location clusters capture this",
]), { x: 2.9, y: 5.0, w: 9.8, h: 1.8, fontSize: 15, color: C.text1, isTextBox: true, paraSpaceAfter: 6 });

s = content(`Cleaning removed ${taxi.cleaning.rows_removed_pct}% of trips; 27 features were built`, "Project B — Taxi");
const cl = taxi.cleaning;
s.addTable([
  [{ text: "Rule", options: { bold: true, color: "FFFFFF", fill: { color: HEX.dk2 } } }, { text: "Rows removed", options: { bold: true, color: "FFFFFF", fill: { color: HEX.dk2 }, align: "right" } }],
  ["Coordinates outside NYC", { text: n0(cl.outside_nyc_bbox), options: { align: "right" } }],
  ["Duration < 1 min or > 4 h", { text: n0(cl.duration_outside_60s_4h), options: { align: "right" } }],
  ["Passenger count not 1–6", { text: n0(cl.passenger_count_not_1_to_6), options: { align: "right" } }],
  ["Pickup = drop-off (< 50 m)", { text: n0(cl.zero_distance_trips), options: { align: "right" } }],
  ["Average speed > 120 km/h", { text: n0(cl.speed_over_120kmh), options: { align: "right" } }],
  [{ text: "Rows kept", options: { bold: true } }, { text: n0(cl.rows_kept), options: { bold: true, align: "right" } }],
], { x: 0.6, y: 1.5, w: 5.6, colW: [3.6, 2.0], fontSize: 14, color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "D5DAE3" }, rowH: 0.5 });
card(s, 6.6, 1.5, 6.15, 5.1, "Feature engineering", bullets([
  "Time: hour, minute of day, weekday, month, day of year, weekend, rush hour",
  "Distance: haversine km, Manhattan km, bearing, lat/lon deltas, trip centre",
  "Places: JFK / LaGuardia pickup and drop-off flags",
  "K-Means (k = 60) pickup and drop-off clusters — unsupervised learning used as a feature",
  `Split 70/15/15: ${n0(taxi.split.train)} / ${n0(taxi.split.validation)} / ${n0(taxi.split.test)}`,
]));

s = content("LightGBM cuts the error of the mean baseline by more than half", "Project B — Taxi");
const tm = Object.keys(taxi.baseline).concat(["LightGBM (tuned)"]);
const tv = tm.map((m) => (m === "LightGBM (tuned)" ? taxi.tuning.tuned_lightgbm_validation.rmsle : taxi.baseline[m].validation.rmsle));
s.addChart(pres.charts.BAR, [{ name: "Validation RMSLE", labels: tm, values: tv }], {
  x: 0.6, y: 1.4, w: 7.6, h: 5.3, barDir: "bar", chartColors: tm.map((m) => (m.includes("tuned") ? HEX.accent2 : HEX.accent1)),
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0, valAxisLabelFormatCode: "0.0",
  valGridLine: { color: "E3E7EE", size: 0.5 }, catGridLine: { style: "none" }, catAxisOrientation: "maxMin", showLegend: false,
  showTitle: true, title: "Validation RMSLE (lower is better)", titleFontSize: 14, titleColor: "14213D", ...CHART_TXT,
});
card(s, 8.5, 1.4, 4.25, 5.3, "Reading the results", bullets([
  `Ridge underfits (RMSLE ${taxi.baseline["Ridge Regression"].validation.rmsle.toFixed(3)}): duration is not linear in location and time`,
  "Tree ensembles capture interactions such as rush hour × Midtown",
  `Tuned LightGBM: ${taxi.tuning.tuned_lightgbm_validation.rmsle.toFixed(3)}, chosen as the final model`,
]));

s = content("Tuning: random search, then a full refit on one million trips", "Project B — Taxi");
fitImage(s, FIG("taxi_learning_curve.png"), 0.6, 1.4, 6.4, 3.4, "LightGBM learning curve with early stopping");
fitImage(s, FIG("taxi_random_search.png"), 0.6, 4.8, 6.4, 2.1, "Random search trials");
const tb = taxi.tuning["LightGBM (Random Search)"].best_params;
card(s, 7.3, 1.4, 5.45, 5.3, "Tuning steps", bullets([
  `Grid search on Ridge alpha → best ${taxi.tuning["Ridge (Grid Search)"].best_params.reg__alpha}`,
  `Random search, 15 LightGBM trials, 3-fold CV on 300k rows → CV RMSLE ${taxi.tuning["LightGBM (Random Search)"].cv_rmsle}`,
  `Best: num_leaves ${tb.num_leaves}, learning_rate ${tb.learning_rate}, min_child_samples ${tb.min_child_samples}`,
  `Refit on all ${n0(taxi.split.train)} training rows with ${n0(taxi.tuning.final_best_iteration)} trees; validation error flattens after ~1,000`,
  `Train vs validation RMSLE ${taxi.tuning.tuned_lightgbm_train.rmsle.toFixed(3)} vs ${taxi.tuning.tuned_lightgbm_validation.rmsle.toFixed(3)} → mild, acceptable overfitting`,
]));

s = content(`Test set: RMSLE ${taxi.test.rmsle.toFixed(3)}, well inside the 0.40 target`, "Project B — Taxi");
stat(s, 0.6, 1.5, 2.6, taxi.test.rmsle.toFixed(3), "RMSLE (target ≤ 0.40)");
stat(s, 0.6, 3.0, 2.6, taxi.test.r2_log.toFixed(3), "R² on log duration (≥ 0.70)");
stat(s, 0.6, 4.5, 2.6, (taxi.test.median_ae_seconds / 60).toFixed(1) + " min", "Median absolute error");
s.addText(`MAE ${(taxi.test.mae_seconds / 60).toFixed(1)} min · RMSE ${(taxi.test.rmse_seconds / 60).toFixed(1)} min`, { x: 0.6, y: 6.0, w: 3, h: 0.5, fontSize: 13, color: C.text1, isTextBox: true, margin: 0 });
fitImage(s, FIG("taxi_test_pred_vs_actual.png"), 3.5, 1.4, 9.2, 5.3, "Predicted vs actual and residuals on the test set");

s = content("Distance dominates; geometry and time refine it", "Project B — Taxi");
fitImage(s, FIG("taxi_feature_importance.png"), 0.6, 1.3, 6.6, 5.5, "Top 15 taxi features by LightGBM gain");
card(s, 7.5, 1.4, 5.25, 5.3, "Insights", bullets([
  "Distance features explain most of the variance",
  "Time of day and day of year capture traffic and seasonality",
  "Engineered geometry features (Manhattan distance, deltas, bearing, trip centre) fill most of the top 15",
  "K-Means clusters and airport flags add little beyond the raw coordinates",
  "Remaining error comes from traffic and events the data does not contain — weather or live traffic would help",
]));

// ===================================================================== Deployment & monitoring
pres.addSection({ title: "Deployment & Monitoring" });
s = content("Deployment: one model package, two front doors", "Deployment & Monitoring");
const box = (x, y, w, h, t, sub, dark) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: dark ? C.accent1 : C.background2 }, line: { type: "none" }, objectName: name("arch") });
  s.addText([{ text: t, options: { bold: true, fontSize: 16, breakLine: true } }, { text: sub, options: { fontSize: 13 } }], { x, y, w, h, align: "center", valign: "middle", color: dark ? C.background1 : C.text1, isTextBox: true });
};
const arrow = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color: C.accent3, width: 1.5, endArrowType: "triangle" }, objectName: name("flow") });
box(0.6, 2.0, 2.6, 1.2, "Training scripts", "src/*_pipeline.py");
box(0.6, 4.2, 2.6, 1.2, "Saved models", "models/*.joblib", true);
arrow(1.9, 3.2, 1.9, 4.2);
box(4.2, 4.2, 2.8, 1.2, "inference.py", "same cleaning + features as training");
arrow(3.2, 4.8, 4.2, 4.8);
box(8.0, 2.4, 2.4, 1.2, "Streamlit app", "web UI + feedback");
box(8.0, 5.0, 2.4, 1.2, "FastAPI", "/predict · /monitoring");
arrow(7.0, 4.6, 8.0, 3.0); arrow(7.0, 5.0, 8.0, 5.6);
box(10.8, 2.4, 1.95, 1.2, "Users", "browser");
box(10.8, 5.0, 1.95, 1.2, "Other systems", "REST JSON");
arrow(10.4, 3.0, 10.8, 3.0); arrow(10.4, 5.6, 10.8, 5.6);
s.addText("Run locally: streamlit run app/streamlit_app.py  ·  uvicorn api.main:app. Free hosting: Streamlit Community Cloud (steps in README).", { x: 0.6, y: 6.4, w: 12.1, h: 0.45, fontSize: 13, color: C.accent3, italic: true, isTextBox: true, margin: 0 });
s.addText(`Single-row latency: airline ${air.latency_ms_single_row.p50} ms, taxi ${taxi.latency_ms_single_row.p50} ms (median)`, { x: 0.6, y: 1.4, w: 12, h: 0.45, fontSize: 15, bold: true, color: C.text2, isTextBox: true, margin: 0 });

if (mon) {
  s = content("Monitoring: latency, live accuracy and data drift", "Deployment & Monitoring");
  fitImage(s, FIG("monitoring_latency_drift.png"), 0.6, 1.3, 12.1, 3.6, "API latency histogram and PSI drift per feature");
  card(s, 0.6, 5.1, 3.9, 1.7, "Live accuracy (500 requests)", `Airline accuracy ${pct(mon.airline_live.accuracy)} · F1 ${mon.airline_live.f1.toFixed(3)}\nTaxi RMSLE ${mon.taxi_live.rmsle.toFixed(3)} (June trips)`, { size: 13 });
  card(s, 4.75, 5.1, 3.9, 1.7, "API latency p95", `Airline ${mon.latency_ms.airline.p95} ms · Taxi ${mon.latency_ms.taxi.p95} ms`, { size: 13 });
  const drifted = Object.entries(mon.psi_status).filter((e) => e[1] !== "stable").map((e) => e[0]);
  card(s, 8.9, 5.1, 3.85, 1.7, "Drift (PSI)", drifted.length ? `Drift flagged: ${drifted.join(", ")} — expected for a new month; retrain monthly` : "All monitored features stable (PSI < 0.1)", { size: 13 });
}

// ===================================================================== conclusion
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Conclusion" });
s.addText("Both models beat their success criteria", { placeholder: "title" });
s.addText(`Airline: ${pct(air.test.accuracy)} accuracy, F1 ${air.test.f1.toFixed(3)}, AUC ${air.test.roc_auc.toFixed(3)}   ·   Taxi: RMSLE ${taxi.test.rmsle.toFixed(3)}, R² ${taxi.test.r2_log.toFixed(3)}\nNext: add weather and traffic data to the taxi model, collect app feedback, and retrain when PSI > 0.25`, { placeholder: "body" });

(async () => {
  const out = path.join(__dirname, "Olyad_Bizuneh_ML_Process.pptx");
  await pres.writeFile({ fileName: out });
  const { applyTheme } = require("./apply_theme.js");
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
