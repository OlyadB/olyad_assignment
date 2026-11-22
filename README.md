# ML Process Assignment — Olyad Bizuneh

Two end-to-end machine-learning projects that follow the full ML process
(problem definition → data → EDA/cleaning/feature engineering → algorithm selection →
training → evaluation & tuning → testing & deployment → monitoring → documentation).

| Project | Task | Final model | Headline test result |
|---|---|---|---|
| Airline Passenger Satisfaction Prediction | Binary classification | Tuned LightGBM | see `reports/airline_metrics.json` |
| NYC Taxi Trip Duration Prediction | Regression | Tuned LightGBM (+ K-Means location clusters) | see `reports/taxi_metrics.json` |

## Folder layout

```
data/airline/            train.csv, test.csv           (Kaggle: teejmahal20/airline-passenger-satisfaction)
data/taxi/               NYC.csv                       (Kaggle: yasserh/nyc-taxi-trip-duration)
src/airline_pipeline.py  classification: EDA → cleaning → features → split → models → tuning → test → save
src/taxi_pipeline.py     regression:     same steps, RMSLE metric, K-Means features
src/inference.py         shared prediction functions used by the app, API and monitor
src/monitor.py           post-deployment monitoring: latency, live accuracy, drift (PSI)
api/main.py              FastAPI service  (/predict/airline, /predict/taxi, /monitoring, /health)
app/streamlit_app.py     Streamlit web app (both models + user feedback collection)
models/                  saved models (*.joblib)
reports/                 metrics JSON + figures used in the report and slides
presentation/            PowerPoint presentation
logs/                    request log (API) and user feedback (app)
```

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. get the data (needs ~/.kaggle/kaggle.json)
kaggle datasets download -d teejmahal20/airline-passenger-satisfaction -p data/airline --unzip
kaggle datasets download -d yasserh/nyc-taxi-trip-duration -p data/taxi --unzip

# 2. train + evaluate (writes models/, reports/)
python src/airline_pipeline.py      # ~10–30 min (random search, 5-fold CV)
python src/taxi_pipeline.py         # ~15–30 min (1.4M rows)

# 3. deploy
streamlit run app/streamlit_app.py              # web UI  → http://localhost:8501
uvicorn api.main:app --port 8000                # REST API → http://localhost:8000/docs

# 4. monitoring simulation (latency, live metrics, PSI drift)
python src/monitor.py
```

### Example API call

```bash
curl -X POST localhost:8000/predict/taxi -H 'Content-Type: application/json' -d '{
  "pickup_datetime": "2016-04-12T17:30:00",
  "pickup_latitude": 40.7580, "pickup_longitude": -73.9855,
  "dropoff_latitude": 40.6413, "dropoff_longitude": -73.7781,
  "passenger_count": 1, "vendor_id": 2}'
```

## Free cloud deployment (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo (models/ included; data/ not needed for the app).
2. On https://share.streamlit.io choose the repo, main file `app/streamlit_app.py`.
3. Streamlit installs `requirements.txt` and serves the app at a public URL.

Note: `models/taxi_model.joblib` must stay under GitHub's 100 MB file limit (it does with the saved settings).
