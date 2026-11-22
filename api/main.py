"""
FastAPI service exposing both models, with built-in monitoring.

Run:   uvicorn api.main:app --reload --port 8000
Docs:  http://localhost:8000/docs
Every request is logged to logs/requests.jsonl (endpoint, latency, inputs, prediction);
GET /monitoring summarises request volume, latency percentiles and prediction drift.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inference import predict_airline, predict_taxi  # noqa: E402

LOG = ROOT / "logs" / "requests.jsonl"
LOG.parent.mkdir(exist_ok=True)

app = FastAPI(title="Olyad Bizuneh — ML Process Assignment API",
              description="Airline passenger satisfaction (classification) and NYC taxi trip duration (regression).",
              version="1.0")

Rating = Field(ge=0, le=5)


class Passenger(BaseModel):
    Gender: Literal["Male", "Female"]
    customer_type: Literal["Loyal Customer", "disloyal Customer"] = Field(alias="Customer Type")
    Age: int = Field(ge=1, le=100)
    type_of_travel: Literal["Business travel", "Personal Travel"] = Field(alias="Type of Travel")
    Class: Literal["Business", "Eco", "Eco Plus"]
    flight_distance: float = Field(alias="Flight Distance", gt=0)
    wifi: int = Field(alias="Inflight wifi service", ge=0, le=5)
    time_convenient: int = Field(alias="Departure/Arrival time convenient", ge=0, le=5)
    online_booking: int = Field(alias="Ease of Online booking", ge=0, le=5)
    gate_location: int = Field(alias="Gate location", ge=0, le=5)
    food: int = Field(alias="Food and drink", ge=0, le=5)
    online_boarding: int = Field(alias="Online boarding", ge=0, le=5)
    seat_comfort: int = Field(alias="Seat comfort", ge=0, le=5)
    entertainment: int = Field(alias="Inflight entertainment", ge=0, le=5)
    onboard_service: int = Field(alias="On-board service", ge=0, le=5)
    leg_room: int = Field(alias="Leg room service", ge=0, le=5)
    baggage: int = Field(alias="Baggage handling", ge=0, le=5)
    checkin: int = Field(alias="Checkin service", ge=0, le=5)
    inflight_service: int = Field(alias="Inflight service", ge=0, le=5)
    cleanliness: int = Field(alias="Cleanliness", ge=0, le=5)
    departure_delay: float = Field(alias="Departure Delay in Minutes", ge=0)
    arrival_delay: float | None = Field(default=None, alias="Arrival Delay in Minutes", ge=0)

    model_config = {"populate_by_name": True, "json_schema_extra": {"example": {
        "Gender": "Female", "Customer Type": "Loyal Customer", "Age": 35, "Type of Travel": "Business travel",
        "Class": "Business", "Flight Distance": 1500, "Inflight wifi service": 4,
        "Departure/Arrival time convenient": 4, "Ease of Online booking": 4, "Gate location": 3,
        "Food and drink": 4, "Online boarding": 5, "Seat comfort": 5, "Inflight entertainment": 5,
        "On-board service": 4, "Leg room service": 4, "Baggage handling": 4, "Checkin service": 4,
        "Inflight service": 4, "Cleanliness": 5, "Departure Delay in Minutes": 0, "Arrival Delay in Minutes": 0}}}


class Trip(BaseModel):
    pickup_datetime: datetime
    pickup_latitude: float = Field(ge=40.49, le=40.92)
    pickup_longitude: float = Field(ge=-74.27, le=-73.68)
    dropoff_latitude: float = Field(ge=40.49, le=40.92)
    dropoff_longitude: float = Field(ge=-74.27, le=-73.68)
    passenger_count: int = Field(1, ge=1, le=6)
    vendor_id: Literal[1, 2] = 2
    store_and_fwd_flag: Literal["Y", "N"] = "N"

    model_config = {"json_schema_extra": {"example": {
        "pickup_datetime": "2016-04-12T17:30:00", "pickup_latitude": 40.7580, "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.6413, "dropoff_longitude": -73.7781, "passenger_count": 1, "vendor_id": 2}}}


def _log(endpoint, payload, result, ms):
    with LOG.open("a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "endpoint": endpoint,
                            "latency_ms": round(ms, 2), "input": payload, "output": result}, default=str) + "\n")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/airline")
def airline(p: Passenger):
    s = time.perf_counter()
    rec = p.model_dump(by_alias=True)
    out = predict_airline([rec])[0]
    ms = (time.perf_counter() - s) * 1000
    _log("airline", rec, out, ms)
    return {**out, "latency_ms": round(ms, 2)}


@app.post("/predict/taxi")
def taxi(t: Trip):
    s = time.perf_counter()
    rec = t.model_dump()
    rec["pickup_datetime"] = rec["pickup_datetime"].strftime("%Y-%m-%d %H:%M:%S")
    out = predict_taxi([rec])[0]
    ms = (time.perf_counter() - s) * 1000
    _log("taxi", rec, out, ms)
    return {**out, "latency_ms": round(ms, 2)}


@app.get("/monitoring")
def monitoring():
    if not LOG.exists():
        return {"requests": 0}
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    summary = {}
    for ep in ("airline", "taxi"):
        r = [x for x in rows if x["endpoint"] == ep]
        if not r:
            continue
        lat = np.array([x["latency_ms"] for x in r])
        s = {"requests": len(r), "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
             "latency_ms_p95": round(float(np.percentile(lat, 95)), 2), "last_request": r[-1]["ts"]}
        if ep == "airline":
            s["share_predicted_satisfied"] = round(float(np.mean([x["output"]["prediction"] == "satisfied" for x in r])), 3)
        else:
            s["mean_predicted_minutes"] = round(float(np.mean([x["output"]["trip_duration_minutes"] for x in r])), 2)
        summary[ep] = s
    return summary
