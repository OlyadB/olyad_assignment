"""
Streamlit front end for both models.
Run:  streamlit run app/streamlit_app.py
"""
import csv
import json
import sys
import time
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inference import predict_airline, predict_taxi  # noqa: E402

FEEDBACK = ROOT / "logs" / "feedback.csv"
FEEDBACK.parent.mkdir(exist_ok=True)

st.set_page_config(page_title="Olyad Bizuneh — ML Assignment", page_icon="📊", layout="wide")
st.title("ML Process Assignment — Olyad Bizuneh")
st.caption("Airline Passenger Satisfaction (classification) · NYC Taxi Trip Duration (regression)")


def save_feedback(project, inputs, prediction, verdict, actual=""):
    new = not FEEDBACK.exists()
    with FEEDBACK.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "project", "inputs", "prediction", "user_verdict", "actual"])
        w.writerow([datetime.now(timezone.utc).isoformat(), project, json.dumps(inputs, default=str),
                    json.dumps(prediction), verdict, actual])


def metrics_card(name):
    p = ROOT / "reports" / f"{name}_metrics.json"
    if p.exists():
        return json.loads(p.read_text()).get("test", {})
    return {}


tab1, tab2, tab3 = st.tabs(["✈️ Airline satisfaction", "🚕 Taxi trip duration", "📈 Model info"])

# ------------------------------------------------------------------ airline
with tab1:
    st.subheader("Will this passenger be satisfied?")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        ctype = st.selectbox("Customer type", ["Loyal Customer", "disloyal Customer"])
        age = st.slider("Age", 7, 85, 35)
        travel = st.selectbox("Type of travel", ["Business travel", "Personal Travel"])
        cls = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
        dist = st.number_input("Flight distance (miles)", 31, 5000, 1200)
        dep = st.number_input("Departure delay (min)", 0, 1600, 0)
        arr = st.number_input("Arrival delay (min)", 0, 1600, 0)
    services = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
                "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment",
                "On-board service", "Leg room service", "Baggage handling", "Checkin service",
                "Inflight service", "Cleanliness"]
    ratings = {}
    for i, s in enumerate(services):
        with (c2 if i < 7 else c3):
            ratings[s] = st.slider(s, 0, 5, 4, help="0 = not applicable")
    rec = {"Gender": gender, "Customer Type": ctype, "Age": age, "Type of Travel": travel, "Class": cls,
           "Flight Distance": dist, **ratings, "Departure Delay in Minutes": dep, "Arrival Delay in Minutes": arr}
    if st.button("Predict satisfaction", type="primary"):
        s = time.perf_counter()
        out = predict_airline([rec])[0]
        st.session_state["air"] = (rec, out, (time.perf_counter() - s) * 1000)
    if "air" in st.session_state:
        rec_, out, ms = st.session_state["air"]
        p = out["satisfied_probability"]
        st.metric("Prediction", out["prediction"].title(), f"P(satisfied) = {p:.1%}")
        st.progress(p)
        st.caption(f"Inference latency: {ms:.1f} ms")
        fb = st.radio("Was this prediction right?", ["—", "Yes", "No"], horizontal=True, key="air_fb")
        if fb != "—" and st.button("Send feedback", key="air_send"):
            save_feedback("airline", rec_, out, fb); st.success("Thanks — feedback saved for model improvement.")

# ------------------------------------------------------------------ taxi
PLACES = {"Times Square": (40.7580, -73.9855), "JFK Airport": (40.6413, -73.7781),
          "LaGuardia Airport": (40.7769, -73.8740), "Wall Street": (40.7060, -74.0086),
          "Central Park (Columbus Circle)": (40.7681, -73.9819), "Brooklyn — Williamsburg": (40.7081, -73.9571),
          "Grand Central": (40.7527, -73.9772), "Upper East Side": (40.7736, -73.9566),
          "Penn Station": (40.7506, -73.9935), "Long Island City": (40.7447, -73.9485)}
with tab2:
    st.subheader("How long will this taxi trip take?")
    c1, c2 = st.columns(2)
    with c1:
        src = st.selectbox("Pickup", list(PLACES), index=0)
        dst = st.selectbox("Drop-off", list(PLACES), index=1)
        custom = st.checkbox("Enter custom coordinates")
        plat, plon = PLACES[src]; dlat, dlon = PLACES[dst]
        if custom:
            plat = st.number_input("Pickup latitude", 40.49, 40.92, plat, format="%.4f")
            plon = st.number_input("Pickup longitude", -74.27, -73.68, plon, format="%.4f")
            dlat = st.number_input("Drop-off latitude", 40.49, 40.92, dlat, format="%.4f")
            dlon = st.number_input("Drop-off longitude", -74.27, -73.68, dlon, format="%.4f")
    with c2:
        d = st.date_input("Pickup date", date(2016, 4, 12))
        t = st.time_input("Pickup time", dtime(17, 30))
        pax = st.slider("Passengers", 1, 6, 1)
        vendor = st.radio("Vendor", [1, 2], index=1, horizontal=True)
    st.map(pd.DataFrame({"lat": [plat, dlat], "lon": [plon, dlon]}), size=80)
    trip = {"pickup_datetime": datetime.combine(d, t).strftime("%Y-%m-%d %H:%M:%S"),
            "pickup_latitude": plat, "pickup_longitude": plon, "dropoff_latitude": dlat,
            "dropoff_longitude": dlon, "passenger_count": pax, "vendor_id": vendor, "store_and_fwd_flag": "N"}
    if st.button("Predict trip duration", type="primary"):
        s = time.perf_counter()
        out = predict_taxi([trip])[0]
        st.session_state["taxi"] = (trip, out, (time.perf_counter() - s) * 1000)
    if "taxi" in st.session_state:
        trip_, out, ms = st.session_state["taxi"]
        a, b = st.columns(2)
        a.metric("Predicted duration", f"{out['trip_duration_minutes']:.1f} min")
        b.metric("Straight-line distance", f"{out['distance_km']:.2f} km")
        st.caption(f"Inference latency: {ms:.1f} ms")
        actual = st.number_input("Actual duration if known (min)", 0.0, 300.0, 0.0, key="taxi_actual")
        if st.button("Send feedback", key="taxi_send"):
            save_feedback("taxi", trip_, out, "actual_reported" if actual else "none", actual or "")
            st.success("Thanks — feedback saved for model improvement.")

# ------------------------------------------------------------------ info
with tab3:
    a, b = st.columns(2)
    with a:
        st.markdown("**Airline — tuned LightGBM, test set**")
        st.json(metrics_card("airline"))
        img = ROOT / "reports" / "figures" / "airline_feature_importance.png"
        if img.exists():
            st.image(str(img))
    with b:
        st.markdown("**Taxi — tuned LightGBM, test set**")
        st.json(metrics_card("taxi"))
        img = ROOT / "reports" / "figures" / "taxi_feature_importance.png"
        if img.exists():
            st.image(str(img))
