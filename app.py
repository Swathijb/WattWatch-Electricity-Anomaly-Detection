from pathlib import Path
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "isolation_forest_model.joblib"
CONFIG_PATH = ARTIFACT_DIR / "model_config.joblib"
DEFAULT_THRESHOLD = -0.081625  # selected 2%-target-FPR candidate from synthetic benchmark

st.set_page_config(page_title="WattWatch | Electricity Anomaly Detection", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp{background:#0b1220;color:#edf2f7}
[data-testid="stSidebar"]{background:#111b2e;border-right:1px solid #283951}
.block-container{padding-top:1.5rem;max-width:1450px}
.hero{padding:1.5rem 1.7rem;border-radius:20px;background:linear-gradient(120deg,#14253e,#123b46);border:1px solid #2b5363;margin-bottom:1.2rem}
.eyebrow{color:#68e0c0;font-size:.78rem;font-weight:700;letter-spacing:.14em}
.hero h1{margin:.35rem 0;color:#f4f8ff;font-size:2.1rem}
.hero p{margin:0;color:#c2d2e5}
.metric{background:#121e31;border:1px solid #283951;border-radius:15px;padding:1rem;min-height:105px}
.label{color:#9eb0c8;font-size:.82rem}.value{color:#f4f8ff;font-size:1.65rem;font-weight:750;margin-top:.3rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
        raise FileNotFoundError("Copy isolation_forest_model.joblib and model_config.joblib into the artifacts/ folder.")
    return joblib.load(MODEL_PATH), joblib.load(CONFIG_PATH)

def read_upload(upload):
    raw = upload.getvalue()
    if not raw:
        raise ValueError("The uploaded file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", na_values=["?", "NA", "NaN", ""])
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), sep=";", na_values=["?", "NA", "NaN", ""])
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def make_features(data):
    # Mirrors make_features() in the user's original Colab notebook.
    x = pd.DataFrame(index=data.index)
    power = data["Global_active_power"].astype(float)
    x["power"] = power
    x["voltage"] = data["Voltage"].astype(float) if "Voltage" in data else np.nan
    x["intensity"] = data["Global_intensity"].astype(float) if "Global_intensity" in data else np.nan
    for col in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        if col in data.columns:
            x[col.lower()] = data[col].astype(float)
    x["hour"] = data.index.hour
    x["day_of_week"] = data.index.dayofweek
    x["is_weekend"] = (data.index.dayofweek >= 5).astype(int)
    past = power.shift(1)
    x["rolling_mean_6h"] = past.rolling(6, min_periods=3).mean()
    x["rolling_std_6h"] = past.rolling(6, min_periods=3).std()
    x["rolling_mean_24h"] = past.rolling(24, min_periods=12).mean()
    x["rolling_std_24h"] = past.rolling(24, min_periods=12).std()
    x["change_from_previous"] = power.diff().abs()
    x["deviation_from_24h_mean"] = (power - x["rolling_mean_24h"]).abs()
    return x.replace([np.inf, -np.inf], np.nan)

def prepare_data(upload_df, feature_columns):
    d = upload_df.copy()
    d.columns = [str(c).strip().replace("\ufeff", "") for c in d.columns]
    if "datetime" in d.columns:
        d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
        d = d.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")
    elif {"Date", "Time"}.issubset(d.columns):
        d["datetime"] = pd.to_datetime(d["Date"].astype(str) + " " + d["Time"].astype(str),
                                       dayfirst=True, errors="coerce")
        d = d.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")
        d = d.drop(columns=["Date", "Time"], errors="ignore")
    else:
        raise ValueError("File must contain a 'datetime' column or both 'Date' and 'Time' columns.")
    if "Global_active_power" not in d.columns:
        raise ValueError("Missing required column 'Global_active_power'. Use the original UCI dataset format.")
    source_for_feature = {"voltage":"Voltage", "intensity":"Global_intensity",
                          "sub_metering_1":"Sub_metering_1", "sub_metering_2":"Sub_metering_2",
                          "sub_metering_3":"Sub_metering_3"}
    missing = [src for feat, src in source_for_feature.items()
               if feat in feature_columns and src not in d.columns]
    if missing:
        raise ValueError("Your saved model expects these sensor columns: " + ", ".join(missing) +
                         ". Upload the original UCI data or a CSV with the same sensor columns.")
    for col in d.columns:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    agg = {col: ("sum" if col in ["Sub_metering_1","Sub_metering_2","Sub_metering_3"] else "mean")
           for col in d.columns}
    d = d.resample("1h").agg(agg).interpolate(method="time", limit=3)
    d = d.dropna(subset=["Global_active_power"])
    x = make_features(d).reindex(columns=feature_columns)
    valid = x.notna().all(axis=1)
    x = x.loc[valid]
    d = d.loc[x.index]
    if x.empty:
        raise ValueError("No rows remain after feature engineering. Upload several days of consecutive readings for rolling features.")
    return d, x

def metric(label, value, note):
    st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="label">{note}</div></div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="eyebrow">SMART ENERGY ANALYTICS · FAI CASE STUDY</div>
<h1>⚡ WattWatch</h1>
<p>Unusual household electricity consumption detection using your saved Isolation Forest model.</p>
</div>
""", unsafe_allow_html=True)

try:
    model, config = load_model()
except Exception as e:
    st.error("The model files could not be loaded.")
    st.markdown("""Expected structure:
```
WattWatch_Streamlit_UI/
├── app.py
├── requirements.txt
└── artifacts/
    ├── isolation_forest_model.joblib
    └── model_config.joblib
```""")
    st.exception(e)
    st.stop()

feature_columns = config.get("feature_columns")
if not feature_columns:
    st.error("model_config.joblib does not contain feature_columns. Re-save the config from your original notebook.")
    st.stop()

st.sidebar.header("Detection settings")
threshold = st.sidebar.number_input("Anomaly-score threshold", min_value=-1.0, max_value=1.0,
                                    value=float(DEFAULT_THRESHOLD), step=0.001, format="%.6f",
                                    help="Scores >= this threshold are flagged.")
st.sidebar.caption(f"Model: {config.get('algorithm','IsolationForest')}")
st.sidebar.caption(f"Resampling: {config.get('resampling','hourly')} · Features: {len(feature_columns)}")
st.sidebar.warning("The default threshold came from a controlled synthetic benchmark. The real-world false-positive rate is unknown.")

c1, c2 = st.columns([1.5,1])
with c1:
    st.subheader("Analyse household readings")
    st.caption("Upload a CSV/TXT with timestamps and the sensor columns expected by the trained model.")
    upload = st.file_uploader("Choose household data", type=["csv","txt"])
with c2:
    st.subheader("Interpretation")
    st.info("A flagged observation is unusual relative to learned patterns. It is not proof of an electrical fault, theft, or a specific cause.")

if upload is None:
    st.markdown("---")
    st.subheader("Ready for analysis")
    st.write("Upload the original UCI Individual Household Electric Power Consumption dataset, or a compatible CSV. The app reuses your saved model and does not retrain it.")
    st.markdown("**Required:** `datetime` or `Date` + `Time`, `Global_active_power`, and the sensor columns required by your saved model (`Voltage`, `Global_intensity`, `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`). Include several days of consecutive readings.")
    st.stop()

try:
    with st.spinner("Preparing hourly features and scoring readings…"):
        uploaded_df = read_upload(upload)
        hourly, features = prepare_data(uploaded_df, feature_columns)
        scores = -model.decision_function(features)
        flags = scores >= threshold
        results = pd.DataFrame({
            "timestamp": features.index,
            "Global_active_power": hourly["Global_active_power"].astype(float).values,
            "anomaly_score": scores,
            "is_anomaly": flags
        })
        results["status"] = np.where(results["is_anomaly"], "Anomaly", "Normal")
except Exception as e:
    st.error("Could not analyse this file.")
    st.warning(str(e))
    st.stop()

n = len(results)
n_flagged = int(results["is_anomaly"].sum())
rate = float(results["is_anomaly"].mean()) if n else 0
mean_power = float(results["Global_active_power"].mean()) if n else 0
peak_power = float(results["Global_active_power"].max()) if n else 0

st.markdown("---")
st.subheader("Detection overview")
cols = st.columns(4)
with cols[0]: metric("Hourly observations", f"{n:,}", "Scored rows")
with cols[1]: metric("Flagged intervals", f"{n_flagged:,}", "Potential anomalies")
with cols[2]: metric("Alert rate", f"{rate:.2%}", "Share of scored hours")
with cols[3]: metric("Average active power", f"{mean_power:.3f} kW", f"Peak: {peak_power:.3f} kW")
st.caption(f"Threshold: {threshold:.6f} · Period: {results['timestamp'].min()} to {results['timestamp'].max()}. Alert rate is descriptive, not a measured false-positive rate.")

left, right = st.columns([1.6,1])
with left:
    st.subheader("Consumption timeline")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results["timestamp"], y=results["Global_active_power"], mode="lines",
                             name="Active power", line=dict(color="#55c8c0", width=1.5)))
    flagged = results[results["is_anomaly"]]
    fig.add_trace(go.Scatter(x=flagged["timestamp"], y=flagged["Global_active_power"], mode="markers",
                             name="Flagged", marker=dict(color="#ff6b6b", size=8),
                             customdata=flagged["anomaly_score"],
                             hovertemplate="%{x}<br>Power: %{y:.3f} kW<br>Score: %{customdata:.5f}<extra>Flagged</extra>"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=390, xaxis_title="Timestamp", yaxis_title="Active power (kW)",
                      margin=dict(l=10,r=10,t=15,b=10), legend=dict(orientation="h",y=1.1))
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Anomaly score distribution")
    hist = px.histogram(results, x="anomaly_score", color="status", nbins=35, barmode="overlay",
                        opacity=.82, color_discrete_map={"Normal":"#4e8bd6","Anomaly":"#ff6b6b"})
    hist.add_vline(x=threshold, line_dash="dash", line_color="#f4c95d", annotation_text="Threshold")
    hist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       height=390, margin=dict(l=10,r=10,t=15,b=10),
                       xaxis_title="Anomaly score (higher = more unusual)", yaxis_title="Hourly observations")
    st.plotly_chart(hist, use_container_width=True)

st.markdown("---")
st.subheader("Detection results")
only_flagged = st.checkbox("Show only flagged observations", value=True)
table = results[results["is_anomaly"]] if only_flagged else results
table = table.sort_values("anomaly_score", ascending=False)
st.dataframe(table, use_container_width=True, hide_index=True,
             column_config={"timestamp":st.column_config.DatetimeColumn("Timestamp",format="YYYY-MM-DD HH:mm"),
                            "Global_active_power":st.column_config.NumberColumn("Active power (kW)",format="%.3f"),
                            "anomaly_score":st.column_config.NumberColumn("Anomaly score",format="%.6f"),
                            "is_anomaly":st.column_config.CheckboxColumn("Flagged")})
st.download_button("⬇ Download all results as CSV", results.to_csv(index=False).encode("utf-8"),
                   file_name="electricity_anomaly_detection_results.csv", mime="text/csv", type="primary")
st.caption("The original dataset has no verified anomaly labels. Metrics such as precision and recall cannot be calculated for uploaded real data without ground-truth labels.")
