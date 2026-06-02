from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from smart_shoe_utils import (  # noqa: E402
    DATA_DIR,
    FSR_ZONES,
    MODELS_DIR,
    add_pressure_features,
    align_features,
    calories_from_gps,
    model_feature_frame,
    simulate_running_dataset,
)


st.set_page_config(page_title="Brain-Sole", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background: #050505;
        color: #f5f7fb;
    }
    [data-testid="stHeader"] {
        background: rgba(5, 5, 5, 0.92);
    }
    [data-testid="stSidebar"] {
        background: #0f1117;
    }
    [data-testid="stSidebar"] * {
        color: #f5f7fb;
    }
    .block-container {
        max-width: 1180px;
        padding: 1.8rem 1rem 2rem;
    }
    h1 {
        font-size: 2rem !important;
        line-height: 1.25 !important;
        margin: 0 0 0.55rem !important;
        padding-top: 0.15rem !important;
        overflow: visible !important;
    }
    h2, h3 {
        font-size: 1.05rem !important;
    }
    [data-testid="stMetric"] {
        background: #11141c;
        border: 1px solid #2a3140;
        border-radius: 8px;
        padding: 0.55rem 0.65rem;
        min-height: 82px;
    }
    [data-testid="stMetric"] * {
        color: #f5f7fb;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1rem;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 0.45rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        overflow-x: auto;
        flex-wrap: nowrap;
    }
    .stTabs [data-baseweb="tab"] {
        min-width: fit-content;
        padding: 0.4rem 0.55rem;
        font-size: 0.82rem;
        color: #dbe4f3;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff;
        border-bottom-color: #38bdf8;
    }
    .js-plotly-plot, .stDataFrame, iframe {
        border-radius: 8px;
        overflow: hidden;
    }
    .chat-panel {
        background: #11141c;
        border: 1px solid #2a3140;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.6rem 0 0.9rem;
    }
    .chat-panel * {
        color: #f5f7fb;
    }
    [data-testid="stChatMessage"] {
        background: #171b25;
        border: 1px solid #293244;
        border-radius: 8px;
    }
    @media (max-width: 520px) {
        .block-container {
            max-width: 100%;
            padding-left: 0.55rem;
            padding-right: 0.55rem;
        }
        [data-testid="stSidebar"] {
            min-width: 82vw !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_prediction_module():
    module_path = PROJECT_DIR / "03_predict_and_llm_advice.py"
    spec = importlib.util.spec_from_file_location("predict_and_llm", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@st.cache_data
def load_data() -> pd.DataFrame:
    test_path = DATA_DIR / "test_running_shoe_dataset.csv"
    train_path = DATA_DIR / "train_running_shoe_dataset.csv"
    path = test_path if test_path.exists() else train_path
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return add_pressure_features(df)


@st.cache_resource
def load_artifacts():
    try:
        from tensorflow import keras

        model = keras.models.load_model(MODELS_DIR / "smart_shoe_injury_model.keras")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
        imputer = joblib.load(MODELS_DIR / "imputer.pkl")
        feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
        return model, scaler, label_encoder, imputer, feature_columns
    except Exception:
        return None


def predict_latest(df: pd.DataFrame):
    artifacts = load_artifacts()
    if artifacts is None or df.empty:
        return None
    model, scaler, label_encoder, imputer, feature_columns = artifacts
    latest = df.tail(1)
    X = align_features(model_feature_frame(latest), feature_columns)
    X = pd.DataFrame(imputer.transform(X), columns=feature_columns)
    probabilities = model.predict(scaler.transform(X), verbose=0)[0]
    class_idx = int(np.argmax(probabilities))
    return label_encoder.inverse_transform([class_idx])[0], float(probabilities[class_idx]), probabilities


def pressure_figure(row: pd.Series, side: str) -> go.Figure:
    zones = list(FSR_ZONES.values())
    values = [float(row[f"{side}_S{i}"]) for i in range(1, 7)]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=zones,
            orientation="h",
            marker={"color": values, "colorscale": "YlOrRd", "showscale": True},
        )
    )
    fig.update_layout(height=320, margin=dict(l=8, r=8, t=24, b=8), xaxis_title="Pressure", yaxis_title="")
    return fig


def warning_level(predicted_class: str, probability: float, latest: pd.Series) -> str:
    if predicted_class != "normal" and (probability >= 0.75 or latest["left_right_imbalance"] >= 0.18):
        return "high"
    if predicted_class != "normal" or latest["pressure_variability"] >= 75:
        return "medium"
    return "low"


def route_map(df: pd.DataFrame):
    import folium

    center = [df["latitude"].mean(), df["longitude"].mean()]
    fmap = folium.Map(location=center, zoom_start=14)
    route = df[["latitude", "longitude"]].dropna().values.tolist()
    folium.PolyLine(route, color="#38bdf8", weight=4, opacity=0.85).add_to(fmap)

    risky = df[
        (df["left_right_imbalance"] > 0.15)
        | (df["pressure_variability"] > 85)
        | ((df["speed_kmh"].diff().fillna(0) > 1.2) & (df["avg_forefoot_pressure"] > 520))
    ]
    for _, point in risky.tail(50).iterrows():
        folium.CircleMarker(
            location=[point["latitude"], point["longitude"]],
            radius=5,
            color="#ef4444",
            fill=True,
            fill_opacity=0.85,
            popup=f"Risk: {point['injury_location']}<br>Imbalance: {point['left_right_imbalance']:.2f}",
        ).add_to(fmap)
    return fmap


def make_daily_summary(source_df: pd.DataFrame) -> pd.DataFrame:
    daily = source_df.copy()
    daily["date"] = daily["timestamp"].dt.date
    if daily["date"].nunique() == 1 and len(daily) >= 120:
        start_date = pd.Timestamp(daily["date"].iloc[0])
        daily["date"] = start_date + pd.to_timedelta(np.arange(len(daily)) // 60, unit="D")
        daily["date"] = daily["date"].dt.date

    summary = daily.groupby("date").agg(
        total_distance_m=("distance_m", lambda s: s.max() - s.min()),
        average_speed=("speed_kmh", "mean"),
        average_pace=("pace_min_per_km", "mean"),
        average_cadence=("cadence_spm", "mean"),
        total_calories=("calories", lambda s: s.max() - s.min()),
        pressure_imbalance=("left_right_imbalance", "mean"),
        pressure_variability=("pressure_variability", "mean"),
        common_injury_risk=("injury_location", lambda s: s.mode().iloc[0]),
    ).reset_index()
    summary["total_distance_km"] = summary["total_distance_m"] / 1000
    summary["pace_change"] = summary["average_pace"].diff()
    summary["calories_change"] = summary["total_calories"].diff()
    summary["imbalance_change"] = summary["pressure_imbalance"].diff()
    return summary


def build_chat_prompt(
    question: str,
    latest_row: pd.Series,
    summary: pd.DataFrame,
    predicted_class: str,
    probability: float,
    runner_profile: dict,
) -> str:
    context = {
        "runner_profile": runner_profile,
        "latest_ai_prediction": {
            "injury_risk": predicted_class,
            "confidence": round(float(probability), 4),
            "warning_level": warning_level(predicted_class, probability, latest_row),
        },
        "latest_running_metrics": {
            "speed_kmh": round(float(latest_row["speed_kmh"]), 2),
            "pace_min_per_km": round(float(latest_row["pace_min_per_km"]), 2),
            "cadence_spm": round(float(latest_row["cadence_spm"]), 1),
            "stride_length_m": round(float(latest_row["stride_length_m"]), 2),
            "left_right_imbalance": round(float(latest_row["left_right_imbalance"]), 3),
            "pressure_variability": round(float(latest_row["pressure_variability"]), 2),
            "high_pressure_zone": str(latest_row["high_pressure_zone"]),
            "forefoot_pressure": round(float(latest_row["avg_forefoot_pressure"]), 2),
            "midfoot_pressure": round(float(latest_row["avg_midfoot_pressure"]), 2),
            "heel_pressure": round(float(latest_row["avg_heel_pressure"]), 2),
        },
        "daily_performance_summary": summary.tail(5).to_dict(orient="records"),
    }
    return f"""
คุณคือ Brain แชตบอตผู้ช่วยวิเคราะห์ประสิทธิภาพการวิ่งและแรงกดเท้า
ตอบภาษาไทยให้กระชับ ใช้งานได้จริง และอิงจากข้อมูลที่ให้เท่านั้น
ห้ามวินิจฉัยโรค ให้ใช้คำว่า ความเสี่ยง หรือ ควรพิจารณา
ถ้าผู้ใช้พูดถึงอาการเจ็บ บวม ชา หรือเจ็บมากขึ้น ให้แนะนำหยุดวิ่งและพบผู้เชี่ยวชาญทางการแพทย์

ข้อมูลสำหรับวิเคราะห์:
{json.dumps(context, ensure_ascii=False, default=str, indent=2)}

คำถามผู้ใช้:
{question}
"""


def render_chatbot(
    latest_row: pd.Series,
    summary: pd.DataFrame,
    predicted_class: str,
    probability: float,
    runner_profile: dict,
) -> None:
    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
    st.subheader("Brain Chat")
    prediction_module = load_prediction_module()
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "สวัสดีครับ ผมช่วยวิเคราะห์ประสิทธิภาพการวิ่ง แรงกดเท้า pace cadence และความเสี่ยงจากข้อมูลรองเท้าอัจฉริยะได้",
            }
        ]

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("ถามเรื่องฟอร์มการวิ่ง ประสิทธิภาพ หรือแรงกดเท้า")
    if user_question:
        st.session_state.chat_messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        prompt = build_chat_prompt(
            user_question,
            latest_row,
            summary,
            str(predicted_class),
            probability,
            runner_profile,
        )
        answer = prediction_module.call_llm(prompt)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
    st.markdown("</div>", unsafe_allow_html=True)


st.title("Brain-Sole")

df = load_data()
if df.empty:
    st.warning("No dataset found. Run 01_simulate_train_dataset.py or 04_simulate_test_dataset.py first.")
    st.stop()

with st.sidebar:
    st.header("Runner Inputs")
    weight_kg = st.number_input("Body weight (kg)", min_value=35.0, max_value=160.0, value=68.0, step=1.0)
    age = st.number_input("Age", min_value=10, max_value=90, value=30, step=1)
    gender = st.selectbox("Gender", ["Not specified", "Female", "Male", "Non-binary"])
    duration_min = st.number_input("Running duration (minutes)", min_value=1, max_value=300, value=45, step=5)
    row_count = st.slider("Rows to display", min_value=30, max_value=len(df), value=min(180, len(df)), step=10)
    st.header("Simulation")
    sim_minutes = st.number_input("Simulated minutes", min_value=60, max_value=2400, value=420, step=60)
    sim_seed = st.number_input("Random seed", min_value=1, max_value=999999, value=2026, step=1)
    if st.button("Generate running data", use_container_width=True):
        with st.spinner("Simulating GPS and FSR sensor data..."):
            simulated_df = simulate_running_dataset(
                n_minutes=int(sim_minutes),
                seed=int(sim_seed),
                weight_kg=float(weight_kg),
            )
            output_path = DATA_DIR / "test_running_shoe_dataset.csv"
            simulated_df.to_csv(output_path, index=False)
            load_data.clear()
            st.session_state["simulation_message"] = (
                f"Generated {len(simulated_df):,} rows and saved to {output_path.name}"
            )
        st.rerun()
    if "simulation_message" in st.session_state:
        st.success(st.session_state["simulation_message"])
    st.header("LLM")
    llm_provider = st.selectbox("Provider", ["placeholder", "phratumma", "ollama", "openai"])
    phratumma_url = st.text_input("Phratumma URL", value=os.getenv("PHRATUMMA_URL", "http://localhost:8000/v1/chat/completions"))
    phratumma_model = st.text_input("Phratumma model", value=os.getenv("PHRATUMMA_MODEL", "phratumma"))

os.environ["LLM_PROVIDER"] = llm_provider
if llm_provider == "phratumma":
    os.environ["PHRATUMMA_URL"] = phratumma_url
    os.environ["PHRATUMMA_MODEL"] = phratumma_model

view_df = df.tail(row_count).copy()
latest = view_df.iloc[-1]
distance_km = float(view_df["distance_m"].iloc[-1] - view_df["distance_m"].iloc[0]) / 1000
calories_estimate = calories_from_gps(weight_kg, max(distance_km, 0))

prediction = predict_latest(view_df)
if prediction:
    predicted_class, probability, _ = prediction
else:
    predicted_class, probability = latest.get("injury_location", "model_not_trained"), 0.0

level = warning_level(str(predicted_class), probability, latest)
daily_summary = make_daily_summary(df)
cols = st.columns(5)
cols[0].metric("Predicted risk", str(predicted_class))
cols[1].metric("Confidence", f"{probability:.1%}" if probability else "N/A")
cols[2].metric("Warning", level)
cols[3].metric("GPS calories", f"{calories_estimate:.0f} kcal")
cols[4].metric("Duration input", f"{duration_min} min")

runner_profile = {
    "weight_kg": weight_kg,
    "age": age,
    "gender": gender,
    "running_duration_min": duration_min,
    "display_distance_km": round(distance_km, 2),
    "estimated_calories": round(calories_estimate, 1),
}

tab_graphs, tab_pressure, tab_daily, tab_map, tab_chatbot = st.tabs(
    ["Running Data", "Foot Pressure", "Daily Summary", "GPS Route", "Chatbot"]
)

with tab_graphs:
    left, right = st.columns(2)
    left.plotly_chart(px.line(view_df, x="timestamp", y="speed_kmh", title="Speed over time"), use_container_width=True)
    right.plotly_chart(px.line(view_df, x="timestamp", y="pace_min_per_km", title="Pace over time"), use_container_width=True)
    left.plotly_chart(px.line(view_df, x="timestamp", y="cadence_spm", title="Cadence over time"), use_container_width=True)
    right.plotly_chart(px.line(view_df, x="timestamp", y="calories", title="Calories over time"), use_container_width=True)
    pressure_long = view_df.melt(
        id_vars="timestamp",
        value_vars=["left_total_pressure", "right_total_pressure"],
        var_name="foot",
        value_name="pressure",
    )
    left.plotly_chart(px.line(pressure_long, x="timestamp", y="pressure", color="foot", title="Left/right pressure over time"), use_container_width=True)
    right.plotly_chart(px.line(view_df, x="timestamp", y="pressure_variability", title="Pressure variability over time"), use_container_width=True)

with tab_pressure:
    st.subheader("AI Injury Recommendation")
    prediction_module = load_prediction_module()
    previous = view_df.iloc[-11] if len(view_df) > 10 else None
    prompt_prediction = {"predicted_class": str(predicted_class), "probability": float(probability)}
    advice_prompt = prediction_module.create_llm_prompt(latest, previous, prompt_prediction)
    st.info(prediction_module.call_llm(advice_prompt))

    high_zone = latest["high_pressure_zone"]
    st.metric("Highest pressure zone", high_zone)
    left, right = st.columns(2)
    left.plotly_chart(pressure_figure(latest, "left"), use_container_width=True)
    right.plotly_chart(pressure_figure(latest, "right"), use_container_width=True)

with tab_daily:
    st.dataframe(daily_summary, use_container_width=True)
    chart_left, chart_right = st.columns(2)
    chart_left.plotly_chart(px.bar(daily_summary, x="date", y="total_distance_km", title="Daily distance comparison"), use_container_width=True)
    chart_right.plotly_chart(px.bar(daily_summary, x="date", y="total_calories", title="Daily calories comparison"), use_container_width=True)
    chart_left.plotly_chart(px.line(daily_summary, x="date", y=["average_speed", "average_cadence"], markers=True, title="Speed and cadence trend"), use_container_width=True)
    chart_right.plotly_chart(px.line(daily_summary, x="date", y="average_pace", markers=True, title="Pace trend"), use_container_width=True)
    chart_left.plotly_chart(px.line(daily_summary, x="date", y=["pressure_imbalance", "pressure_variability"], markers=True, title="Pressure risk trend"), use_container_width=True)
    risk_counts = daily_summary.groupby(["date", "common_injury_risk"]).size().reset_index(name="count")
    chart_right.plotly_chart(px.bar(risk_counts, x="date", y="count", color="common_injury_risk", title="Daily injury risk trend"), use_container_width=True)

    if len(daily_summary) >= 2:
        today = daily_summary.iloc[-1]
        previous_day = daily_summary.iloc[-2]
        c1, c2, c3 = st.columns(3)
        c1.metric("Calories difference", f"{today['total_calories'] - previous_day['total_calories']:.1f} kcal")
        c2.metric("Pace difference", f"{today['average_pace'] - previous_day['average_pace']:.2f} min/km")
        c3.metric("Imbalance difference", f"{today['pressure_imbalance'] - previous_day['pressure_imbalance']:.3f}")

with tab_map:
    st_folium(route_map(view_df), use_container_width=True, height=520)

with tab_chatbot:
    render_chatbot(latest, daily_summary, str(predicted_class), probability, runner_profile)

st.caption(
    f"Runner profile: age {age}, gender {gender}. This prototype estimates injury risk; it is not a medical diagnosis."
)
