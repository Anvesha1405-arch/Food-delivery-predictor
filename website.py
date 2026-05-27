import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
import random
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍕 DeliverIQ – Smart Delivery Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Google Fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ---- Root Variables ---- */
:root {
    --bg-dark:      #0d0f1a;
    --bg-card:      #161929;
    --bg-card2:     #1e2235;
    --accent1:      #ff6b35;
    --accent2:      #f7c59f;
    --accent3:      #4cc9f0;
    --accent4:      #7b2d8b;
    --green:        #2ecc71;
    --yellow:       #f1c40f;
    --red:          #e74c3c;
    --text:         #e8eaf0;
    --text-muted:   #8b93b0;
    --border:       rgba(255,107,53,0.18);
    --glow:         rgba(255,107,53,0.35);
}

/* ---- Base ---- */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: var(--bg-dark);
    color: var(--text);
}

/* Remove default Streamlit padding */
.block-container { padding: 1.5rem 2rem 2rem; }

/* ---- Hero Banner ---- */
.hero {
    background: linear-gradient(135deg, #1a0533 0%, #0d1b3e 40%, #0d0f1a 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(255,107,53,0.12), inset 0 1px 0 rgba(255,255,255,0.05);
}
.hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(255,107,53,0.25) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute; bottom: -80px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(76,201,240,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(90deg, #ff6b35, #f7c59f, #4cc9f0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
}
.hero-sub {
    color: var(--text-muted); font-size: 1rem; font-weight: 300;
    margin: 0; letter-spacing: 0.5px;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,107,53,0.15);
    border: 1px solid rgba(255,107,53,0.4);
    color: var(--accent1);
    font-size: 0.72rem; font-weight: 600;
    padding: 4px 12px; border-radius: 20px;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 0.8rem;
}

/* ---- Cards ---- */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: border-color 0.3s, box-shadow 0.3s;
}
.card:hover {
    border-color: rgba(255,107,53,0.4);
    box-shadow: 0 8px 40px rgba(255,107,53,0.1);
}
.card-title {
    font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: var(--accent1);
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 1.2rem;
}

/* ---- Result Box ---- */
.result-box {
    background: linear-gradient(135deg, #1a2744 0%, #0f1e38 100%);
    border: 2px solid rgba(76,201,240,0.4);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(76,201,240,0.12), inset 0 1px 0 rgba(255,255,255,0.05);
    animation: fadeIn 0.6s ease;
}
.result-time {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 5rem; font-weight: 800;
    background: linear-gradient(90deg, #4cc9f0, #7b2d8b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1; margin: 0.3rem 0;
}
.result-label {
    color: var(--text-muted); font-size: 0.85rem;
    text-transform: uppercase; letter-spacing: 2px;
    margin-bottom: 0.5rem;
}
.result-sub {
    color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;
}

/* ---- Speed Badge ---- */
.speed-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 8px 20px; border-radius: 30px; font-weight: 700;
    font-size: 0.9rem; margin-top: 1rem; letter-spacing: 0.5px;
}
.speed-fast   { background: rgba(46,204,113,0.15); border: 1px solid rgba(46,204,113,0.5); color: #2ecc71; }
.speed-avg    { background: rgba(241,196,15,0.15);  border: 1px solid rgba(241,196,15,0.5);  color: #f1c40f; }
.speed-slow   { background: rgba(231,76,60,0.15);  border: 1px solid rgba(231,76,60,0.5);  color: #e74c3c; }

/* ---- Stat Tiles ---- */
.stat-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-top: 1.5rem; }
.stat-tile {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem;
    text-align: center;
}
.stat-val  { font-size: 1.5rem; font-weight: 700; color: var(--accent3); }
.stat-key  { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }

/* ---- Timeline / Progress Bar ---- */
.timeline { margin: 1.5rem 0 0.5rem; }
.tl-bar-wrap { background: rgba(255,255,255,0.06); border-radius: 8px; height: 10px; overflow: hidden; }
.tl-bar { height: 10px; border-radius: 8px;
    background: linear-gradient(90deg, #ff6b35, #f7c59f, #4cc9f0);
    transition: width 0.8s cubic-bezier(.4,0,.2,1); }
.tl-labels { display: flex; justify-content: space-between;
    font-size: 0.68rem; color: var(--text-muted); margin-top: 4px; }

/* ---- Sidebar styling ---- */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ---- Streamlit widgets overrides ---- */
div[data-baseweb="slider"] > div { background: rgba(255,107,53,0.2) !important; }
div[data-baseweb="slider"] [role="slider"] {
    background: var(--accent1) !important;
    box-shadow: 0 0 12px rgba(255,107,53,0.6) !important;
}
div[data-baseweb="select"] > div {
    background: var(--bg-card2) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
}
.stButton > button {
    width: 100%; font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem; font-weight: 700; letter-spacing: 1px;
    background: linear-gradient(135deg, #ff6b35 0%, #e84393 100%);
    color: white; border: none; border-radius: 12px;
    padding: 0.75rem; cursor: pointer;
    box-shadow: 0 0 24px rgba(255,107,53,0.4);
    transition: all 0.3s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 32px rgba(255,107,53,0.6);
}

/* ---- Animations ---- */
@keyframes fadeIn { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:none; } }
@keyframes pulse  { 0%,100% { opacity:1; } 50% { opacity:0.5; } }

/* ---- Factor list ---- */
.factor-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; border-radius: 8px; margin-bottom: 6px;
    background: var(--bg-card2); border: 1px solid rgba(255,255,255,0.05);
    font-size: 0.85rem;
}
.factor-icon { font-size: 1.1rem; }
.factor-impact { margin-left: auto; font-weight: 700; font-size: 0.8rem; }
.impact-pos { color: #e74c3c; }
.impact-neg { color: #2ecc71; }
.impact-neu { color: var(--text-muted); }

/* ---- Fun facts ticker ---- */
.ticker {
    background: rgba(255,107,53,0.08); border: 1px solid rgba(255,107,53,0.2);
    border-radius: 8px; padding: 0.6rem 1rem;
    font-size: 0.8rem; color: var(--text-muted);
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ── Load assets ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model")

@st.cache_data
def load_data():
    return pd.read_csv("Food_Delivery_Times.csv")

model = load_model()
df = load_data()

# ── Helper functions ────────────────────────────────────────────────────────────
def build_input(distance, weather, traffic, time_of_day, vehicle, prep_time, experience):
    cols = model.feature_names_in_
    row = {c: 0 for c in cols}
    row['Distance_km']             = distance
    row['Preparation_Time_min']    = prep_time
    row['Courier_Experience_yrs']  = experience
    # one-hot
    wmap = {'Clear':'Weather_Clear','Foggy':'Weather_Foggy','Rainy':'Weather_Rainy',
            'Snowy':'Weather_Snowy','Windy':'Weather_Windy'}
    tmap = {'High':'Traffic_Level_High','Low':'Traffic_Level_Low','Medium':'Traffic_Level_Medium'}
    dmap = {'Afternoon':'Time_of_Day_Afternoon','Evening':'Time_of_Day_Evening',
            'Morning':'Time_of_Day_Morning','Night':'Time_of_Day_Night'}
    vmap = {'Bike':'Vehicle_Type_Bike','Car':'Vehicle_Type_Car','Scooter':'Vehicle_Type_Scooter'}
    for mp, val in [(wmap,weather),(tmap,traffic),(dmap,time_of_day),(vmap,vehicle)]:
        k = mp.get(val)
        if k and k in row: row[k] = 1
    return pd.DataFrame([row])[cols]

def speed_category(mins):
    if mins < 40:   return "⚡ Lightning Fast", "speed-fast"
    elif mins < 65: return "✅ On Schedule",    "speed-avg"
    else:           return "🐢 Running Late",   "speed-slow"

def get_progress(mins):
    return min(int((mins / 120) * 100), 100)

WEATHER_ICONS = {'Clear':'☀️','Foggy':'🌫️','Rainy':'🌧️','Snowy':'❄️','Windy':'💨'}
VEHICLE_ICONS = {'Scooter':'🛵','Bike':'🚲','Car':'🚗'}
TRAFFIC_ICONS = {'Low':'🟢','Medium':'🟡','High':'🔴'}
TIME_ICONS    = {'Morning':'🌅','Afternoon':'🌤️','Evening':'🌆','Night':'🌙'}

FUN_FACTS = [
    "🍕 On average, pizza is the most ordered food for delivery worldwide.",
    "🚴 Bike couriers can be faster than cars in dense city traffic.",
    "🌧️ Rainy days see a 30% spike in food delivery orders globally.",
    "⭐ Couriers with 5+ years experience are 20% faster on average.",
    "📍 Most deliveries travel less than 5 km from the restaurant.",
]

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:3rem;'>🛵</div>
        <div style='font-family:"Space Grotesk",sans-serif; font-size:1.2rem; font-weight:700;
                    background:linear-gradient(90deg,#ff6b35,#4cc9f0);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            DeliverIQ
        </div>
        <div style='font-size:0.7rem; color:#8b93b0; text-transform:uppercase; letter-spacing:2px; margin-top:2px;'>
            Prediction Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-title">🗺️ ROUTE DETAILS</div>', unsafe_allow_html=True)
    distance = st.slider("Distance (km)", 0.5, 20.0, 8.0, 0.1)
    prep_time = st.slider("Preparation Time (min)", 5, 45, 15, 1)
    experience = st.slider("Courier Experience (years)", 0, 10, 4, 1)

    st.markdown('<div class="card-title" style="margin-top:1rem;">🌤️ CONDITIONS</div>', unsafe_allow_html=True)
    weather    = st.selectbox("Weather",       ['Clear','Foggy','Rainy','Snowy','Windy'],   format_func=lambda x: f"{WEATHER_ICONS[x]}  {x}")
    traffic    = st.selectbox("Traffic Level", ['Low','Medium','High'],                     format_func=lambda x: f"{TRAFFIC_ICONS[x]}  {x}")
    time_of_day= st.selectbox("Time of Day",   ['Morning','Afternoon','Evening','Night'],   format_func=lambda x: f"{TIME_ICONS[x]}  {x}")
    vehicle    = st.selectbox("Vehicle Type",  ['Scooter','Bike','Car'],                    format_func=lambda x: f"{VEHICLE_ICONS[x]}  {x}")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔮  PREDICT DELIVERY TIME", use_container_width=True)

    # Mini dataset stats
    st.markdown("""
    <div style='margin-top:2rem; padding-top:1rem; border-top:1px solid rgba(255,107,53,0.15);'>
    <div style='font-size:0.7rem; color:#8b93b0; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.8rem;'>📊 Dataset Overview</div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class='stat-tile' style='margin-bottom:6px;'>
        <div class='stat-val'>{len(df):,}</div>
        <div class='stat-key'>Total Orders</div>
    </div>
    <div class='stat-tile' style='margin-bottom:6px;'>
        <div class='stat-val'>{int(df.Delivery_Time_min.mean())} min</div>
        <div class='stat-key'>Avg Delivery</div>
    </div>
    <div class='stat-tile'>
        <div class='stat-val'>{int(df.Distance_km.mean())} km</div>
        <div class='stat-key'>Avg Distance</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN CONTENT ────────────────────────────────────────────────────────────────
# Hero
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>🤖 AI-Powered Prediction</div>
    <div class='hero-title'>🛵 DeliverIQ</div>
    <p class='hero-sub'>Real-time food delivery time predictions using machine learning &mdash; tune the inputs on the left and hit Predict.</p>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout
col_left, col_right = st.columns([1.1, 1], gap="large")

# ── LEFT: Prediction result or placeholder ─────────────────────────────────────
with col_left:
    if predict_btn or "last_pred" in st.session_state:
        if predict_btn:
            with st.spinner(""):
                time.sleep(0.7)  # dramatic pause
            X = build_input(distance, weather, traffic, time_of_day, vehicle, prep_time, experience)
            pred = float(model.predict(X)[0])
            pred = max(5, pred)
            st.session_state["last_pred"] = pred
            st.session_state["last_inputs"] = {
                "distance": distance, "weather": weather, "traffic": traffic,
                "time_of_day": time_of_day, "vehicle": vehicle,
                "prep_time": prep_time, "experience": experience
            }
        else:
            pred = st.session_state["last_pred"]

        mins_val = int(round(pred))
        label, badge_cls = speed_category(pred)
        progress = get_progress(pred)

        range_lo = max(5, mins_val - 5)
        range_hi = mins_val + 8

        st.markdown(f"""
        <div class='result-box'>
            <div class='result-label'>Estimated Delivery Time</div>
            <div class='result-time'>{mins_val}<span style='font-size:2rem; color:#4cc9f0;'> min</span></div>
            <div class='result-sub'>Likely range: {range_lo}–{range_hi} minutes</div>
            <div style='margin-top:0.8rem;'>
                <span class='speed-badge {badge_cls}'>{label}</span>
            </div>

            <div class='timeline' style='margin-top:1.8rem; text-align:left;'>
                <div style='font-size:0.72rem; color:#8b93b0; text-transform:uppercase;
                            letter-spacing:1.5px; margin-bottom:6px;'>Delivery Urgency Scale</div>
                <div class='tl-bar-wrap'>
                    <div class='tl-bar' style='width:{progress}%;'></div>
                </div>
                <div class='tl-labels'>
                    <span>0 min</span><span>30</span><span>60</span><span>90</span><span>120+</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Breakdown stats
        st.markdown(f"""
        <div class='stat-grid' style='margin-top:1.2rem;'>
            <div class='stat-tile'>
                <div class='stat-val'>{distance} km</div>
                <div class='stat-key'>Distance</div>
            </div>
            <div class='stat-tile'>
                <div class='stat-val'>{prep_time} min</div>
                <div class='stat-key'>Prep Time</div>
            </div>
            <div class='stat-tile'>
                <div class='stat-val'>{experience} yr</div>
                <div class='stat-key'>Experience</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding: 4rem 2rem; background:var(--bg-card);
                    border: 2px dashed rgba(255,107,53,0.25); border-radius:20px;'>
            <div style='font-size:4rem; margin-bottom:1rem;'>🔮</div>
            <div style='font-size:1.1rem; font-weight:600; color:#e8eaf0;'>Ready to Predict</div>
            <div style='color:#8b93b0; font-size:0.85rem; margin-top:0.5rem;'>
                Configure your order details in the sidebar<br>and hit <b style='color:#ff6b35;'>Predict Delivery Time</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── RIGHT: Factor analysis & insights ─────────────────────────────────────────
with col_right:
    # Impact factors
    st.markdown("""
    <div class='card'>
        <div class='card-title'>🧠 Factor Impact Analysis</div>
    """, unsafe_allow_html=True)

    # Build impact cards dynamically
    w_impact = {"Clear": "−3 min", "Foggy": "+5 min", "Rainy": "+8 min", "Snowy": "+12 min", "Windy": "+4 min"}
    w_cls    = {"Clear": "impact-neg", "Foggy": "impact-pos", "Rainy": "impact-pos", "Snowy": "impact-pos", "Windy": "impact-pos"}
    t_impact = {"Low": "−5 min", "Medium": "Neutral", "High": "+10 min"}
    t_cls    = {"Low": "impact-neg", "Medium": "impact-neu", "High": "impact-pos"}
    dist_mins = f"+{int(distance * 2.5)} min (est)"

    factors = [
        (WEATHER_ICONS[weather],    f"Weather: {weather}",         w_impact[weather],   w_cls[weather]),
        (TRAFFIC_ICONS[traffic],    f"Traffic: {traffic}",         t_impact[traffic],   t_cls[traffic]),
        (VEHICLE_ICONS[vehicle],    f"Vehicle: {vehicle}",         "Varies",            "impact-neu"),
        (TIME_ICONS[time_of_day],   f"Time: {time_of_day}",        "Seasonal",          "impact-neu"),
        ("📍",                      f"Distance: {distance} km",    dist_mins,           "impact-pos"),
        ("👨‍🍳",                     f"Prep Time: {prep_time} min", f"+{prep_time} min",  "impact-pos"),
        ("🏅",                      f"Courier Exp: {experience}y", f"−{experience} min","impact-neg"),
    ]
    html_factors = ""
    for icon, label, impact, cls in factors:
        html_factors += f"""
        <div class='factor-item'>
            <span class='factor-icon'>{icon}</span>
            <span>{label}</span>
            <span class='factor-impact {cls}'>{impact}</span>
        </div>"""
    st.markdown(html_factors + "</div>", unsafe_allow_html=True)

    # Historical comparison
    st.markdown("""
    <div class='card'>
        <div class='card-title'>📈 Historical Comparison</div>
    """, unsafe_allow_html=True)

    similar = df[
        (df.Weather == weather) &
        (df.Traffic_Level == traffic) &
        (df.Vehicle_Type == vehicle)
    ]['Delivery_Time_min']

    if len(similar) > 0:
        avg_similar = similar.mean()
        median_sim  = similar.median()
        count_sim   = len(similar)
        st.markdown(f"""
        <div class='stat-grid'>
            <div class='stat-tile'>
                <div class='stat-val'>{int(avg_similar)}</div>
                <div class='stat-key'>Avg (min)</div>
            </div>
            <div class='stat-tile'>
                <div class='stat-val'>{int(median_sim)}</div>
                <div class='stat-key'>Median</div>
            </div>
            <div class='stat-tile'>
                <div class='stat-val'>{count_sim}</div>
                <div class='stat-key'>Samples</div>
            </div>
        </div>
        <div style='margin-top:1rem; font-size:0.8rem; color:#8b93b0;'>
            Based on <b style='color:#e8eaf0;'>{count_sim}</b> similar orders with
            <b style='color:#e8eaf0;'>{WEATHER_ICONS[weather]} {weather}</b> weather,
            <b style='color:#e8eaf0;'>{TRAFFIC_ICONS[traffic]} {traffic}</b> traffic,
            via <b style='color:#e8eaf0;'>{VEHICLE_ICONS[vehicle]} {vehicle}</b>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#8b93b0; font-size:0.85rem;'>No matching historical data.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Fun fact ticker
    fact = random.choice(FUN_FACTS)
    st.markdown(f"""
    <div class='ticker'>
        💡 <b>Did you know?</b> {fact}
    </div>
    """, unsafe_allow_html=True)

# ── Bottom: Full distribution chart using Streamlit native ────────────────────
st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>📊 Delivery Time Distribution by Condition</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🌤️ By Weather", "🚦 By Traffic", "🚗 By Vehicle"])

with tab1:
    weather_avg = df.groupby('Weather')['Delivery_Time_min'].mean().reset_index()
    weather_avg.columns = ['Weather', 'Avg Delivery Time (min)']
    weather_avg['Icon'] = weather_avg['Weather'].map(WEATHER_ICONS)
    weather_avg['Label'] = weather_avg['Icon'] + ' ' + weather_avg['Weather']
    st.bar_chart(weather_avg.set_index('Label')['Avg Delivery Time (min)'])

with tab2:
    traffic_avg = df.groupby('Traffic_Level')['Delivery_Time_min'].mean().reset_index()
    traffic_avg.columns = ['Traffic', 'Avg Delivery Time (min)']
    traffic_avg['Icon'] = traffic_avg['Traffic'].map(TRAFFIC_ICONS)
    traffic_avg['Label'] = traffic_avg['Icon'] + ' ' + traffic_avg['Traffic']
    st.bar_chart(traffic_avg.set_index('Label')['Avg Delivery Time (min)'])

with tab3:
    vehicle_avg = df.groupby('Vehicle_Type')['Delivery_Time_min'].mean().reset_index()
    vehicle_avg.columns = ['Vehicle', 'Avg Delivery Time (min)']
    vehicle_avg['Icon'] = vehicle_avg['Vehicle'].map(VEHICLE_ICONS)
    vehicle_avg['Label'] = vehicle_avg['Icon'] + ' ' + vehicle_avg['Vehicle']
    st.bar_chart(vehicle_avg.set_index('Label')['Avg Delivery Time (min)'])

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center; margin-top:3rem; padding:1.5rem;
            border-top:1px solid rgba(255,107,53,0.12); color:#8b93b0; font-size:0.75rem;'>
    🛵 <b style='color:#ff6b35;'>DeliverIQ</b> &nbsp;·&nbsp; Powered by Linear Regression
    &nbsp;·&nbsp; {len(df):,} orders trained &nbsp;·&nbsp;
    Built with ❤️ using Streamlit &nbsp;·&nbsp;
    {datetime.now().strftime("%B %d, %Y")}
</div>
""", unsafe_allow_html=True)
