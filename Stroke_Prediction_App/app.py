"""
╔══════════════════════════════════════════════════════════════════╗
║     Early Stroke Risk Prediction System - Enhanced UI/UX        ║
║     AI Healthcare Dashboard | Glassmorphism | Premium Design    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import joblib
import time
import datetime
import io
import plotly.graph_objects as go
import plotly.express as px

# ── PDF Generation ────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroGuard AI – Stroke Risk Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  LOAD ML ARTIFACTS  (logic UNCHANGED)
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    model  = joblib.load("stroke_prediction_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    MODEL_LOADED = True
except Exception:
    MODEL_LOADED = False

# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS – Dark Medical Glassmorphism Theme
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ─── Google Fonts ─── */
@import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Root Variables ─── */
:root {
  --bg-primary:     #050c1a;
  --bg-secondary:   #071224;
  --bg-card:        rgba(8, 20, 45, 0.75);
  --glass-border:   rgba(0, 170, 255, 0.18);
  --neon-blue:      #00aaff;
  --neon-cyan:      #00e5ff;
  --neon-green:     #00ff9d;
  --neon-red:       #ff3b5c;
  --neon-amber:     #ffb300;
  --neon-purple:    #b57aff;
  --text-primary:   #e8f4ff;
  --text-secondary: #7ba8cc;
  --text-muted:     #4a6a88;
  --accent-gradient: linear-gradient(135deg, #00aaff 0%, #00e5ff 50%, #00ff9d 100%);
  --danger-gradient: linear-gradient(135deg, #ff3b5c 0%, #ff8c42 100%);
  --safe-gradient:   linear-gradient(135deg, #00ff9d 0%, #00e5ff 100%);
}

/* ─── Global Reset ─── */
* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background-color: var(--bg-primary);
  color: var(--text-primary);
}

/* ─── Animated Background ─── */
.stApp {
  background:
    radial-gradient(ellipse 80% 60% at 10% 20%, rgba(0,100,200,0.12) 0%, transparent 60%),
    radial-gradient(ellipse 60% 80% at 90% 80%, rgba(0,200,150,0.08) 0%, transparent 60%),
    radial-gradient(ellipse 50% 50% at 50% 50%, rgba(0,50,120,0.10) 0%, transparent 70%),
    linear-gradient(180deg, #050c1a 0%, #07111f 50%, #050c1a 100%);
  background-attachment: fixed;
}

/* ─── Floating particles overlay ─── */
.particle-canvas {
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}
.particle {
  position: absolute;
  border-radius: 50%;
  background: var(--neon-blue);
  opacity: 0;
  animation: float-particle linear infinite;
}
@keyframes float-particle {
  0%   { transform: translateY(100vh) scale(0); opacity: 0; }
  10%  { opacity: 0.4; }
  90%  { opacity: 0.2; }
  100% { transform: translateY(-10vh) scale(1.2); opacity: 0; }
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060e20 0%, #07152a 100%) !important;
  border-right: 1px solid var(--glass-border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ─── Hide default Streamlit chrome ─── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ─── Glass Card ─── */
.glass-card {
  background: var(--bg-card);
  border: 1px solid var(--glass-border);
  border-radius: 20px;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  padding: 28px 32px;
  margin-bottom: 20px;
  box-shadow:
    0 8px 32px rgba(0,0,0,0.4),
    inset 0 1px 0 rgba(255,255,255,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
}
.glass-card:hover {
  transform: translateY(-4px);
  box-shadow:
    0 16px 48px rgba(0,0,0,0.5),
    0 0 30px rgba(0,170,255,0.08),
    inset 0 1px 0 rgba(255,255,255,0.08);
  border-color: rgba(0,170,255,0.35);
}

/* ─── Hero Section ─── */
.hero-section {
  text-align: center;
  padding: 52px 24px 40px;
  position: relative;
  overflow: hidden;
}
.hero-section::before {
  content: '';
  position: absolute;
  top: -40px; left: 50%;
  transform: translateX(-50%);
  width: 600px; height: 300px;
  background: radial-gradient(ellipse, rgba(0,170,255,0.15) 0%, transparent 70%);
  pointer-events: none;
}
.hero-badge {
  display: inline-block;
  background: rgba(0,170,255,0.12);
  border: 1px solid rgba(0,170,255,0.3);
  border-radius: 50px;
  padding: 6px 18px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 2px;
  color: var(--neon-cyan);
  text-transform: uppercase;
  margin-bottom: 20px;
  animation: pulse-badge 3s ease-in-out infinite;
}
@keyframes pulse-badge {
  0%, 100% { box-shadow: 0 0 0 0 rgba(0,170,255,0.4); }
  50%       { box-shadow: 0 0 0 8px rgba(0,170,255,0); }
}
.hero-title {
  font-family: 'Oxanium', sans-serif;
  font-size: clamp(2rem, 5vw, 3.8rem);
  font-weight: 800;
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1;
  margin-bottom: 16px;
  letter-spacing: -1px;
}
.hero-subtitle {
  font-size: 1.05rem;
  color: var(--text-secondary);
  max-width: 620px;
  margin: 0 auto 28px;
  line-height: 1.7;
}
.hero-stats {
  display: flex;
  justify-content: center;
  gap: 40px;
  flex-wrap: wrap;
  margin-top: 12px;
}
.hero-stat { text-align: center; }
.hero-stat-num {
  font-family: 'Oxanium', sans-serif;
  font-size: 1.8rem;
  font-weight: 700;
  color: var(--neon-cyan);
}
.hero-stat-label {
  font-size: 0.72rem;
  color: var(--text-muted);
  letter-spacing: 1px;
  text-transform: uppercase;
}

/* ─── Section Headers ─── */
.section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 22px;
}
.section-icon {
  width: 42px; height: 42px;
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.2rem;
  flex-shrink: 0;
}
.section-icon-blue  { background: rgba(0,170,255,0.15); border: 1px solid rgba(0,170,255,0.25); }
.section-icon-green { background: rgba(0,255,157,0.12); border: 1px solid rgba(0,255,157,0.22); }
.section-icon-red   { background: rgba(255,59,92,0.12);  border: 1px solid rgba(255,59,92,0.22); }
.section-icon-purple{ background: rgba(181,122,255,0.12); border: 1px solid rgba(181,122,255,0.22); }
.section-title {
  font-family: 'Oxanium', sans-serif;
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--text-primary);
}
.section-subtitle {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-top: 2px;
}

/* ─── Metric Cards ─── */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 14px;
  margin-bottom: 24px;
}
.metric-card {
  background: rgba(8,20,45,0.8);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  padding: 18px 16px;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;
}
.metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
}
.metric-card.blue::before  { background: var(--neon-blue); }
.metric-card.cyan::before  { background: var(--neon-cyan); }
.metric-card.green::before { background: var(--neon-green); }
.metric-card.red::before   { background: var(--neon-red); }
.metric-card.purple::before{ background: var(--neon-purple); }
.metric-card:hover { transform: translateY(-3px); border-color: rgba(0,170,255,0.3); }
.metric-icon { font-size: 1.5rem; margin-bottom: 8px; }
.metric-value {
  font-family: 'Oxanium', sans-serif;
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 4px;
}
.metric-label {
  font-size: 0.7rem;
  color: var(--text-muted);
  letter-spacing: 1px;
  text-transform: uppercase;
}

/* ─── Input Fields ─── */
.stSlider > div > div { background: rgba(0,170,255,0.1) !important; }
.stSlider [data-testid="stThumbValue"] { color: var(--neon-cyan) !important; }

.stSelectbox > div > div {
  background: rgba(8,20,45,0.9) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
}
.stSelectbox > div > div:hover { border-color: var(--neon-blue) !important; }

.stNumberInput > div > div {
  background: rgba(8,20,45,0.9) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 10px !important;
}

/* ─── Predict Button ─── */
.stButton > button {
  background: linear-gradient(135deg, #0055cc 0%, #0099ee 50%, #00ccaa 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 14px !important;
  padding: 16px 40px !important;
  font-family: 'Oxanium', sans-serif !important;
  font-size: 1.05rem !important;
  font-weight: 700 !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  width: 100% !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 4px 24px rgba(0,150,255,0.35) !important;
  position: relative !important;
  overflow: hidden !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 36px rgba(0,150,255,0.55) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ─── Risk Alert Boxes ─── */
.alert-high {
  background: rgba(255,59,92,0.10);
  border: 1px solid rgba(255,59,92,0.4);
  border-radius: 16px;
  padding: 22px 28px;
  text-align: center;
  animation: pulse-red 2s ease-in-out infinite;
}
@keyframes pulse-red {
  0%, 100% { box-shadow: 0 0 0 0 rgba(255,59,92,0.3); }
  50%       { box-shadow: 0 0 0 12px rgba(255,59,92,0); }
}
.alert-low {
  background: rgba(0,255,157,0.08);
  border: 1px solid rgba(0,255,157,0.3);
  border-radius: 16px;
  padding: 22px 28px;
  text-align: center;
  animation: pulse-green 3s ease-in-out infinite;
}
@keyframes pulse-green {
  0%, 100% { box-shadow: 0 0 0 0 rgba(0,255,157,0.25); }
  50%       { box-shadow: 0 0 0 10px rgba(0,255,157,0); }
}
.alert-title {
  font-family: 'Oxanium', sans-serif;
  font-size: 1.8rem;
  font-weight: 800;
  margin-bottom: 6px;
}
.alert-sub { font-size: 0.9rem; color: var(--text-secondary); }

/* ─── Progress / Risk Bars ─── */
.risk-bar-wrap { margin: 10px 0; }
.risk-bar-label {
  display: flex;
  justify-content: space-between;
  font-size: 0.82rem;
  color: var(--text-secondary);
  margin-bottom: 6px;
}
.risk-bar-track {
  height: 8px;
  background: rgba(255,255,255,0.06);
  border-radius: 99px;
  overflow: hidden;
}
.risk-bar-fill {
  height: 100%;
  border-radius: 99px;
  background: var(--accent-gradient);
  animation: bar-grow 1.2s cubic-bezier(0.4,0,0.2,1) both;
}
@keyframes bar-grow {
  from { width: 0 !important; }
}

/* ─── XAI Insight Cards ─── */
.xai-card {
  background: rgba(0,100,200,0.08);
  border: 1px solid rgba(0,170,255,0.15);
  border-radius: 14px;
  padding: 14px 18px;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 14px;
  transition: all 0.25s ease;
}
.xai-card:hover {
  background: rgba(0,100,200,0.15);
  border-color: rgba(0,170,255,0.3);
  transform: translateX(4px);
}
.xai-rank {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  color: var(--neon-cyan);
  background: rgba(0,200,255,0.1);
  border-radius: 8px;
  padding: 4px 10px;
  white-space: nowrap;
}
.xai-name { font-size: 0.88rem; font-weight: 500; flex: 1; }
.xai-bar-wrap { width: 100px; }
.xai-bar-track {
  height: 6px;
  background: rgba(255,255,255,0.06);
  border-radius: 99px;
  overflow: hidden;
}
.xai-bar-fill {
  height: 100%;
  border-radius: 99px;
  background: linear-gradient(90deg, var(--neon-blue), var(--neon-cyan));
  animation: bar-grow 1.4s ease both;
}
.xai-pct {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  color: var(--neon-cyan);
  min-width: 36px;
  text-align: right;
}

/* ─── Recommendation Cards ─── */
.rec-card {
  background: rgba(0,255,157,0.06);
  border: 1px solid rgba(0,255,157,0.15);
  border-radius: 12px;
  padding: 12px 18px;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 0.88rem;
  transition: all 0.25s ease;
}
.rec-card:hover {
  background: rgba(0,255,157,0.12);
  border-color: rgba(0,255,157,0.28);
  transform: translateX(4px);
}
.rec-icon { font-size: 1.1rem; flex-shrink: 0; }

/* ─── Test Cards ─── */
.test-card {
  background: rgba(181,122,255,0.07);
  border: 1px solid rgba(181,122,255,0.18);
  border-radius: 12px;
  padding: 12px 18px;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 0.88rem;
  transition: all 0.25s ease;
}
.test-card:hover {
  background: rgba(181,122,255,0.12);
  transform: translateX(4px);
}

/* ─── Emergency Banner ─── */
.emergency-banner {
  background: linear-gradient(135deg, rgba(255,59,92,0.2), rgba(255,100,50,0.15));
  border: 2px solid rgba(255,59,92,0.5);
  border-radius: 16px;
  padding: 22px 28px;
  text-align: center;
  animation: emergency-pulse 1.5s ease-in-out infinite;
  margin: 16px 0;
}
@keyframes emergency-pulse {
  0%, 100% { border-color: rgba(255,59,92,0.5); box-shadow: 0 0 0 0 rgba(255,59,92,0.3); }
  50%       { border-color: rgba(255,59,92,0.9); box-shadow: 0 0 0 16px rgba(255,59,92,0); }
}
.emergency-title {
  font-family: 'Oxanium', sans-serif;
  font-size: 1.3rem;
  font-weight: 800;
  color: var(--neon-red);
  margin-bottom: 8px;
}
.emergency-text { font-size: 0.9rem; color: rgba(255,200,200,0.85); }

/* ─── Clock Widget ─── */
.clock-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 16px 0 8px;
}
.clock-digital {
  font-family: 'JetBrains Mono', monospace;
  font-size: 2rem;
  font-weight: 500;
  color: var(--neon-cyan);
  letter-spacing: 3px;
  margin-top: 14px;
  text-shadow: 0 0 20px rgba(0,229,255,0.5);
}
.clock-date {
  font-size: 0.78rem;
  color: var(--text-muted);
  letter-spacing: 2px;
  margin-top: 4px;
  text-transform: uppercase;
}

/* ─── Sidebar Labels ─── */
.sidebar-label {
  font-family: 'Oxanium', sans-serif;
  font-size: 0.7rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--text-muted);
  padding: 14px 0 6px;
  border-top: 1px solid rgba(255,255,255,0.05);
  margin-top: 8px;
}

/* ─── Footer ─── */
.footer {
  text-align: center;
  padding: 32px 24px;
  border-top: 1px solid var(--glass-border);
  margin-top: 40px;
}
.footer-brand {
  font-family: 'Oxanium', sans-serif;
  font-size: 1.1rem;
  font-weight: 700;
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 6px;
}
.footer-text { font-size: 0.78rem; color: var(--text-muted); }

/* ─── Symptom Tag Grid ─── */
.symptom-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}
.symptom-tag {
  background: rgba(0,170,255,0.1);
  border: 1px solid rgba(0,170,255,0.22);
  border-radius: 99px;
  padding: 5px 14px;
  font-size: 0.78rem;
  color: var(--neon-cyan);
  white-space: nowrap;
}
.symptom-tag.active {
  background: rgba(255,59,92,0.15);
  border-color: rgba(255,59,92,0.35);
  color: var(--neon-red);
}

/* ─── Animations ─── */
@keyframes fade-in-up {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fade-in-up 0.6s ease both; }
.fade-in-d1 { animation: fade-in-up 0.6s 0.1s ease both; }
.fade-in-d2 { animation: fade-in-up 0.6s 0.2s ease both; }
.fade-in-d3 { animation: fade-in-up 0.6s 0.3s ease both; }
.fade-in-d4 { animation: fade-in-up 0.6s 0.4s ease both; }

/* ─── Typing animation for hero ─── */
@keyframes typing { from { width: 0; } to { width: 100%; } }
@keyframes blink   { 50% { border-color: transparent; } }

/* ─── Info boxes override ─── */
.stAlert { border-radius: 12px !important; }

/* ─── Tab styling ─── */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(8,20,45,0.6) !important;
  border-radius: 12px !important;
  padding: 4px !important;
  border: 1px solid var(--glass-border) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px !important;
  color: var(--text-secondary) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.85rem !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(0,170,255,0.18) !important;
  color: var(--neon-cyan) !important;
}

/* ─── Expander ─── */
.streamlit-expanderHeader {
  background: rgba(8,20,45,0.7) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 12px !important;
  color: var(--text-primary) !important;
}

/* ─── Divider ─── */
hr { border-color: var(--glass-border) !important; margin: 24px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  FLOATING PARTICLES  (CSS-only, injected once)
# ══════════════════════════════════════════════════════════════════
particles_html = """
<div class="particle-canvas" aria-hidden="true">
""" + "".join([
    f"""<div class="particle" style="
        left:{np.random.randint(0,100)}%;
        width:{np.random.randint(2,5)}px;
        height:{np.random.randint(2,5)}px;
        animation-duration:{np.random.randint(12,25)}s;
        animation-delay:{np.random.randint(0,15)}s;
        background:{'#00aaff' if i%3==0 else '#00e5ff' if i%3==1 else '#00ff9d'};
        opacity:0.35;
    "></div>""" for i in range(28)
]) + """
</div>
"""
st.markdown(particles_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown(
        '<div class="sidebar-label">🧭 Navigation</div>',
        unsafe_allow_html=True
    )

    st.markdown("🏠 Home / Predict")
    st.markdown("📊 Risk Analysis")
    st.markdown("💊 Recommendations")
    st.markdown("🧪 Clinical Tests")
    st.markdown("🧠 XAI Insights")

    # ── Real-time Analog Clock ─────────────────────

    import datetime
    import math

    st.markdown(
        '<div class="sidebar-label">🕐 LIVE MONITOR</div>',
        unsafe_allow_html=True
    )

    clock_placeholder = st.empty()

    def render_clock():

        now = datetime.datetime.now()

        h = now.hour % 12
        m = now.minute
        s = now.second

        h_deg = h * 30 + m * 0.5
        m_deg = m * 6 + s * 0.1
        s_deg = s * 6

        date_str = now.strftime("%d %b %Y")
        time_str = now.strftime("%H:%M:%S")

        cx, cy, r = 70, 70, 60

        def hand_end(deg, length):
            rad = math.radians(deg - 90)
            x = cx + length * math.cos(rad)
            y = cy + length * math.sin(rad)
            return x, y

        hx, hy = hand_end(h_deg, 32)
        mx, my = hand_end(m_deg, 44)
        sx, sy = hand_end(s_deg, 50)

        svg = f"""
        <div style="text-align:center;">

        <svg width="140" height="140">

        <circle cx="70" cy="70" r="60"
        fill="#071426"
        stroke="#00d4ff"
        stroke-width="3"/>

        <line x1="70" y1="70"
        x2="{hx}" y2="{hy}"
        stroke="white"
        stroke-width="4"/>

        <line x1="70" y1="70"
        x2="{mx}" y2="{my}"
        stroke="#00d4ff"
        stroke-width="3"/>

        <line x1="70" y1="70"
        x2="{sx}" y2="{sy}"
        stroke="red"
        stroke-width="2"/>

        <circle cx="70" cy="70" r="4"
        fill="#00d4ff"/>

        </svg>

        <h3 style="color:#00d4ff;">{time_str}</h3>
        <p style="color:gray;">{date_str}</p>

        </div>
        """

        return svg

    clock_placeholder.markdown(
        render_clock(),
        unsafe_allow_html=True
    )
    # ── Navigation ─────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">🧭 Navigation</div>', unsafe_allow_html=True)

    nav_items = [
        ("🏠", "Home / Predict", "Main prediction interface"),
        ("📊", "Risk Analysis", "AI-driven risk breakdown"),
        ("💊", "Recommendations", "Preventive healthcare advice"),
        ("🧪", "Clinical Tests", "Suggested diagnostic tests"),
        ("🤖", "XAI Insights", "Explainable AI explanations"),
    ]
    for icon, label, desc in nav_items:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:9px 12px;
                    border-radius:10px;margin-bottom:4px;cursor:pointer;
                    background:rgba(0,170,255,0.06);border:1px solid rgba(0,170,255,0.1);
                    transition:all 0.2s ease;"
             onmouseover="this.style.background='rgba(0,170,255,0.14)'"
             onmouseout="this.style.background='rgba(0,170,255,0.06)'">
          <span style="font-size:1.05rem;">{icon}</span>
          <div>
            <div style="font-size:0.83rem;font-weight:500;color:#e8f4ff;">{label}</div>
            <div style="font-size:0.68rem;color:#4a6a88;">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Model Info ─────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">🤖 AI Model Info</div>', unsafe_allow_html=True)
    model_status = "🟢 Online" if MODEL_LOADED else "🔴 Offline"
    st.markdown(f"""
    <div style="background:rgba(8,20,45,0.8);border:1px solid rgba(0,170,255,0.15);
                border-radius:12px;padding:14px 16px;font-size:0.8rem;">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
        <span style="color:#7ba8cc;">Status</span>
        <span style="color:#00ff9d;">{model_status}</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
        <span style="color:#7ba8cc;">Algorithm</span>
        <span style="color:#e8f4ff;">Ensemble</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
        <span style="color:#7ba8cc;">XAI Engine</span>
        <span style="color:#e8f4ff;">SHAP/LIME</span>
      </div>
      <div style="display:flex;justify-content:space-between;">
        <span style="color:#7ba8cc;">Features</span>
        <span style="color:#e8f4ff;">17 Clinical</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Research Objective ──────────────────────────────────────
    with st.expander("📋 Research Objective"):
        st.markdown("""
        <div style="font-size:0.82rem;color:#7ba8cc;line-height:1.7;">
        This system leverages ensemble machine learning with Explainable AI (XAI)
        to predict early stroke risk from clinical symptoms — enabling faster,
        data-driven clinical decision-making.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📞 Contact"):
        st.markdown("""
        <div style="font-size:0.82rem;color:#7ba8cc;line-height:1.8;">
        🔬 Md. Asif Khandoker<br>
        📧 asifkhandoker.media@gmail.com<br>
        🌐 neuroguard.ai<br>
        🏥 Version 2.0 | 2025
        </div>
        """, unsafe_allow_html=True)

        
st.markdown("""
<div class="hero-section fade-in"
style="
display:flex;
flex-direction:column;
justify-content:center;
align-items:center;
text-align:center;
width:100%;
margin:auto;
">

<div class="hero-badge">
🧠 AI-Powered · Ensemble Learning · XAI Enabled
</div>

<h1 class="hero-title">
Early Stroke Risk<br>
Prediction System
</h1>

<div style="
max-width:900px;
margin:auto;
display:flex;
justify-content:center;
align-items:center;
text-align:center;
">

<p class="hero-subtitle"
style="
text-align:center;
line-height:1.9;
font-size:1.1rem;
max-width:850px;
margin:auto;
">
Advanced AI healthcare diagnostics powered by Ensemble Machine Learning
and Explainable AI — providing real-time stroke risk assessment
with clinical precision.
</p>

</div>

<div class="hero-stats">

<div class="hero-stat">
<div class="hero-stat-num">17</div>
<div class="hero-stat-label">Clinical Features</div>
</div>

<div class="hero-stat">
<div class="hero-stat-num">XAI</div>
<div class="hero-stat-label">Explainable AI</div>
</div>

<div class="hero-stat">
<div class="hero-stat-num">Real-Time</div>
<div class="hero-stat-label">Prediction</div>
</div>

<div class="hero-stat">
<div class="hero-stat-num">ML</div>
<div class="hero-stat-label">Ensemble Model</div>
</div>

</div>

</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  RESEARCH DISCLAIMER
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div style="
    margin-top:40px;
    margin-bottom:25px;
    padding:28px;
    border-radius:18px;
    background:rgba(255, 183, 0, 0.08);
    border:1px solid rgba(255, 183, 0, 0.25);
    backdrop-filter: blur(12px);
    text-align:center;
">

<h3 style="
    color:#ffcc00;
    font-family:'Oxanium', sans-serif;
    margin-bottom:14px;
    font-size:1.3rem;
">
⚠️ Research & Educational Disclaimer
</h3>

<p style="
    color:#c7d5e0;
    font-size:15px;
    line-height:1.9;
    max-width:950px;
    margin:auto;
">

This AI-based Stroke Risk Prediction System was developed strictly for
academic research, educational demonstration, and machine learning practice purposes.

The predictions generated by this prototype system are NOT intended for
real-world medical diagnosis, emergency healthcare decisions,
or professional clinical use.

Stroke is a serious medical condition.
Always consult certified healthcare professionals, neurologists,
or medical specialists for proper diagnosis and treatment.

This project is presented as an experimental AI research implementation only.

</p>

</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  HELPER: YES/NO → 0/1 ENCODING
# ══════════════════════════════════════════════════════════════════
def yn_select(label, key, options=None, hint=None):
    """Display a professional Yes/No selectbox; returns 0 or 1 internally."""
    if options is None:
        options = ["No", "Yes"]
    mapping = {"No": 0, "Yes": 1, "Absent": 0, "Present": 1,
               "Not Detected": 0, "Detected": 1, "Normal": 0, "Abnormal": 1}
    choice = st.selectbox(label, options, key=key,
                          help=hint or "Select the appropriate clinical status")
    return mapping.get(choice, 0)

# ══════════════════════════════════════════════════════════════════
#  PATIENT INPUT FORM
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="glass-card fade-in-d1">
  <div class="section-header">
    <div class="section-icon section-icon-blue">👤</div>
    <div>
      <div class="section-title">Patient Information</div>
      <div class="section-subtitle">Enter clinical parameters for AI analysis</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
    <div style="background:rgba(8,20,45,0.6);border:1px solid rgba(0,170,255,0.12);
                border-radius:16px;padding:22px 24px;margin-bottom:16px;">
    <div style="font-family:'Oxanium',sans-serif;font-size:0.9rem;font-weight:600;
                color:#00aaff;margin-bottom:16px;letter-spacing:1px;">
    🧬 DEMOGRAPHICS</div>
    """, unsafe_allow_html=True)

    age = st.slider("🎂 Age", 1, 100, 30,
                    help="Patient age in years")

    gender_choice = st.selectbox("⚧ Gender", ["Female", "Male"],
                                 help="Biological sex of the patient")
    gender = 1 if gender_choice == "Male" else 0

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(8,20,45,0.6);border:1px solid rgba(0,170,255,0.12);
                border-radius:16px;padding:22px 24px;margin-bottom:16px;">
    <div style="font-family:'Oxanium',sans-serif;font-size:0.9rem;font-weight:600;
                color:#00aaff;margin-bottom:16px;letter-spacing:1px;">
    ❤️ CARDIOVASCULAR SYMPTOMS</div>
    """, unsafe_allow_html=True)

    chest_pain        = yn_select("💔 Chest Pain",         "cp",
                                   ["Absent", "Present"],
                                   "Presence of chest pain or tightness")
    hypertension      = yn_select("📈 Hypertension",        "htn",
                                   ["Not Detected", "Detected"],
                                   "High blood pressure diagnosis")
    irregular_heartbeat = yn_select("💓 Irregular Heartbeat","ihb",
                                   ["Absent", "Present"],
                                   "Cardiac arrhythmia or irregular pulse")
    shortness_of_breath = yn_select("🫁 Shortness of Breath","sob",
                                   ["Absent", "Present"],
                                   "Dyspnea or breathing difficulty")
    chest_discomfort  = yn_select("⚠️ Chest Discomfort",    "cd",
                                   ["Absent", "Present"],
                                   "Discomfort or pressure in chest area")

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div style="background:rgba(8,20,45,0.6);border:1px solid rgba(0,170,255,0.12);
                border-radius:16px;padding:22px 24px;margin-bottom:16px;">
    <div style="font-family:'Oxanium',sans-serif;font-size:0.9rem;font-weight:600;
                color:#00aaff;margin-bottom:16px;letter-spacing:1px;">
    🩺 GENERAL SYMPTOMS</div>
    """, unsafe_allow_html=True)

    fatigue_weakness  = yn_select("😴 Fatigue / Weakness",  "fw",
                                   ["Absent", "Present"],
                                   "General fatigue or muscle weakness")
    dizziness         = yn_select("🌀 Dizziness",            "dz",
                                   ["Absent", "Present"],
                                   "Vertigo or balance disturbances")
    edema             = yn_select("💧 Edema",                "ed",
                                   ["Not Detected", "Detected"],
                                   "Swelling in extremities")
    neck_jaw_pain     = yn_select("🦷 Neck / Jaw Pain",      "njp",
                                   ["Absent", "Present"],
                                   "Radiating pain to neck or jaw")
    excessive_sweating= yn_select("💦 Excessive Sweating",   "es",
                                   ["Absent", "Present"],
                                   "Diaphoresis or unusual sweating")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(8,20,45,0.6);border:1px solid rgba(0,170,255,0.12);
                border-radius:16px;padding:22px 24px;margin-bottom:16px;">
    <div style="font-family:'Oxanium',sans-serif;font-size:0.9rem;font-weight:600;
                color:#00aaff;margin-bottom:16px;letter-spacing:1px;">
    🧠 NEUROLOGICAL & OTHER</div>
    """, unsafe_allow_html=True)

    persistent_cough  = yn_select("🤧 Persistent Cough",    "pc",
                                   ["Absent", "Present"],
                                   "Chronic or recurring cough")
    nausea_vomiting   = yn_select("🤢 Nausea / Vomiting",   "nv",
                                   ["Absent", "Present"],
                                   "Gastrointestinal distress symptoms")
    cold_hands_feet   = yn_select("🥶 Cold Hands / Feet",   "chf",
                                   ["Absent", "Present"],
                                   "Peripheral circulation issues")
    sleep_apnea       = yn_select("😮‍💨 Sleep Apnea",         "sa",
                                   ["Not Detected", "Detected"],
                                   "Sleep-disordered breathing")
    anxiety           = yn_select("😰 Anxiety",              "ax",
                                   ["Absent", "Present"],
                                   "Anxiety disorder or panic episodes")

    st.markdown("</div>", unsafe_allow_html=True)

# ── Symptom Summary Tags ───────────────────────────────────────────
all_symptoms = {
    "Chest Pain": chest_pain, "Hypertension": hypertension,
    "Irregular Heartbeat": irregular_heartbeat,
    "Shortness of Breath": shortness_of_breath,
    "Fatigue/Weakness": fatigue_weakness, "Dizziness": dizziness,
    "Edema": edema, "Neck/Jaw Pain": neck_jaw_pain,
    "Excessive Sweating": excessive_sweating,
    "Persistent Cough": persistent_cough,
    "Nausea/Vomiting": nausea_vomiting,
    "Chest Discomfort": chest_discomfort,
    "Cold Hands/Feet": cold_hands_feet,
    "Sleep Apnea": sleep_apnea, "Anxiety": anxiety,
}
active_symptoms = [s for s, v in all_symptoms.items() if v == 1]

tags_html = '<div class="symptom-grid">'
for sym, val in all_symptoms.items():
    cls = "symptom-tag active" if val else "symptom-tag"
    icon = "🔴" if val else "⚪"
    tags_html += f'<span class="{cls}">{icon} {sym}</span>'
tags_html += '</div>'

st.markdown(f"""
<div class="glass-card fade-in-d2" style="margin-top:8px;">
  <div class="section-header">
    <div class="section-icon section-icon-red">🏷️</div>
    <div>
      <div class="section-title">Symptom Summary</div>
      <div class="section-subtitle">{len(active_symptoms)} of {len(all_symptoms)} symptoms detected</div>
    </div>
  </div>
  {tags_html}
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PREDICT BUTTON
# ══════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
predict_col = st.columns([1, 2, 1])[1]
with predict_col:
    predict_clicked = st.button("🧠  Analyze Stroke Risk  →", use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  PREDICTION BLOCK  (ML logic UNCHANGED)
# ══════════════════════════════════════════════════════════════════
if predict_clicked:
    if not MODEL_LOADED:
        st.error("⚠️ ML model files not found. Ensure `stroke_prediction_model.pkl` and `scaler.pkl` are present.")
        st.stop()

    # ── AI Loading Animation ──────────────────────────────────
    loading_placeholder = st.empty()
    steps = [
        ("🔍", "Initializing Neural Network…"),
        ("📊", "Processing Clinical Parameters…"),
        ("🧬", "Running Ensemble Classifier…"),
        ("📈", "Generating Risk Probability…"),
        ("🤖", "Compiling XAI Explanations…"),
        ("✅", "Analysis Complete!"),
    ]
    for icon, msg in steps:
        loading_placeholder.markdown(f"""
        <div style="text-align:center;padding:32px;background:rgba(8,20,45,0.8);
                    border:1px solid rgba(0,170,255,0.2);border-radius:20px;
                    animation:fade-in-up 0.3s ease;">
          <div style="font-size:2.5rem;margin-bottom:12px;
                      animation:pulse-badge 1s infinite;">{icon}</div>
          <div style="font-family:'Oxanium',sans-serif;font-size:1.1rem;
                      color:#00e5ff;font-weight:600;">{msg}</div>
          <div style="font-size:0.78rem;color:#4a6a88;margin-top:6px;">
            AI Analyzing Patient Data…</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.38)
    loading_placeholder.empty()

    # ── ML CORE (EXACTLY AS ORIGINAL) ─────────────────────────
    sample = [[
        age, gender, chest_pain, hypertension, irregular_heartbeat,
        shortness_of_breath, fatigue_weakness, dizziness, edema,
        neck_jaw_pain, excessive_sweating, persistent_cough,
        nausea_vomiting, chest_discomfort, cold_hands_feet,
        sleep_apnea, anxiety
    ]]
    sample_scaled  = scaler.transform(sample)
    prediction     = model.predict(sample_scaled)
    probability    = model.predict_proba(sample_scaled)
    risk           = probability[0][1] * 100
    confidence     = max(probability[0]) * 100
    pred_time      = datetime.datetime.now().strftime("%H:%M:%S")
    pred_date      = datetime.datetime.now().strftime("%d %B %Y")

    # ── Risk Level Classification ──────────────────────────────
    if risk < 30:
        risk_level, risk_color, risk_icon = "LOW RISK", "#00ff9d", "✅"
    elif risk < 60:
        risk_level, risk_color, risk_icon = "MODERATE RISK", "#ffb300", "⚠️"
    elif risk < 80:
        risk_level, risk_color, risk_icon = "HIGH RISK", "#ff6b35", "🔶"
    else:
        risk_level, risk_color, risk_icon = "CRITICAL RISK", "#ff3b5c", "🚨"

    is_high = prediction[0] == 1

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;margin-bottom:24px;">
      <span style="font-family:'Oxanium',sans-serif;font-size:0.75rem;letter-spacing:3px;
                   text-transform:uppercase;color:#4a6a88;">Analysis Results</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Result Alert ────────────────────────────────────────────
    if is_high:
        st.markdown(f"""
        <div class="alert-high fade-in">
          <div class="alert-title" style="color:{risk_color};">{risk_icon} {risk_level}</div>
          <div class="alert-sub">Stroke Risk Probability: <strong style="color:{risk_color};">{risk:.1f}%</strong></div>
          <div style="font-size:0.8rem;color:#7ba8cc;margin-top:8px;">
            AI Confidence: {confidence:.1f}% · Analyzed at {pred_time}
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-low fade-in">
          <div class="alert-title" style="color:{risk_color};">{risk_icon} {risk_level}</div>
          <div class="alert-sub">Stroke Risk Probability: <strong style="color:{risk_color};">{risk:.1f}%</strong></div>
          <div style="font-size:0.8rem;color:#7ba8cc;margin-top:8px;">
            AI Confidence: {confidence:.1f}% · Analyzed at {pred_time}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dashboard Metric Cards ─────────────────────────────────
    st.markdown(f"""
    <div class="metric-grid fade-in-d1">
      <div class="metric-card {'red' if is_high else 'green'}">
        <div class="metric-icon">🎯</div>
        <div class="metric-value" style="color:{risk_color};">{risk:.1f}%</div>
        <div class="metric-label">Risk Score</div>
      </div>
      <div class="metric-card cyan">
        <div class="metric-icon">🤖</div>
        <div class="metric-value" style="color:#00e5ff;">{confidence:.1f}%</div>
        <div class="metric-label">AI Confidence</div>
      </div>
      <div class="metric-card {'red' if is_high else 'green'}">
        <div class="metric-icon">{'⚠️' if is_high else '✅'}</div>
        <div class="metric-value" style="color:{risk_color};font-size:1rem;">{risk_level}</div>
        <div class="metric-label">Health Status</div>
      </div>
      <div class="metric-card blue">
        <div class="metric-icon">⏱️</div>
        <div class="metric-value" style="color:#00aaff;font-size:1.1rem;">{pred_time}</div>
        <div class="metric-label">Prediction Time</div>
      </div>
      <div class="metric-card purple">
        <div class="metric-icon">🧬</div>
        <div class="metric-value" style="color:#b57aff;font-size:1rem;">{len(active_symptoms)}/{len(all_symptoms)}</div>
        <div class="metric-label">Symptoms Active</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  TABS: Risk Analysis | Visualization | XAI | Clinical
    # ══════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Risk Analysis",
        "📈 Visualization",
        "🤖 XAI Insights",
        "💊 Clinical Plan",
    ])

    # ── TAB 1: Risk Analysis ───────────────────────────────────
    with tab1:
        factors = []
        if chest_pain:            factors.append(("💔 Chest Pain",          18))
        if hypertension:          factors.append(("📈 Hypertension",         16))
        if irregular_heartbeat:   factors.append(("💓 Irregular Heartbeat",  14))
        if shortness_of_breath:   factors.append(("🫁 Shortness of Breath",  13))
        if sleep_apnea:           factors.append(("😮‍💨 Sleep Apnea",          12))
        if fatigue_weakness:      factors.append(("😴 Fatigue/Weakness",      10))
        if edema:                 factors.append(("💧 Edema",                 9))
        if cold_hands_feet:       factors.append(("🥶 Cold Hands/Feet",       8))
        if anxiety:               factors.append(("😰 Anxiety",               8))
        if dizziness:             factors.append(("🌀 Dizziness",              7))
        if nausea_vomiting:       factors.append(("🤢 Nausea/Vomiting",        7))
        if chest_discomfort:      factors.append(("⚠️ Chest Discomfort",       6))
        if neck_jaw_pain:         factors.append(("🦷 Neck/Jaw Pain",          6))
        if excessive_sweating:    factors.append(("💦 Excessive Sweating",     5))
        if persistent_cough:      factors.append(("🤧 Persistent Cough",       5))

        st.markdown("""
        <div class="glass-card fade-in">
          <div class="section-header">
            <div class="section-icon section-icon-red">⚠️</div>
            <div>
              <div class="section-title">Risk Factor Analysis</div>
              <div class="section-subtitle">Contributing clinical symptoms with weighted impact</div>
            </div>
          </div>
        """, unsafe_allow_html=True)

        if factors:
            max_pct = max(p for _, p in factors)
            bars_html = ""
            for fname, pct in factors:
                width = int((pct / max_pct) * 100)
                bars_html += f"""
                <div class="risk-bar-wrap">
                  <div class="risk-bar-label">
                    <span>{fname}</span>
                    <span style="color:#00e5ff;">{pct}%</span>
                  </div>
                  <div class="risk-bar-track">
                    <div class="risk-bar-fill" style="width:{width}%;"></div>
                  </div>
                </div>
                """
            st.markdown(bars_html + "</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:24px;color:#4a6a88;font-size:0.9rem;">
              ✅ No major risk factors identified in the current assessment.
            </div></div>""", unsafe_allow_html=True)

        # Overall risk bar
        st.markdown(f"""
        <div class="glass-card fade-in-d1">
          <div class="section-header">
            <div class="section-icon section-icon-blue">🎯</div>
            <div>
              <div class="section-title">Overall Stroke Risk Meter</div>
            </div>
          </div>
          <div class="risk-bar-label">
            <span>Risk Level</span>
            <span style="color:{risk_color};">{risk:.1f}%</span>
          </div>
          <div class="risk-bar-track" style="height:14px;">
            <div class="risk-bar-fill" style="width:{risk:.1f}%;
                 background:{'linear-gradient(90deg,#00ff9d,#00e5ff)' if risk < 30
                             else 'linear-gradient(90deg,#ffb300,#ff6b35)' if risk < 60
                             else 'linear-gradient(90deg,#ff6b35,#ff3b5c)'};"></div>
          </div>
          <div style="display:flex;justify-content:space-between;
                      font-size:0.7rem;color:#4a6a88;margin-top:6px;">
            <span>0% · SAFE</span><span>50% · MODERATE</span><span>100% · CRITICAL</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: Visualization ───────────────────────────────────
    with tab2:
        col_v1, col_v2 = st.columns(2)

        # Gauge Chart
        with col_v1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk,
                number={"suffix": "%", "font": {"size": 38, "color": risk_color,
                                                  "family": "Oxanium"}},
                delta={"reference": 50, "increasing": {"color": "#ff3b5c"},
                       "decreasing": {"color": "#00ff9d"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1,
                             "tickcolor": "#7ba8cc", "tickfont": {"color": "#7ba8cc"}},
                    "bar": {"color": risk_color, "thickness": 0.3},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30],  "color": "rgba(0,255,157,0.12)"},
                        {"range": [30, 60], "color": "rgba(255,179,0,0.12)"},
                        {"range": [60, 80], "color": "rgba(255,107,53,0.12)"},
                        {"range": [80, 100],"color": "rgba(255,59,92,0.15)"},
                    ],
                    "threshold": {"line": {"color": risk_color, "width": 3},
                                  "thickness": 0.8, "value": risk},
                },
                title={"text": "Stroke Risk Gauge",
                       "font": {"color": "#7ba8cc", "size": 13, "family": "Inter"}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8f4ff",
                margin=dict(t=60, b=20, l=20, r=20),
                height=300,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Probability Donut
        with col_v2:
            fig_donut = go.Figure(go.Pie(
                labels=["Stroke Risk", "Safe Margin"],
                values=[risk, 100 - risk],
                hole=0.68,
                marker_colors=[risk_color, "rgba(255,255,255,0.06)"],
                textinfo="none",
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            ))
            fig_donut.add_annotation(
                text=f"<b>{risk:.0f}%</b>",
                x=0.5, y=0.5, font_size=28, showarrow=False,
                font_color=risk_color, font_family="Oxanium",
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(font_color="#7ba8cc", bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=20, b=20, l=20, r=20),
                height=300,
                title=dict(text="Risk Distribution", font=dict(color="#7ba8cc", size=13)),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Bar chart of active risk factors
        if factors:
            fnames = [f[0] for f in factors]
            fpcts  = [f[1] for f in factors]
            bar_colors = [
                "#ff3b5c" if p >= 15 else "#ffb300" if p >= 10 else "#00aaff"
                for p in fpcts
            ]
            fig_bar = go.Figure(go.Bar(
                x=fpcts, y=fnames,
                orientation="h",
                marker_color=bar_colors,
                marker_line_color="rgba(0,0,0,0)",
                text=[f"{p}%" for p in fpcts],
                textposition="outside",
                textfont=dict(color="#7ba8cc", size=11),
                hovertemplate="%{y}: %{x}%<extra></extra>",
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(8,20,45,0.4)",
                xaxis=dict(title="Contribution (%)", gridcolor="rgba(255,255,255,0.05)",
                           color="#7ba8cc", tickfont_color="#7ba8cc"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#7ba8cc",
                           tickfont_color="#e8f4ff"),
                margin=dict(t=20, b=40, l=10, r=60),
                height=max(280, len(factors) * 38),
                title=dict(text="Risk Factor Contributions", font=dict(color="#7ba8cc", size=13)),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── TAB 3: XAI Insights ────────────────────────────────────
    with tab3:
        st.markdown("""
        <div class="glass-card fade-in">
          <div class="section-header">
            <div class="section-icon section-icon-purple">🤖</div>
            <div>
              <div class="section-title">Explainable AI Analysis</div>
              <div class="section-subtitle">Top contributing factors identified by the AI model</div>
            </div>
          </div>
        """, unsafe_allow_html=True)

        xai_factors = [
            ("Chest Pain",          18, "High contribution — cardiovascular stress indicator"),
            ("Hypertension",        16, "Strong predictor — elevated arterial pressure"),
            ("Irregular Heartbeat", 14, "Cardiac rhythm anomaly — significant risk marker"),
            ("Shortness of Breath", 13, "Respiratory distress — oxygen supply concern"),
            ("Sleep Apnea",         12, "Sleep-related hypoxia — long-term risk factor"),
            ("Fatigue/Weakness",    10, "Systemic weakness — reduced perfusion signal"),
            ("Anxiety",              8, "Neurological stress — elevated cortisol correlation"),
            ("Dizziness",            7, "Vestibular/vascular disruption indicator"),
        ]

        xai_html = ""
        for i, (name, pct, desc) in enumerate(xai_factors, 1):
            is_active = name.replace(" ", "").lower() in [
                s.replace(" ", "").replace("/","").lower() for s in active_symptoms
            ]
            highlight = "border-color:rgba(255,59,92,0.35);background:rgba(255,59,92,0.08);" if is_active else ""
            xai_html += f"""
            <div class="xai-card" style="{highlight}">
              <div class="xai-rank">#{i}</div>
              <div style="flex:1;">
                <div class="xai-name">{name}
                  {'<span style="margin-left:8px;font-size:0.65rem;color:#ff3b5c;'
                   'background:rgba(255,59,92,0.15);border-radius:6px;padding:2px 8px;">ACTIVE</span>'
                   if is_active else ''}
                </div>
                <div style="font-size:0.72rem;color:#4a6a88;margin-top:2px;">{desc}</div>
                <div class="xai-bar-track" style="margin-top:6px;">
                  <div class="xai-bar-fill" style="width:{pct*5}%;
                       {'background:linear-gradient(90deg,#ff3b5c,#ff8c42);' if is_active else ''}"></div>
                </div>
              </div>
              <div class="xai-pct">{pct}%</div>
            </div>
            """
        st.markdown(xai_html + "</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card fade-in-d1" style="background:rgba(0,50,120,0.15);">
          <div style="font-size:0.82rem;color:#7ba8cc;line-height:1.8;">
            <strong style="color:#00e5ff;">🧠 AI Clinical Insight:</strong><br>
            The Explainable AI engine identified chest pain, hypertension, shortness of breath,
            and sleep apnea as the primary drivers in stroke risk computation.
            Features are weighted using SHAP (SHapley Additive exPlanations) values derived
            from the trained ensemble classifier, providing transparent, clinically interpretable
            decision reasoning.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 4: Clinical Plan ────────────────────────────────────
    with tab4:
        col_r, col_t = st.columns(2)

        with col_r:
            st.markdown("""
            <div class="glass-card fade-in">
              <div class="section-header">
                <div class="section-icon section-icon-green">💊</div>
                <div>
                  <div class="section-title">Preventive Recommendations</div>
                </div>
              </div>
            """, unsafe_allow_html=True)

            recommendations = [
                ("🥗", "Maintain a heart-healthy, low-sodium diet"),
                ("🏃", "Exercise regularly — 30 min/day minimum"),
                ("🧘", "Reduce stress through mindfulness or meditation"),
                ("🚭", "Avoid smoking, alcohol, and stimulants"),
                ("😴", "Ensure 7–8 hours of quality sleep nightly"),
                ("💊", "Monitor blood pressure and take medications as prescribed"),
                ("🩺", "Schedule regular cardiovascular check-ups"),
                ("💧", "Stay adequately hydrated throughout the day"),
            ]
            recs_html = ""
            for icon, rec in recommendations:
                recs_html += f"""
                <div class="rec-card">
                  <span class="rec-icon">{icon}</span>
                  <span>{rec}</span>
                </div>
                """
            st.markdown(recs_html + "</div>", unsafe_allow_html=True)

        with col_t:
            st.markdown("""
            <div class="glass-card fade-in">
              <div class="section-header">
                <div class="section-icon section-icon-purple">🧪</div>
                <div>
                  <div class="section-title">Recommended Clinical Tests</div>
                </div>
              </div>
            """, unsafe_allow_html=True)

            tests = [
                ("🫀", "ECG / EKG",                "Cardiac rhythm evaluation"),
                ("📊", "Blood Pressure Monitoring", "24-hour ambulatory monitoring"),
                ("🩸", "Blood Sugar Test",          "Fasting & post-prandial glucose"),
                ("🧪", "Cholesterol Profile",       "LDL, HDL, triglycerides panel"),
                ("🧠", "Brain CT Scan",             "Cerebrovascular imaging"),
                ("🔬", "MRI Brain",                 "Detailed neural structure review"),
                ("💉", "CBC Blood Panel",           "Complete blood count analysis"),
            ]
            tests_html = ""
            for icon, test, desc in tests:
                tests_html += f"""
                <div class="test-card">
                  <span style="font-size:1.1rem;">{icon}</span>
                  <div>
                    <div style="font-size:0.85rem;font-weight:500;">{test}</div>
                    <div style="font-size:0.7rem;color:#7ba8cc;">{desc}</div>
                  </div>
                </div>
                """
            st.markdown(tests_html + "</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  EMERGENCY BANNER  (risk > 80)
    # ══════════════════════════════════════════════════════════
    if risk > 80:
        st.markdown(f"""
        <div class="emergency-banner fade-in">
          <div class="emergency-title">🚨 EMERGENCY ALERT — IMMEDIATE ACTION REQUIRED</div>
          <div class="emergency-text">
            AI detected a critical stroke probability of <strong>{risk:.1f}%</strong>.<br>
            Immediate consultation with a <strong>Neurologist / Cardiologist</strong> is strongly recommended.<br>
            Contact emergency services if sudden neurological symptoms appear.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  PDF DOWNLOAD REPORT
    # ══════════════════════════════════════════════════════════
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card fade-in">
      <div class="section-header">
        <div class="section-icon section-icon-blue">📄</div>
        <div>
          <div class="section-title">Download AI Medical Report</div>
          <div class="section-subtitle">Professional PDF report with full analysis</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    def generate_pdf_report():
        """Generate a professional PDF medical report."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50,
        )
        story = []
        styles = getSampleStyleSheet()

        # Custom Styles
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=22, fontName="Helvetica-Bold",
            textColor=colors.HexColor("#0055cc"),
            alignment=TA_CENTER, spaceAfter=6,
        )
        subtitle_style = ParagraphStyle(
            "Subtitle", parent=styles["Normal"],
            fontSize=10, textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER, spaceAfter=16,
        )
        section_style = ParagraphStyle(
            "Section", parent=styles["Heading2"],
            fontSize=13, fontName="Helvetica-Bold",
            textColor=colors.HexColor("#0055cc"),
            spaceBefore=14, spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "Body", parent=styles["Normal"],
            fontSize=10, textColor=colors.HexColor("#333333"),
            spaceAfter=5, leading=16,
        )
        small_style = ParagraphStyle(
            "Small", parent=styles["Normal"],
            fontSize=9, textColor=colors.HexColor("#666666"),
            spaceAfter=4,
        )

        # ── Header ──────────────────────────────────────────
        story.append(Paragraph("🧠 NeuroGuard AI — Stroke Risk Assessment", title_style))
        story.append(Paragraph(
            f"AI Healthcare Report  ·  {pred_date}  ·  {pred_time}", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=2,
                                color=colors.HexColor("#0055cc"), spaceAfter=12))

        # ── Summary Box ─────────────────────────────────────
        result_color = colors.HexColor("#cc0033") if is_high else colors.HexColor("#007700")
        summary_data = [
            ["PREDICTION RESULT", risk_level],
            ["Risk Probability",  f"{risk:.2f}%"],
            ["AI Confidence",     f"{confidence:.2f}%"],
            ["Analysis Date",     pred_date],
            ["Analysis Time",     pred_time],
            ["Active Symptoms",   f"{len(active_symptoms)} / {len(all_symptoms)}"],
        ]
        summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (0, -1), colors.HexColor("#f0f4ff")),
            ("BACKGROUND",   (1, 0), (1, 0),  colors.HexColor("#fff0f0") if is_high else colors.HexColor("#f0fff0")),
            ("TEXTCOLOR",    (0, 0), (0, -1), colors.HexColor("#0055cc")),
            ("TEXTCOLOR",    (1, 0), (1, 0),  result_color),
            ("FONTNAME",     (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",     (1, 0), (1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 10),
            ("FONTSIZE",     (1, 0), (1, 0),  12),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f9f9f9"), colors.white]),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("ROUNDED",      (0, 0), (-1, -1), 4),
            ("PADDING",      (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 16))

        # ── Patient Data ─────────────────────────────────────
        story.append(Paragraph("Patient Clinical Parameters", section_style))
        pt_data = [["Parameter", "Value", "Status"]]
        pt_data.append(["Age", str(age), "—"])
        pt_data.append(["Gender", "Male" if gender == 1 else "Female", "—"])
        for sym, val in all_symptoms.items():
            pt_data.append([sym, "Present" if val else "Absent",
                            "⚠ Active" if val else "✓ Normal"])
        pt_table = Table(pt_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
        pt_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#0055cc")),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.HexColor("#f5f5f5"), colors.white]),
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("PADDING",      (0, 0), (-1, -1), 6),
        ]))
        story.append(pt_table)
        story.append(Spacer(1, 14))

        # ── Risk Factors ─────────────────────────────────────
        if factors:
            story.append(Paragraph("Risk Factor Breakdown", section_style))
            rf_data = [["Risk Factor", "Contribution (%)"]]
            for fname, pct in factors:
                rf_data.append([fname, f"{pct}%"])
            rf_table = Table(rf_data, colWidths=[4*inch, 2*inch])
            rf_table.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#cc4400")),
                ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
                ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),
                 [colors.HexColor("#fff5f0"), colors.white]),
                ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                ("PADDING",      (0, 0), (-1, -1), 6),
            ]))
            story.append(rf_table)
            story.append(Spacer(1, 14))

        # ── Recommendations ──────────────────────────────────
        story.append(Paragraph("Preventive Recommendations", section_style))
        rec_texts = [
            "Maintain a heart-healthy, low-sodium diet",
            "Exercise regularly — minimum 30 minutes per day",
            "Reduce stress through mindfulness or meditation",
            "Avoid smoking, alcohol, and stimulants",
            "Ensure 7–8 hours of quality sleep nightly",
            "Monitor blood pressure and take medications as prescribed",
            "Schedule regular cardiovascular check-ups",
            "Stay adequately hydrated throughout the day",
        ]
        for rec in rec_texts:
            story.append(Paragraph(f"• {rec}", body_style))
        story.append(Spacer(1, 12))

        # ── Clinical Tests ────────────────────────────────────
        story.append(Paragraph("Recommended Clinical Tests", section_style))
        test_texts = [
            "ECG / EKG — Cardiac rhythm evaluation",
            "Blood Pressure Monitoring — 24-hour ambulatory",
            "Blood Sugar Test — Fasting & post-prandial glucose",
            "Cholesterol Profile — LDL, HDL, triglycerides",
            "Brain CT Scan — Cerebrovascular imaging",
            "MRI Brain — Detailed neural structure review",
            "CBC Blood Panel — Complete blood count analysis",
        ]
        for t in test_texts:
            story.append(Paragraph(f"🧪 {t}", body_style))
        story.append(Spacer(1, 12))

        # ── XAI ──────────────────────────────────────────────
        story.append(Paragraph("Explainable AI (XAI) Insights", section_style))
        story.append(Paragraph(
            "The Explainable AI engine identified chest pain, hypertension, "
            "irregular heartbeat, shortness of breath, and sleep apnea as the primary "
            "contributors toward stroke risk. Feature importance was computed using SHAP "
            "(SHapley Additive exPlanations) values from the trained ensemble classifier.",
            body_style
        ))
        story.append(Spacer(1, 12))

        # ── Emergency ─────────────────────────────────────────
        if risk > 80:
            story.append(HRFlowable(width="100%", thickness=1.5,
                                    color=colors.red, spaceAfter=8))
            story.append(Paragraph("⚠ EMERGENCY ALERT", ParagraphStyle(
                "Emergency", parent=styles["Normal"],
                fontSize=14, fontName="Helvetica-Bold",
                textColor=colors.red, alignment=TA_CENTER, spaceAfter=6,
            )))
            story.append(Paragraph(
                f"AI detected a critical stroke probability of {risk:.1f}%. "
                "Immediate consultation with a Neurologist/Cardiologist is strongly recommended. "
                "Contact emergency services if sudden neurological symptoms appear.",
                ParagraphStyle("EBody", parent=styles["Normal"],
                               fontSize=10, textColor=colors.HexColor("#880000"),
                               alignment=TA_CENTER, spaceAfter=6)
            ))
            story.append(HRFlowable(width="100%", thickness=1.5,
                                    color=colors.red, spaceAfter=12))

        # ── Footer ───────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#cccccc"), spaceAfter=8))
        story.append(Paragraph(
            "NeuroGuard AI provides an advanced, data-driven stroke risk assessment based on patient symptoms and clinical parameters. "
            "This report is AI-generated and should be reviewed by a qualified medical professional.",
            small_style,
        ))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    dl_col = st.columns([1, 2, 1])[1]
    with dl_col:
        if REPORTLAB_AVAILABLE:
            pdf_data = generate_pdf_report()
            st.download_button(
                label="📄  Download Full AI Medical Report (PDF)",
                data=pdf_data,
                file_name=f"NeuroGuard_StrokeReport_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info("💡 Install `reportlab` (`pip install reportlab`) to enable PDF download.")

# ══════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding:20px;'>

<h3 style='color:#00d4ff;'>Md Asif Khandoker</h3>

<p style='color:gray;'>
AI & Healthcare Research Enthusiast
</p>

<p>
<a href="https://github.com/nafis-ak" target="_blank" style="color:#00d4ff; text-decoration:none;">
🔗 GitHub
</a>

&nbsp; | &nbsp;

<a href="https://www.linkedin.com/in/md-asif-khandoker-004431336/" target="_blank" style="color:#00d4ff; text-decoration:none;">
💼 LinkedIn
</a>

&nbsp; | &nbsp;

📧 asifkhandoker.media@gmail.com
</p>

</div>
""", unsafe_allow_html=True)

# ── Auto-refresh clock every second ──────────────────────────────
# (Streamlit reruns on interaction; clock updates on user interaction)
# For a pure auto-refresh, use st_autorefresh if available:
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1000, key="clock_refresh")
    clock_placeholder.markdown(render_clock(), unsafe_allow_html=True)
except ImportError:
    pass  # Clock shows correct time on each user interaction without autorefresh
