import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreastCare AI · Survival Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&family=Share+Tech+Mono&display=swap');

  :root {
    --neon-cyan: #00f5ff;
    --neon-pink: #ff006e;
    --neon-purple: #7b2fff;
    --dark-bg: #020812;
    --panel-bg: rgba(0, 245, 255, 0.03);
    --border: rgba(0, 245, 255, 0.15);
    --text-primary: #e0f7ff;
    --text-dim: #4a7a8a;
  }

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: var(--dark-bg) !important;
    color: var(--text-primary) !important;
  }

  .stApp {
    background: radial-gradient(ellipse at 10% 0%, rgba(0,245,255,0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 90% 100%, rgba(123,47,255,0.08) 0%, transparent 50%),
                var(--dark-bg) !important;
  }

  /* ── Grid lines background ── */
  .stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
      linear-gradient(rgba(0,245,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,245,255,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
  }

  /* ── Header ── */
  .hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: 0.1em;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple), var(--neon-pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0;
    line-height: 1.1;
  }

  .hero-sub {
    font-family: 'Share Tech Mono', monospace;
    color: var(--text-dim);
    text-align: center;
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    margin-top: 0.4rem;
    margin-bottom: 2rem;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple));
  }
  .metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: var(--text-dim);
    text-transform: uppercase;
  }
  .metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--neon-cyan);
    line-height: 1.1;
  }
  .metric-delta {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #00ff88;
  }

  /* ── Section headers ── */
  .section-header {
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    letter-spacing: 0.15em;
    color: var(--neon-cyan);
    border-left: 3px solid var(--neon-cyan);
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
    text-transform: uppercase;
  }

  /* ── Result panel ── */
  .result-alive {
    background: rgba(0, 255, 136, 0.06);
    border: 1px solid rgba(0, 255, 136, 0.3);
    border-radius: 2px;
    padding: 1.5rem 2rem;
    text-align: center;
  }
  .result-dead {
    background: rgba(255, 0, 110, 0.06);
    border: 1px solid rgba(255, 0, 110, 0.3);
    border-radius: 2px;
    padding: 1.5rem 2rem;
    text-align: center;
  }
  .result-number {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    line-height: 1;
  }
  .alive-number { color: #00ff88; }
  .dead-number  { color: var(--neon-pink); }
  .result-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    margin-top: 0.3rem;
    color: var(--text-dim);
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: rgba(2, 8, 18, 0.95) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] .stSlider > div > div > div {
    background: var(--neon-cyan) !important;
  }

  /* ── Streamlit native overrides ── */
  .stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: var(--neon-cyan) !important;
    border-color: var(--neon-cyan) !important;
  }
  div[data-testid="metric-container"] {
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 1rem !important;
  }
  div[data-testid="metric-container"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    color: var(--text-dim) !important;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.8rem !important;
    color: var(--neon-cyan) !important;
  }

  /* ── Divider ── */
  hr { border-color: var(--border) !important; }

  /* ── Tab styling ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--border);
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    color: var(--text-dim) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
  }
  .stTabs [aria-selected="true"] {
    color: var(--neon-cyan) !important;
    border-bottom: 2px solid var(--neon-cyan) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    background: transparent !important;
    border: 1px solid var(--neon-cyan) !important;
    color: var(--neon-cyan) !important;
    border-radius: 2px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: rgba(0, 245, 255, 0.1) !important;
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.3) !important;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--dark-bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  /* ── Info box ── */
  .info-box {
    background: rgba(123,47,255,0.07);
    border: 1px solid rgba(123,47,255,0.25);
    border-radius: 2px;
    padding: 1rem 1.2rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #c0a0ff;
    line-height: 1.7;
  }
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADER ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Breast_Cancer.csv")
    except FileNotFoundError:
        # Generate synthetic data matching the real dataset structure
        np.random.seed(42)
        n = 4024
        df = pd.DataFrame({
            "Age": np.random.randint(30, 80, n),
            "Race": np.random.choice(["White", "Black", "Other"], n, p=[0.85, 0.10, 0.05]),
            "Marital Status": np.random.choice(["Married", "Single", "Divorced", "Widowed"], n),
            "T Stage": np.random.choice(["T1", "T2", "T3", "T4"], n, p=[0.45, 0.35, 0.15, 0.05]),
            "N Stage": np.random.choice(["N1", "N2", "N3"], n, p=[0.6, 0.3, 0.1]),
            "6th Stage": np.random.choice(["IIA", "IIB", "IIIA", "IIIB", "IIIC"], n),
            "differentiate": np.random.choice(
                ["Well differentiated", "Moderately differentiated", "Poorly differentiated", "Undifferentiated"], n
            ),
            "Grade": np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.4, 0.4, 0.1]),
            "A Stage": np.random.choice(["Regional", "Distant"], n, p=[0.9, 0.1]),
            "Tumor Size": np.random.randint(1, 140, n),
            "Estrogen Status": np.random.choice(["Positive", "Negative"], n, p=[0.85, 0.15]),
            "Progesterone Status": np.random.choice(["Positive", "Negative"], n, p=[0.75, 0.25]),
            "Regional Node Examined": np.random.randint(1, 50, n),
            "Reginol Node Positive": np.random.randint(1, 25, n),
            "Survival Months": np.random.randint(1, 107, n),
            "Status": np.random.choice(["Alive", "Dead"], n, p=[0.85, 0.15])
        })
    return df

@st.cache_resource
def train_models(df):
    X = df[['Tumor Size', 'Regional Node Examined']]
    y = df['Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, preds)
        }
    return results


# ─── PLOTLY THEME ───────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Rajdhani, sans-serif", color="#e0f7ff"),
    xaxis=dict(gridcolor="rgba(0,245,255,0.07)", zerolinecolor="rgba(0,245,255,0.15)"),
    yaxis=dict(gridcolor="rgba(0,245,255,0.07)", zerolinecolor="rgba(0,245,255,0.15)"),
    margin=dict(l=20, r=20, t=40, b=20),
)

CYAN   = "#00f5ff"
PINK   = "#ff006e"
PURPLE = "#7b2fff"
GREEN  = "#00ff88"


# ─── MAIN ───────────────────────────────────────────────────────────────────
df = load_data()
model_results = train_models(df)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">BREASTCARE · AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">// SURVIVAL ANALYSIS SYSTEM · ONCOLOGY DATA MODULE //</div>', unsafe_allow_html=True)

# ── Sidebar controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">INPUT PARAMETERS</div>', unsafe_allow_html=True)

    tumor_min, tumor_max = int(df["Tumor Size"].min()), int(df["Tumor Size"].max())
    node_min,  node_max  = int(df["Regional Node Examined"].min()), int(df["Regional Node Examined"].max())

    tumor_range = st.slider(
        "Tumor Size Range (mm)",
        min_value=tumor_min, max_value=tumor_max,
        value=(tumor_min, min(tumor_min + 30, tumor_max)),
        step=1
    )
    node_range = st.slider(
        "Regional Nodes Examined Range",
        min_value=node_min, max_value=node_max,
        value=(node_min, min(node_min + 15, node_max)),
        step=1
    )

    st.markdown("---")
    st.markdown('<div class="section-header">MODEL SELECT</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Classifier",
        options=list(model_results.keys()),
        index=2,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    📡 DATASET · Breast Cancer Surveillance<br>
    🔬 FEATURES · Tumor Size + Regional Nodes<br>
    🎯 TARGET · Patient Survival Status<br>
    ⚙️ MODELS · LR · DT · RF Ensemble
    </div>
    """, unsafe_allow_html=True)


# ── Filter data ─────────────────────────────────────────────────────────────
filtered = df[
    (df["Tumor Size"].between(*tumor_range)) &
    (df["Regional Node Examined"].between(*node_range))
]
total     = len(filtered)
alive     = (filtered["Status"] == "Alive").sum()
dead      = (filtered["Status"] == "Dead").sum()
mortality = (dead / total * 100) if total > 0 else 0


# ── KPI row ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("PATIENTS IN RANGE", f"{total:,}")
with c2:
    st.metric("ALIVE", f"{alive:,}", delta=f"{100-mortality:.1f}% survival")
with c3:
    st.metric("DECEASED", f"{dead:,}", delta=f"{mortality:.1f}% mortality", delta_color="inverse")
with c4:
    acc = model_results[selected_model]["accuracy"]
    st.metric("MODEL ACCURACY", f"{acc:.2%}")

st.markdown("---")

# ── Result spotlight ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">SURVIVAL ANALYSIS · SELECTED RANGE</div>', unsafe_allow_html=True)

ra, rb = st.columns(2)
with ra:
    st.markdown(f"""
    <div class="result-alive">
      <div class="result-number alive-number">{alive:,}</div>
      <div class="result-label">▲ PATIENTS ALIVE</div>
      <div style="font-family:'Share Tech Mono';font-size:1rem;color:#00ff88;margin-top:0.5rem;">
        {100-mortality:.1f}% SURVIVAL RATE
      </div>
    </div>""", unsafe_allow_html=True)

with rb:
    st.markdown(f"""
    <div class="result-dead">
      <div class="result-number dead-number">{dead:,}</div>
      <div class="result-label">▼ PATIENTS DECEASED</div>
      <div style="font-family:'Share Tech Mono';font-size:1rem;color:#ff006e;margin-top:0.5rem;">
        {mortality:.1f}% MORTALITY RATE
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["DISTRIBUTION", "MODEL PERFORMANCE", "DATA EXPLORER"])

# ─ Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        # Survival donut
        fig_donut = go.Figure(go.Pie(
            labels=["Alive", "Deceased"],
            values=[alive, dead],
            hole=0.65,
            marker=dict(colors=[GREEN, PINK],
                        line=dict(color="#020812", width=2)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
        ))
        fig_donut.add_annotation(
            text=f"<b>{mortality:.1f}%</b><br><span style='font-size:10px'>MORTALITY</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="Orbitron", size=16, color=PINK)
        )
        fig_donut.update_layout(**PLOTLY_LAYOUT, title=dict(text="Survival Distribution", font=dict(family="Orbitron", size=12, color=CYAN)))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        # Tumor size histogram by status
        fig_hist = go.Figure()
        for status, color in [("Alive", GREEN), ("Dead", PINK)]:
            sub = filtered[filtered["Status"] == status]["Tumor Size"]
            fig_hist.add_trace(go.Histogram(
                x=sub, name=status,
                marker_color=color,
                opacity=0.75,
                nbinsx=20,
                hovertemplate=f"<b>{status}</b><br>Tumor Size: %{{x}}<br>Count: %{{y}}<extra></extra>"
            ))
        fig_hist.update_layout(**PLOTLY_LAYOUT,
            title=dict(text="Tumor Size Distribution by Status", font=dict(family="Orbitron", size=12, color=CYAN)),
            barmode="overlay",
            legend=dict(font=dict(family="Rajdhani")))
        st.plotly_chart(fig_hist, use_container_width=True)

    # Scatter
    if total > 0 and total <= 5000:
        sample = filtered.sample(min(500, total), random_state=42)
    else:
        sample = filtered.sample(500, random_state=42) if total > 500 else filtered

    fig_scatter = px.scatter(
        sample, x="Tumor Size", y="Regional Node Examined",
        color="Status",
        color_discrete_map={"Alive": GREEN, "Dead": PINK},
        opacity=0.6,
        hover_data=["Tumor Size", "Regional Node Examined", "Status"]
    )
    fig_scatter.update_traces(marker=dict(size=5))
    fig_scatter.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Tumor Size vs Nodes Examined  ·  Scatter Plot", font=dict(family="Orbitron", size=12, color=CYAN)),
        legend=dict(font=dict(family="Rajdhani")))
    st.plotly_chart(fig_scatter, use_container_width=True)


# ─ Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    names = list(model_results.keys())
    accs  = [model_results[n]["accuracy"] for n in names]

    colors = [CYAN if n == selected_model else PURPLE for n in names]

    fig_bar = go.Figure(go.Bar(
        x=names, y=accs,
        marker=dict(color=colors, line=dict(color="#020812", width=1)),
        text=[f"{a:.2%}" for a in accs],
        textposition="outside",
        textfont=dict(family="Orbitron", size=11, color=CYAN),
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2%}<extra></extra>"
    ))
    fig_bar.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Model Accuracy Comparison  ·  test_size=0.2", font=dict(family="Orbitron", size=12, color=CYAN)),
        showlegend=False)
    fig_bar.update_yaxes(tickformat=".0%", range=[0.7, 0.92])
    st.plotly_chart(fig_bar, use_container_width=True)

    # test_size accuracy table from notebook
    st.markdown('<div class="section-header">ACCURACY · MULTIPLE TEST SIZES</div>', unsafe_allow_html=True)
    test_sizes_data = {
        "Test Size": ["0.1", "0.2", "0.5", "0.7", "0.9"],
        "Logistic Regression": [0.8536, 0.8447, 0.8434, 0.8431, 0.8462],
        "Decision Tree":       [0.7990, 0.7925, 0.7763, 0.7820, 0.7662],
        "Random Forest":       [0.8065, 0.7963, 0.7858, 0.8001, 0.7998],
    }
    ts_df = pd.DataFrame(test_sizes_data)

    fig_line = go.Figure()
    colors_line = [CYAN, PINK, PURPLE]
    for i, col in enumerate(["Logistic Regression", "Decision Tree", "Random Forest"]):
        fig_line.add_trace(go.Scatter(
            x=ts_df["Test Size"], y=ts_df[col],
            mode="lines+markers",
            name=col,
            line=dict(color=colors_line[i], width=2),
            marker=dict(size=8, symbol="diamond"),
            hovertemplate=f"<b>{col}</b><br>Test Size: %{{x}}<br>Accuracy: %{{y:.2%}}<extra></extra>"
        ))
    fig_line.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Accuracy vs Test Size", font=dict(family="Orbitron", size=12, color=CYAN)),
        legend=dict(font=dict(family="Rajdhani", size=13)))
    fig_line.update_yaxes(tickformat=".0%", range=[0.74, 0.88])
    st.plotly_chart(fig_line, use_container_width=True)


# ─ Tab 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">FILTERED DATASET PREVIEW</div>', unsafe_allow_html=True)
    st.markdown(f"<p style='font-family:Share Tech Mono;font-size:0.75rem;color:#4a7a8a;'>Showing {min(200, total)} of {total} records matching current filter</p>", unsafe_allow_html=True)
    st.dataframe(
        filtered.head(200),
        use_container_width=True,
        height=380
    )

    # Status breakdown by node range
    if total > 0:
        st.markdown('<div class="section-header">NODE BINS · SURVIVAL BREAKDOWN</div>', unsafe_allow_html=True)
        bins = pd.cut(filtered["Regional Node Examined"], bins=5)
        group = filtered.groupby([bins, "Status"], observed=True).size().unstack(fill_value=0).reset_index()
        group.columns = [str(c) for c in group.columns]
        node_col = group.columns[0]
        group[node_col] = group[node_col].astype(str)

        fig_stacked = go.Figure()
        for status, color in [("Alive", GREEN), ("Dead", PINK)]:
            if status in group.columns:
                fig_stacked.add_trace(go.Bar(
                    x=group[node_col], y=group[status],
                    name=status, marker_color=color,
                    hovertemplate=f"<b>{status}</b><br>Node range: %{{x}}<br>Count: %{{y}}<extra></extra>"
                ))
        fig_stacked.update_layout(**PLOTLY_LAYOUT,
            title=dict(text="Survival by Node Examination Range", font=dict(family="Orbitron", size=12, color=CYAN)),
            barmode="stack",
            legend=dict(font=dict(family="Rajdhani")))
        st.plotly_chart(fig_stacked, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#4a7a8a;padding:1rem 0;">
  BREASTCARE · AI SYSTEM &nbsp;·&nbsp; BUILT WITH STREAMLIT + SCIKIT-LEARN + PLOTLY
  &nbsp;·&nbsp; DATA: BREAST CANCER SURVEILLANCE DATASET
</div>
""", unsafe_allow_html=True)
