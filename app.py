"""
Environmental Factors K-Means Clustering — Streamlit App
==========================================================
Interactive AI Dashboard for clustering environmental sensor readings.
"""

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Environmental Factors — K-Means Clustering",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set global Matplotlib dark-mode defaults
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "text.color": "#f8fafc",
    "axes.labelcolor": "#cbd5e1",
    "xtick.color": "#cbd5e1",
    "ytick.color": "#cbd5e1",
    "axes.edgecolor": "#475569",
    "grid.color": "#334155",
    "grid.alpha": 0.4
})

# ============================================================
# CUSTOM CSS — DARK UI WITH HIGH CONTRAST ACCENTS
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 15%, rgba(56,189,248,.12), transparent 28%),
            radial-gradient(circle at 85% 20%, rgba(34,197,94,.12), transparent 28%),
            radial-gradient(circle at 50% 90%, rgba(168,85,247,.10), transparent 32%),
            linear-gradient(135deg, #07111f 0%, #0b1728 45%, #071827 100%);
        color: #f8fafc;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Hero Banner */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.2rem 2.5rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(14,165,233,.22), rgba(34,197,94,.18)), rgba(15,23,42,.78);
        border: 1px solid rgba(148,163,184,.25);
        box-shadow: 0 20px 60px rgba(0,0,0,.35);
        backdrop-filter: blur(16px);
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 800;
        letter-spacing: -1.2px;
        background: linear-gradient(90deg, #38bdf8, #4ade80, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1.05rem;
        max-width: 800px;
        line-height: 1.6;
        margin-top: .6rem;
    }

    .pill {
        display: inline-block;
        padding: .35rem .8rem;
        margin-bottom: .6rem;
        border-radius: 999px;
        background: rgba(56,189,248,.15);
        border: 1px solid rgba(56,189,248,.4);
        color: #38bdf8;
        font-size: .8rem;
        font-weight: 700;
        letter-spacing: .5px;
    }

    /* Cards */
    .glass-card {
        background: rgba(15,23,42,.68);
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 18px;
        padding: 1.25rem;
        box-shadow: 0 12px 35px rgba(0,0,0,.22);
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: .3rem;
    }

    /* Metric Overrides */
    div[data-testid="stMetric"] {
        background: rgba(30,41,59,.78) !important;
        border: 1px solid rgba(148,163,184,.25) !important;
        border-radius: 16px !important;
        padding: 1rem 1.15rem !important;
    }

    div[data-testid="stMetricLabel"] > label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 2.5rem 0 .5rem;
        font-size: .85rem;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS & SETUP
# ============================================================
DEFAULT_FILE = "environmental_factors.csv"
FEATURE_COLS = [
    "temperature",
    "humidity",
    "wind_speed",
    "carbon_emissions",
    "solar_irradiance",
    "pollution_level",
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================
@st.cache_data
def load_data(path_or_buffer):
    return pd.read_csv(path_or_buffer)

@st.cache_data
def generate_synthetic_data(num_samples=250, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "temperature": rng.uniform(10.0, 45.0, num_samples),
        "humidity": rng.uniform(20.0, 95.0, num_samples),
        "wind_speed": rng.uniform(1.0, 30.0, num_samples),
        "carbon_emissions": rng.uniform(200.0, 800.0, num_samples),
        "solar_irradiance": rng.uniform(100.0, 1200.0, num_samples),
        "pollution_level": rng.uniform(10.0, 300.0, num_samples),
    })

@st.cache_data
def compute_elbow(data_scaled, max_k):
    inertias = []
    for kk in range(1, max_k + 1):
        km = KMeans(n_clusters=kk, random_state=42, n_init=10)
        km.fit(data_scaled)
        inertias.append(km.inertia_)
    return inertias

@st.cache_resource
def fit_kmeans(data_scaled, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(data_scaled)
    return km, labels

# ============================================================
# SIDEBAR — DATA SOURCE & CONTROLS
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="
            padding:1rem;
            border-radius:16px;
            background:linear-gradient(135deg, rgba(56,189,248,.16), rgba(34,197,94,.10));
            border:1px solid rgba(56,189,248,.25);
            margin-bottom:1rem;
        ">
            <div style="font-size:2rem;">🌍</div>
            <div style="font-size:1.2rem;font-weight:800;color:#f8fafc;">Cluster AI</div>
            <div style="color:#94a3b8;font-size:.8rem;">Environmental Factor Engine</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Data Source")
    
    df = None
    if os.path.exists(DEFAULT_FILE):
        df = load_data(DEFAULT_FILE)
        st.success(f"✓ Loaded local file ({len(df)} rows)")
    else:
        uploaded = st.file_uploader(f"Upload '{DEFAULT_FILE}'", type=["csv"])
        if uploaded is not None:
            df = load_data(uploaded)
            st.success(f"✓ Uploaded {len(df)} rows")
        else:
            st.info("Local file missing. Using synthetic data.")
            df = generate_synthetic_data()

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    data = df[FEATURE_COLS].dropna().reset_index(drop=True)

    st.markdown("---")
    st.markdown("### 2. Model Settings")
    max_k = min(10, len(data) - 1) if len(data) > 1 else 2
    k = st.slider("Number of clusters (k)", 2, max_k, value=min(4, max_k))
    show_elbow = st.checkbox("Show Elbow Method Chart", value=True)

# ============================================================
# DATA PREPARATION & SCALING
# ============================================================
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# ============================================================
# HERO HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="pill">● ENVIRONMENT INTELLIGENCE</div>
        <h1>Environmental Factors Clustering</h1>
        <p>
            Uncover patterns in temperature, air quality, solar irradiance, and emission indicators 
            using Unsupervised K-Means Machine Learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("🔍 Preview Dataset", expanded=False):
    st.dataframe(data.head(20), use_container_width=True)
    st.caption(f"Total observations: {len(data):,} rows × {len(FEATURE_COLS)} features")

# ============================================================
# ELBOW METHOD
# ============================================================
if show_elbow:
    st.markdown('<div class="section-title">📉 Optimal Cluster Selection (Elbow Method)</div>', unsafe_allow_html=True)
    inertias = compute_elbow(data_scaled, max_k)
    
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(range(1, max_k + 1), inertias, marker="o", color="#38bdf8", linewidth=2, markersize=6)
    ax.axvline(k, color="#4ade80", linestyle="--", linewidth=1.5, label=f"Selected k={k}")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.grid(True)
    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ============================================================
# MODEL FITTING & METRICS
# ============================================================
kmeans, labels = fit_kmeans(data_scaled, k)
data_clustered = data.copy()
data_clustered["cluster"] = labels

sil_score = silhouette_score(data_scaled, labels) if len(set(labels)) > 1 else float("nan")

c1, c2, c3 = st.columns(3)
c1.metric("Active Clusters (k)", k)
c2.metric("Silhouette Score", f"{sil_score:.3f}")
c3.metric("Total Data Samples", f"{len(data):,}")

# ============================================================
# CLUSTER VISUALIZATION (PCA)
# ============================================================
st.markdown("---")
st.markdown('<div class="section-title">🗺️ Cluster Projection (2D PCA)</div>', unsafe_allow_html=True)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(data_scaled)
centers_2d = pca.transform(kmeans.cluster_centers_)

fig2, ax2 = plt.subplots(figsize=(8, 4.5))
scatter = ax2.scatter(
    coords[:, 0], coords[:, 1], 
    c=labels, cmap="tab10", s=25, alpha=0.75, edgecolors="none"
)
ax2.scatter(
    centers_2d[:, 0], centers_2d[:, 1],
    c="#f8fafc", marker="X", s=220, edgecolors="#000000", linewidth=1.5, label="Centroids"
)
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
ax2.grid(True)
ax2.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)

# ============================================================
# CLUSTER PROFILES
# ============================================================
st.markdown("---")
st.markdown('<div class="section-title">📊 Cluster Profiles (Feature Averages)</div>', unsafe_allow_html=True)

profile = data_clustered.groupby("cluster")[FEATURE_COLS].mean().round(2)
profile["sample_count"] = data_clustered.groupby("cluster").size()
st.dataframe(profile, use_container_width=True)

with st.expander("📥 Download Full Clustered Dataset"):
    st.dataframe(data_clustered, use_container_width=True)
    st.download_button(
        "⬇️ Download Clustered CSV",
        data=data_clustered.to_csv(index=False).encode("utf-8"),
        file_name="clustered_environmental_data.csv",
        mime="text/csv",
    )

# ============================================================
# INFERENCE — PREDICT NEW READINGS
# ============================================================
st.markdown("---")
st.markdown('<div class="section-title">🔮 Classify New Environmental Reading</div>', unsafe_allow_html=True)
st.caption("Adjust sliders below to classify real-time sensor metrics into one of the trained clusters.")

pred_cols = st.columns(3)
input_values = {}

for i, feat in enumerate(FEATURE_COLS):
    col = pred_cols[i % 3]
    lo, hi = float(data[feat].min()), float(data[feat].max())
    default = float(data[feat].mean())
    step = (hi - lo) / 100 if hi != lo else 0.1
    input_values[feat] = col.slider(
        feat.replace("_", " ").title(),
        min_value=round(lo, 2),
        max_value=round(hi, 2),
        value=round(default, 2),
        step=round(step, 2)
    )

if st.button("Classify Reading", type="primary", use_container_width=True):
    new_point = pd.DataFrame([input_values])[FEATURE_COLS]
    new_scaled = scaler.transform(new_point)
    pred_cluster = int(kmeans.predict(new_scaled)[0])

    st.success(f"🎯 Assigned to **Cluster {pred_cluster}**")
    
    st.markdown("#### Belonging Cluster Means:")
    st.dataframe(profile.loc[[pred_cluster]], use_container_width=True)

    new_2d = pca.transform(new_scaled)
    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    ax3.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=20, alpha=0.35)
    ax3.scatter(centers_2d[:, 0], centers_2d[:, 1], c="#f8fafc", marker="X", s=180, edgecolors="#000000")
    ax3.scatter(
        new_2d[:, 0], new_2d[:, 1],
        c="#ef4444", marker="*", s=350, edgecolors="#ffffff", label=f"New Input (Cluster {pred_cluster})"
    )
    ax3.set_xlabel("Principal Component 1")
    ax3.set_ylabel("Principal Component 2")
    ax3.grid(True)
    ax3.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        🌍 Environmental Analytics Dashboard &nbsp;•&nbsp; K-Means Unsupervised Learning &nbsp;•&nbsp; Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)