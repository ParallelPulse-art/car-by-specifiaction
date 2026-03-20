"""
🚗 Car Price Prediction App
============================
Run with:  streamlit run car_price_app.py
Install :  pip install streamlit scikit-learn pandas seaborn matplotlib plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── Page config ───────────────────────────
st.set_page_config(
    page_title="🚗 Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background: #0f0f1a; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] { color: #a0aec0 !important; font-size: 13px; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 28px; font-weight: 700; }
    [data-testid="stMetricDelta"] { color: #68d391 !important; }

    /* Prediction banner */
    .pred-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(102,126,234,0.4);
    }
    .pred-banner h1 { color: white; font-size: 3rem; margin: 0; }
    .pred-banner p  { color: rgba(255,255,255,0.85); font-size: 1rem; margin: 4px 0 0; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 24px 0 8px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
        border-right: 1px solid #0f3460;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e;
        border-radius: 8px;
        color: #a0aec0;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }

    /* Info boxes */
    .insight-box {
        background: #1a1a2e;
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e2e8f0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Data Loading ──────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("car_price_prediction_.csv")
    df.dropna(inplace=True)
    df["Car Age"] = 2024 - df["Year"]
    df["Price_log"] = np.log1p(df["Price"])
    return df

@st.cache_resource
def train_models(df):
    features = ["Year", "Engine Size", "Mileage", "Car Age",
                "Brand", "Fuel Type", "Transmission", "Condition"]
    target = "Price"

    X = df[features].copy()
    y = df[target]

    # Encode categoricals
    le_map = {}
    for col in ["Brand", "Fuel Type", "Transmission", "Condition"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_map[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.08,
            max_depth=5, subsample=0.85, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42
        ),
        "Ridge Regression": Ridge(alpha=10.0),
    }

    results = {}
    for name, model in models.items():
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_train) if name == "Ridge Regression" else X_train
        Xte_s = scaler.transform(X_test)       if name == "Ridge Regression" else X_test

        model.fit(Xtr_s, y_train)
        preds = model.predict(Xte_s)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        results[name] = {
            "model": model,
            "scaler": scaler if name == "Ridge Regression" else None,
            "preds": preds,
            "MAE": mae, "RMSE": rmse, "R²": r2,
            "MAPE (%)": mape,
            "X_test": Xte_s, "y_test": y_test,
        }

    # Feature importance from best model (GB)
    gb = results["Gradient Boosting"]["model"]
    importances = pd.Series(gb.feature_importances_, index=features).sort_values(ascending=False)

    return results, le_map, features, importances, X_train, y_train

# ─────────────────────────── Load ──────────────────────────────────
df = load_data()
results, le_map, features, importances, X_train, y_train = train_models(df)

BEST_MODEL = "Gradient Boosting"

# ═══════════════════════════ SIDEBAR ═══════════════════════════════
with st.sidebar:
    st.markdown("### 🚗 Car Price Predictor")
    st.markdown("---")

    st.markdown("#### 📋 Enter Car Details")

    brand        = st.selectbox("Brand",        sorted(df["Brand"].unique()))
    fuel_type    = st.selectbox("Fuel Type",     sorted(df["Fuel Type"].unique()))
    transmission = st.selectbox("Transmission",  sorted(df["Transmission"].unique()))
    condition    = st.selectbox("Condition",     sorted(df["Condition"].unique()))

    st.markdown("---")
    year         = st.slider("Year",         2000, 2023, 2015)
    engine_size  = st.slider("Engine Size (L)", 1.0, 6.0, 2.5, 0.1)
    mileage      = st.slider("Mileage (km)", 0, 300_000, 80_000, 1_000)

    st.markdown("---")
    model_choice = st.selectbox("🤖 Model", list(results.keys()))

    predict_btn = st.button("🔮 Predict Price", use_container_width=True, type="primary")


# ─────────────────────────── Prediction ────────────────────────────
def make_prediction(brand, fuel_type, transmission, condition,
                    year, engine_size, mileage, model_choice):
    car_age = 2024 - year
    row = pd.DataFrame([{
        "Year": year, "Engine Size": engine_size,
        "Mileage": mileage, "Car Age": car_age,
        "Brand": brand, "Fuel Type": fuel_type,
        "Transmission": transmission, "Condition": condition,
    }])
    for col in ["Brand", "Fuel Type", "Transmission", "Condition"]:
        row[col] = le_map[col].transform(row[col].astype(str))

    info = results[model_choice]
    X_in = info["scaler"].transform(row) if info["scaler"] else row
    pred = info["model"].predict(X_in)[0]
    return max(pred, 0)


if predict_btn:
    pred_price = make_prediction(
        brand, fuel_type, transmission, condition,
        year, engine_size, mileage, model_choice
    )
    st.session_state["pred_price"]  = pred_price
    st.session_state["model_used"]  = model_choice
    st.session_state["input_brand"] = brand


# ═══════════════════════════ MAIN AREA ═════════════════════════════
st.markdown("# 🚗 Car Price Prediction Dashboard")
st.markdown("*Predict market value · Explore data · Compare models*")

if "pred_price" in st.session_state:
    p = st.session_state["pred_price"]
    m = st.session_state["model_used"]
    b = st.session_state["input_brand"]
    st.markdown(f"""
    <div class="pred-banner">
        <p>💡 Predicted Price using <b>{m}</b></p>
        <h1>$ {p:,.0f}</h1>
        <p>Estimated market value for your <b>{b}</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Confidence range
    mae = results[m]["MAE"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Lower Estimate",  f"${max(p - mae,0):,.0f}", "conservative")
    c2.metric("Predicted Price", f"${p:,.0f}",             "estimate")
    c3.metric("Upper Estimate",  f"${p + mae:,.0f}",       "optimistic")
    st.markdown("---")

# ─────────────────────────── Tabs ──────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🔍 EDA", "🤖 Model Performance",
    "📈 Feature Importance", "🔗 Correlations"
])


# ══════════════════ TAB 1 — OVERVIEW ══════════════════
with tab1:
    st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",    f"{len(df):,}")
    c2.metric("Avg Price",        f"${df['Price'].mean():,.0f}")
    c3.metric("Price Range",      f"${df['Price'].min():,.0f} – ${df['Price'].max():,.0f}")
    c4.metric("Avg Mileage",      f"{df['Mileage'].mean():,.0f} km")

    st.markdown('<p class="section-header">Price Distribution by Brand</p>', unsafe_allow_html=True)
    fig = px.box(
        df, x="Brand", y="Price", color="Brand",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        template="plotly_dark"
    )
    fig.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
        showlegend=False, height=420,
        yaxis_title="Price ($)", xaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Fuel Type Distribution</p>', unsafe_allow_html=True)
        fuel_counts = df["Fuel Type"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=fuel_counts.index, values=fuel_counts.values,
            hole=0.5,
            marker_colors=["#667eea","#764ba2","#f093fb","#f5576c"],
        ))
        fig2.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
            font_color="#e2e8f0", height=300, showlegend=True
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Condition Split</p>', unsafe_allow_html=True)
        cond_counts = df["Condition"].value_counts()
        fig3 = go.Figure(go.Bar(
            x=cond_counts.index, y=cond_counts.values,
            marker_color=["#667eea","#764ba2","#f5576c"],
            text=cond_counts.values, textposition="auto"
        ))
        fig3.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
            font_color="#e2e8f0", height=300,
            xaxis_title="", yaxis_title="Count"
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<p class="section-header">Raw Data Sample</p>', unsafe_allow_html=True)
    st.dataframe(
        df.drop(columns=["Car Age","Price_log"]).sample(10, random_state=1).reset_index(drop=True),
        use_container_width=True
    )


# ══════════════════ TAB 2 — EDA ══════════════════
with tab2:
    st.markdown('<p class="section-header">Price vs Mileage</p>', unsafe_allow_html=True)
    fig = px.scatter(
        df, x="Mileage", y="Price", color="Fuel Type", size="Engine Size",
        hover_data=["Brand","Year","Condition"],
        color_discrete_sequence=px.colors.qualitative.Vivid,
        template="plotly_dark", opacity=0.7
    )
    fig.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
        font_color="#e2e8f0", height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Price by Year</p>', unsafe_allow_html=True)
        year_avg = df.groupby("Year")["Price"].mean().reset_index()
        fig4 = px.line(
            year_avg, x="Year", y="Price",
            color_discrete_sequence=["#667eea"],
            template="plotly_dark", markers=True
        )
        fig4.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
            font_color="#e2e8f0", height=320
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Engine Size vs Price</p>', unsafe_allow_html=True)
        fig5 = px.scatter(
            df, x="Engine Size", y="Price", color="Transmission",
            color_discrete_sequence=["#667eea","#f5576c"],
            template="plotly_dark", opacity=0.6
        )
        fig5.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
            font_color="#e2e8f0", height=320
        )
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<p class="section-header">Price Distribution</p>', unsafe_allow_html=True)
    fig6 = px.histogram(
        df, x="Price", nbins=60, color="Condition",
        color_discrete_sequence=["#667eea","#764ba2","#f5576c"],
        template="plotly_dark", barmode="overlay", opacity=0.75
    )
    fig6.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
        font_color="#e2e8f0", height=350
    )
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown('<p class="section-header">Avg Price by Brand & Fuel Type (Heatmap)</p>', unsafe_allow_html=True)
    pivot = df.pivot_table(index="Brand", columns="Fuel Type", values="Price", aggfunc="mean")
    fig_heat, ax = plt.subplots(figsize=(10, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    sns.heatmap(
        pivot, annot=True, fmt=",.0f", cmap="viridis",
        linewidths=0.5, linecolor="#0f0f1a",
        ax=ax, annot_kws={"color":"white","size":9}
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#0f0f1a")
    plt.title("Average Price ($)", color="white", pad=10)
    plt.xticks(color="white"); plt.yticks(color="white")
    plt.tight_layout()
    st.pyplot(fig_heat)


# ══════════════════ TAB 3 — MODEL PERFORMANCE ══════════════════
with tab3:
    st.markdown('<p class="section-header">Model Comparison</p>', unsafe_allow_html=True)

    metrics_rows = []
    for name, info in results.items():
        metrics_rows.append({
            "Model": name,
            "R² Score": round(info["R²"], 4),
            "MAE ($)": f"{info['MAE']:,.0f}",
            "RMSE ($)": f"{info['RMSE']:,.0f}",
            "MAPE (%)": round(info["MAPE (%)"], 2),
        })
    metrics_df = pd.DataFrame(metrics_rows)
    st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

    # Bar chart of R² scores
    names_  = list(results.keys())
    r2s_    = [results[n]["R²"]   for n in names_]
    maes_   = [results[n]["MAE"]  for n in names_]
    rmses_  = [results[n]["RMSE"] for n in names_]

    fig_m = make_subplots(rows=1, cols=3,
        subplot_titles=["R² Score (higher=better)",
                         "MAE ($) (lower=better)",
                         "RMSE ($) (lower=better)"])
    colors = ["#667eea","#764ba2","#f5576c"]

    for i, (vals, title) in enumerate(
        [(r2s_, "R²"), (maes_, "MAE"), (rmses_, "RMSE")], start=1
    ):
        fig_m.add_trace(
            go.Bar(x=names_, y=vals, marker_color=colors,
                   text=[f"{v:,.3f}" if i==1 else f"${v:,.0f}" for v in vals],
                   textposition="auto"),
            row=1, col=i
        )
    fig_m.update_layout(
        template="plotly_dark", showlegend=False, height=380,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a", font_color="#e2e8f0"
    )
    st.plotly_chart(fig_m, use_container_width=True)

    # Actual vs Predicted scatter for chosen model
    st.markdown(f'<p class="section-header">Actual vs Predicted — {model_choice}</p>', unsafe_allow_html=True)
    info = results[model_choice]
    y_test_vals = info["y_test"].values
    y_pred_vals = info["preds"]

    fig_ap = go.Figure()
    fig_ap.add_trace(go.Scatter(
        x=y_test_vals, y=y_pred_vals, mode="markers",
        marker=dict(color="#667eea", opacity=0.5, size=5),
        name="Predictions"
    ))
    lo, hi = y_test_vals.min(), y_test_vals.max()
    fig_ap.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="#f5576c", dash="dash", width=2),
        name="Perfect fit"
    ))
    fig_ap.update_layout(
        template="plotly_dark", height=420,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a", font_color="#e2e8f0",
        xaxis_title="Actual Price ($)", yaxis_title="Predicted Price ($)"
    )
    st.plotly_chart(fig_ap, use_container_width=True)

    # Residuals
    st.markdown('<p class="section-header">Residuals Distribution</p>', unsafe_allow_html=True)
    residuals = y_test_vals - y_pred_vals
    fig_res = px.histogram(
        x=residuals, nbins=60,
        color_discrete_sequence=["#764ba2"],
        template="plotly_dark", labels={"x": "Residual ($)"}
    )
    fig_res.add_vline(x=0, line_color="#f5576c", line_dash="dash", line_width=2)
    fig_res.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
        font_color="#e2e8f0", height=320
    )
    st.plotly_chart(fig_res, use_container_width=True)


# ══════════════════ TAB 4 — FEATURE IMPORTANCE ══════════════════
with tab4:
    st.markdown('<p class="section-header">Feature Importance (Gradient Boosting)</p>', unsafe_allow_html=True)

    imp_df = importances.reset_index()
    imp_df.columns = ["Feature", "Importance"]

    fig_fi = px.bar(
        imp_df, x="Importance", y="Feature",
        orientation="h", color="Importance",
        color_continuous_scale=["#667eea","#764ba2","#f5576c"],
        template="plotly_dark", text="Importance"
    )
    fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_fi.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
        font_color="#e2e8f0", height=460,
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<p class="section-header">Insights</p>', unsafe_allow_html=True)
    for feat, score in importances.items():
        pct = score / importances.sum() * 100
        st.markdown(
            f'<div class="insight-box">🔹 <b>{feat}</b> — contributes <b>{pct:.1f}%</b> to the model\'s predictions</div>',
            unsafe_allow_html=True
        )

    # Partial dependence — price vs mileage
    st.markdown('<p class="section-header">Price Sensitivity: Mileage</p>', unsafe_allow_html=True)
    mileage_range = np.linspace(df["Mileage"].min(), df["Mileage"].max(), 200)
    base_row = pd.DataFrame([{
        "Year": 2015, "Engine Size": 2.5, "Mileage": 0, "Car Age": 9,
        "Brand": le_map["Brand"].transform(["Toyota"])[0],
        "Fuel Type": le_map["Fuel Type"].transform(["Petrol"])[0],
        "Transmission": le_map["Transmission"].transform(["Automatic"])[0],
        "Condition": le_map["Condition"].transform(["Used"])[0],
    }] * 200)
    base_row["Mileage"] = mileage_range

    gb_model = results["Gradient Boosting"]["model"]
    pdp_preds = gb_model.predict(base_row)

    fig_pdp = go.Figure(go.Scatter(
        x=mileage_range, y=pdp_preds,
        mode="lines", line=dict(color="#667eea", width=3)
    ))
    fig_pdp.add_trace(go.Scatter(
        x=mileage_range, y=pdp_preds,
        fill="tozeroy", fillcolor="rgba(102,126,234,0.15)", mode="none"
    ))
    fig_pdp.update_layout(
        template="plotly_dark", height=320,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a", font_color="#e2e8f0",
        xaxis_title="Mileage (km)", yaxis_title="Predicted Price ($)"
    )
    st.plotly_chart(fig_pdp, use_container_width=True)


# ══════════════════ TAB 5 — CORRELATIONS ══════════════════
with tab5:
    st.markdown('<p class="section-header">Numeric Correlation Matrix</p>', unsafe_allow_html=True)

    num_cols = ["Year","Engine Size","Mileage","Car Age","Price"]
    corr = df[num_cols].corr()

    fig_corr, ax = plt.subplots(figsize=(8, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        linewidths=0.5, linecolor="#0f0f1a", ax=ax,
        annot_kws={"color":"white","size":11}
    )
    ax.tick_params(colors="white")
    plt.title("Feature Correlations", color="white", pad=12)
    plt.xticks(color="white", rotation=30)
    plt.yticks(color="white", rotation=0)
    plt.tight_layout()
    st.pyplot(fig_corr)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Avg Price by Transmission</p>', unsafe_allow_html=True)
        trans_avg = df.groupby("Transmission")["Price"].mean().reset_index()
        fig_t = px.bar(
            trans_avg, x="Transmission", y="Price",
            color="Transmission",
            color_discrete_sequence=["#667eea","#764ba2"],
            template="plotly_dark", text="Price"
        )
        fig_t.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_t.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
            font_color="#e2e8f0", height=320, showlegend=False
        )
        st.plotly_chart(fig_t, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Avg Price by Condition</p>', unsafe_allow_html=True)
        cond_avg = df.groupby("Condition")["Price"].mean().reset_index()
        fig_c = px.bar(
            cond_avg, x="Condition", y="Price",
            color="Condition",
            color_discrete_sequence=["#667eea","#764ba2","#f5576c"],
            template="plotly_dark", text="Price"
        )
        fig_c.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_c.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
            font_color="#e2e8f0", height=320, showlegend=False
        )
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown('<p class="section-header">Price by Brand & Year (Violin)</p>', unsafe_allow_html=True)
    fig_v = px.violin(
        df[df["Year"] >= 2015], x="Brand", y="Price",
        color="Brand", box=True, points=False,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        template="plotly_dark"
    )
    fig_v.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
        font_color="#e2e8f0", height=400, showlegend=False
    )
    st.plotly_chart(fig_v, use_container_width=True)


# ─────────────────────────── Footer ────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4a5568;font-size:0.8rem;'>"
    "🚗 Car Price Predictor | Built with Streamlit, Scikit-learn, Pandas, Seaborn, Matplotlib & Plotly"
    "</p>",
    unsafe_allow_html=True
)
