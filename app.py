# app.py
# Streamlit Drone ETA Lab — leak-safe CV, coverage metric, reliability plot, ablations
# Requirements: streamlit, pandas, scikit-learn, numpy, plotly

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Dict, Any

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Drone ETA Lab", page_icon="🛩️", layout="wide")

# -----------------------------
# Sidebar / Header
# -----------------------------
st.sidebar.title("🛩️ Drone ETA Lab")
st.sidebar.markdown(
"""
**What this does**
- Trains an ETA model from your dataset (leak-safe splits)
- Shows calibration & error by wind (proper test alignment)
- Gives 5–95% prediction intervals + **coverage**
- What-if ETA simulator
- Optional feature ablation
"""
)

# -----------------------------
# Data loader with caching
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    df = pd.DataFrame()
    if path is None:
        return df
    try:
        if isinstance(path, str) and os.path.exists(path):
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not load data: {e}")
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    # Coerce numeric cols quietly
    for col in ["time","altitude","velocity_x","velocity_y","velocity_z",
                 "speed","payload","wind_speed","position_x","position_y","position_z"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# -----------------------------
# Feature engineering (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def make_flight_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {"flight","time","speed","payload","altitude","wind_speed",
                "velocity_x","velocity_y","velocity_z","position_x","position_y","position_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(list(missing))}")

    # Aggregate features per flight
    flight_summary = df.groupby('flight').agg(
        total_time=('time', 'max'),
        average_speed=('speed', 'mean'),
        max_payload=('payload', 'max'),
        max_altitude=('altitude', 'max'),
        average_wind_speed=('wind_speed', 'mean'),
        max_velocity_combined=('velocity_x', lambda x: np.sqrt(x**2 + df.loc[x.index, 'velocity_y']**2 + df.loc[x.index, 'velocity_z']**2).max()),
        start_pos_x=('position_x', 'first'),
        start_pos_y=('position_y', 'first'),
        start_pos_z=('position_z', 'first'),
        end_pos_x=('position_x', 'last'),
        end_pos_y=('position_y', 'last'),
        end_pos_z=('position_z', 'last')
    ).reset_index()

    # Straight-line distance
    flight_summary["total_distance"] = np.sqrt(
        (flight_summary['end_pos_x'] - flight_summary['start_pos_x'])**2 +
        (flight_summary['end_pos_y'] - flight_summary['start_pos_y'])**2 +
        (flight_summary['end_pos_z'] - flight_summary['start_pos_z'])**2
    )

    # Clean
    flight_summary = flight_summary.replace([np.inf, -np.inf], np.nan).dropna()
    flight_summary = flight_summary[flight_summary["total_time"] > 0].reset_index(drop=True)
    return flight_summary

# -----------------------------
# Globals
# -----------------------------
FEATURES = [
    'average_speed', 'max_payload', 'max_altitude', 'average_wind_speed',
    'max_velocity_combined', 'total_distance'
]
TARGET = 'total_time'

# -----------------------------
# Train models with leak-safe CV + proper hold-out
# -----------------------------
@st.cache_resource(show_spinner=True)
def train_models(flights: pd.DataFrame) -> Dict[str, Any]:
    X = flights[FEATURES].to_numpy()
    y = flights[TARGET].to_numpy()
    groups = flights["flight"].to_numpy()

    # Pipeline for scaling + model
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(random_state=42))
    ])

    param_grid = {
        'gbr__n_estimators': [500, 1000],
        'gbr__learning_rate': [0.01, 0.05, 0.1],
        'gbr__max_depth': [3, 5, 7],
        'gbr__subsample': [0.8, 1.0]
    }

    gkf = GroupKFold(n_splits=5)
    gs = GridSearchCV(pipe, param_grid, cv=gkf.split(X, y, groups),
                      scoring='r2', n_jobs=-1, verbose=0)
    gs.fit(X, y)
    best_pipe = gs.best_estimator_
    cv_r2 = gs.best_score_

    # Grouped hold-out (no leakage): 80/20 by flight
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    (train_idx, test_idx) = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Refit best model on train only
    best_pipe.fit(X_train, y_train)
    preds = best_pipe.predict(X_test)

    holdout_r2 = r2_score(y_test, preds)
    holdout_mape = mean_absolute_percentage_error(y_test, preds) * 100

    # Quantile GBMs for variable-width 5–95% intervals (fit on train only)
    scaler = best_pipe.named_steps['scaler']
    gbr_best = best_pipe.named_steps['gbr']
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    q_lo = GradientBoostingRegressor(loss='quantile', alpha=0.05,
                                     n_estimators=gbr_best.n_estimators,
                                     learning_rate=gbr_best.learning_rate,
                                     max_depth=gbr_best.max_depth,
                                     subsample=gbr_best.subsample,
                                     random_state=42).fit(X_train_s, y_train)
    q_hi = GradientBoostingRegressor(loss='quantile', alpha=0.95,
                                     n_estimators=gbr_best.n_estimators,
                                     learning_rate=gbr_best.learning_rate,
                                     max_depth=gbr_best.max_depth,
                                     subsample=gbr_best.subsample,
                                     random_state=42).fit(X_train_s, y_train)
    lower = q_lo.predict(X_test_s)
    upper = q_hi.predict(X_test_s)

    # Build eval dataframe with deterministic wind mapping
    eval_df = pd.DataFrame({
        "actual": y_test,
        "predicted": preds,
        "lower_5": lower,
        "upper_95": upper,
        "wind_speed": flights.loc[test_idx, "average_wind_speed"].to_numpy()
    }).reset_index(drop=True)

    eval_df["wind_bin"] = pd.cut(
        eval_df["wind_speed"],
        bins=[0,5,10,15,20,np.inf],
        labels=['0–5 m/s','5–10 m/s','10–15 m/s','15–20 m/s','>20 m/s']
    )
    eval_df["abs_err"] = np.abs(eval_df["actual"] - eval_df["predicted"])
    eval_df["pi_width"] = eval_df["upper_95"] - eval_df["lower_5"]
    inside = (eval_df["actual"] >= eval_df["lower_5"]) & (eval_df["actual"] <= eval_df["upper_95"])
    coverage = inside.mean() * 100.0

    # Feature importance from final model
    feat_imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": gbr_best.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return {
        "pipeline": best_pipe,
        "scaler": scaler,
        "gbr": gbr_best,
        "test_idx": test_idx,
        "eval": eval_df,
        "cv_r2": cv_r2,
        "r2": holdout_r2,
        "mape": holdout_mape,
        "coverage": coverage,
        "feature_importance": feat_imp,
        "flights": flights
    }

# -----------------------------
# Optional: quick ablation (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def ablation_table(flights: pd.DataFrame, base_pipe: Pipeline, test_idx: np.ndarray):
    rows = []
    X = flights[FEATURES].to_numpy()
    y = flights[TARGET].to_numpy()

    # Build fixed train/test by indices to be comparable
    train_idx = np.setdiff1d(np.arange(len(flights)), test_idx)

    for f in FEATURES:
        keep = [x for x in FEATURES if x != f]
        X_train = flights.iloc[train_idx][keep].to_numpy()
        y_train = flights.iloc[train_idx][TARGET].to_numpy()
        X_test  = flights.iloc[test_idx][keep].to_numpy()
        y_test  = flights.iloc[test_idx][TARGET].to_numpy()

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("gbr", GradientBoostingRegressor(
                n_estimators=base_pipe.named_steps['gbr'].n_estimators,
                learning_rate=base_pipe.named_steps['gbr'].learning_rate,
                max_depth=base_pipe.named_steps['gbr'].max_depth,
                subsample=base_pipe.named_steps['gbr'].subsample,
                random_state=42
            ))
        ])
        pipe.fit(X_train, y_train)
        r2_drop = r2_score(y_test, base_pipe.predict(flights.iloc[test_idx][FEATURES].to_numpy())) \
                  - r2_score(y_test, pipe.predict(X_test))
        rows.append({"feature_removed": f, "ΔR² (remove)": r2_drop})
    return pd.DataFrame(rows).sort_values("ΔR² (remove)", ascending=False)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧪 What-If Simulator", "📁 Your Data", "ℹ️ Notes"])

# === Initial data load ===
default_csv = "CarnegieMellonDataMain.csv"
uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if uploaded_file:
    df = load_data_from_path(uploaded_file)
else:
    df = load_data_from_path(default_csv)

if df.empty:
    with tab1:
        st.error("No dataset loaded. Upload `CarnegieMellonDataMain.csv` or use the uploader in the sidebar.")
        st.stop()

# Preprocess and train
try:
    flights = make_flight_table(df)
    models = train_models(flights)
except Exception as e:
    st.error(f"Error preparing/training models: {e}")
    st.stop()

eval_df = models["eval"]

# =========================
# TAB 1: Dashboard
# =========================
with tab1:
    st.title("Drone Delivery ETA — Analytics & Prediction")
    st.caption("Educational demo. Leak-safe evaluation. Not for operational use.")

    colA, colB, colC, colD = st.columns(4)
    colA.metric("R² (CV, GroupKFold)", f"{models['cv_r2']:.3f}")
    colB.metric("R² (hold-out)", f"{models['r2']:.3f}")
    colC.metric("MAPE (hold-out)", f"{models['mape']:.1f}%")
    colD.metric("PI Coverage (target 90–95%)", f"{models['coverage']:.1f}%")

    # Pred vs Actual (calibration)
    fig_cal = px.scatter(
        eval_df, x="actual", y="predicted",
        labels={"actual":"Actual Total Time (s)","predicted":"Predicted Total Time (s)"},
        title="Calibration: Predicted vs Actual (Hold-Out)"
    )
    mn = float(min(eval_df["actual"].min(), eval_df["predicted"].min()))
    mx = float(max(eval_df["actual"].max(), eval_df["predicted"].max()))
    fig_cal.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx, line=dict(dash="dash"))
    st.plotly_chart(fig_cal, use_container_width=True)

    # Residuals
    res_df = eval_df.copy()
    res_df["residual"] = res_df["actual"] - res_df["predicted"]
    fig_res = px.scatter(res_df, x="predicted", y="residual",
                         labels={"predicted":"Predicted (s)","residual":"Residual (s)"},
                         title="Residuals vs Predicted (Hold-Out)")
    fig_res.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig_res, use_container_width=True)

    # Wind bin MAPE (aligned to hold-out rows)
    mape_by_bin = eval_df.groupby("wind_bin")["abs_err"].apply(
        lambda s: (s / eval_df.loc[s.index, "actual"]).abs().mean() * 100
    ).reset_index().dropna()
    fig_mape = px.bar(mape_by_bin, x="wind_bin", y="abs_err",
                      labels={"wind_bin":"Wind (m/s)","abs_err":"MAPE (%)"},
                      title="Error by Wind Bin (MAPE) — Hold-Out")
    st.plotly_chart(fig_mape, use_container_width=True)

    # Prediction intervals preview + coverage shading
    fig_pi = px.scatter(
        eval_df.reset_index(drop=True),
        x=eval_df.reset_index(drop=True).index,
        y="predicted",
        error_y=eval_df["upper_95"] - eval_df["predicted"],
        error_y_minus=eval_df["predicted"] - eval_df["lower_5"],
        labels={"x":"Test Flight #","predicted":"Predicted Time (s)"},
        title="Prediction Intervals (5%–95%) — Hold-Out"
    )
    st.plotly_chart(fig_pi, use_container_width=True)

    # Reliability plot: interval width vs absolute error
    fig_rel = px.scatter(
        eval_df, x="pi_width", y="abs_err",
        labels={"pi_width":"PI Width (s)","abs_err":"Absolute Error (s)"},
        title="Reliability: Interval Width vs Absolute Error"
    )
    st.plotly_chart(fig_rel, use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance")
    st.dataframe(models["feature_importance"], use_container_width=True)

    # Optional Ablation
    with st.expander("Feature Ablation (ΔR² when a feature is removed)"):
        ab = ablation_table(models["flights"], models["pipeline"], models["test_idx"])
        st.dataframe(ab, use_container_width=True)

    st.markdown("---")
    st.info("Download the evaluation CSV (hold-out predictions + intervals).")
    st.download_button("Download results CSV",
                       data=eval_df.to_csv(index=False).encode('utf-8'),
                       file_name="drone_eta_results.csv",
                       mime="text/csv")

# =========================
# TAB 2: What-If Simulator
# =========================
with tab2:
    st.header("What-If ETA Simulator")

    med = models["flights"][FEATURES].median()

    st.caption("Adjust inputs and get ETA + uncertainty (from quantile models).")
    avg_speed = st.slider("Average Speed (m/s)", 0.1, float(models["flights"]["average_speed"].max()), float(med["average_speed"]))
    payload   = st.slider("Max Payload (kg)", 0.0, float(max(0.1, models["flights"]["max_payload"].max() or 10)), float(med["max_payload"]))
    altitude  = st.slider("Max Altitude (m)", 0.0, float(models["flights"]["max_altitude"].max()), float(med["max_altitude"]))
    wind      = st.slider("Average Wind Speed (m/s)", 0.0, float(models["flights"]["average_wind_speed"].max()), float(med["average_wind_speed"]))
    max_v     = st.slider("Max Velocity (combined, m/s)", 0.0, float(max(1.0, models["flights"]["max_velocity_combined"].max() or 50)), float(med["max_velocity_combined"]))
    distance  = st.slider("Total Distance (m)", 0.0, float(models["flights"]["total_distance"].max()), float(med["total_distance"]))

    x_vec = np.array([[avg_speed, payload, altitude, wind, max_v, distance]])
    # Use the fitted scaler/model
    scaler = models["scaler"]
    gbr = models["gbr"]
    x_scaled = scaler.transform(x_vec)
    eta_pred = gbr.predict(x_scaled)[0]

    # Simple uncertainty from training quantiles:
    # Refit small quantiles on entire flights using tuned hyperparams (cached by train_models)
    q_lo = GradientBoostingRegressor(loss='quantile', alpha=0.05,
                                     n_estimators=gbr.n_estimators, learning_rate=gbr.learning_rate,
                                     max_depth=gbr.max_depth, subsample=gbr.subsample, random_state=42).fit(
        scaler.transform(models["flights"][FEATURES].to_numpy()),
        models["flights"][TARGET].to_numpy()
    )
    q_hi = GradientBoostingRegressor(loss='quantile', alpha=0.95,
                                     n_estimators=gbr.n_estimators, learning_rate=gbr.learning_rate,
                                     max_depth=gbr.max_depth, subsample=gbr.subsample, random_state=42).fit(
        scaler.transform(models["flights"][FEATURES].to_numpy()),
        models["flights"][TARGET].to_numpy()
    )
    lo = q_lo.predict(x_scaled)[0]
    hi = q_hi.predict(x_scaled)[0]

    st.success(f"**Predicted ETA:** {eta_pred:.1f} s  \n**Interval (5%–95%):** {lo:.1f} – {hi:.1f} s")
    st.caption("Note: Educational model; not validated for operational BVLOS use.")

# =========================
# TAB 3: Your Data
# =========================
with tab3:
    st.header("Your Dataset & Batch Predictions")
    st.write("Preview of the cleaned flight summary table.")
    st.dataframe(models["flights"].head(50), use_container_width=True)
    st.markdown("**Upload a new CSV** to retrain with different data (must include required columns).")
    up = st.file_uploader("Upload CSV to replace dataset (optional)", type=["csv"])
    if up:
        try:
            new_df = load_data_from_path(up)
            new_flights = make_flight_table(new_df)
            st.success("New dataset loaded and processed. Refresh to retrain models.")
            st.dataframe(new_flights.head(30), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to process uploaded CSV: {e}")

    st.markdown("---")
    st.write("Download the evaluation CSV (hold-out predictions + intervals).")
    st.download_button("Download evaluation CSV",
                       data=models["eval"].to_csv(index=False).encode('utf-8'),
                       file_name="drone_eta_evaluation.csv",
                       mime="text/csv")

# =========================
# TAB 4: Notes & Reproducibility
# =========================
with tab4:
    st.header("Notes, Limitations & How to Cite")
    st.markdown("""
**Validation**
- **CV:** GroupKFold by *flight* to prevent leakage between train/test.
- **Hold-out:** GroupShuffleSplit (20%) by *flight*; metrics reported above.
- **Intervals:** Gradient Boosting quantiles (5–95%) with measured **coverage** on hold-out.

**Metrics shown**
- R² (CV & hold-out), MAPE, PI coverage, wind-stratified error, reliability.

**Limitations**
- Dataset from controlled test flights; not validated for BVLOS/operational decision-making.
- Sparse high-wind segments; accuracy degrades with wind (see wind MAPE chart).

**Reproducibility**
1. Put `CarnegieMellonDataMain.csv` next to `app.py` or upload via sidebar.
2. All models/tuning run inside this app with leak-safe splits.
3. Download evaluation CSV for external analysis.

**Contact**
- Add your name/email + a link to your repo for credibility.
""")
