# app.py
# Drone ETA Lab — Physics + ML residuals, leak-safe CV/hold-out, fixed What-If, stable caching
# Requirements (requirements.txt):
# streamlit
# pandas
# scikit-learn
# numpy
# plotly

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Dict, Any, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Drone ETA Lab", page_icon="🛩️", layout="wide")

# =========================================
# Sidebar / Header
# =========================================
st.sidebar.title("🛩️ Drone ETA Lab")
st.sidebar.markdown(
"""
**What this app does**
- Uses **physics baseline** (time ≈ distance / speed) + ML residuals
- Leak-safe **GroupKFold CV** and **grouped hold-out**
- Wind-aware errors, 5–95% prediction intervals + **coverage**
- A reliable **What-If** simulator (distance ↑ → ETA ↑)
"""
)

# =========================================
# Helpers
# =========================================
EPS = 1e-6  # guard for division

FEATURES = [
    'average_speed', 'max_payload', 'max_altitude', 'average_wind_speed',
    'max_velocity_combined', 'total_distance'
]
TARGET = 'total_time'

def physics_baseline(distance: np.ndarray, avg_speed: np.ndarray) -> np.ndarray:
    """time ≈ distance / speed, with guards to avoid divide-by-zero."""
    sp = np.maximum(avg_speed.astype(float), 0.1)  # cap min speed
    return distance.astype(float) / sp

def make_eval_df(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 lo: np.ndarray,
                 hi: np.ndarray,
                 wind_speed: np.ndarray) -> pd.DataFrame:
    e = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
        "lower_5": lo,
        "upper_95": hi,
        "wind_speed": wind_speed
    })
    e["abs_err"] = np.abs(e["actual"] - e["predicted"])
    e["pi_width"] = e["upper_95"] - e["lower_5"]
    e["wind_bin"] = pd.cut(
        e["wind_speed"],
        bins=[0,5,10,15,20,np.inf],
        labels=['0–5 m/s','5–10 m/s','10–15 m/s','15–20 m/s','>20 m/s']
    )
    return e.reset_index(drop=True)

# =========================================
# Caching: data (cache_data) vs models (cache_resource)
# =========================================
@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
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


@st.cache_data(show_spinner=False)
def make_flight_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {"flight","time","speed","payload","altitude","wind_speed",
                "velocity_x","velocity_y","velocity_z","position_x","position_y","position_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(list(missing))}")

    # Aggregate per flight
    flight_summary = df.groupby('flight').agg(
        total_time=('time', 'max'),
        average_speed=('speed', 'mean'),
        max_payload=('payload', 'max'),
        max_altitude=('altitude', 'max'),
        average_wind_speed=('wind_speed', 'mean'),
        max_velocity_combined=('velocity_x', lambda x: np.sqrt(
            x**2 +
            df.loc[x.index, 'velocity_y']**2 +
            df.loc[x.index, 'velocity_z']**2
        ).max()),
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


@st.cache_resource(show_spinner=True)
def train_models(flights: pd.DataFrame) -> Dict[str, Any]:
    """
    Physics + ML residuals:
      residual = total_time - (total_distance / average_speed)
      Model learns residuals (GBR), intervals via quantile GBRs on residuals.
    Leak-safe GroupKFold CV and GroupShuffleSplit hold-out.
    """
    X = flights[FEATURES].to_numpy()
    y = flights[TARGET].to_numpy()
    groups = flights["flight"].to_numpy()

    # Physics baseline
    base_time = physics_baseline(
        distance=flights["total_distance"].to_numpy(),
        avg_speed=flights["average_speed"].to_numpy()
    )
    resid = y - base_time  # what ML needs to learn

    # Pipeline for residuals
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

    # Leak-safe CV on flights
    gkf = GroupKFold(n_splits=5)
    gs = GridSearchCV(pipe, param_grid, cv=gkf.split(X, resid, groups),
                      scoring='r2', n_jobs=-1, verbose=0)
    gs.fit(X, resid)
    best_pipe = gs.best_estimator_
    cv_r2_resid = gs.best_score_

    # Grouped hold-out (no leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    (train_idx, test_idx) = next(gss.split(X, resid, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    resid_train, resid_test = resid[train_idx], resid[test_idx]
    base_train, base_test = base_time[train_idx], base_time[test_idx]
    y_test = y[test_idx]

    # Refit on train residuals
    best_pipe.fit(X_train, resid_train)
    resid_pred_test = best_pipe.predict(X_test)
    y_pred_test = base_test + resid_pred_test

    holdout_r2 = r2_score(y_test, y_pred_test)
    holdout_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100

    # Quantile models on residuals (same hyperparams)
    scaler = best_pipe.named_steps['scaler']
    gbr_best = best_pipe.named_steps['gbr']
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    q_lo = GradientBoostingRegressor(
        loss='quantile', alpha=0.05,
        n_estimators=gbr_best.n_estimators, learning_rate=gbr_best.learning_rate,
        max_depth=gbr_best.max_depth, subsample=gbr_best.subsample, random_state=42
    ).fit(X_train_s, resid_train)

    q_hi = GradientBoostingRegressor(
        loss='quantile', alpha=0.95,
        n_estimators=gbr_best.n_estimators, learning_rate=gbr_best.learning_rate,
        max_depth=gbr_best.max_depth, subsample=gbr_best.subsample, random_state=42
    ).fit(X_train_s, resid_train)

    lo_resid = q_lo.predict(X_test_s)
    hi_resid = q_hi.predict(X_test_s)
    lo_time = base_test + lo_resid
    hi_time = base_test + hi_resid

    eval_df = make_eval_df(
        y_true=y_test,
        y_pred=y_pred_test,
        lo=lo_time,
        hi=hi_time,
        wind_speed=flights.loc[test_idx, "average_wind_speed"].to_numpy()
    )
    coverage = ((eval_df["actual"] >= eval_df["lower_5"]) & (eval_df["actual"] <= eval_df["upper_95"])).mean() * 100

    # Feature importance (residual model)
    feat_imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": gbr_best.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Return everything needed (only primitives/arrays/DFs; no custom classes outside pipeline)
    return {
        "pipeline": best_pipe,             # residuals pipeline (scaler + gbr)
        "cv_r2_resid": cv_r2_resid,        # R² on residuals (FYI)
        "r2": holdout_r2,                  # R² on total time (hold-out)
        "mape": holdout_mape,              # MAPE on total time (hold-out)
        "coverage": coverage,              # PI coverage on total time
        "eval": eval_df,                   # eval table
        "feature_importance": feat_imp,    # residual model feature importances
        "flights": flights,                # flights DF
        "test_idx": test_idx,              # indices for hold-out rows
        "base_all": base_time,             # physics baseline for all rows
    }

# =========================================
# Tabs
# =========================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧪 What-If Simulator", "📁 Your Data", "ℹ️ Notes"])

# === Initial data load ===
default_csv = "CarnegieMellonDataMain.csv"
uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if uploaded_file:
    raw = load_data_from_path(uploaded_file)
else:
    raw = load_data_from_path(default_csv)

if raw.empty:
    with tab1:
        st.error("No dataset loaded. Upload `CarnegieMellonDataMain.csv` or use the uploader in the sidebar.")
        st.stop()

# Preprocess and train
try:
    flights = make_flight_table(raw)
    models = train_models(flights)
except Exception as e:
    st.error(f"Error preparing/training models: {e}")
    st.stop()

eval_df = models["eval"]

# =========================================
# TAB 1: Dashboard
# =========================================
with tab1:
    st.title("Drone Delivery ETA — Physics + ML Residuals")
    st.caption("Leak-safe evaluation. Physics-consistent predictions. Not for operational use.")

    colA, colB, colC = st.columns(3)
    colA.metric("R² (hold-out)", f"{models['r2']:.3f}")
    colB.metric("MAPE (hold-out)", f"{models['mape']:.1f}%")
    colC.metric("PI Coverage (5–95%)", f"{models['coverage']:.1f}%")

    # Calibration
    fig_cal = px.scatter(
        eval_df, x="actual", y="predicted",
        labels={"actual":"Actual Total Time (s)", "predicted":"Predicted Total Time (s)"},
        title="Calibration: Predicted vs Actual (Hold-Out)"
    )
    mn = float(min(eval_df["actual"].min(), eval_df["predicted"].min()))
    mx = float(max(eval_df["actual"].max(), eval_df["predicted"].max()))
    fig_cal.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx, line=dict(dash="dash"))
    st.plotly_chart(fig_cal, use_container_width=True)

    # Residuals plot
    res_df = eval_df.copy()
    res_df["residual"] = res_df["actual"] - res_df["predicted"]
    fig_res = px.scatter(
        res_df, x="predicted", y="residual",
        labels={"predicted":"Predicted (s)", "residual":"Residual (s)"},
        title="Residuals vs Predicted (Hold-Out)"
    )
    fig_res.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig_res, use_container_width=True)

    # Wind bin MAPE (aligned to hold-out rows)
    tmp = eval_df.copy()
    tmp["mape"] = np.abs(tmp["actual"] - tmp["predicted"]) / np.maximum(tmp["actual"], EPS) * 100
    mape_by_bin = tmp.groupby("wind_bin", observed=True)["mape"].mean().reset_index().dropna()
    fig_mape = px.bar(
        mape_by_bin, x="wind_bin", y="mape",
        labels={"wind_bin":"Wind (m/s)", "mape":"MAPE (%)"},
        title="Error by Wind Bin (MAPE) — Hold-Out"
    )
    st.plotly_chart(fig_mape, use_container_width=True)

    # Prediction intervals preview
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

    # Reliability: PI width vs absolute error
    fig_rel = px.scatter(
        eval_df, x="pi_width", y="abs_err",
        labels={"pi_width":"PI Width (s)","abs_err":"Absolute Error (s)"},
        title="Reliability: Interval Width vs Absolute Error"
    )
    st.plotly_chart(fig_rel, use_container_width=True)

    # Feature importance (residual model)
    st.subheader("Feature Importance (Residual Model)")
    st.dataframe(models["feature_importance"], use_container_width=True)

    # (Optional) quick ablation — not cached, avoids unhashable issues
    with st.expander("Feature Ablation (ΔR² when a feature is removed) — Optional"):
        try:
            # Build a comparable train/test using saved indices
            test_idx = models["test_idx"]
            train_idx = np.setdiff1d(np.arange(len(flights)), test_idx)

            base_pipe = models["pipeline"]
            # Full model predictions on test
            full_resid_pred = base_pipe.predict(flights.iloc[test_idx][FEATURES].to_numpy())
            full_pred = models["base_all"][test_idx] + full_resid_pred
            y_test_all = flights.iloc[test_idx][TARGET].to_numpy()
            full_r2 = r2_score(y_test_all, full_pred)

            rows = []
            for f in FEATURES:
                keep = [x for x in FEATURES if x != f]
                X_train = flights.iloc[train_idx][keep].to_numpy()
                y_train = (flights.iloc[train_idx][TARGET].to_numpy() -
                           physics_baseline(
                               flights.iloc[train_idx]["total_distance"].to_numpy(),
                               flights.iloc[train_idx]["average_speed"].to_numpy()
                           ))
                X_test  = flights.iloc[test_idx][keep].to_numpy()
                base_test = physics_baseline(
                    flights.iloc[test_idx]["total_distance"].to_numpy(),
                    flights.iloc[test_idx]["average_speed"].to_numpy()
                )

                # clone lightweight pipe with same hyperparams
                gbr_best = base_pipe.named_steps['gbr']
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("gbr", GradientBoostingRegressor(
                        n_estimators=gbr_best.n_estimators,
                        learning_rate=gbr_best.learning_rate,
                        max_depth=gbr_best.max_depth,
                        subsample=gbr_best.subsample,
                        random_state=42
                    ))
                ])
                pipe.fit(X_train, y_train)
                pred = base_test + pipe.predict(X_test)
                r2 = r2_score(y_test_all, pred)
                rows.append({"feature_removed": f, "ΔR² (remove)": float(full_r2 - r2)})
            ablation_df = pd.DataFrame(rows).sort_values("ΔR² (remove)", ascending=False)
            st.dataframe(ablation_df, use_container_width=True)
        except Exception as e:
            st.info(f"Ablation skipped: {e}")

    st.markdown("---")
    st.info("Download the evaluation CSV (hold-out predictions + intervals).")
    st.download_button(
        "Download results CSV",
        data=eval_df.to_csv(index=False).encode('utf-8'),
        file_name="drone_eta_results.csv",
        mime="text/csv"
    )

# =========================================
# TAB 2: What-If Simulator (fixed physics behavior)
# =========================================
with tab2:
    st.header("What-If ETA Simulator (Physics + ML Residuals)")
    med = flights[FEATURES].median()

    st.caption("Adjust inputs — baseline enforces time ≈ distance / speed; ML adjusts for wind, altitude, etc.")

    avg_speed = st.slider("Average Speed (m/s)", 0.1, float(max(0.2, flights["average_speed"].max() or 1.0)), float(max(0.3, med["average_speed"])))
    payload   = st.slider("Max Payload (kg)", 0.0, float(max(0.1, flights["max_payload"].max() or 10)), float(med["max_payload"]))
    altitude  = st.slider("Max Altitude (m)", 0.0, float(flights["max_altitude"].max()), float(med["max_altitude"]))
    wind      = st.slider("Average Wind Speed (m/s)", 0.0, float(flights["average_wind_speed"].max()), float(med["average_wind_speed"]))
    max_v     = st.slider("Max Velocity (combined, m/s)", 0.0, float(max(1.0, flights["max_velocity_combined"].max() or 50)), float(med["max_velocity_combined"]))
    distance  = st.slider("Total Distance (m)", 0.0, float(max(1.0, flights["total_distance"].max() or 1.0)), float(med["total_distance"]))

    # Physics baseline (guarantees monotonicity w.r.t distance & speed)
    base_eta = float(physics_baseline(np.array([distance]), np.array([avg_speed]))[0])

    # Residual prediction using trained pipeline
    x_vec = np.array([[avg_speed, payload, altitude, wind, max_v, distance]])
    resid_pipe = models["pipeline"]
    resid_pred = float(resid_pipe.predict(x_vec)[0])

    # Total ETA
    eta_pred = base_eta + resid_pred

    # Quick uncertainty from quantiles refit on ALL flights residuals (not cached per move)
    # (This keeps UI responsive while giving reasonable bounds.)
    X_all = flights[FEATURES].to_numpy()
    y_all = flights[TARGET].to_numpy()
    base_all = physics_baseline(
        flights["total_distance"].to_numpy(),
        flights["average_speed"].to_numpy()
    )
    resid_all = y_all - base_all

    scaler = StandardScaler().fit(X_all)
    X_all_s = scaler.transform(X_all)

    gbr_best = resid_pipe.named_steps['gbr']
    q_lo = GradientBoostingRegressor(
        loss='quantile', alpha=0.05,
        n_estimators=gbr_best.n_estimators, learning_rate=gbr_best.learning_rate,
        max_depth=gbr_best.max_depth, subsample=gbr_best.subsample, random_state=42
    ).fit(X_all_s, resid_all)

    q_hi = GradientBoostingRegressor(
        loss='quantile', alpha=0.95,
        n_estimators=gbr_best.n_estimators, learning_rate=gbr_best.learning_rate,
        max_depth=gbr_best.max_depth, subsample=gbr_best.subsample, random_state=42
    ).fit(X_all_s, resid_all)

    x_vec_s = scaler.transform(x_vec)
    lo = base_eta + float(q_lo.predict(x_vec_s)[0])
    hi = base_eta + float(q_hi.predict(x_vec_s)[0])

    st.success(f"**Predicted ETA:** {eta_pred:.1f} s  \n**Interval (5%–95%):** {lo:.1f} – {hi:.1f} s")
    st.caption("Because ETA = distance/speed + residuals, increasing distance or lowering speed will always increase ETA.")

# =========================================
# TAB 3: Your Data
# =========================================
with tab3:
    st.header("Your Dataset & Batch Predictions")
    st.write("Preview the cleaned flight summary table.")
    st.dataframe(flights.head(50), use_container_width=True)

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
    st.download_button(
        "Download evaluation CSV",
        data=models["eval"].to_csv(index=False).encode('utf-8'),
        file_name="drone_eta_evaluation.csv",
        mime="text/csv"
    )

# =========================================
# TAB 4: Notes & Reproducibility
# =========================================
with tab4:
    st.header("Notes, Limitations & How to Cite")
    st.markdown("""
**Method**
- **Physics baseline:** `ETA_base = distance / average_speed`
- **ML residuals:** GBR learns `ETA_residual = total_time - ETA_base`
- Final prediction: `ETA = ETA_base + ETA_residual`

**Validation**
- **CV:** 5-fold **GroupKFold** by *flight* (leak-safe).
- **Hold-out:** **GroupShuffleSplit** 20% by *flight*.
- **Intervals:** Quantile Gradient Boosting on **residuals** (5–95%), reported **coverage** on hold-out.

**Why the What-If is now sane**
- Physics baseline forces monotonic behavior: distance ↑ or speed ↓ ⇒ ETA ↑.
- ML adds corrections for wind, altitude, payload, etc.

**Limitations**
- Dataset from controlled test flights; not validated for BVLOS/operational use.
- Sparse extreme winds can degrade accuracy (see wind-bin MAPE).

**Reproducibility**
1. Put `CarnegieMellonDataMain.csv` next to `app.py` or upload via sidebar.
2. Everything trains inside the app with leak-safe splits.
3. Download evaluation CSV for external analysis.

**Contact**
- Add your name/email + repo link here.
""")
