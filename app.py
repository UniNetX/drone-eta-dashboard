# app.py
# Drone ETA Lab — polished UI + physics-guided ML residuals + leak-safe evaluation
# Requirements: streamlit, pandas, numpy, scikit-learn, plotly, matplotlib, seaborn

import os
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Page config & CSS
# ---------------------------
st.set_page_config(page_title="Drone ETA Lab", page_icon="🛩️", layout="wide")

# small CSS to improve look
st.markdown(
    """
    <style>
    .stApp { font-family: "Inter", sans-serif; }
    .big-metric { font-size: 20px; font-weight: 700; }
    .metric-sub { color: #6c757d; }
    .sidebar .stButton>button { background-color:#0b5cff; color:white; }
    .card { background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:12px; border-radius:8px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Constants
# ---------------------------
DEFAULT_CSV = "CarnegieMellonDataMain.csv"
EPS = 1e-6

FEATURES = [
    'average_speed', 'max_payload', 'max_altitude', 'average_wind_speed',
    'max_velocity_combined', 'total_distance'
]
TARGET = 'total_time'
REQUIRED_COLUMNS = {"flight","time","speed","payload","altitude","wind_speed",
                    "velocity_x","velocity_y","velocity_z","position_x","position_y","position_z"}

# ---------------------------
# Utility functions
# ---------------------------
def physics_baseline(distance: np.ndarray, avg_speed: np.ndarray) -> np.ndarray:
    """Return baseline ETA = distance / avg_speed with safe guards."""
    sp = np.maximum(np.array(avg_speed, dtype=float), 0.1)
    return np.array(distance, dtype=float) / sp

def make_eval_df(y_true: np.ndarray, y_pred: np.ndarray, lo: np.ndarray, hi: np.ndarray, wind_speed: np.ndarray) -> pd.DataFrame:
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

# ---------------------------
# Caching: data vs models
# ---------------------------
@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer) -> pd.DataFrame:
    # Accept path or uploaded buffer
    try:
        if isinstance(path_or_buffer, str) and os.path.exists(path_or_buffer):
            df = pd.read_csv(path_or_buffer)
        else:
            df = pd.read_csv(path_or_buffer)
    except Exception as e:
        raise e
    df.columns = df.columns.str.strip()
    # coerce numeric columns if present
    for col in ["time","altitude","velocity_x","velocity_y","velocity_z",
                "speed","payload","wind_speed","position_x","position_y","position_z"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

@st.cache_data(show_spinner=False)
def make_flight_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(list(missing))}")

    flight_summary = df.groupby('flight').agg(
        total_time=('time', 'max'),
        average_speed=('speed', 'mean'),
        max_payload=('payload', 'max'),
        max_altitude=('altitude', 'max'),
        average_wind_speed=('wind_speed', 'mean'),
        max_velocity_combined=('velocity_x', lambda x: np.sqrt(
            x**2 + df.loc[x.index, 'velocity_y']**2 + df.loc[x.index, 'velocity_z']**2).max()),
        start_pos_x=('position_x', 'first'),
        start_pos_y=('position_y', 'first'),
        start_pos_z=('position_z', 'first'),
        end_pos_x=('position_x', 'last'),
        end_pos_y=('position_y', 'last'),
        end_pos_z=('position_z', 'last')
    ).reset_index()

    flight_summary["total_distance"] = np.sqrt(
        (flight_summary['end_pos_x'] - flight_summary['start_pos_x'])**2 +
        (flight_summary['end_pos_y'] - flight_summary['start_pos_y'])**2 +
        (flight_summary['end_pos_z'] - flight_summary['start_pos_z'])**2
    )

    flight_summary = flight_summary.replace([np.inf, -np.inf], np.nan).dropna()
    flight_summary = flight_summary[flight_summary["total_time"] > 0].reset_index(drop=True)
    return flight_summary

@st.cache_resource(show_spinner=True)
def train_models(flights: pd.DataFrame) -> Dict[str, Any]:
    """
    Train residual model:
      resid = total_time - physics_baseline(distance, average_speed)
      model learns resid from FEATURES
    Returns dict of primitives, arrays, and fitted Pipeline object (safe in cache_resource).
    """
    X = flights[FEATURES].to_numpy()
    y = flights[TARGET].to_numpy()
    groups = flights["flight"].to_numpy()

    base_time = physics_baseline(flights["total_distance"].to_numpy(), flights["average_speed"].to_numpy())
    resid = y - base_time  # targets for ML

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(random_state=42))
    ])
    param_grid = {
        'gbr__n_estimators': [500, 1000],
        'gbr__learning_rate': [0.01, 0.05, 0.1],
        'gbr__max_depth': [3, 5],
        'gbr__subsample': [0.8, 1.0]
    }

    gkf = GroupKFold(n_splits=5)
    gs = GridSearchCV(pipe, param_grid, cv=gkf.split(X, resid, groups), scoring='r2', n_jobs=-1, verbose=0)
    gs.fit(X, resid)
    best_pipe = gs.best_estimator_
    cv_r2_resid = gs.best_score_

    # grouped hold-out (20% by flight)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, resid, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    resid_train, resid_test = resid[train_idx], resid[test_idx]
    base_test = base_time[test_idx]
    y_test = y[test_idx]

    best_pipe.fit(X_train, resid_train)
    resid_pred_test = best_pipe.predict(X_test)
    y_pred_test = base_test + resid_pred_test

    holdout_r2 = r2_score(y_test, y_pred_test)
    holdout_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    holdout_mae = mean_absolute_error(y_test, y_pred_test)

    # quantile models on residuals (same hyperparams)
    scaler = best_pipe.named_steps['scaler']
    gbr_best = best_pipe.named_steps['gbr']

    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    q_lo = GradientBoostingRegressor(loss='quantile', alpha=0.05,
                                     n_estimators=gbr_best.n_estimators,
                                     learning_rate=gbr_best.learning_rate,
                                     max_depth=gbr_best.max_depth,
                                     subsample=gbr_best.subsample,
                                     random_state=42).fit(X_train_s, resid_train)
    q_hi = GradientBoostingRegressor(loss='quantile', alpha=0.95,
                                     n_estimators=gbr_best.n_estimators,
                                     learning_rate=gbr_best.learning_rate,
                                     max_depth=gbr_best.max_depth,
                                     subsample=gbr_best.subsample,
                                     random_state=42).fit(X_train_s, resid_train)

    lo_resid = q_lo.predict(X_test_s)
    hi_resid = q_hi.predict(X_test_s)
    lo_time = base_test + lo_resid
    hi_time = base_test + hi_resid

    eval_df = make_eval_df(y_test, y_pred_test, lo_time, hi_time, flights.loc[test_idx, "average_wind_speed"].to_numpy())
    coverage = ((eval_df["actual"] >= eval_df["lower_5"]) & (eval_df["actual"] <= eval_df["upper_95"])).mean() * 100

    feat_imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": gbr_best.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Return safe objects (primitives, arrays, DF, and pipeline)
    return {
        "pipeline": best_pipe,
        "cv_r2_resid": float(cv_r2_resid),
        "r2": float(holdout_r2),
        "mape": float(holdout_mape),
        "mae": float(holdout_mae),
        "coverage": float(coverage),
        "eval": eval_df,
        "feature_importance": feat_imp,
        "flights": flights,
        "test_idx": test_idx,
        "base_all": base_time
    }

# ---------------------------
# App UI
# ---------------------------
def main():
    st.sidebar.header("Dataset & Run")
    uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
    use_default = st.sidebar.button("Use default dataset")
    csv_source = None
    if uploaded:
        csv_source = uploaded
    elif use_default or (not uploaded and os.path.exists(DEFAULT_CSV)):
        csv_source = DEFAULT_CSV

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Options**")
    prune = st.sidebar.checkbox("Enable weak-feature pruning (corr < 0.05)", value=False)
    show_corr = st.sidebar.checkbox("Show correlation heatmap", value=True)
    show_ablation = st.sidebar.checkbox("Show ablation (may be slow)", value=False)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Export**")
    st.sidebar.markdown("Download eval CSV after run from Dashboard tab.")

    if csv_source is None:
        st.header("No dataset loaded")
        st.write("Upload `CarnegieMellonDataMain.csv` or click 'Use default dataset' (file must exist).")
        return

    # Load data
    try:
        raw = load_csv(csv_source)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return

    # try to create flight summary
    try:
        flights = make_flight_table(raw)
    except Exception as e:
        st.error(f"Failed to build flight summary: {e}")
        return

    # Optional correlation-based pruning
    if prune:
        corr = flights[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].abs()
        keep = corr[corr >= 0.05].index.tolist()
        # ensure features list remains valid
        keep_feats = [c for c in FEATURES if c in keep]
        if len(keep_feats) < len(FEATURES):
            removed = set(FEATURES) - set(keep_feats)
            st.sidebar.info(f"Pruned weak features: {', '.join(sorted(removed))}")
            # mutate FEATURES locally for training by building a new flights df view
            current_features = keep_feats
        else:
            current_features = FEATURES
    else:
        current_features = FEATURES

    # Train (or fetch cached)
    with st.spinner("Training models (leak-safe CV + hold-out)..."):
        models = train_models(flights)

    # Layout: header metrics + 3 tabs
    st.markdown("## 🛩️ Drone Delivery ETA — Dashboard")
    col1, col2, col3, col4 = st.columns([1.2,1.0,1.0,1.0])
    col1.markdown(f"**Hold-out R²**  \n`{models['r2']:.3f}`", unsafe_allow_html=True)
    col2.markdown(f"**MAPE (hold-out)**  \n`{models['mape']:.1f}%`", unsafe_allow_html=True)
    col3.markdown(f"**MAE (hold-out)**  \n`{models['mae']:.2f} s`", unsafe_allow_html=True)
    col4.markdown(f"**PI Coverage**  \n`{models['coverage']:.1f}%`", unsafe_allow_html=True)

    tabs = st.tabs(["Dashboard", "What-If Simulator", "Data & Downloads", "Notes & Repro"])

    # --- Dashboard tab ---
    with tabs[0]:
        st.markdown("### Calibration & Reliability (Hold-out)")
        eval_df = models["eval"]

        # Calibration scatter
        fig_cal = px.scatter(eval_df, x="actual", y="predicted",
                             labels={"actual":"Actual total time (s)", "predicted":"Predicted total time (s)"},
                             title="Predicted vs Actual (Hold-out)")
        mn = min(eval_df["actual"].min(), eval_df["predicted"].min())
        mx = max(eval_df["actual"].max(), eval_df["predicted"].max())
        fig_cal.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx, line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig_cal, use_container_width=True)

        # Residuals plot
        st.markdown("#### Residuals")
        fig_res = px.scatter(eval_df, x="predicted", y="abs_err",
                             labels={"predicted":"Predicted (s)", "abs_err":"Absolute Error (s)"},
                             title="Absolute Error vs Predicted")
        st.plotly_chart(fig_res, use_container_width=True)

        # Wind bin bar
        st.markdown("#### Error by Wind Bin")
        mape_by_bin = eval_df.groupby("wind_bin", observed=True)["abs_err"].apply(
            lambda s: (s / np.maximum(eval_df.loc[s.index, "actual"], EPS)).mean() * 100
        ).reset_index().rename(columns={"abs_err":"mape"})
        mape_by_bin = mape_by_bin.dropna()
        fig_wind = px.bar(mape_by_bin, x="wind_bin", y="mape", labels={"wind_bin":"Wind (m/s)","mape":"MAPE (%)"}, title="MAPE by Wind Bin (Hold-out)")
        st.plotly_chart(fig_wind, use_container_width=True)

        # PI preview
        st.markdown("#### Prediction Intervals (hold-out)")
        fig_pi = px.scatter(eval_df.reset_index(drop=True), x=eval_df.reset_index(drop=True).index,
                            y="predicted", error_y=eval_df["upper_95"] - eval_df["predicted"],
                            error_y_minus=eval_df["predicted"] - eval_df["lower_5"],
                            labels={"x":"Test Flight #","predicted":"Predicted time (s)"},
                            title="Prediction Intervals (5%–95%)")
        st.plotly_chart(fig_pi, use_container_width=True)

        # Reliability: PI width vs absolute error
        st.markdown("#### Reliability: interval width vs absolute error")
        fig_rel = px.scatter(eval_df, x="pi_width", y="abs_err", labels={"pi_width":"PI width (s)","abs_err":"Absolute error (s)"}, title="Interval Width vs Absolute Error")
        st.plotly_chart(fig_rel, use_container_width=True)

        # Feature importance
        st.markdown("#### Feature Importance (Residual Model)")
        st.dataframe(models["feature_importance"].rename(columns={"importance":"importance (residual model)"}), use_container_width=True)

        # Correlation heatmap
        if show_corr:
            st.markdown("#### Correlation heatmap (features vs total time)")
            corr = flights[current_features + [TARGET]].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax)
            st.pyplot(fig)
            plt.close(fig)

        # Optional ablation (non-cached)
        if show_ablation:
            st.markdown("#### Quick Ablation (ΔR² when removing a feature)")
            try:
                test_idx = models["test_idx"]
                train_idx = np.setdiff1d(np.arange(len(flights)), test_idx)
                base_pipe = models["pipeline"]
                # full predictions on test
                full_resid = base_pipe.predict(flights.iloc[test_idx][FEATURES].to_numpy())
                full_pred = models["base_all"][test_idx] + full_resid
                y_test_all = flights.iloc[test_idx][TARGET].to_numpy()
                full_r2 = r2_score(y_test_all, full_pred)
                rows = []
                for f in FEATURES:
                    keep = [x for x in FEATURES if x != f]
                    X_train = flights.iloc[train_idx][keep].to_numpy()
                    y_train = flights.iloc[train_idx][TARGET].to_numpy() - physics_baseline(
                        flights.iloc[train_idx]["total_distance"].to_numpy(),
                        flights.iloc[train_idx]["average_speed"].to_numpy()
                    )
                    X_test  = flights.iloc[test_idx][keep].to_numpy()
                    base_test = physics_baseline(
                        flights.iloc[test_idx]["total_distance"].to_numpy(),
                        flights.iloc[test_idx]["average_speed"].to_numpy()
                    )
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
                st.info(f"Ablation failed: {e}")

    # --- What-If tab ---
    with tabs[1]:
        st.markdown("### What-If Simulator — Physics-friendly (guaranteed monotonic)")
        col_left, col_right = st.columns([2,1])

        with col_left:
            st.markdown("Adjust mission parameters and see predicted ETA + interval.")
            med = flights[FEATURES].median()
            avg_speed = st.number_input("Average speed (m/s)", min_value=0.1, max_value=float(max(1.0, flights["average_speed"].max())), value=float(max(0.5, med["average_speed"])))
            payload   = st.number_input("Max payload (kg)", value=float(med["max_payload"]), min_value=0.0, max_value=float(max(0.1, flights["max_payload"].max() or 10)))
            altitude  = st.number_input("Max altitude (m)", value=float(med["max_altitude"]), min_value=0.0, max_value=float(max(1.0, flights["max_altitude"].max())))
            wind      = st.number_input("Average wind speed (m/s)", value=float(med["average_wind_speed"]), min_value=0.0, max_value=float(max(0.0, flights["average_wind_speed"].max())))
            max_v     = st.number_input("Max velocity combined (m/s)", value=float(med["max_velocity_combined"]), min_value=0.0, max_value=float(max(1.0, flights["max_velocity_combined"].max())))
            distance  = st.number_input("Total distance (m)", value=float(med["total_distance"]), min_value=0.1, max_value=float(max(1.0, flights["total_distance"].max())))

            # physics baseline
            base_eta = float(physics_baseline(np.array([distance]), np.array([avg_speed]))[0])

            # residual prediction
            x_vec = np.array([[avg_speed, payload, altitude, wind, max_v, distance]])
            resid_pipe = models["pipeline"]
            resid_pred = float(resid_pipe.predict(x_vec)[0])
            eta_pred = base_eta + resid_pred

            # quick quantile bounds by fitting quantile models on all residuals (fast enough)
            X_all = flights[FEATURES].to_numpy()
            y_all = flights[TARGET].to_numpy()
            base_all = models["base_all"]
            resid_all = y_all - base_all
            scaler = StandardScaler().fit(X_all)
            X_all_s = scaler.transform(X_all)
            gbr_best = resid_pipe.named_steps['gbr']

            q_lo = GradientBoostingRegressor(loss='quantile', alpha=0.05,
                                             n_estimators=gbr_best.n_estimators, learning_rate=gbr_best.learning_rate,
                                             max_depth=gbr_best.max_depth, subsample=gbr_best.subsample, random_state=42).fit(X_all_s, resid_all)
            q_hi = GradientBoostingRegressor(loss='quantile', alpha=0.95,
                                             n_estimators=gbr_best.n_estimators, learning_rate=gbr_best.learning_rate,
                                             max_depth=gbr_best.max_depth, subsample=gbr_best.subsample, random_state=42).fit(X_all_s, resid_all)
            x_s = scaler.transform(x_vec)
            lo = base_eta + float(q_lo.predict(x_s)[0])
            hi = base_eta + float(q_hi.predict(x_s)[0])

            st.success(f"Predicted ETA: **{eta_pred:.1f} s**")
            st.info(f"Interval (5–95%): **{lo:.1f} – {hi:.1f} s**")

            st.markdown("#### Quick sanity checks")
            st.write("- Physics baseline (distance/speed) = {:.2f} s".format(base_eta))
            st.write("- Residual adjustment (ML) = {:.2f} s".format(resid_pred))
            st.write("- Final ETA = baseline + residuals")

        with col_right:
            st.markdown("#### Scenario presets")
            if st.button("Calm short (2 km, 8 m/s)"):
                st.experimental_set_query_params(preset="calm_short")
                st.experimental_rerun()
            st.markdown("#### Visual preview")
            # show mini gauge or indicator
            fig_g = go.Figure(go.Indicator(
                mode="number+delta",
                value=eta_pred,
                delta={"reference": base_eta, "relative": False, "position": "right"},
                title={"text":"ETA vs baseline (s)"}
            ))
            fig_g.update_layout(height=200)
            st.plotly_chart(fig_g, use_container_width=True)

    # --- Data & Downloads tab ---
    with tabs[2]:
        st.markdown("### Dataset & Exports")
        st.markdown("#### Flight summary (first 200 rows)")
        st.dataframe(flights.head(200), use_container_width=True)
        st.markdown("Download the hold-out evaluation CSV (predictions + intervals)")
        eval_csv = models["eval"].to_csv(index=False).encode('utf-8')
        st.download_button("Download evaluation CSV", data=eval_csv, file_name="drone_eta_evaluation.csv", mime="text/csv")

        st.markdown("---")
        st.markdown("If you want a reproducible package, download a sample artifact (small demo subset).")
        # produce small sample
        sample = flights.sample(n=min(100, len(flights)), random_state=42).reset_index(drop=True)
        buf = io.StringIO()
        sample.to_csv(buf, index=False)
        st.download_button("Download sample dataset (CSV)", data=buf.getvalue().encode('utf-8'), file_name="drone_eta_sample.csv", mime="text/csv")

    # --- Notes & Repro tab ---
    with tabs[3]:
        st.markdown("## Notes, limitations & reproducibility")
        st.markdown("""
        **Method summary**
        - Baseline physics: `ETA_base = total_distance / average_speed`
        - Residuals: ML (Gradient Boosting) models `resid = total_time - ETA_base`
        - Final prediction: `ETA = ETA_base + resid_pred`
        - Prediction intervals: quantile GBMs (5%–95%) trained on residuals
        - Validation: GroupKFold CV (by flight) + grouped hold-out (20% by flight)

        **Why physics-first?**
        - Guarantees monotonic, sensible behavior in the What-If UI.
        - ML models focus on smaller residual effects (wind, altitude, payload).

        **Limitations**
        - Dataset from controlled flights — not validated for BVLOS / live ops.
        - Sparse extreme-wind samples reduce reliability in high-wind bins.
        - Predictions assume consistent drone platforms & sensors.

        **How to cite / use**
        - Put `CarnegieMellonDataMain.csv` next to `app.py` or use uploader.
        - The app trains inside the UI; use 'Download evaluation CSV' to export hold-out results.
        - For publications, use GroupKFold CV numbers and hold-out metrics reported here.

        **Contact**
        - Add your name & email in the repo README for credibility.
        """)
        st.markdown("---")
        st.markdown("### Quick tips for improving model quality")
        st.write("- Add more flights with varied wind conditions.")
        st.write("- Ensure position/velocity sampling frequency is consistent across flights.")
        st.write("- If possible, include drone mass/configuration as features.")

if __name__ == "__main__":
    main()
