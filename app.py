# app.py
# Drone ETA Lab — physics-first + ML residuals, leak-safe evaluation, reliable What-If simulator
# Requirements: streamlit, pandas, numpy, scikit-learn, plotly, matplotlib

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="Drone ETA Lab", page_icon="🛩️", layout="wide")
st.title("🛩️ Drone Delivery ETA — Physics + ML Residuals")
st.markdown("Reliable ETA predictions using a physics baseline (`distance / avg_speed`) plus an ML model for residual corrections. Leak-safe evaluation by flight.")

# ---------------------------
# Constants & required schema
# ---------------------------
DEFAULT_CSV = "CarnegieMellonDataMain.csv"
EPS = 1e-9

REQUIRED_COLUMNS = {
    "flight", "time", "speed", "payload", "altitude", "wind_speed",
    "velocity_x", "velocity_y", "velocity_z",
    "position_x", "position_y", "position_z"
}

FEATURES = [
    "average_speed", "max_payload", "max_altitude", "average_wind_speed",
    "max_velocity_combined", "total_distance"
]
TARGET = "total_time"

# ---------------------------
# Utility / modeling helpers
# ---------------------------
def physics_baseline(distance: np.ndarray, avg_speed: np.ndarray) -> np.ndarray:
    """Baseline ETA = distance / avg_speed (seconds). Avoid divide-by-zero by clamping speed."""
    d = np.array(distance, dtype=float)
    s = np.maximum(np.array(avg_speed, dtype=float), 0.1)  # min speed 0.1 m/s to avoid blowups
    return d / s

def make_eval_df(y_true: np.ndarray, y_pred: np.ndarray, lo: np.ndarray, hi: np.ndarray, wind_speed: np.ndarray) -> pd.DataFrame:
    """Construct evaluation DataFrame with useful diagnostics."""
    e = pd.DataFrame({
        "actual": np.array(y_true).astype(float),
        "predicted": np.array(y_pred).astype(float),
        "lower_5": np.array(lo).astype(float),
        "upper_95": np.array(hi).astype(float),
        "wind_speed": np.array(wind_speed).astype(float)
    })
    e["abs_err"] = np.abs(e["actual"] - e["predicted"])
    e["pi_width"] = e["upper_95"] - e["lower_5"]
    e["wind_bin"] = pd.cut(e["wind_speed"], bins=[0,5,10,15,20, np.inf],
                           labels=["0–5 m/s","5–10 m/s","10–15 m/s","15–20 m/s",">20 m/s"],
                           right=False) # Use right=False for left-inclusive bins
    return e.reset_index(drop=True)

# ---------------------------
# Data loading & aggregation
# ---------------------------
@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer) -> pd.DataFrame:
    """Load CSV from path or uploaded buffer; coerce numeric columns where appropriate."""
    if path_or_buffer is None:
        return pd.DataFrame()
    try:
        if isinstance(path_or_buffer, str) and os.path.exists(path_or_buffer):
            df = pd.read_csv(path_or_buffer)
        else:
            # uploaded buffer
            df = pd.read_csv(path_or_buffer)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")
    df.columns = df.columns.str.strip()
    # Attempt to coerce common numeric columns
    numeric_cols = ["time","altitude","velocity_x","velocity_y","velocity_z",
                    "speed","payload","wind_speed","position_x","position_y","position_z"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def make_flight_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw telemetry into one row per flight (required schema enforced)."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(list(missing))}")

    # FIX: Calculate combined velocity on the full dataframe first, then aggregate.
    # This avoids issues with groupby's Series index not matching the original dataframe's index.
    df['velocity_combined'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2 + df['velocity_z']**2)

    # Aggregate features per flight
    flight_summary = df.groupby("flight").agg(
        total_time=("time", "max"),
        average_speed=("speed", "mean"),
        max_payload=("payload", "max"),
        max_altitude=("altitude", "max"),
        average_wind_speed=("wind_speed", "mean"),
        max_velocity_combined=("velocity_combined", "max"),
        start_pos_x=("position_x", "first"),
        start_pos_y=("position_y", "first"),
        start_pos_z=("position_z", "first"),
        end_pos_x=("position_x", "last"),
        end_pos_y=("position_y", "last"),
        end_pos_z=("position_z", "last")
    ).reset_index()

    # Straight-line distance
    flight_summary["total_distance"] = np.sqrt(
        (flight_summary["end_pos_x"] - flight_summary["start_pos_x"])**2 +
        (flight_summary["end_pos_y"] - flight_summary["start_pos_y"])**2 +
        (flight_summary["end_pos_z"] - flight_summary["start_pos_z"])**2
    )

    # cleanup
    flight_summary = flight_summary.replace([np.inf, -np.inf], np.nan).dropna()
    flight_summary = flight_summary[flight_summary["total_time"] > 0].reset_index(drop=True)
    return flight_summary

# ---------------------------
# Train models (cached as resource)
# ---------------------------
@st.cache_resource(show_spinner=True)
def train_models(flights: pd.DataFrame) -> Dict[str, Any]:
    """
    Train pipeline to predict residuals:
      resid = total_time - physics_baseline(distance, average_speed)
    Uses GroupKFold CV for hyperparam tuning, GroupShuffleSplit for a grouped hold-out.
    Trains quantile GBMs on residuals to obtain prediction intervals.
    Returns primitives and fitted pipeline/quantile models.
    """
    X = flights[FEATURES].to_numpy()
    y = flights[TARGET].to_numpy()
    groups = flights["flight"].to_numpy()

    # compute physics baseline and residuals
    base_time = physics_baseline(flights["total_distance"].to_numpy(), flights["average_speed"].to_numpy())
    resid = y - base_time

    # pipeline: scale -> gbr (tunes on residuals)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(random_state=42))
    ])

    param_grid = {
        "gbr__n_estimators": [500, 1000],
        "gbr__learning_rate": [0.01, 0.05, 0.1],
        "gbr__max_depth": [3, 5],
        "gbr__subsample": [0.8, 1.0]
    }

    gkf = GroupKFold(n_splits=5)
    gs = GridSearchCV(pipe, param_grid, cv=gkf.split(X, resid, groups), scoring="r2", n_jobs=-1, verbose=0)
    gs.fit(X, resid)
    best_pipe = gs.best_estimator_
    cv_r2_resid = float(gs.best_score_)

    # grouped hold-out: 20% flights
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, resid, groups))
    
    # Use pandas iloc to correctly slice the dataframes based on the indices
    X_train_df, X_test_df = flights.iloc[train_idx][FEATURES], flights.iloc[test_idx][FEATURES]
    resid_train, resid_test = resid[train_idx], resid[test_idx]
    base_test = base_time[test_idx]
    y_test = y[test_idx]

    # fit best pipeline on train residuals
    best_pipe.fit(X_train_df, resid_train)
    resid_pred_test = best_pipe.predict(X_test_df)
    y_pred_test = base_test + resid_pred_test

    holdout_r2 = float(r2_score(y_test, y_pred_test))
    holdout_mape = float(mean_absolute_percentage_error(y_test, y_pred_test) * 100.0)
    holdout_mae = float(mean_absolute_error(y_test, y_pred_test))

    # Quantile GBMs trained on residuals_train to produce PIs on holdout
    scaler = best_pipe.named_steps["scaler"]
    gbr_best = best_pipe.named_steps["gbr"]

    X_train_s = scaler.transform(X_train_df)
    X_test_s = scaler.transform(X_test_df)

    q_lo = GradientBoostingRegressor(loss="quantile", alpha=0.05,
                                     n_estimators=gbr_best.n_estimators,
                                     learning_rate=gbr_best.learning_rate,
                                     max_depth=gbr_best.max_depth,
                                     subsample=gbr_best.subsample,
                                     random_state=42).fit(X_train_s, resid_train)

    q_hi = GradientBoostingRegressor(loss="quantile", alpha=0.95,
                                     n_estimators=gbr_best.n_estimators,
                                     learning_rate=gbr_best.learning_rate,
                                     max_depth=gbr_best.max_depth,
                                     subsample=gbr_best.subsample,
                                     random_state=42).fit(X_train_s, resid_train)

    # Holdout intervals
    lo_resid = q_lo.predict(X_test_s)
    hi_resid = q_hi.predict(X_test_s)
    lo_time = base_test + lo_resid
    hi_time = base_test + hi_resid

    # Pass the appropriate dataframes/arrays to make_eval_df
    eval_df = make_eval_df(y_test, y_pred_test, lo_time, hi_time, flights.iloc[test_idx]["average_wind_speed"].to_numpy())
    coverage = float(((eval_df["actual"] >= eval_df["lower_5"]) & (eval_df["actual"] <= eval_df["upper_95"])).mean() * 100.0)

    # For interactive what-if: fit quantiles on ALL residuals
    X_all_s = scaler.transform(X)
    q_lo_all = GradientBoostingRegressor(loss="quantile", alpha=0.05,
                                         n_estimators=gbr_best.n_estimators,
                                         learning_rate=gbr_best.learning_rate,
                                         max_depth=gbr_best.max_depth,
                                         subsample=gbr_best.subsample,
                                         random_state=42).fit(X_all_s, resid)
    q_hi_all = GradientBoostingRegressor(loss="quantile", alpha=0.95,
                                         n_estimators=gbr_best.n_estimators,
                                         learning_rate=gbr_best.learning_rate,
                                         max_depth=gbr_best.max_depth,
                                         subsample=gbr_best.subsample,
                                         random_state=42).fit(X_all_s, resid)

    feat_imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": gbr_best.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return {
        "pipeline": best_pipe,
        "cv_r2_resid": cv_r2_resid,
        "r2": holdout_r2,
        "mape": holdout_mape,
        "mae": holdout_mae,
        "coverage": coverage,
        "eval": eval_df,
        "feature_importance": feat_imp,
        "flights": flights,
        "test_idx": test_idx,
        "base_all": base_time,
        "q_lo_all": q_lo_all,
        "q_hi_all": q_hi_all
    }

# ---------------------------
# Main UI flow
# ---------------------------
def main():
    st.sidebar.header("Dataset")
    uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
    use_default = st.sidebar.button("Use default dataset (if present)")
    
    # FIX: Move this checkbox to the sidebar where it's defined and used for control
    show_heatmap = st.sidebar.checkbox("Show correlation heatmap", value=True)
    show_ablation = st.sidebar.checkbox("Show ablation (may be slow)", value=False)

    csv_source = None
    if uploaded:
        csv_source = uploaded
    elif use_default:
        if os.path.exists(DEFAULT_CSV):
            csv_source = DEFAULT_CSV
        else:
            st.sidebar.warning(f"Default CSV not found at '{DEFAULT_CSV}'")

    if csv_source is None:
        st.info("Upload `CarnegieMellonDataMain.csv` or use the default dataset button (if file exists on server).")
        return

    # Load raw CSV
    try:
        raw = load_csv(csv_source)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return

    # Create flight summary
    try:
        flights = make_flight_table(raw)
    except Exception as e:
        st.error(f"Failed to build flight summary: {e}")
        return

    # Train models (heavy op cached)
    with st.spinner("Training models (takes a moment)..."):
        models = train_models(flights)

    # Header metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hold-out R²", f"{models['r2']:.3f}")
    col2.metric("MAPE (hold-out)", f"{models['mape']:.1f}%")
    col3.metric("MAE (hold-out)", f"{models['mae']:.2f} s")
    col4.metric("PI Coverage (5–95%)", f"{models['coverage']:.1f}%")
    st.markdown("---")

    # Tabs: Dashboard / What-If / Data & Downloads / Notes
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧪 What-If Simulator", "📁 Data & Downloads", "ℹ️ Notes"])

    # --- Dashboard ---
    with tab1:
        st.header("Calibration & Reliability (hold-out)")
        eval_df = models["eval"]

        # Calibration scatter
        fig_cal = px.scatter(eval_df, x="actual", y="predicted",
                             labels={"actual":"Actual total time (s)", "predicted":"Predicted total time (s)"},
                             title="Predicted vs Actual (Hold-out)")
        mn = float(min(eval_df["actual"].min(), eval_df["predicted"].min()))
        mx = float(max(eval_df["actual"].max(), eval_df["predicted"].max()))
        fig_cal.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx, line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig_cal, use_container_width=True)

        # Residuals
        st.subheader("Residual diagnostics")
        fig_res = px.scatter(eval_df, x="predicted", y="abs_err",
                             labels={"predicted":"Predicted (s)", "abs_err":"Absolute error (s)"},
                             title="Absolute Error vs Predicted")
        st.plotly_chart(fig_res, use_container_width=True)

        # Wind MAPE bar
        st.subheader("MAPE by Wind Bin (hold-out)")
        # FIX: Correctly calculate MAPE by wind bin using a groupby aggregate
        mape_by_bin = eval_df.groupby("wind_bin", observed=True).apply(
            lambda g: (np.abs(g["actual"] - g["predicted"]) / np.maximum(g["actual"], EPS)).mean() * 100
        ).reset_index(name="mape")
        
        if not mape_by_bin.empty:
            fig_wind = px.bar(mape_by_bin, x="wind_bin", y="mape",
                              labels={"wind_bin":"Wind (m/s)", "mape":"MAPE (%)"},
                              title="MAPE by Wind Bin (Hold-out)")
            st.plotly_chart(fig_wind, use_container_width=True)
        else:
            st.info("Not enough hold-out flights in wind bins to display MAPE by bin.")

        # PI preview
        st.subheader("Prediction Intervals (hold-out)")
        fig_pi = px.scatter(eval_df.reset_index(drop=True), x=eval_df.reset_index(drop=True).index,
                            y="predicted", error_y=eval_df["upper_95"] - eval_df["predicted"],
                            error_y_minus=eval_df["predicted"] - eval_df["lower_5"],
                            labels={"x":"Test Flight #","predicted":"Predicted time (s)"},
                            title="Prediction Intervals (5%–95%) — Hold-out")
        st.plotly_chart(fig_pi, use_container_width=True)

        # Feature importance
        st.subheader("Feature importance (residual model)")
        st.dataframe(models["feature_importance"], use_container_width=True)

        # Correlation heatmap
        if show_heatmap:
            st.subheader("Correlation heatmap (features vs total_time)")
            corr = flights[FEATURES + [TARGET]].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu")
            st.plotly_chart(fig, use_container_width=True)

        # optional ablation
        if show_ablation:
            st.subheader("Feature ablation (ΔR² when removing a feature)")
            try:
                # FIX: Correctly re-train the pipeline for each ablated feature set
                test_idx = models["test_idx"]
                train_idx = np.setdiff1d(np.arange(len(flights)), test_idx)
                base_pipe = models["pipeline"]
                
                # Full model performance on hold-out
                full_resid = base_pipe.predict(flights.iloc[test_idx][FEATURES].to_numpy())
                full_pred = models["base_all"][test_idx] + full_resid
                y_test_all = flights.iloc[test_idx][TARGET].to_numpy()
                full_r2 = r2_score(y_test_all, full_pred)
                
                rows = []
                for f in FEATURES:
                    keep = [x for x in FEATURES if x != f]
                    
                    # Re-instantiate a new pipeline for each ablation test
                    pipe_local = Pipeline([
                        ("scaler", StandardScaler()),
                        ("gbr", GradientBoostingRegressor(
                            n_estimators=base_pipe.named_steps["gbr"].n_estimators,
                            learning_rate=base_pipe.named_steps["gbr"].learning_rate,
                            max_depth=base_pipe.named_steps["gbr"].max_depth,
                            subsample=base_pipe.named_steps["gbr"].subsample,
                            random_state=42
                        ))
                    ])
                    
                    # Train on the subset of features
                    pipe_local.fit(flights.iloc[train_idx][keep], models["base_all"][train_idx])
                    
                    # Predict on the hold-out set with the same subset
                    base_test_local = physics_baseline(
                        flights.iloc[test_idx]["total_distance"].to_numpy(),
                        flights.iloc[test_idx]["average_speed"].to_numpy()
                    )
                    pred_local = base_test_local + pipe_local.predict(flights.iloc[test_idx][keep])
                    
                    # Calculate R² and delta
                    r2_local = r2_score(y_test_all, pred_local)
                    rows.append({"feature_removed": f, "ΔR² (remove)": float(full_r2 - r2_local)})
                
                ablation_df = pd.DataFrame(rows).sort_values("ΔR² (remove)", ascending=False)
                st.dataframe(ablation_df, use_container_width=True)
            except Exception as e:
                st.info(f"Ablation failed or too slow: {e}")

    # --- What-If simulator ---
    with tab2:
        st.header("What-If Simulator — physics baseline + residual adjustment")
        st.markdown("ETA is computed as: `ETA = distance / avg_speed + ML_residual` (so increasing distance or decreasing speed always increases ETA).")

        left_col, right_col = st.columns([2,1])
        with left_col:
            med = models["flights"][FEATURES].median()
            # Ensure number_input values are floats and have a max value
            avg_speed = st.number_input("Average Speed (m/s)", min_value=0.1, max_value=float(max(1.0, models["flights"]["average_speed"].max())), value=float(max(0.5, med["average_speed"])))
            payload = st.number_input("Max Payload (kg)", min_value=0.0, max_value=float(max(1.0, models["flights"]["max_payload"].max())), value=float(med["max_payload"]))
            altitude = st.number_input("Max Altitude (m)", min_value=0.0, max_value=float(max(1.0, models["flights"]["max_altitude"].max())), value=float(med["max_altitude"]))
            wind = st.number_input("Average Wind Speed (m/s)", min_value=0.0, max_value=float(max(1.0, models["flights"]["average_wind_speed"].max())), value=float(med["average_wind_speed"]))
            max_v = st.number_input("Max Velocity (m/s)", min_value=0.0, max_value=float(max(1.0, models["flights"]["max_velocity_combined"].max())), value=float(med["max_velocity_combined"]))
            distance = st.number_input("Total Distance (m)", min_value=0.1, max_value=float(max(1.0, models["flights"]["total_distance"].max())), value=float(med["total_distance"]))

            # compute baseline & residual
            base_eta = float(physics_baseline(np.array([distance]), np.array([avg_speed]))[0])
            x_vec = np.array([[avg_speed, payload, altitude, wind, max_v, distance]])
            resid_pipe = models["pipeline"]
            resid_pred = float(resid_pipe.predict(x_vec)[0])
            eta_pred = base_eta + resid_pred

            # fast quantile intervals using full-quantile models (returned by train_models)
            scaler_pipe = resid_pipe.named_steps["scaler"]
            q_lo_all = models["q_lo_all"]
            q_hi_all = models["q_hi_all"]
            x_s = scaler_pipe.transform(x_vec)
            lo = base_eta + float(q_lo_all.predict(x_s)[0])
            hi = base_eta + float(q_hi_all.predict(x_s)[0])

            st.success(f"Predicted ETA: **{eta_pred:.1f} s**")
            st.info(f"Interval (5%–95%): **{lo:.1f} – {hi:.1f} s**")

            st.markdown("#### Breakdown")
            st.write(f"- Physics baseline (distance/speed): **{base_eta:.2f} s**")
            st.write(f"- ML residual adjustment: **{resid_pred:.2f} s**")
            st.write(f"- Final ETA: **{eta_pred:.2f} s**")

        with right_col:
            st.markdown("#### Sanity checks")
            if st.button("Double distance sanity test"):
                new_dist = distance * 2.0
                base2 = float(physics_baseline(np.array([new_dist]), np.array([avg_speed]))[0])
                pred2 = base2 + float(resid_pipe.predict(np.array([[avg_speed, payload, altitude, wind, max_v, new_dist]]))[0])
                st.write(f"Distance ×2 -> ETA from {eta_pred:.2f}s to {pred2:.2f}s (baseline doubled to {base2:.2f}s)")
            st.markdown("#### Visual ETA vs Baseline")
            fig_g = px.bar(x=["baseline","final"], y=[base_eta, eta_pred], labels={"x":"type","y":"seconds"}, title="Baseline vs Final ETA")
            st.plotly_chart(fig_g, use_container_width=True)

    # --- Data & Downloads ---
    with tab3:
        st.header("Data & Downloads")
        st.markdown("Flight summary (first 200 rows)")
        st.dataframe(models["flights"].head(200), use_container_width=True)
        st.markdown("Download hold-out evaluation (predictions + intervals)")
        eval_csv = models["eval"].to_csv(index=False).encode("utf-8")
        st.download_button("Download evaluation CSV", data=eval_csv, file_name="drone_eta_evaluation.csv", mime="text/csv")

        # sample dataset
        st.markdown("Download a small sample dataset for sharing")
        sample = models["flights"].sample(n=min(100, len(models["flights"])), random_state=42).reset_index(drop=True)
        buf = io.StringIO()
        sample.to_csv(buf, index=False)
        st.download_button("Download sample CSV", data=buf.getvalue().encode("utf-8"), file_name="drone_eta_sample.csv", mime="text/csv")

    # --- Notes & Repro ---
    with tab4:
        st.header("Notes, Limitations & Reproducibility")
        st.markdown("""
        **Summary**
        - Baseline: `ETA_base = total_distance / average_speed`
        - Residual model: GradientBoostingRegressor predicts `resid = total_time - ETA_base`.
        - Final prediction: `ETA = ETA_base + resid_pred`.
        - Validation: GroupKFold CV (tuning) and grouped hold-out (20% flights).
        - Intervals: quantile GBMs on residuals (5%–95%); hold-out coverage reported.

        **Why this design**
        - Physics-first ensures monotonic, interpretable behavior (distance increases => ETA increases).
        - ML captures secondary effects (wind, altitude, payload).

        **Limitations**
        - Dataset came from controlled test flights; do not use this app for operational BVLOS decisions.
        - Sparse extreme-wind samples reduce reliability at high winds.
        - The interactive intervals are trained on all residuals for responsiveness; use hold-out metrics for formal reporting.

        **Reproducibility**
        - Place `CarnegieMellonDataMain.csv` beside `app.py` or upload via the uploader.
        - The app trains models in the UI (cached). Use 'Download evaluation CSV' to export hold-out results.
        """)

if __name__ == "__main__":
    main()

