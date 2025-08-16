# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Drone ETA Lab", page_icon="🛩️", layout="wide")

st.sidebar.title("🛩️ Drone ETA Lab")
st.sidebar.markdown("""
**What this does**
- Trains an ETA model from your dataset (XGBoost)  
- Shows calibration & error by wind  
- Provides prediction intervals (quantile GB)  
- Lets you test “what-if” inputs  
- Upload your own CSV and batch-predict
""")

# -------------------------
# Utility: load CSV (local path or uploaded file)
# -------------------------
@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer):
    try:
        df = pd.read_csv(path_or_buffer)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()
    df.columns = df.columns.str.strip()
    # Coerce important numeric columns (silent)
    for c in ["time","altitude","velocity_x","velocity_y","velocity_z","speed","payload","wind_speed","position_x","position_y","position_z","flight"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# -------------------------
# Feature engineering (NO LEAKAGE)
# -------------------------
@st.cache_data(show_spinner=False)
def build_flight_summary(df):
    required = {"flight","time","speed","payload","altitude","wind_speed",
                "velocity_x","velocity_y","velocity_z","position_x","position_y","position_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(list(missing))}")

    # aggregate per flight (target: total_time)
    flight_summary = df.groupby("flight").agg(
        total_time=("time","max"),
        average_speed=("speed","mean"),
        speed_std=("speed","std"),
        max_payload=("payload","max"),
        max_altitude=("altitude","max"),
        average_wind_speed=("wind_speed","mean"),
        max_velocity_combined=("velocity_x", lambda x: np.sqrt(x**2 + df.loc[x.index,"velocity_y"]**2 + df.loc[x.index,"velocity_z"]**2).max()),
        path_start_x=("position_x","first"),
        path_start_y=("position_y","first"),
        path_start_z=("position_z","first"),
        path_end_x=("position_x","last"),
        path_end_y=("position_y","last"),
        path_end_z=("position_z","last")
    ).reset_index()

    # straight-line distance (meters)
    flight_summary["total_distance"] = np.sqrt(
        (flight_summary["path_end_x"] - flight_summary["path_start_x"])**2 +
        (flight_summary["path_end_y"] - flight_summary["path_start_y"])**2 +
        (flight_summary["path_end_z"] - flight_summary["path_start_z"])**2
    )

    # Derived but non-leaky features (no use of target total_time)
    # Interactions & nonlinear features:
    flight_summary["wind_x_dist"] = flight_summary["average_wind_speed"] * flight_summary["total_distance"]
    flight_summary["payload_x_dist"] = flight_summary["max_payload"] * flight_summary["total_distance"]
    flight_summary["dist_sq"] = flight_summary["total_distance"] ** 2
    flight_summary["wind_sq"] = flight_summary["average_wind_speed"] ** 2
    flight_summary["speed_sq"] = flight_summary["average_speed"] ** 2

    # Binned wind categorical (helpful)
    flight_summary["wind_bin"] = pd.cut(
        flight_summary["average_wind_speed"],
        bins=[-1,2,5,10,15,100],
        labels=["Calm","Light","Moderate","Strong","Very Strong"]
    )

    flight_summary = flight_summary.replace([np.inf, -np.inf], np.nan).dropna()
    flight_summary = flight_summary[flight_summary["total_time"] > 0]
    return flight_summary

# -------------------------
# Model training & evaluation
# -------------------------
FEATURES = [
    "average_speed","speed_std","max_payload","max_altitude","average_wind_speed",
    "max_velocity_combined","total_distance","wind_x_dist","payload_x_dist",
    "dist_sq","wind_sq","speed_sq"
]
TARGET = "total_time"

@st.cache_resource(show_spinner=True)
def train_and_eval(flights_df):
    X = flights_df[FEATURES].copy()
    y = flights_df[TARGET].copy().astype(float)

    # Optional polynomial expansion (degree=2) for interactions (kept small)
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X)
    poly_feature_names = poly.get_feature_names_out(FEATURES)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_poly)

    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)

    # Use XGBoost with small GridSearch for speed / reliability
    xgb = XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42, verbosity=0, n_jobs=1)
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1]
    }
    gs = GridSearchCV(xgb, param_grid, cv=3, scoring="r2", n_jobs=1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    # Train quantile-like models for intervals using GradientBoostingRegressor (quantile)
    q_lo = GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=100, max_depth=3, random_state=42)
    q_hi = GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=100, max_depth=3, random_state=42)
    q_lo.fit(X_train, y_train)
    q_hi.fit(X_train, y_train)

    # Predictions on hold-out
    preds = best.predict(X_test)
    lower = q_lo.predict(X_test)
    upper = q_hi.predict(X_test)

    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100

    # Cross-validated R2 (5-fold) on whole dataset for robust metric
    cv_r2 = cross_val_score(best, Xs, y, cv=5, scoring="r2").mean()

    eval_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": preds,
        "lower_5": lower,
        "upper_95": upper
    }).reset_index(drop=True)

    # compute feature importance approx (from XGBoost)
    try:
        fi = best.get_booster().get_score(importance_type="gain")
        # map to feature names (poly names) - best effort
        fi_series = pd.Series(fi).sort_values(ascending=False)
    except Exception:
        fi_series = pd.Series([])

    return {
        "model": best,
        "scaler": scaler,
        "poly": poly,
        "eval_df": eval_df,
        "r2": r2,
        "mape": mape,
        "cv_r2": cv_r2,
        "feature_importance": fi_series,
        "X_test_shape": X_test.shape
    }

# -------------------------
# UI: Tabs
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard","🧪 What-If Simulator","📁 Data","ℹ️ Notes"])

# Default local CSV name expected in repo root
default_csv = "CarnegieMellonDataMain.csv"
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
csv_source = uploaded if uploaded is not None else (default_csv if os.path.exists(default_csv) else None)

if csv_source is None:
    st.sidebar.error("No dataset. Upload CSV in sidebar or add CarnegieMellonDataMain.csv to the repo.")
    st.stop()

df = load_csv(csv_source)
if df.empty:
    st.sidebar.error("Loaded CSV is empty or invalid.")
    st.stop()

try:
    flights = build_flight_summary(df)
except Exception as e:
    st.error(f"Feature build failed: {e}")
    st.stop()

with st.spinner("Training model (may take 30–90s depending on data)..."):
    results = train_and_eval(flights)

eval_df = results["eval_df"]

# ---------- TAB 1: Dashboard ----------
with tab1:
    st.title("Drone Delivery ETA — Performance Dashboard")
    st.caption("Educational research demo — not certified for real-world operations (BVLOS).")

    c1, c2, c3 = st.columns(3)
    c1.metric("Hold-out R²", f"{results['r2']:.3f}")
    c2.metric("Cross-val R²", f"{results['cv_r2']:.3f}")
    c3.metric("MAPE", f"{results['mape']:.1f}%")

    # Calibration (Pred vs Actual)
    fig_cal = px.scatter(eval_df, x="actual", y="predicted", trendline="ols",
                         labels={"actual":"Actual Total Time (s)","predicted":"Predicted Time (s)"},
                         title="Calibration: Predicted vs Actual")
    fig_cal.add_shape(type="line", x0=eval_df["actual"].min(), y0=eval_df["actual"].min(),
                      x1=eval_df["actual"].max(), y1=eval_df["actual"].max(),
                      line=dict(dash="dash"))
    st.plotly_chart(fig_cal, use_container_width=True)

    # Residuals
    res_df = eval_df.copy()
    res_df["residual"] = res_df["actual"] - res_df["predicted"]
    fig_res = px.scatter(res_df, x="predicted", y="residual", title="Residuals vs Predicted",
                         labels={"predicted":"Predicted (s)","residual":"Residual (s)"})
    fig_res.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig_res, use_container_width=True)

    # MAPE by wind bin (map wind info back to eval)
    # Build wind_test aligned with eval via same train_test_split RNG
    wind_arr = flights["average_wind_speed"].to_numpy()
    _, wind_test = train_test_split(wind_arr, test_size=0.2, random_state=42)
    eval_df["wind_speed"] = wind_test
    eval_df["wind_bin"] = pd.cut(eval_df["wind_speed"], bins=[-1,2,5,10,15,100],
                                 labels=["Calm","Light","Moderate","Strong","Very Strong"])
    eval_df["ape"] = np.abs((eval_df["actual"] - eval_df["predicted"]) / eval_df["actual"]) * 100
    mape_by_bin = eval_df.groupby("wind_bin")["ape"].mean().reset_index().dropna()
    fig_mape = px.bar(mape_by_bin, x="wind_bin", y="ape", labels={"wind_bin":"Wind category","ape":"MAPE (%)"},
                      title="MAPE by Wind Category")
    st.plotly_chart(fig_mape, use_container_width=True)

    # Prediction intervals plot
    fig_pi = px.scatter(eval_df.reset_index(drop=True), x=eval_df.reset_index(drop=True).index, y="predicted",
                        error_y=eval_df["upper_95"] - eval_df["predicted"],
                        error_y_minus=eval_df["predicted"] - eval_df["lower_5"],
                        title="Prediction Intervals (5%–95%)",
                        labels={"x":"Test flight #","predicted":"Predicted Time (s)"})
    st.plotly_chart(fig_pi, use_container_width=True)

    # Feature importance (top 10)
    st.subheader("Feature importance (XGBoost gain)")
    if not results["feature_importance"].empty:
        fi = results["feature_importance"].rename_axis("feature").reset_index(name="gain")
        st.dataframe(fi.head(12), use_container_width=True)
    else:
        st.info("Feature importance not available for this model.")

    st.markdown("---")
    st.download_button("Download evaluation CSV", data=eval_df.to_csv(index=False).encode("utf-8"),
                       file_name="drone_eta_evaluation.csv", mime="text/csv")

# ---------- TAB 2: What-If ----------
with tab2:
    st.header("What-If ETA Simulator")
    med = flights[FEATURES].median()

    avg_speed = st.slider("Average speed (m/s)", 0.1, float(flights["average_speed"].max()), float(med["average_speed"]))
    payload = st.slider("Payload (kg)", 0.0, float(max(0.1, flights["max_payload"].max() or 10)), float(med["max_payload"]))
    max_alt = st.slider("Max altitude (m)", 0.0, float(flights["max_altitude"].max()), float(med["max_altitude"]))
    wind = st.slider("Average wind (m/s)", 0.0, float(flights["average_wind_speed"].max()), float(med["average_wind_speed"]))
    dist = st.slider("Straight-line distance (m)", 0.0, float(flights["total_distance"].max()), float(med["total_distance"]))

    # advanced optional
    max_v = st.number_input("Max combined velocity (m/s)", value=float(med["max_velocity_combined"]))
    speed_std = st.number_input("Speed std (m/s)", value=float(med["speed_std"] if "speed_std" in flights.columns else 1.0))

    # create features consistent with training (no leakage)
    wind_x_dist = wind * dist
    payload_x_dist = payload * dist
    dist_sq = dist ** 2
    wind_sq = wind ** 2
    speed_sq = avg_speed ** 2

    X_live = pd.DataFrame([{
        "average_speed": avg_speed,
        "speed_std": speed_std,
        "max_payload": payload,
        "max_altitude": max_alt,
        "average_wind_speed": wind,
        "max_velocity_combined": max_v,
        "total_distance": dist,
        "wind_x_dist": wind_x_dist,
        "payload_x_dist": payload_x_dist,
        "dist_sq": dist_sq,
        "wind_sq": wind_sq,
        "speed_sq": speed_sq
    }])
    # Apply poly & scaler and predict
    X_poly_live = results["poly"].transform(X_live[FEATURES])
    Xs_live = results["scaler"].transform(X_poly_live)
    pred = results["model"].predict(Xs_live)[0]

    # quantile intervals quick fit (we trained q models on the held-out pattern inside training)
    # For simplicity reuse same quantile approach by refitting small models on full X,y (fast)
    # (acceptable for demo; for publication train once and persist)
    X_full = flights[FEATURES].to_numpy()
    poly_full = results["poly"].transform(flights[FEATURES])
    scaler_full = StandardScaler().fit(poly_full)
    Xs_full = scaler_full.transform(poly_full)
    y_full = flights[TARGET].to_numpy()
    q_lo = GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=100, max_depth=3, random_state=42).fit(Xs_full, y_full)
    q_hi = GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=100, max_depth=3, random_state=42).fit(Xs_full, y_full)
    lo = q_lo.predict(results["scaler"].transform(results["poly"].transform(X_live[FEATURES])))[0]
    hi = q_hi.predict(results["scaler"].transform(results["poly"].transform(X_live[FEATURES])))[0]

    st.success(f"Predicted ETA: {pred:.1f} s  — Interval (5%–95%): {lo:.1f} – {hi:.1f} s")
    st.caption("Note: model trained on test-flight data. Not validated for operational BVLOS use.")

# ---------- TAB 3: Data ----------
with tab3:
    st.header("Flight summary (cleaned)")
    st.write("Top rows of the cleaned flight-level table used for modeling.")
    st.dataframe(flights.head(80), use_container_width=True)
    st.markdown("Upload a different CSV to retrain / test (must contain required columns).")

    up = st.file_uploader("Upload CSV to replace dataset (optional)", type=["csv"])
    if up:
        try:
            newdf = load_csv(up)
            new_flights = build_flight_summary(newdf)
            st.success("New dataset processed. Refresh to retrain.")
            st.dataframe(new_flights.head(50), use_container_width=True)
        except Exception as e:
            st.error(f"Processing failed: {e}")

# ---------- TAB 4: Notes ----------
with tab4:
    st.header("Notes, limitations & reproducibility")
    st.markdown("""
    **Summary**
    - Model: XGBoost with polynomial features (degree=2). Quantile GB used for intervals.
    - Metrics: hold-out R², cross-val R², MAPE (displayed in Dashboard tab).
    - This app intentionally **avoids** target-derived features to prevent label leakage.

    **Caveats**
    - Data are controlled test flights; not certified for operational decision-making.
    - Prediction intervals are empirical quantiles from quantile regression models.

    **How to reproduce**
    1. Add `CarnegieMellonDataMain.csv` to the repo root OR upload via sidebar.
    2. Deploy this repo to Streamlit Community Cloud.
    3. Use the "Download evaluation CSV" button to export hold-out predictions.

    **For applications**: if you claim a metric (R²), be ready to show the repo + evaluation CSV as proof.
    """)

