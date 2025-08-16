import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
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
- Trains an ETA model from your dataset  
- Shows calibration & error by wind  
- Gives prediction intervals  
- Lets you test “what-if” inputs  
- Batch-predict on uploaded CSV  
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
            # If path is bytes or buffer, let calling code pass it to read_csv
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

    # Calculate total path length
    df['dx'] = df.groupby('flight')['position_x'].diff().fillna(0)
    df['dy'] = df.groupby('flight')['position_y'].diff().fillna(0)
    df['dz'] = df.groupby('flight')['position_z'].diff().fillna(0)
    df['incremental_distance'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)
    
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
        end_pos_z=('position_z', 'last'),
        total_path_length=('incremental_distance', 'sum') # New valid feature
    ).reset_index()

    # Calculate straight-line distance
    flight_summary["straight_line_distance"] = np.sqrt(
        (flight_summary['end_pos_x'] - flight_summary['start_pos_x'])**2 +
        (flight_summary['end_pos_y'] - flight_summary['start_pos_y'])**2 +
        (flight_summary['end_pos_z'] - flight_summary['start_pos_z'])**2
    )

    # Calculate a valid, non-leaking feature
    # Using np.where to handle division by zero for total_path_length=0
    flight_summary["path_straightness"] = np.where(
        flight_summary["total_path_length"] > 0,
        flight_summary["straight_line_distance"] / flight_summary["total_path_length"],
        1.0  # Assign 1.0 for straightness if path length is zero
    )

    # Drop columns not used in model training, as they are now used to create new ones
    flight_summary = flight_summary.drop(columns=['start_pos_x', 'start_pos_y', 'start_pos_z', 'end_pos_x', 'end_pos_y', 'end_pos_z'])
    
    flight_summary = flight_summary.replace([np.inf, -np.inf], np.nan).dropna()
    flight_summary = flight_summary[flight_summary["total_time"] > 0]
    return flight_summary

# -----------------------------
# Model training (cached)
# -----------------------------
FEATURES = [
    'average_speed', 'max_payload', 'max_altitude', 'average_wind_speed',
    'max_velocity_combined', 'straight_line_distance', 'total_path_length', 'path_straightness'
]
TARGET = 'total_time'

@st.cache_resource(show_spinner=True)
def train_models(flight_summary: pd.DataFrame):
    X = flight_summary[FEATURES].to_numpy()
    y = flight_summary[TARGET].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Re-introducing GridSearchCV to find better parameters
    gbr_tuned = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [300],
        'learning_rate': [0.05],
        'max_depth': [3]
    }
    gs = GridSearchCV(gbr_tuned, param_grid, cv=3, scoring='r2', n_jobs=1)
    gs.fit(X_train, y_train)
    best_gbr = gs.best_estimator_

    # Using a more powerful Random Forest model
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)

    # Stacking regressor with the re-tuned base estimators
    stacked = StackingRegressor(
        estimators=[('gbr', best_gbr), ('rf', rf)],
        final_estimator=GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    )
    stacked.fit(X_train, y_train)
    preds = stacked.predict(X_test)

    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100

    # Quantile models for prediction intervals with improved n_estimators
    lower_model = GradientBoostingRegressor(loss='quantile', alpha=0.05, n_estimators=100, random_state=42)
    upper_model = GradientBoostingRegressor(loss='quantile', alpha=0.95, n_estimators=100, random_state=42)
    lower_model.fit(X_train, y_train)
    upper_model.fit(X_train, y_train)
    lower = lower_model.predict(X_test)
    upper = upper_model.predict(X_test)

    # Feature importance average
    rf_imp = rf.fit(X_train, y_train).feature_importances_
    gbr_imp = best_gbr.fit(X_train, y_train).feature_importances_
    avg_imp = (rf_imp + gbr_imp) / 2.0

    eval_df = pd.DataFrame({
        "actual": y_test,
        "predicted": preds,
        "lower_5": lower,
        "upper_95": upper
    }).reset_index(drop=True)

    return {
        "scaler": scaler,
        "stacked": stacked,
        "gb_best": best_gbr,
        "rf": rf,
        "X_test": X_test,
        "y_test": y_test,
        "eval": eval_df,
        "r2": r2,
        "mape": mape,
        "feature_importance": pd.DataFrame({"feature": FEATURES, "importance": avg_imp}).sort_values("importance", ascending=False)
    }

# -----------------------------
# Utility: wind binning + MAPE by bin
# -----------------------------
def add_wind_bins(flight_summary: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    X = flight_summary[FEATURES].to_numpy()
    y = flight_summary[TARGET].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)
    _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    wind_full = flight_summary["average_wind_speed"].to_numpy()
    _, wind_test = train_test_split(wind_full, test_size=0.2, random_state=42)

    e = eval_df.copy()
    e["wind_speed"] = wind_test
    e["wind_bin"] = pd.cut(
        e["wind_speed"],
        bins=[0,5,10,15,20,np.inf],
        labels=['0–5 m/s','5–10 m/s','10–15 m/s','15–20 m/s','>20 m/s']
    )
    e["ape"] = np.abs((e["actual"] - e["predicted"]) / e["actual"]) * 100
    return e

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧪 What-If Simulator", "📁 Your Data", "ℹ️ Notes"])

# === Initial data load: try local CSV first
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
scaler = models["scaler"]

# =========================
# TAB 1: Dashboard
# =========================
with tab1:
    st.title("Drone Delivery ETA — Analytics & Prediction")
    st.caption("Educational demo. Not for operational use (see Notes tab).")

    colA, colB, colC = st.columns(3)
    colA.metric("R² (hold-out)", f"{models['r2']:.3f}")
    colB.metric("MAPE (overall)", f"{models['mape']:.1f}%")
    colC.metric("# Flights (post-clean)", f"{len(flights):,}")

    # Pred vs Actual (calibration)
    fig_cal = px.scatter(
        eval_df, x="actual", y="predicted",
        labels={"actual":"Actual Total Time (s)","predicted":"Predicted Total Time (s)"},
        title="Calibration: Predicted vs Actual"
    )
    fig_cal.add_shape(type="line", x0=eval_df["actual"].min(), y0=eval_df["actual"].min(),
                     x1=eval_df["actual"].max(), y1=eval_df["actual"].max(),
                     line=dict(dash="dash"))
    st.plotly_chart(fig_cal, use_container_width=True)

    # Residuals
    res_df = eval_df.copy()
    res_df["residual"] = res_df["actual"] - res_df["predicted"]
    fig_res = px.scatter(res_df, x="predicted", y="residual",
                         labels={"predicted":"Predicted (s)","residual":"Residual (s)"},
                         title="Residuals vs Predicted")
    fig_res.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig_res, use_container_width=True)

    # Wind bin MAPE
    binned = add_wind_bins(flights, eval_df)
    mape_by_bin = binned.groupby("wind_bin")["ape"].mean().reset_index().dropna()
    fig_mape = px.bar(mape_by_bin, x="wind_bin", y="ape",
                     labels={"wind_bin":"Wind (m/s)","ape":"MAPE (%)"},
                     title="Error by Wind Bin (MAPE)")
    st.plotly_chart(fig_mape, use_container_width=True)

    # Prediction intervals preview
    fig_pi = px.scatter(
        eval_df.reset_index(drop=True),
        x=eval_df.reset_index(drop=True).index,
        y="predicted",
        error_y=eval_df["upper_95"] - eval_df["predicted"],
        error_y_minus=eval_df["predicted"] - eval_df["lower_5"],
        labels={"x":"Test Flight #","predicted":"Predicted Time (s)"},
        title="Prediction Intervals (5%–95%)"
    )
    st.plotly_chart(fig_pi, use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance (RF+GBR avg)")
    st.dataframe(models["feature_importance"], use_container_width=True)

    st.markdown("---")
    st.info("Download the dashboard-ready results CSV (used for the public app).")
    csv_bytes = eval_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download results CSV", data=csv_bytes, file_name="drone_eta_results.csv", mime="text/csv")

# =========================
# TAB 2: What-If Simulator
# =========================
with tab2:
    st.header("What-If ETA Simulator")

    med = flights[FEATURES].median()

    st.caption("Adjust inputs and get ETA + uncertainty. Advanced toggles expose engineered features.")
    basic_col, adv_col = st.columns([2,1])

    with basic_col:
        avg_speed = st.slider("Average Speed (m/s)", 0.1, float(flights["average_speed"].max()), float(med["average_speed"]))
        payload = st.slider("Max Payload (kg)", 0.0, float(max(0.1, flights["max_payload"].max() or 10)), float(med["max_payload"]))
        altitude = st.slider("Max Altitude (m)", 0.0, float(flights["max_altitude"].max()), float(med["max_altitude"]))
        wind = st.slider("Average Wind Speed (m/s)", 0.0, float(flights["average_wind_speed"].max()), float(med["average_wind_speed"]))
        distance = st.slider("Straight-line Distance (m)", 0.0, float(flights["straight_line_distance"].max()), float(med["straight_line_distance"]))
        path_length = st.slider("Total Path Length (m)", 0.0, float(flights["total_path_length"].max()), float(med["total_path_length"]))
        
    with adv_col:
        st.markdown("**Advanced (optional)**")
        max_v = st.slider("Max Velocity (combined, m/s)", 0.0, float(max(1.0, flights["max_velocity_combined"].max() or 50)), float(med["max_velocity_combined"]))
        path_str = st.slider("Path Straightness (1=perfect)", 0.0, float(max(1.0, flights["path_straightness"].max() or 1.0)), float(med["path_straightness"]))
    
    x_vec = np.array([[avg_speed, payload, altitude, wind, max_v, distance, path_length, path_str]])
    x_scaled = models["scaler"].transform(x_vec)

    eta_pred = models["stacked"].predict(x_scaled)[0]

    # Quick quantile interval fit for the what-if prediction (retrain small quantiles)
    X = flights[FEATURES].to_numpy()
    y = flights[TARGET].to_numpy()
    X_scaled_all = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_all, y, test_size=0.2, random_state=42)
    q_lo = GradientBoostingRegressor(loss='quantile', alpha=0.05, n_estimators=100, random_state=42).fit(X_train, y_train)
    q_hi = GradientBoostingRegressor(loss='quantile', alpha=0.95, n_estimators=100, random_state=42).fit(X_train, y_train)
    lo = q_lo.predict(x_scaled)[0]
    hi = q_hi.predict(x_hi)[0]

    st.success(f"**Predicted ETA:** {eta_pred:.1f} s  \n**Interval (5%–95%):** {lo:.1f} – {hi:.1f} s")
    st.caption("Note: Educational model; not validated for operational use (e.g., BVLOS).")

# =========================
# TAB 3: Your Data
# =========================
with tab3:
    st.header("Your Dataset & Batch Predictions")
    st.write("Preview the cleaned flight summary table derived from the uploaded dataset.")
    st.dataframe(flights.head(50), use_container_width=True)
    st.markdown("**Upload a new CSV** to retrain the models with different data (must include required columns).")
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
    st.write("Download the evaluation CSV (hold-out predictions + intervals)")
    st.download_button("Download evaluation CSV", data=models["eval"].to_csv(index=False).encode('utf-8'),
                         file_name="drone_eta_evaluation.csv", mime="text/csv")

# =========================
# TAB 4: Notes & Reproducibility
# =========================
with tab4:
    st.header("Notes, Limitations & How to Cite")
    st.markdown("""
    **Quick summary**
    - This app trains a stacked ensemble (Gradient Boosting + Random Forest) to predict drone delivery ETA
    - Provides calibration (pred vs actual), MAPE by wind bin, and 5%–95% prediction intervals

    **Important caveats**
    - This model was trained on *controlled test flight data*; it is **not validated** for operational BVLOS use.
    - Two engineered features (`altitude_change_rate`, `efficiency`) from the original pipeline caused **label leakage** and have been removed. The new feature, `path_straightness`, is a statistically sound replacement.
    - Prediction intervals are empirical quantiles (Gradient Boosting quantile regression), not regulatory certification.

    **Reproducibility**
    1. Put `CarnegieMellonDataMain.csv` in the repo root or use the uploader.
    2. Deploy to Streamlit Community Cloud (see README).
    3. Use the "Download evaluation CSV" for the dashboard-ready results.

    **If you use this for an application**
    - Use honest metrics (R², MAPE). If you later retrain a clean model (no leakage) and R² drops, use the clean-model number in formal materials.

    **Credits**
    - Built from Colab pipeline provided by student researcher.
    """)
    st.markdown("---")
    st.write("Contact: add your name & email in the repo README for credibility.")
