# To deploy this app on a platform like Streamlit Cloud,
# create a file named 'requirements.txt' with the following content:
# streamlit
# pandas
# scikit-learn
# numpy
# matplotlib
# seaborn

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Streamlit app title and description
st.set_page_config(layout="wide", page_title="Drone ETA Dashboard")
st.title("Drone Estimated Time of Arrival (ETA) Dashboard �")
st.markdown("""
This dashboard provides a comprehensive view of the stacked ensemble model's performance for predicting drone flight times.
Explore the interactive prediction tool and review key performance metrics and visualizations below.
""")

# IMPORTANT: This path is for a local file.
# Replace with the path to your 'CarnegieMellonDataMain.csv' file.
FILE_PATH = "CarnegieMellonDataMain.csv"

# === 1. Data Loading and Feature Engineering (Cached for Performance) ===
@st.cache_data
def load_and_preprocess_data():
    """
    Loads data and performs all necessary preprocessing and feature engineering.
    This function is cached to prevent re-running on every app interaction.
    """
    with st.spinner('Loading and preprocessing data...'):
        try:
            df = pd.read_csv(FILE_PATH)
            df.columns = df.columns.str.strip()
        except FileNotFoundError:
            st.error(f"Error: The file '{FILE_PATH}' was not found. Please upload the file or change the file path.")
            return None, None

        # Convert necessary columns to numeric types
        for col in ['time', 'altitude', 'velocity_x', 'velocity_y', 'velocity_z']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['time', 'altitude', 'velocity_x', 'velocity_y', 'velocity_z'])

        # Group by 'flight' to create aggregated features
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

        # Calculate total flight distance
        flight_summary['total_distance'] = np.sqrt(
            (flight_summary['end_pos_x'] - flight_summary['start_pos_x'])**2 +
            (flight_summary['end_pos_y'] - flight_summary['start_pos_y'])**2 +
            (flight_summary['end_pos_z'] - flight_summary['start_pos_z'])**2
        )

        # Extra feature engineering
        flight_summary['altitude_change_rate'] = flight_summary['max_altitude'] / flight_summary['total_time']
        flight_summary['efficiency'] = flight_summary['total_distance'] / flight_summary['total_time']

        flight_summary = flight_summary.dropna()
        flight_summary = flight_summary[flight_summary['total_time'] > 0]

        features = [
            'average_speed', 'max_payload', 'max_altitude', 'average_wind_speed',
            'max_velocity_combined', 'total_distance', 'altitude_change_rate', 'efficiency'
        ]
        target = 'total_time'

        X = flight_summary[features]
        y = flight_summary[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test, X.columns, scaler, flight_summary

# === 2. Model Training (Cached for Performance) ===
@st.cache_resource
def train_model(X_train, y_train):
    """
    Trains and returns the stacked ensemble model.
    This function is cached and will only run once.
    """
    with st.spinner('Training the stacked ensemble model...'):
        # Model tuning for Gradient Boosting Regressor
        gbr_tuned = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [300, 500],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
        grid_search = GridSearchCV(gbr_tuned, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_gbr = grid_search.best_estimator_

        # Use a Random Forest Regressor as a base estimator
        rf = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42)

        # Create a Stacking Regressor
        stacked = StackingRegressor(
            estimators=[('gbr', best_gbr), ('rf', rf)],
            final_estimator=GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
        )
        stacked.fit(X_train, y_train)

        return stacked

# Load data and train the model
X_train, X_test, y_train, y_test, feature_names, scaler, flight_summary = load_and_preprocess_data()

if X_train is not None:
    stacked_model = train_model(X_train, y_train)
    stacked_preds = stacked_model.predict(X_test)
    r2 = r2_score(y_test, stacked_preds)
    st.success(f"Model Training Complete! The R² score is: {r2:.4f}")

    # === 3. Dashboard Layout ===
    st.header("Interactive ETA Prediction")
    st.write("Adjust the sliders on the left to see a real-time prediction based on your custom input.")
    
    # Create a sidebar for user input
    with st.sidebar:
        st.header("Input Features")
        input_data = {}
        for feature in feature_names:
            min_val = flight_summary[feature].min()
            max_val = flight_summary[feature].max()
            mean_val = flight_summary[feature].mean()
            
            # Use a slider for user input
            input_data[feature] = st.slider(
                f"**{feature.replace('_', ' ').title()}**",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                step=(max_val - min_val) / 100
            )

    # Convert input to a DataFrame and scale it
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Predict the ETA
    predicted_time = stacked_model.predict(input_scaled)[0]
    
    st.markdown(f"### Predicted Total Flight Time: **{predicted_time:.2f} seconds**")

    st.markdown("---")
    
    # === 4. Model Performance Metrics and Plots ===
    st.header("Model Performance Metrics")
    col1, col2 = st.columns(2)
    
    # Calculate and display MAPE
    mape = mean_absolute_percentage_error(y_test, stacked_preds) * 100
    
    with col1:
        st.metric(label="R² Score", value=f"{r2:.4f}")
    with col2:
        st.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{mape:.2f}%")
        
    st.markdown("---")
    
    # === 5. Visualizations ===
    st.header("Model Performance Visualizations")
    
    # Create two columns for plots
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        # Predicted vs. Actual Plot
        st.subheader("Predicted vs. Actual Values")
        fig_pa, ax_pa = plt.subplots(figsize=(8, 6))
        ax_pa.scatter(y_test, stacked_preds, alpha=0.5)
        ax_pa.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax_pa.set_xlabel('Actual Total Time (s)')
        ax_pa.set_ylabel('Predicted Total Time (s)')
        ax_pa.set_title('Predicted vs. Actual Values (Stacked Ensemble)')
        ax_pa.grid(True)
        st.pyplot(fig_pa)
    
    with plot_col2:
        # Residual Plot
        st.subheader("Residual Plot")
        fig_res, ax_res = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=stacked_preds, y=y_test.values - stacked_preds, color="blue", alpha=0.5, ax=ax_res)
        ax_res.axhline(y=0, color='red', linestyle='--')
        ax_res.set_xlabel('Predicted Total Time (s)')
        ax_res.set_ylabel('Residuals (Actual - Predicted)')
        ax_res.set_title('Residual Plot: Stacked Ensemble Prediction Errors')
        ax_res.grid(True)
        st.pyplot(fig_res)
        
    st.markdown("---")
    
    # Prediction Intervals Plot in a full-width section
    st.subheader("Prediction Intervals")
    with st.spinner('Calculating prediction intervals...'):
        lower_model = GradientBoostingRegressor(loss='quantile', alpha=0.05, n_estimators=100, random_state=42)
        upper_model = GradientBoostingRegressor(loss='quantile', alpha=0.95, n_estimators=100, random_state=42)
        lower_model.fit(X_train, y_train)
        upper_model.fit(X_train, y_train)
        lower_preds = lower_model.predict(X_test)
        upper_preds = upper_model.predict(X_test)
    
    # Calculate error bars
    yerr_lower = np.abs(stacked_preds - lower_preds)
    yerr_upper = np.abs(upper_preds - stacked_preds)

    fig_err, ax_err = plt.subplots(figsize=(12, 7))
    ax_err.errorbar(
        range(len(stacked_preds)),
        stacked_preds,
        yerr=[yerr_lower, yerr_upper],
        fmt='o',
        ecolor='red',
        alpha=0.6,
        capsize=4
    )
    ax_err.set_xlabel("Test Flight #")
    ax_err.set_ylabel("Predicted Total Time (s)")
    ax_err.set_title("Prediction Intervals for Drone ETA")
    ax_err.grid(True)
    st.pyplot(fig_err)
    
    st.markdown("---")

    # MAPE by Wind Speed Table
    st.subheader("MAPE by Wind Speed Bin")
    df_test = pd.DataFrame({
        'predicted_time': stacked_preds,
        'actual_time': y_test.reset_index(drop=True),
        'wind_speed': flight_summary.loc[y_test.index, 'average_wind_speed'].values
    })
    df_test['wind_bin'] = pd.cut(
        df_test['wind_speed'],
        bins=[0, 5, 10, 15, 20, np.inf],
        labels=['0-5 m/s', '5-10 m/s', '10-15 m/s', '15-20 m/s', '>20 m/s']
    )
    mape_by_bin = df_test.groupby('wind_bin').apply(
        lambda g: np.mean(np.abs((g['actual_time'] - g['predicted_time']) / g['actual_time'])) * 100
    )
    mape_by_bin = mape_by_bin.fillna("No flights in this range")
    st.dataframe(mape_by_bin)
