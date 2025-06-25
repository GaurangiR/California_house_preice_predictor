import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load model, scaler, and feature column list
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.title("🏡 California House Price Predictor")
st.markdown("Enter the details of the house below to estimate its **Median House Value**.")

# User inputs
longitude = st.number_input("📍 Longitude", value=-120.0)
latitude = st.number_input("📍 Latitude", value=37.0)
housing_median_age = st.number_input("🏠 Housing Median Age", min_value=1.0, value=25.0)
total_rooms = st.number_input("🚪 Total Rooms", min_value=1.0, value=1000.0)
total_bedrooms = st.number_input("🛏 Total Bedrooms", min_value=1.0, value=300.0)
population = st.number_input("👥 Population", min_value=1.0, value=1000.0)
households = st.number_input("🏘 Households", min_value=1.0, value=400.0)
median_income = st.number_input("💰 Median Income", min_value=0.0, value=3.0)

ocean_proximity = st.selectbox(
    "🌊 Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

# Log-transform skewed inputs
log_total_rooms = np.log(total_rooms)
log_total_bedrooms = np.log(total_bedrooms)
log_population = np.log(population)
log_households = np.log(households)

# Feature engineering
bedroom_ratio = log_total_bedrooms / log_total_rooms
household_rooms = log_total_rooms / log_households

# One-hot encode ocean proximity
ocean_features = {
    '<1H OCEAN': 0,
    'INLAND': 0,
    'ISLAND': 0,
    'NEAR BAY': 0,
    'NEAR OCEAN': 0
}
if ocean_proximity in ocean_features:
    ocean_features[ocean_proximity] = 1

# Create input DataFrame
input_dict = {
    'longitude': longitude,
    'latitude': latitude,
    'housing_median_age': housing_median_age,
    'total_rooms': log_total_rooms,
    'total_bedrooms': log_total_bedrooms,
    'population': log_population,
    'households': log_households,
    'median_income': median_income,
    'bedroom_ratio': bedroom_ratio,
    'household_rooms': household_rooms,
    **ocean_features
}

input_df = pd.DataFrame([input_dict])

# Ensure all feature columns are present
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing dummy column

# Reorder to match training data
input_df = input_df[feature_columns]

# Scale features
input_scaled = scaler.transform(input_df)

# Predict
if st.button("🔮 Predict Median House Value"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"🏠 Estimated Median House Value: ${prediction:,.2f}")
