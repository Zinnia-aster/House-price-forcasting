import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model
model = joblib.load("model.pkl")

st.title("🏠 California House Value Predictor")

# User inputs
MedInc = st.number_input("Median Income", min_value=0.0, format="%.2f")
HouseAge = st.number_input("House Age", min_value=0.0, format="%.2f")
AveRooms = st.number_input("Average Rooms", min_value=0.0, format="%.2f")
AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, format="%.2f")
Population = st.number_input("Population", min_value=0.0, format="%.2f")
AveOccup = st.number_input("Average Occupants", min_value=0.1, format="%.2f")
Latitude = st.number_input("Latitude", format="%.4f")
Longitude = st.number_input("Longitude", format="%.4f")

if st.button("Predict"):
    # Feature engineering
    Rooms_per_household = AveRooms / AveOccup
    Population_per_household = Population / AveOccup

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude,
        "Rooms_per_household": Rooms_per_household,
        "Population_per_household": Population_per_household
    }])

    # Prediction
    prediction = model.predict(input_data)[0]
    st.success(f"🏡 Predicted Median House Value: **${prediction:,.2f}**")
