import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Title
st.title("🏡 Dubai Real Estate Price Prediction")

st.write("Enter property details below:")

# Inputs
area = st.number_input("Area Size", min_value=0.0)
beds = st.number_input("Bedrooms", min_value=0)
baths = st.number_input("Bathrooms", min_value=0)

# Prediction
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "area_size": [area],
        "beds": [beds],
        "baths": [baths]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Price per Sq Ft: ${prediction[0]:.2f}")