import streamlit as st
import pandas as pd
import pickle

# Load model + columns
model = pickle.load(open("model.pkl", "rb"))
model_columns = pickle.load(open("columns.pkl", "rb"))

# Load dataset to get valid year-month values
df = pd.read_csv("area_prices_monthly.csv")

st.title("🏡 Dubai Real Estate Price Prediction")
st.write("Fill property details below:")

# ---------------- YEAR-MONTH (AUTO FROM DATASET) ----------------
year_month = st.selectbox(
    "Year-Month",
    sorted(df["year_month"].unique(), reverse=True),
    index=0
)

# ---------------- COMMUNITY / ZONE ----------------
community = st.text_input("Community (e.g. Palm Jumeirah)")
zone = st.text_input("Zone (e.g. Palm Jumeirah)")

# ---------------- FREEHOLD ----------------
is_freehold = st.selectbox("Freehold", ["True", "False"])
is_freehold = True if is_freehold == "True" else False

# ---------------- NUMERIC INPUTS ----------------
offplan_price = st.number_input("Offplan Price per sqft", min_value=0.0)
rental_price = st.number_input("Rental Price per sqft", min_value=0.0)

n_sec = st.number_input("Secondary Listings", min_value=0)
n_off = st.number_input("Offplan Listings", min_value=0)
n_rent = st.number_input("Rental Listings", min_value=0)

cbuae = st.number_input("CBUAE Base Rate %", min_value=0.0)
mortgage = st.number_input("Mortgage Rate %", min_value=0.0)

# ---------------- PREDICT ----------------
if st.button("Predict Price"):

    input_df = pd.DataFrame([{
        "year_month": year_month,
        "community": community,
        "zone": zone,
        "is_freehold": is_freehold,
        "offplan_price_per_sqft_usd": offplan_price,
        "rental_price_per_sqft_annual_usd": rental_price,
        "n_listings_secondary": n_sec,
        "n_listings_offplan": n_off,
        "n_listings_rental": n_rent,
        "cbuae_base_rate_pct": cbuae,
        "avg_mortgage_rate_pct": mortgage
    }])

    # one-hot encoding (same as training)
    input_df = pd.get_dummies(input_df)

    # align columns with training
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # prediction
    prediction = model.predict(input_df)

    st.success(f"🏡 Predicted Price per Sq Ft: ${prediction[0]:.2f}")