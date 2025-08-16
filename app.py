import streamlit as st
import pandas as pd
import pickle

# load converter and model
with open("converter.pkl", "rb") as f:
    ct = pickle.load(f)

with open("rajanmodel.pkl", "rb") as f:
    model = pickle.load(f)

# load dataset for dropdown values
df = pd.read_csv("cleaned_data_car.csv")
unique_names = sorted(df["name"].unique().tolist())
unique_companies = sorted(df["company"].unique().tolist())

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Predictor")
st.write("Enter your car details and get the predicted price!")

# user inputs
name = st.selectbox("Car Name", unique_names, index=0)
company = st.selectbox("Company", unique_companies, index=0)
year = st.number_input("Year", min_value=2000, max_value=2025, value=2018)
kms_driven = st.number_input("Kms Driven", min_value=0, value=25000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])

if st.button("Predict Price"):
    # create dataframe
    new_data = pd.DataFrame(
        {
            "name": [name],
            "company": [company],
            "year": [year],
            "kms_driven": [kms_driven],
            "fuel_type": [fuel_type],
        }
    )

    # transform and predict
    new_data_transformed = ct.transform(new_data)
    predicted_price = model.predict(new_data_transformed)[0]
    predicted_price = float(predicted_price)  # convert ndarray -> float

    st.success(f"💰 Estimated Price: ₹ {predicted_price:,.2f}")
