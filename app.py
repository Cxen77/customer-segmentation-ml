import streamlit as st
import pandas as pd
import joblib


# load model and scaler
kmeans_model = joblib.load(
    "model/kmeans_model.pkl"
)

scaler = joblib.load(
    "model/scaler.pkl"
)


# app title
st.title("Customer Segmentation App")

st.write("Predict Customer Type")


# user inputs
age = st.number_input(
    "Enter Age",
    min_value=1,
    max_value=100
)

income = st.number_input(
    "Enter Annual Income"
)

spending = st.number_input(
    "Enter Spending Score",
    min_value=1,
    max_value=100
)


# prediction button
if st.button("Predict Cluster"):

    new_customer = pd.DataFrame(
        [[age, income, spending]],
        columns=[
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    # scale
    new_customer_scaled = scaler.transform(
        new_customer
    )

    # predict
    predicted_cluster = kmeans_model.predict(
        new_customer_scaled
    )

    cluster_id = predicted_cluster[0]

    cluster_descriptions = {

        0: "Older Moderate Spenders",

        1: "Rich High Spenders",

        2: "Young Active Spenders",

        3: "Rich Low Spenders"
    }

    st.success(
        f"Predicted Cluster: {cluster_id}"
    )

    st.write(
        f"Customer Type: {cluster_descriptions[cluster_id]}"
    )