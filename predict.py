import joblib
import pandas as pd

kmeans_model=joblib.load("model/kmeans_model.pkl")
scaler=joblib.load("model/scaler.pkl")

a=int(input("Enter Age:"))
if a < 0 or a > 100:
    print("Invalid Age")
    exit()

b=int(input("Enter Income:"))
if b < 0:
    print("Invalid Income")
    exit()
    
c=int(input("Enter Spending Score:"))
if c < 0 or c > 100:
    print("Invalid Spending Score")
    exit()

new_customer=pd.DataFrame([[a, b, c]],columns=[
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"])


new_customer_scaled=scaler.transform(new_customer)
predicted_cluster=kmeans_model.predict(new_customer_scaled)

cluster_descriptions = {

    0: "Older Moderate Spenders",

    1: "Rich High Spenders",

    2: "Young Active Spenders",

    3: "Rich Low Spenders"
}

print(
    "Predicted Cluster:",
    predicted_cluster[0]
)

cluster_id = predicted_cluster[0]

print(
    "Predicted Cluster:",
    cluster_id
)

print(
    "Customer Type:",
    cluster_descriptions[cluster_id]
)
