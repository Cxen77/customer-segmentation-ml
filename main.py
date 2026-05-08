from src.preprocess import load_data, select_features, scale_data
from src.clustering import run_kmeans, run_dbscan
from src.visualize import plot_cluster
from src.tuning import Elbow_Method,find_eps
from src.analysis import analysis_clusters
import joblib
# load data
df = load_data("data/Mall_Customers.csv")

# select features
X = select_features(df)

# scale
scaled_x, scaler = scale_data(X)
#elbow_method
Elbow_Method(scaled_x)
find_eps(scaled_x)
# clustering
kmeans_labels, kmeans_model = run_kmeans(scaled_x)
dbscan_labels, _ = run_dbscan(scaled_x)


# add results to df
df["kmeans_cluster"] = kmeans_labels
df["dbscan_cluster"] = dbscan_labels

print(df.head())
analysis_clusters(df)
# visulization 
plot_cluster(df, "kmeans_cluster")
plot_cluster(df, "dbscan_cluster")

import pandas as pd
new_customer =pd.DataFrame([[28, 85, 82]],columns=[
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
)
new_customer_scaled=scaler.transform(new_customer)
predicted_cluster = kmeans_model.predict(new_customer_scaled)
print("predicted cluster:",predicted_cluster[0])


joblib.dump(
    kmeans_model,"model/kmeans_model.pkl"
)

joblib.dump(
    scaler,"model/scaler.pkl"
)

print("Model saved successfully")