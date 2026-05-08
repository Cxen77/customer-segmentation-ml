def analysis_clusters(df):
    
    summary=df.groupby("kmeans_cluster")[
            ["Age",
             "Annual Income (k$)",
             "Spending Score (1-100)"
            ]
        ].mean()
    
    print(summary)

    print("\nCluster Insights:\n")

    print("Cluster 0 -> Older moderate spenders")
    print("Cluster 1 -> Rich high spenders")
    print("Cluster 2 -> Young active spenders")
    print("Cluster 3 -> Rich low spenders")