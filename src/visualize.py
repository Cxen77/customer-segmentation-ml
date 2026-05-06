from matplotlib import pyplot as plt 
def plot_cluster(df,clusters):
    for k in df[clusters].unique():
        clustered_data=df[df[clusters]==k]
        plt.scatter(clustered_data["Annual Income (k$)"],clustered_data["Spending Score (1-100)"],label=f"clusters{k}")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100")
    plt.title(f"{clusters} Visualization")
    plt.legend()

    plt.show()
