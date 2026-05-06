from sklearn.cluster import KMeans, DBSCAN

def run_kmeans(scaled_x,n_clusters=4):
    model=KMeans(n_clusters=n_clusters,random_state=42,n_init=10)
    clusters=model.fit_predict(scaled_x)
    return clusters,model

def run_dbscan(scaled_x,eps=0.65,min_samples=5):
    model=DBSCAN(eps=eps, min_samples=min_samples)
    clusters=model.fit_predict(scaled_x)
    return clusters,model

