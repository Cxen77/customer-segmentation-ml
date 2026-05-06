from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
def Elbow_Method(scaled_x):
    inertia=[]
    K=range(1,9)
    for k in K:
        model=KMeans(n_clusters=k,random_state=42,n_init=10)
        model.fit(scaled_x)
        inertia.append(model.inertia_)
    plt.plot(K,inertia,marker="o")
    
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")

    plt.title("Elbow Method")

    plt.show()


def find_eps(scaled_x,k=5):
    neighbour=NearestNeighbors(n_neighbors=k)
    neighbour_fit= neighbour.fit(scaled_x)
    distance,_=neighbour_fit.kneighbors(scaled_x)
    distance=distance[:,k-1]
    distance=np.sort(distance)

    plt.plot(distance)
    plt.xlabel("point sorted ")
    plt.ylabel("eps")
    plt.title("k-distance graph")
    plt.show()