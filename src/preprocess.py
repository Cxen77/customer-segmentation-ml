import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df=pd.read_csv(path)
    return df

def select_features(df):
    X= df[["Age","Annual Income (k$)","Spending Score (1-100)"]]
    return X

def scale_data(X):
    scaler=StandardScaler()
    scaled_x=scaler.fit_transform(X)
    return(scaled_x,scaler)




