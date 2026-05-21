# preprocessing.py - data loading and normalization 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler 
from scipy.io import loadmat 
 
def load_anomaly_dataset(filepath): 
    data = loadmat(filepath) 
    X = data['X'].astype(float) 
    y = data['y'].ravel().astype(int) 
    return X, y 
 
def load_clustering_dataset(filepath, label_col=-1): 
    df = pd.read_csv(filepath, header=None) 
    X = df.iloc[:, :label_col].values.astype(float) 
    y = df.iloc[:, label_col].values 
    return X, y 
 
def normalize(X): 
    return MinMaxScaler().fit_transform(X) 
