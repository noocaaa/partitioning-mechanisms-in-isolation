# evaluation.py - metrics and timing 
import time 
import numpy as np 
from sklearn.metrics import roc_auc_score, adjusted_rand_score, normalized_mutual_info_score 
 
def compute_auc(y_true, scores): 
    return roc_auc_score(y_true, scores) 
 
def compute_ari(y_true, labels): 
    return adjusted_rand_score(y_true, labels) 
 
def compute_nmi(y_true, labels): 
    return normalized_mutual_info_score(y_true, labels) 
 
def measure_time(func, *args, **kwargs): 
    start = time.time() 
    result = func(*args, **kwargs) 
    elapsed = time.time() - start 
    return result, elapsed 
