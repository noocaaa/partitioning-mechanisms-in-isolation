# run_anomaly.py 
import sys, os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
import pandas as pd 
from src.partitions import get_kernel 
from src.preprocessing import load_anomaly_dataset, normalize 
from src.evaluation import compute_auc, measure_time 
 
DATASETS = ['thyroid', 'shuttle', 'breastw', 'ionosphere', 'arrhythmia', 
            'satellite', 'annthyroid', 'lympho', 'musk', 'kddcup99'] 
PARTITIONS = ['anne', 'inne', 'iforest'] 
RESULTS = [] 
 
for dataset in DATASETS: 
    X, y = load_anomaly_dataset(f'data/anomaly_detection/{dataset}.mat') 
    X = normalize(X) 
    for method in PARTITIONS: 
        K, t = measure_time(get_kernel, X, method=method) 
        auc = compute_auc(y, K.diagonal()) 
        RESULTS.append({'dataset': dataset, 'partition': method, 'auc': auc, 'time_s': t}) 
