# partitions.py - core partition wrapper 
from ikpykit.kernel import IsoKernel 
 
def get_kernel(X, method='anne', n_estimators=200, random_state=42): 
    Compute IsoKernel similarity matrix using specified partition method. 
    method: 'anne' (Voronoi), 'inne' (Hypersphere), 'iforest' (Hyperplane) 
    ik = IsoKernel(method=method, n_estimators=n_estimators, random_state=random_state) 
    ik.fit(X) 
    return ik.similarity(X) 
