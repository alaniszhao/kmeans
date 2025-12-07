from scipy.io import loadmat
import numpy as np

for name in ["2018", "3063", "5096", "6046", "8068"]:
    data = loadmat(f"{name}.mat")
    gt = data["groundTruth"]

    print(type(gt))
    print(gt.shape)

    for i in range(gt.shape[1]):
        entry = gt[0, i] 
        seg = entry["Segmentation"][0,0]
        num_clusters = len(np.unique(seg))
        
        print(f"Segmentation #{i+1}: {num_clusters} clusters")