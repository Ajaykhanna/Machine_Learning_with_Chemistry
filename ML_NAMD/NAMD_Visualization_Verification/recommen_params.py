# Add this before hyperparameter optimization:
import numpy as np
from scipy.spatial.distance import pdist

R = np.load("acn_R.npy")
Z = np.load("acn_Z.npy")

# Compute all pairwise distances
all_distances = []
for config in R:
    dists = pdist(config)
    all_distances.extend(dists)

all_distances = np.array(all_distances)

print(f"Data distance statistics:")
print(f"  Min: {all_distances.min():.3f} Å")
print(f"  5th percentile: {np.percentile(all_distances, 5):.3f} Å")
print(f"  Median: {np.median(all_distances):.3f} Å")
print(f"  95th percentile: {np.percentile(all_distances, 95):.3f} Å")
print(f"  Max: {all_distances.max():.3f} Å")

# Set hyperparameter bounds based on data:
recommended_dist_soft_min = all_distances.min() * 0.9  # 10% safety margin
recommended_dist_soft_max = np.percentile(all_distances, 75)
recommended_dist_hard_max = all_distances.max() * 1.1

print(f"\nRecommended hyperparameter bounds:")
print(f"  dist_soft_min: {recommended_dist_soft_min:.9f} Å")
print(f"  dist_soft_max: {recommended_dist_soft_max:.9f} Å")
print(f"  dist_hard_max: {recommended_dist_hard_max:.9f} Å")


