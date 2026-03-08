import numpy as np
R = np.load("acn_R.npy")  # Positions
Z = np.load("acn_Z.npy")  # Atomic numbers

# Find all H-H distances
min_HH_distances = []
for config in R:
    H_positions = config[Z[0] == 1]  # Hydrogen atoms (Z=1)
    if len(H_positions) > 1:
        from scipy.spatial.distance import pdist
        dists = pdist(H_positions)
        min_HH_distances.append(dists.min())

print(f"Minimum H-H distance in training data: {np.min(min_HH_distances):.3f} Å")
print(f"5th percentile: {np.percentile(min_HH_distances, 5):.3f} Å")
