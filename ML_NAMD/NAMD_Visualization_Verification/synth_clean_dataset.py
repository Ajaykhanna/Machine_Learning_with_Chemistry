import numpy as np
from scipy.spatial.distance import pdist
import os

# Load your training data
data_dir = "./"
R = np.load(os.path.join(data_dir, "acn_R.npy"))
Z = np.load(os.path.join(data_dir, "acn_Z.npy"))

print(f"Total configurations: {len(R)}")
print(f"Shape: {R.shape}")

# Find configurations with suspiciously short distances
MIN_PHYSICAL_DISTANCE = 0.8  # Å - anything below this is likely unphysical
bad_configs = []
all_min_distances = []

for i, config in enumerate(R):
    dists = pdist(config)
    min_dist = dists.min()
    all_min_distances.append(min_dist)
    
    if min_dist < MIN_PHYSICAL_DISTANCE:
        bad_configs.append(i)
        if len(bad_configs) <= 10:  # Print first 10
            print(f"Config {i}: minimum distance = {min_dist:.4f} Å")

all_min_distances = np.array(all_min_distances)

print(f"\n{'='*60}")
print(f"Data Quality Analysis:")
print(f"{'='*60}")
print(f"Configurations with dist < {MIN_PHYSICAL_DISTANCE} Å: {len(bad_configs)} ({100*len(bad_configs)/len(R):.2f}%)")
print(f"Overall minimum distance: {all_min_distances.min():.4f} Å")
print(f"5th percentile: {np.percentile(all_min_distances, 5):.4f} Å")
print(f"Median: {np.median(all_min_distances):.4f} Å")

# Histogram of minimum distances
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(all_min_distances, bins=100, edgecolor='black', alpha=0.7)
plt.axvline(x=MIN_PHYSICAL_DISTANCE, color='r', linestyle='--', 
            label=f'Physical limit ({MIN_PHYSICAL_DISTANCE} Å)')
plt.axvline(x=1.4, color='g', linestyle='--', 
            label='dist_soft_min (1.4 Å)')
plt.xlabel('Minimum interatomic distance (Å)')
plt.ylabel('Number of configurations')
plt.title('Distribution of Minimum Distances in Training Data')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('data_quality_check.pdf')
print(f"\nPlot saved: data_quality_check.pdf")

# If there are bad configurations, create cleaned dataset
if len(bad_configs) > 0:
    print(f"\n{'='*60}")
    print("RECOMMENDATION: Remove bad configurations!")
    print(f"{'='*60}")
    print(f"Configurations to remove: {len(bad_configs)}")
    print(f"Remaining: {len(R) - len(bad_configs)}")
    
    # Create boolean mask for good configurations
    good_mask = np.ones(len(R), dtype=bool)
    good_mask[bad_configs] = False
    
    # Save cleaned dataset
    cleaned_dir = os.path.join(data_dir, "cleaned")
    os.makedirs(cleaned_dir, exist_ok=True)
    
    # Save all arrays with bad configs removed
    for filename in os.listdir(data_dir):
        if filename.endswith('.npy'):
            data = np.load(os.path.join(data_dir, filename))
            if len(data) == len(R):  # Only filter arrays with same length
                cleaned_data = data[good_mask]
                np.save(os.path.join(cleaned_dir, filename), cleaned_data)
                print(f"Cleaned {filename}: {len(data)} → {len(cleaned_data)}")
    
    print(f"\nCleaned data saved to: {cleaned_dir}")
    print("\nUpdate your training script to use the cleaned directory!")
else:
    print(f"\n✓ All configurations pass minimum distance check!")

