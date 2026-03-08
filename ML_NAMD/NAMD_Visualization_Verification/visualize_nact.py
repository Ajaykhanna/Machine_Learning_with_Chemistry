import numpy as np
import matplotlib.pyplot as plt

# Load the data
nact_data = np.loadtxt('nact.out')

# Handle single row vs multiple rows
if nact_data.ndim == 1:
    # Single timestep - reshape to 2D with one row
    nact_data = nact_data.reshape(1, -1)

# First column is time, rest is flattened NACT matrix
n_timesteps = nact_data.shape[0]
n_values = nact_data.shape[1] - 1  # subtract 1 for time column
n_states = int(np.sqrt(n_values))

print(f"Number of timesteps: {n_timesteps}")
print(f"Number of states: {n_states}")
print(f"Matrix values per timestep: {n_values} ({n_states}x{n_states})")

# Extract time and NACT matrices
times = nact_data[:, 0]  # First column is time (fs)
nact_flat = nact_data[:, 1:]  # Rest is the flattened NACT matrix

# Reshape into 3D array: (n_timesteps, n_states, n_states)
nact_matrices = nact_flat.reshape(n_timesteps, n_states, n_states)

# For a single timestep, get the 2D matrix
nact_matrix = nact_matrices[0]  # First (or only) timestep

print(f"\nTime: {times[0]} fs")
print(f"Matrix shape: {nact_matrix.shape}")
print(f"Antisymmetric: {np.allclose(nact_matrix, -nact_matrix.T)}")
print(f"Diagonal is zero: {np.allclose(np.diag(nact_matrix), 0)}")

# Find strongest couplings
upper_tri = np.triu_indices(n_states, k=1)
coupling_values = np.abs(nact_matrix[upper_tri])
sorted_idx = np.argsort(coupling_values)[::-1]

print(f"\nTop 5 strongest couplings:")
for i in range(5):
    idx = sorted_idx[i]
    row, col = upper_tri[0][idx], upper_tri[1][idx]
    # States are 1-indexed in NEXMD output (S1, S2, etc.)
    print(f"  S{row+1}-S{col+1}: {nact_matrix[row, col]:.10f}")

# Visualize
plt.figure(figsize=(10, 8))
plt.imshow(nact_matrix, cmap='RdBu_r', aspect='equal')
plt.colorbar(label='NACT (a.u.)')
plt.xlabel('State j')
plt.ylabel('State i')
plt.title(f'Non-Adiabatic Coupling Matrix at t = {times[0]} fs')
plt.xticks(range(n_states), [f'S{i+1}' for i in range(n_states)], rotation=45)
plt.yticks(range(n_states), [f'S{i+1}' for i in range(n_states)])
plt.tight_layout()
plt.savefig('nact_matrix.png', dpi=150)
plt.show()


