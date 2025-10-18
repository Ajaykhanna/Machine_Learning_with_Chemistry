# Input/Output Dimensions Guide for HIPPYNN Training Script

## Quick Reference

### Network Inputs (Always Required)

```
Z (Species):     (N_samples, N_atoms)        - Atomic numbers
R (Positions):   (N_samples, N_atoms, 3)     - Atomic coordinates (Å)
```

### Network Outputs (Depends on training_targets)

```
S0 (Energy GS):  (N_samples,)                - Scalar
S1, S2 (Energy): (N_samples,)                - Scalars per state

F0, F1, F2:      (N_samples, N_atoms*3)      - Forces (computed from ∇E)
                                               Forces are flattened per atom

Q1, Q2:          (N_samples, N_atoms)        - Charges (internal for dipoles)
                                               Not compared directly to targets

D1, D2:          (N_samples, 3)              - Dipole moments (x,y,z)
                                               Computed as: D = Σ(q_i * R_i)

NACRdE_i_j:      (N_samples, N_atoms*3)      - Non-adiabatic couplings
                                               Flattened like forces
```

## Expected File Structure

### Minimal Dataset (Energy Only)
```
dataset_location/
├── acn_Z.npy        # (N, 15)
├── acn_R.npy        # (N, 15, 3)
├── acn_S0.npy       # (N,)
└── acn_S1.npy       # (N,)
```

### Full Dataset (Energy + Dipole + Force + NACR)
```
dataset_location/
├── acn_Z.npy           # (N, 15)           [REQUIRED]
├── acn_R.npy           # (N, 15, 3)        [REQUIRED]
├── acn_S0.npy          # (N,)              [energy]
├── acn_S1.npy          # (N,)              [energy]
├── acn_S2.npy          # (N,)              [energy]
├── acn_F0.npy          # (N, 45)           [force]
├── acn_F1.npy          # (N, 45)           [force]
├── acn_F2.npy          # (N, 45)           [force]
├── acn_Q1.npy          # (N, 15)           [internal for dipole]
├── acn_Q2.npy          # (N, 15)           [internal for dipole]
├── acn_D1.npy          # (N, 3)            [dipole]
├── acn_D2.npy          # (N, 3)            [dipole]
└── acn_NACRdE_1_2.npy  # (N, 45)           [nacr]
```

## State Pair Enumeration

For N states, NACR creates pairs for all state combinations:

| N States | # NACR Pairs | Pairs |
|----------|-------------|-------|
| 2 | 1 | (1,2) |
| 3 | 3 | (1,2), (1,3), (2,3) |
| 4 | 6 | (1,2), (1,3), (1,4), (2,3), (2,4), (3,4) |
| 5 | 10 | All unique pairs |

Formula: $\binom{n}{2} = \frac{n(n-1)}{2}$

## Flattening Convention

### Forces and NACRs
```
Per atom:    (x, y, z)  →  3 values
N atoms:     (x1,y1,z1, x2,y2,z2, ..., xN,yN,zN)  →  N*3 values

Example (15 atoms = 45 values):
index:  0   1   2   3   4   5  ...  42  43  44
atom:   1   1   1   2   2   2  ...  15  15  15
coord:  x   y   z   x   y   z  ...  x   y   z
```

### Dipoles
```
Single vector:  (Dx, Dy, Dz)  →  3 values (NOT flattened further)
```

## Common Configuration Examples

### Example 1: Simple Energy Training (2 States)
```python
python training.py \
    --n-states 2 \
    --n-atoms 15 \
    --training-targets energy \
    --dataset-name acn_
```

**Files needed**:
- acn_Z.npy: (N, 15)
- acn_R.npy: (N, 15, 3)
- acn_S0.npy: (N,)
- acn_S1.npy: (N,)
- acn_S2.npy: (N,)

**Network outputs**: 3 energy scalars

---

### Example 2: Energy + Dipole (2 States)
```python
python training.py \
    --n-states 2 \
    --n-atoms 15 \
    --training-targets energy,dipole \
    --dataset-name acn_
```

**Additional files needed**:
- acn_Q1.npy: (N, 15) [internal charges for S1]
- acn_Q2.npy: (N, 15) [internal charges for S2]
- acn_D1.npy: (N, 3) [dipole moment S1]
- acn_D2.npy: (N, 3) [dipole moment S2]

**Network outputs**:
- 3 energies: (N,)
- 2 dipoles: (N, 3) each

---

### Example 3: Full Training (2 States, All Properties)
```python
python training.py \
    --n-states 2 \
    --n-atoms 15 \
    --training-targets energy,dipole,force,nacr \
    --dataset-name acn_
```

**All files needed** (see Full Dataset above)

**Network outputs**:
- 3 energies: (N,) each
- 2 dipoles: (N, 3) each
- 2 forces: (N, 45) each
- 1 NACR vector: (N, 45)

---

### Example 4: 3-State Training (Multi-Target Mode)
```python
python training.py \
    --n-states 3 \
    --n-atoms 15 \
    --training-targets energy,dipole,nacr \
    --multi-targets \
    --dataset-name acn_
```

**Required files**:
- acn_Z.npy: (N, 15)
- acn_R.npy: (N, 15, 3)
- acn_S0.npy: (N,)
- acn_S1.npy: (N,)
- acn_S2.npy: (N,)
- acn_S3.npy: (N,)
- acn_Q1.npy: (N, 15)
- acn_Q2.npy: (N, 15)
- acn_Q3.npy: (N, 15)
- acn_D1.npy: (N, 3)
- acn_D2.npy: (N, 3)
- acn_D3.npy: (N, 3)
- acn_NACRdE_1_2.npy: (N, 45)
- acn_NACRdE_1_3.npy: (N, 45)
- acn_NACRdE_2_3.npy: (N, 45)

**Network outputs** (with multi-target):
- 1 energy output: (N, 4) - predicts all 4 states at once
- 1 dipole output: (N, 3) - shared charges
- 3 NACR outputs: (N, 45) each

---

## Data Loading and Slicing

### Automatic Slicing

If your dataset has more states than requested:

```python
# Dataset has 10 states, you request 2
python training.py --n-states 2

# Script automatically slices:
# acn_S0.npy (10,) → (10,)    [keep state 0]
# acn_S1.npy (10,) → (10,)    [keep state 1]
# acn_S2.npy (10,) → (10,)    [keep state 2]
# acn_S*.npy (3-9) are ignored
```

### Multi-Target Slicing

For NACR pairs with `multi_targets=True`:

```python
# Dataset has pairs for 5 states (10 NACR files)
# You request 3 states (need 3 NACR files)

Original indices: (0,1), (0,2), (0,3), (0,4), 
                 (1,2), (1,3), (1,4),
                 (2,3), (2,4),
                 (3,4)

Requested indices: (0,1), (0,2), (1,2)

Script extracts: positions 0, 1, 4 from original array
```

---

## Batch Processing

### During Training

```
Batch size: B=32 (default init_batch_size)

Per-batch shapes:
  Z: (32, 15)
  R: (32, 15, 3)
  S0: (32,)
  S1: (32,)
  S2: (32,)
  F0: (32, 45)
  D1: (32, 3)
  etc.
```

### Validation/Testing

```
Eval batch size: 512 (default max_batch_size)

Same shapes as training, just different B value
```

---

## Memory Requirements

### Per-Batch Memory (B=32, N=15 atoms)

| Component | Size |
|-----------|------|
| Z batch | 3.8 KB |
| R batch | 57.6 KB |
| Energy targets | 384 B |
| Force targets | 5.8 KB |
| Dipole targets | 384 B |
| Charges (internal) | 1.9 KB |
| NACR targets | 5.8 KB |
| **Total** | **~75 KB** |

### Full Dataset (N=10000 samples, N=15 atoms)

| Component | Size |
|-----------|------|
| Z | 1.2 MB |
| R | 1.8 MB |
| All targets | 5-10 MB |
| **Total** | **~20-30 MB** |

Use `--db-to-gpu` for full dataset on GPU (easily fits on modern GPUs with >1 GB VRAM)

---

## Troubleshooting

### Error: "shape mismatch"
- Check that N_atoms in `--n-atoms` matches array dimensions
- Arrays should have shape (N_samples, N_atoms, 3) for positions
- Example: `--n-atoms 15` expects R.shape = (N, 15, 3)

### Error: "number of states included in training is larger than dataset"
- Dataset doesn't have enough states for `--n-states`
- If `--n-states 3`, need acn_S0.npy, acn_S1.npy, acn_S2.npy, acn_S3.npy

### Missing dipole or force files
- These are computed internally from energies if not provided
- But script still expects them if in `--training-targets`
- Provide all required .npy files matching training_targets

### Charges (Q*.npy) not needed?
- Charges are never directly compared to targets
- They're internal representations used to compute dipoles
- Must exist in dataset but are only used internally

---

## Data Format Specifications

### NumPy Array Format (.npy)

```python
import numpy as np

# Correct format
Z = np.array([[1, 6, 7, 8, ...],   # molecule 1 atoms
              [1, 6, 7, 8, ...],   # molecule 2 atoms
              ...])
Z = Z.astype(np.int64)  # or np.float32 is also OK
Z.dump('acn_Z.npy')

# Positions
R = np.array([[[x1,y1,z1], [x2,y2,z2], ...],  # molecule 1 (Ångströms)
              [[x1,y1,z1], [x2,y2,z2], ...],  # molecule 2
              ...])
R = R.astype(np.float32)
R.dump('acn_R.npy')

# Energies
S0 = np.array([E1, E2, E3, ...])  # scalar per molecule
S0 = S0.astype(np.float32)
S0.dump('acn_S0.npy')

# Forces
F0 = np.array([[fx1,fy1,fz1, fx2,fy2,fz2, ...],  # molecule 1 (N*3 values)
               [fx1,fy1,fz1, fx2,fy2,fz2, ...],  # molecule 2
               ...])
F0 = F0.astype(np.float32)
F0.dump('acn_F0.npy')

# Dipoles
D1 = np.array([[Dx1, Dy1, Dz1],  # molecule 1 dipole
               [Dx2, Dy2, Dz2],  # molecule 2 dipole
               ...])
D1 = D1.astype(np.float32)
D1.dump('acn_D1.npy')
```

---

## Loading and Inspecting Your Data

```python
import numpy as np

# Check dimensions
Z = np.load('acn_Z.npy')
R = np.load('acn_R.npy')
S0 = np.load('acn_S0.npy')

print(f"Z shape: {Z.shape}")          # Should be (N, 15)
print(f"R shape: {R.shape}")          # Should be (N, 15, 3)
print(f"S0 shape: {S0.shape}")        # Should be (N,)

print(f"Z dtype: {Z.dtype}")          # int64 or float32
print(f"R dtype: {R.dtype}")          # float32
print(f"S0 dtype: {S0.dtype}")        # float32

print(f"Z range: {Z.min()} to {Z.max()}")  # Atomic numbers
print(f"R range: {R.min():.2f} to {R.max():.2f}")  # Distances in Å
print(f"S0 range: {S0.min():.4f} to {S0.max():.4f}")  # Energies
```

