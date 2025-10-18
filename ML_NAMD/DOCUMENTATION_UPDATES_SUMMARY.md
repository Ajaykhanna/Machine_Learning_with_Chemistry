# Documentation Updates - Input/Output Dimensions

## Summary of Changes

This document summarizes the additions made to the README regarding input/output dimensions for the HIPPYNN training script.

## Files Modified/Created

### 1. README.md (Main Documentation)
**New Section Added**: "Input/Output Dimensions and Data Shapes" (Lines 435-704)

**Content includes**:
- Network inputs specification (Z and R)
- Detailed output dimensions for each property:
  - Energy outputs (single vs multi-target modes)
  - Force outputs with autodiff computation
  - Dipole outputs with charge-position computation
  - NACR outputs with state pair enumeration
- Complete 2-state ACN example with data flow diagram
- Dataset array organization with standard naming convention
- Batch processing during training
- Multi-state slicing logic
- Memory considerations for RAM/VRAM

### 2. INPUT_OUTPUT_DIMENSIONS_GUIDE.md (New Quick Reference)
**Created**: A standalone companion guide for quick reference

**Contains**:
- Quick reference tables for all input/output shapes
- Expected file structures for different configurations
- State pair enumeration formulas and examples
- Flattening conventions (forces vs dipoles)
- Four complete configuration examples with all needed files
- Data loading and slicing explanations
- Batch processing specifications
- Memory requirements breakdown
- Troubleshooting guide
- Data format specifications with Python code examples
- Data inspection and validation code

## Key Information Added

### Input Dimensions

```
Z (Species):     (N_samples, N_atoms)           [int64 or float32]
R (Positions):   (N_samples, N_atoms, 3)        [float32, Ångströms]
```

### Output Dimensions by Property

| Property | Single-Target | Multi-Target | Notes |
|----------|---------------|--------------|-------|
| Energy | (N,) per state | (N, n_states) | Scalar per molecule |
| Force | (N, N_atoms×3) | (N, N_atoms×3) | Computed from ∇E |
| Dipole | (N, 3) per state | (N, 3) | Vector (x,y,z) |
| NACR | (N, N_atoms×3) | (N, N_atoms×3) | Per state pair |

### Example for ACN (N=10000, atoms=15, 2 states)

```
Inputs:
  Z: (10000, 15)              - Atomic numbers
  R: (10000, 15, 3)           - Atomic positions

Energy Outputs:
  S0, S1, S2: (10000,)        - 3 scalars (with ground state)

Force Outputs:
  F0, F1, F2: (10000, 45)     - Forces for 15 atoms × 3 coords

Dipole Outputs:
  D1, D2: (10000, 3)          - x,y,z components

NACR Outputs:
  NACRdE_1_2: (10000, 45)     - Coupling between S1 and S2
```

### State Pair Counting for NACR

Formula: $\binom{n}{2} = \frac{n(n-1)}{2}$

| # States | NACR Pairs |
|----------|-----------|
| 2 | 1 |
| 3 | 3 |
| 4 | 6 |
| 5 | 10 |

### Dataset File Structure

Required files (always):
- `{dataset_name}Z.npy`
- `{dataset_name}R.npy`

Conditional files (depends on `--training-targets`):
- Energy: `{dataset_name}S0.npy`, `{dataset_name}S1.npy`, etc.
- Force: `{dataset_name}F0.npy`, `{dataset_name}F1.npy`, etc.
- Dipole: `{dataset_name}Q*.npy` (charges), `{dataset_name}D*.npy` (dipoles)
- NACR: `{dataset_name}NACRdE_i_j.npy` (one per state pair)

### Batch Processing

```
Training batch (B=32, N=15 atoms):
  Input batch:  Z(32,15), R(32,15,3)
  Output batch: S0(32,), F0(32,45), D1(32,3), etc.
  Total per-batch memory: ~75 KB
```

### Memory Considerations

Full dataset (10000 samples, 15 atoms):
- Z: 1.2 MB
- R: 1.8 MB
- Targets: 5-10 MB
- **Total: ~20-30 MB** (fits easily on GPU with `--db-to-gpu`)

## Integration with Existing Documentation

The new section is well-integrated with existing documentation:

1. **Table of Contents** updated to include new section
2. **Cross-references** maintain consistency with Training Targets section
3. **Formatting** matches existing markdown style
4. **Code examples** follow Python/HIPPYNN conventions
5. **Tables** use consistent markdown table format

## Practical Use Cases

### Case 1: Validating Your Dataset
```python
# Check if your files match expected shapes
import numpy as np
Z = np.load('acn_Z.npy')
assert Z.shape == (10000, 15), "Z should be (N_samples, 15)"
```

### Case 2: Understanding Data Flow
The new "Complete Example: 2-State ACN Training" section shows:
- What data each file should contain
- How it flows through the network
- What outputs are compared to which targets
- Complete visual ASCII diagram

### Case 3: Configuring Your Training
The "Common Configuration Examples" show:
- What files you need for each training mode
- Exact command-line arguments
- Expected network outputs
- File structure required

### Case 4: Troubleshooting Errors
The troubleshooting section addresses:
- Shape mismatch errors
- Missing state files
- Missing dipole/force files
- Unexpected charge file requirements

## Key Insights

1. **Flattening Convention**: Forces and NACRs are flattened (N×3 → N*3), but dipoles are kept as 3D vectors

2. **Charges are Internal**: Charge arrays (Q*.npy) are never directly compared to targets; they're internal representations

3. **Physics Constraints**: Forces are always computed as ∇E, not predicted separately

4. **Automatic Slicing**: If dataset has more states than requested, script automatically selects the correct subset

5. **NACR Pair Enumeration**: Script correctly enumerates state pairs using triangular indexing when slicing multi-state datasets

## Files for Reference

1. **README.md** - Main comprehensive documentation (now 1658 lines)
   - Contains "Input/Output Dimensions and Data Shapes" section

2. **INPUT_OUTPUT_DIMENSIONS_GUIDE.md** - Quick reference guide (standalone)
   - For practitioners who just need quick shape/file reference
   - Includes troubleshooting and code examples

## Usage Recommendations

**For comprehensive understanding**: Read the new section in README.md with full explanations, diagrams, and context

**For quick reference**: Consult INPUT_OUTPUT_DIMENSIONS_GUIDE.md for quick shape lookups, file structures, and configuration examples

**For implementation**: Copy code examples from INPUT_OUTPUT_DIMENSIONS_GUIDE.md to validate and inspect your data before training

