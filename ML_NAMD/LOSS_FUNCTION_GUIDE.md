# Loss Function Reference Guide

## Quick Reference: All Loss Functions at a Glance

### Energy Loss
```
Formula: L_energy = RMSE + MAE
         = √(1/N Σ(E_pred - E_true)²) + 1/N Σ|E_pred - E_true|

MSELoss: (power=0.5) → Takes square root to get RMSE in energy units
MAELoss: (power=1) → Linear absolute error for robustness

Normalization: 1.0 (scalars, no per-component normalization)

Typical Range: 1-100 eV
Target RMSE: < 0.05 eV (ground state)
```

### Force Loss  
```
Formula: L_force = RMSE + MAE / √(3n_atoms)
         = √(1/N Σ||F_pred - F_true||²) + (1/N Σ||F_pred - F_true||₁) / √(3n_atoms)

MSELoss: (power=0.5) → RMSE for large errors
MAELoss: (power=1) → MAE normalized by vector dimensionality

Normalization: √(3n_atoms) → Scale-invariant across molecule sizes
              For n_atoms=15: factor = √45 ≈ 6.7

Physics Constraint: F = -∇E (automatic differentiation)
                   NOT independently predicted
                   
Typical Range: -5 to +5 eV/Å
Target RMSE: < 0.1 eV/Å

Key Feature: Physics-constrained (forces always gradient of energy)
             Reduces parameters, improves generalization
```

### Dipole Loss
```
Formula: L_dipole = MSEPhaseLoss + MAEPhaseLoss / √3

Phase-Aware MSE: min(||D_pred - D_true||², ||D_pred + D_true||²)
Phase-Aware MAE: min(||D_pred - D_true||₁, ||D_pred + D_true||₁)

Normalization: √3 → Accounts for 3D vector components

Typical Range: 0-10 Debye
Target RMSE: < 0.5 Debye

Special Feature: Phase ambiguity handling
                D and -D represent same physical dipole
                Network can naturally explore both orientations
                Phase-aware loss prevents false penalties
```

### NACR Loss
```
Formula: L_NACR = SMAPEPhaseLoss

SMAPE: 2/3 Σ |a_j - b_j| / (|a_j| + |b_j| + ε)  [phase-aware]

Key: Percentage error (scale-invariant)
     Ideal for small values (NACR ~ 10⁻⁴ to 10⁻³)
     Regular MSE/MAE would give huge relative errors

Normalization: 1.0 (SMAPE is already scale-invariant)

Typical Range: 10⁻⁴ to 10⁻³ Å⁻¹
Target SMAPE: < 20% (relative error)

Special Feature: Scale-insensitive to absolute magnitude
                ε prevents division by zero
                Handles phase ambiguity (V and -V)
```

---

## Loss Function Design Philosophy

### 1. Dual Loss Strategy (RMSE + MAE)

**Why combine them?**
```
RMSE: Emphasizes large errors (quadratic penalty)
      → Drives overall accuracy
      → Sensitive to outliers

MAE: Linear penalty
     → Robust to outliers
     → Smooth gradient flow

Combined: Best of both worlds
          Accuracy + Robustness
          Prevents overfitting to noisy data
```

### 2. Vector Normalization

**Why normalize by vector dimension?**
```
Forces: √(3n_atoms)
Dipoles: √3
NACRs: 1.0 (already scale-invariant)

Reason: Larger molecules have more coordinates
        Without normalization: bigger molecules → bigger loss
        Would bias training toward smaller systems
        Normalization = scale-independent training
```

### 3. Phase-Aware Loss Functions

**Why phase matters?**
```
Dipole D = (D_x, D_y, D_z)
Physical meaning: Same for D and -D (just opposite direction)

Without phase-aware loss:
  Regular MSE(D, -D) = huge error
  → Network avoids natural phase flip
  → Learns to stay in one orientation

With phase-aware loss:
  min(MSE(D, -D), MSE(-D, -D)) = MSE(0, 0) ≈ 0
  → Network free to explore both orientations
  → Better training dynamics, generalization
```

### 4. Percent-Based Loss for Small Values

**Why SMAPE for NACRs?**
```
NACR values: ~10⁻⁴ to 10⁻³ Å⁻¹  (very small!)

Absolute Error Problem:
  |0.0002 - 0.0001| = 0.0001  → "small" error
  But it's 100% relative error!

SMAPE Solution:
  0.0001 / (0.0002 + 0.0001) ≈ 33% error  → "large" relative error
  Correctly identifies significant mismatches
  Scale-invariant by design
```

---

## Multi-Target Loss Weighting

### Default Behavior (all weights = 1.0)

```
L_total = L_energy + L_dipole + L_force + L_NACR + λ_L2 * L_L2
        = 1.0 * L_E + 1.0 * L_D + 1.0 * L_F + 1.0 * L_N + 2e-5 * L_L2

Result: Equal importance to all properties
```

### Custom Weighting Strategy

```
High-priority property:    weight = 2.0
Normal priority:           weight = 1.0
Lower priority (noisy):    weight = 0.5
Minimal priority:          weight = 0.1

Example - Training on [energy, dipole, force, NACR]:
  --target-weights 1.0,1.0,1.0,0.5

Result: L_total = 1.0*L_E + 1.0*L_D + 1.0*L_F + 0.5*L_N + L_L2
        NACR has half the impact (often harder to predict)
```

---

## L2 Regularization Strategy

### Why 2e-5?

```
Too large (e.g., 0.01):
  → Network heavily penalized for any weights
  → Learns very simple (underfit) model
  → Poor training accuracy
  → Won't capture complex patterns

Too small (e.g., 1e-7):
  → Little penalty for large weights
  → Network overfits to training data
  → Poor generalization (validation worse than training)
  → Learns noise, not patterns

Goldilocks value (2e-5):
  → Empirically found to work well
  → Balances model complexity with data fit
  → Good generalization
  → Network learns meaningful patterns without memorizing
```

### How L2 Works in Training

```
Step 1: Forward pass
        y = network(x)

Step 2: Compute loss
        loss = L_targets + 2e-5 * sum(w²)

Step 3: Backward pass
        ∇loss includes both target loss + weight penalty

Step 4: Optimizer step
        w = w - lr * (∇L_targets + 2e-5 * 2w)
        
Result: Optimizer naturally prefers smaller weights
        While still fitting the data
```

---

## Practical Guidelines for Loss Selection

### Scenario 1: Energy-Only Training
```
Use: RMSE + MAE
Rationale: Standard dual-loss for accuracy + robustness
Weight: 1.0
```

### Scenario 2: Energy + Forces (Dynamics)
```
Use: Energy (RMSE+MAE) + Force (RMSE+MAE)
Weights: [1.0, 1.0]
Rationale: Forces equally important for MD accuracy
Normalization: Forces auto-scaled by √(3n_atoms)
```

### Scenario 3: Energy + Dipole + Force (Full)
```
Use: RMSE+MAE for all three
Weights: [1.0, 1.0, 1.0]
Rationale: All properties equally important
Note: Each normalized appropriately (1, √3, √(3n_atoms))
```

### Scenario 4: Multi-State with NACRs
```
Use: Energy + Dipole + Force + NACR
Weights: [1.0, 1.0, 1.0, 0.5]
Rationale: NACRs are hardest to predict (usually noisier)
         Reduce weight to prevent training collapse
SMAPE handles small NACR values automatically
```

### Scenario 5: Prioritizing Energy
```
Use: Energy + Dipole + Force + NACR
Weights: [2.0, 1.0, 0.5, 0.5]
Rationale: Energy is foundation, most critical
          Other properties secondary
          Useful if NACR data is noisy
```

---

## Debugging Loss Issues

### Problem: Energy loss stuck at high value
```
Possible causes:
  1. Normalization/scaling of energy data
  2. Network architecture too small
  3. Learning rate too high (oscillating)
  4. L2 regularization too strong
  
Solutions:
  → Check energy range in dataset
  → Increase n_features, n_interactions
  → Reduce init_learning_rate (try 5e-4)
  → Reduce L2 coefficient temporarily
```

### Problem: Force loss not improving
```
Possible causes:
  1. Energy not trained well first
  2. Force normalization issue
  3. Training time too short
  
Solutions:
  → Ensure energy RMSE < 0.1 eV first
  → Check √(3n_atoms) normalization manually
  → Increase max_epochs, reduce termination_patience
  → Verify force data quality
```

### Problem: Dipole loss saturating
```
Possible causes:
  1. Phase ambiguity not handled
  2. Data has phase flips
  3. Network confused between D and -D
  
Solutions:
  → Confirm using MSEPhaseLoss (NOT regular MSE)
  → Check if dipole data is consistently oriented
  → Increase n_sensitivities (more basis functions)
```

### Problem: NACR loss very high
```
Possible causes:
  1. NACR values too small (numerical issues)
  2. Data quality issues
  3. SMAPE epsilon too large/small
  
Solutions:
  → Use SMAPE (already handles scale)
  → Check NACR data for NaNs/infinities
  → Increase training time (NACRs converge slowly)
  → Try weight 0.5 to reduce impact
```

---

## Summary Table: Loss Function Purposes

| Property | Primary Goal | Secondary Goal | Special Feature |
|----------|---|---|---|
| **Energy** | Predict electronic energies | Robustness to outliers | Dual RMSE+MAE |
| **Force** | Predict atomic forces | Consistent with energies | Physics-constrained (autodiff) |
| **Dipole** | Predict charge distribution | Handle orientation ambiguity | Phase-aware |
| **NACR** | Predict state coupling | Handle small values | Percentage-based (SMAPE) |
