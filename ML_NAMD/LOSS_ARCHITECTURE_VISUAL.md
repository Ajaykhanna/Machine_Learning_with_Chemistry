# Loss Function Architecture Diagram

## Loss Function Hierarchy

```
╔══════════════════════════════════════════════════════════════════════╗
║                        TOTAL TRAINING LOSS                          ║
║  L_total = λ_E·L_E + λ_D·L_D + λ_F·L_F + λ_N·L_N + λ_L2·L_L2      ║
║                                                                      ║
║  where λ parameters are target_weights (default: 1.0 each)         ║
║        λ_L2 = 2e-5 (L2 regularization coefficient)                 ║
╚══════════════════════════════════════════════════════════════════════╝
                                  │
                    ┌─────────────┼─────────────┬──────────┐
                    │             │             │          │
                    ▼             ▼             ▼          ▼
          ┌─────────────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐
          │  ENERGY LOSS    │ │FORCE LOSS│ │DIPOLE  │ │NACR LOSS │
          │  L_E = (RMSE)   │ │L_F=(RMSE)│ │L_D=    │ │L_N =     │
          │      + (MAE)    │ │ + (MAE)  │ │(MSEPh) │ │(SMAPEPh) │
          │                 │ │         /│ │ +(MAEPh)│ │         │
          │ Normalized: 1.0 │ │norm:√3n │ │norm:√3 │ │norm: 1.0 │
          │                 │ │  atoms  │ │        │ │          │
          └─────────────────┘ └─────────┘ └────────┘ └──────────┘
                    │             │             │          │
         ┌──────────┴──────────┐  │             │          │
         │                     │  │             │          │
         ▼                     ▼  ▼             ▼          ▼
    RMSE Energy           RMSE Force      Phase-Aware    SMAPE
    MSE→power 0.5        MSE→power 0.5    MSE→min(D,-D)  (percentage)
    MAE Energy           MAE→÷√(3n)       
                         atoms             Phase-Aware
                                          MAE→min(D,-D)
```

## Loss Function Detail Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ENERGY LOSS L_E                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  RMSE Component:                      MAE Component:               │
│  ┌──────────────────────────────┐    ┌──────────────────────────┐ │
│  │ MSELoss                      │    │ MAELoss                  │ │
│  │ ────────────────────────     │    │ ──────────────────────   │ │
│  │ Input: Predicted energies    │    │ Input: Predicted energy │ │
│  │ Output: Mean squared error   │    │ Output: Mean absolute   │ │
│  │ Power applied: 0.5           │    │         error            │ │
│  │ ✓ Emphasizes large errors    │    │ Power applied: 1.0      │ │
│  │ ✓ Sensitive to outliers      │    │ ✓ Robust to outliers    │ │
│  │                              │    │ ✓ Linear gradient flow  │ │
│  └──────────────────────────────┘    └──────────────────────────┘ │
│                                                                     │
│  Per-State Tracking:                  Normalization:               │
│  S0-RMSE, S0-MAE                      1.0 (scalars, not vectors)   │
│  S1-RMSE, S1-MAE                      Energy scale preserved       │
│  S2-RMSE, S2-MAE                                                   │
│  ...                                  Typical Target: < 0.05 eV   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                      FORCE LOSS L_F                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  RMSE Component:                    MAE Component:                 │
│  ┌────────────────────────────┐    ┌──────────────────────────┐   │
│  │ GradientNode               │    │ GradientNode             │   │
│  │ ───────────────────────    │    │ ──────────────────────   │   │
│  │ Computes: F = -∇E          │    │ Computes: F = -∇E        │   │
│  │ Uses: Autograd (Physics!)  │    │ Uses: Autograd           │   │
│  │ Power: 0.5 → RMSE          │    │ Power: 1.0 → MAE         │   │
│  │                            │    │ Normalized by:           │   │
│  │ ✓ Large error penalty      │    │ √(3n_atoms)              │   │
│  │ ✓ Ensures F consistent     │    │ = √(3*15) ≈ 6.7 if n=15 │   │
│  │   with energy gradients    │    │ ✓ Scale-independent      │   │
│  └────────────────────────────┘    └──────────────────────────┘   │
│                                                                     │
│  PHYSICS CONSTRAINT:                Normalization:                 │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Forces NOT independently predicted by network            │   │
│  │ Forces computed through automatic differentiation        │   │
│  │ F = -∂E/∂R (energy gradient w.r.t. positions)            │   │
│  │                                                          │   │
│  │ Benefits:                                                │   │
│  │ • Energy-force consistency guaranteed                    │   │
│  │ • Reduces network parameters                            │   │
│  │ • Improves generalization to new geometries             │   │
│  │ • Follows fundamental molecular dynamics                │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Typical Target: < 0.1 eV/Å RMSE                                   │
│  Range: -5 to +5 eV/Å (depends on molecule/state)                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DIPOLE LOSS L_D                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MSEPhaseLoss Component:            MAEPhaseLoss Component:        │
│  ┌──────────────────────────────┐  ┌────────────────────────────┐ │
│  │ Computes both:               │  │ Computes both:             │ │
│  │ • MSE(D_pred, D_true)        │  │ • MAE(D_pred, D_true)      │ │
│  │ • MSE(D_pred, -D_true)       │  │ • MAE(D_pred, -D_true)     │ │
│  │ Takes minimum of both        │  │ Takes minimum of both      │ │
│  │ Power: 0.5 → RMSE            │  │ Power: 1.0 → MAE           │ │
│  │                              │  │ Normalized by: √3          │ │
│  │ ✓ Handles phase ambiguity    │  │ (3D vector components)     │ │
│  │ ✓ Network explores both D    │  │ ✓ Prevents false penalties │ │
│  │   and -D naturally            │  │   for direction flips       │ │
│  └──────────────────────────────┘  └────────────────────────────┘ │
│                                                                     │
│  PHASE AMBIGUITY PROBLEM:                                          │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Dipole D = (Dx, Dy, Dz) has NO preferred direction       │   │
│  │ D and -D are physically EQUIVALENT                        │   │
│  │                                                          │   │
│  │ Without Phase-Aware Loss:                                │   │
│  │   MSE(D, -D) = HUGE                                      │   │
│  │   Network learns to avoid phase flips                   │   │
│  │   Gets stuck in one orientation                         │   │
│  │   Poor generalization                                   │   │
│  │                                                          │   │
│  │ With Phase-Aware Loss:                                   │   │
│  │   min(MSE(D, -D), MSE(-D, -D)) ≈ 0                       │   │
│  │   Network free to explore both orientations             │   │
│  │   Better training dynamics                              │   │
│  │   Better generalization to test data                    │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Typical Target: < 0.5 Debye RMSE                                  │
│  Range: 0-10 Debye (varies by state)                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                       NACR LOSS L_N                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SMAPEPhaseLoss Component:                                          │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ SMAPE = Symmetric Mean Absolute Percentage Error          │   │
│  │                                                          │   │
│  │ Formulation:                                             │   │
│  │ SMAPE = (2/3) Σ |a_j - b_j| / (|a_j| + |b_j| + ε)       │   │
│  │                                                          │   │
│  │ Phase-Aware Version:                                     │   │
│  │ min(SMAPE(V, V_true), SMAPE(V, -V_true))                 │   │
│  │                                                          │   │
│  │ Power: 1.0 (percentage error, no sqrt needed)            │   │
│  │ Normalized: 1.0 (SMAPE is already scale-invariant)       │   │
│  │                                                          │   │
│  │ ✓ Handles percentage errors (0.01 vs 0.001 same ratio)   │   │
│  │ ✓ Works for tiny values (NACR ~ 10⁻⁴)                    │   │
│  │ ✓ Prevents division by zero (epsilon term)              │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  WHY SMAPE FOR NACR:                                                │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ NACR values are TINY: 10⁻⁴ to 10⁻³ Å⁻¹                     │   │
│  │                                                          │   │
│  │ Absolute Error Problem:                                  │   │
│  │   |0.0002 - 0.0001| = 0.0001   → looks small            │   │
│  │   But actually 100% relative error!                      │   │
│  │                                                          │   │
│  │ SMAPE Solution:                                          │   │
│  │   0.0001 / (0.0002 + 0.0001) ≈ 33% relative error       │   │
│  │   Correctly identifies as significant                    │   │
│  │   Scale-invariant by design                             │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Multiple NACR Pairs:                                               │
│  n_states = 2  →  1 pair: NACR(01)                                │
│  n_states = 3  →  3 pairs: NACR(01), NACR(02), NACR(12)           │
│  n_states = N  →  N(N-1)/2 pairs                                   │
│                                                                     │
│  Typical Target: < 20% SMAPE (relative error)                      │
│  Range: 10⁻⁴ to 10⁻³ Å⁻¹                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                   L2 REGULARIZATION TERM                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Formula: L_L2 = 2e-5 × Σ w²  (sum over all network weights)       │
│           Added to total loss: L_total += L_L2                    │
│                                                                     │
│  Purpose: Prevent Overfitting                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Without L2:                  │  With L2:                  │   │
│  │ • Large weights allowed       │  • Large weights penalized │   │
│  │ • Network learns all noise    │  • Simpler models preferred│   │
│  │ • High training accuracy      │  • Better generalization  │   │
│  │ • High validation error       │  • Lower validation error  │   │
│  │ • Overfits                    │  • Better real-world pred  │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Coefficient = 2e-5 Design:                                         │
│  • Empirically found to work well for molecular properties         │
│  • Balances data fit vs model complexity                           │
│  • Too large (0.01): Underfits, poor training                      │
│  • Too small (1e-7): Overfits, poor validation                     │
│  • 2e-5: Goldilocks value for molecular ML                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Loss Computation Flow During Training

```
FORWARD PASS:
  Input: Atomic species Z, Positions R
    ↓
  [Network Processing]
    ↓
  Predictions: E_pred, F_pred, D_pred, V_pred
    ↓
  Target Loading: E_true, F_true, D_true, V_true


LOSS COMPUTATION:
  
  Energy Loss:
    MSE = (E_pred - E_true)²           → power 0.5 → RMSE
    MAE = |E_pred - E_true|            → power 1.0 → MAE
    L_E = RMSE + MAE
  
  Force Loss:
    MSE = ||F_pred - F_true||²         → power 0.5 → RMSE
    MAE = ||F_pred - F_true||₁ / √(3n) → power 1.0 → normalized MAE
    L_F = RMSE + MAE
  
  Dipole Loss:
    MSEph = min(||D_pred - D_true||², ||D_pred + D_true||²)  → RMSE
    MAEph = min(||D_pred - D_true||₁, ||D_pred + D_true||₁) / √3 → MAE
    L_D = MSEph + MAEph
  
  NACR Loss:
    SMAPE_ph = min(SMAPE(V_pred, V_true), SMAPE(V_pred, -V_true))
    L_N = SMAPE_ph
  
  L2 Regularization:
    L_L2 = 2e-5 × Σ(w²)  where w = network weights


TOTAL LOSS:
  L_total = λ_E·L_E + λ_D·L_D + λ_F·L_F + λ_N·L_N + L_L2
          = 1.0·L_E + 1.0·L_D + 1.0·L_F + 1.0·L_N + L_L2  (default)


BACKWARD PASS:
  ∇L_total computed by PyTorch autograd
    ↓
  Gradients flow through each component loss
    ↓
  Network weights updated via AdamW optimizer


METRICS RECORDED:
  S0-RMSE, S0-MAE
  S1-RMSE, S1-MAE
  F-RMSE, F-MAE (or per-state F-RMSE, F-MAE)
  D-RMSE, D-MAE (or per-state)
  NACR-SMAPE (or per-pair)
  L2 (magnitude of weight penalty)
  Loss_wo_L2 (total without L2)
  Loss (total with L2)
```

## Multi-Target Loss Weighting Example

```
Training on: Energy, Dipole, Force, NACR
With weights: [1.0, 1.0, 1.0, 0.5]

L_total = 1.0 × L_E + 1.0 × L_D + 1.0 × L_F + 0.5 × L_N + L_L2

Interpretation:
┌─────────────────────────────────────────────────────────────┐
│ Energy:  Weight = 1.0  →  Full importance                   │
│ Dipole:  Weight = 1.0  →  Full importance                   │
│ Force:   Weight = 1.0  →  Full importance                   │
│ NACR:    Weight = 0.5  →  Half importance (noisier property)│
│                                                             │
│ Training focuses equally on first 3 properties              │
│ NACR weighted lower to prevent training collapse            │
│ L2 regularization applied uniformly                         │
└─────────────────────────────────────────────────────────────┘
```

## Loss Function Decision Tree

```
                    What property to train?
                              │
                 ┌────────────┼────────────┬────────────┬─────────────┐
                 │            │            │            │             │
              ENERGY       FORCE        DIPOLE        NACR      ALL COMBINED
                 │            │            │            │             │
                 ▼            ▼            ▼            ▼             ▼
            RMSE + MAE   Autodiff      Phase-Aware   SMAPE      Weighted Sum
            (norm: 1)    (norm: √3n)   (norm: √3)    (norm: 1)   + L2 Reg
                 │            │            │            │             │
            Typical      Typical       Typical       Typical       All combined
            Scale:       Scale:        Scale:        Scale:
            1-100 eV     -5 to +5      0-10          1e-4 to
                         eV/Å          Debye         1e-3 Å⁻¹
                 │            │            │            │             │
            Target:     Target:     Target:       Target:        Strategy:
            <0.05 eV    <0.1 eV/Å   <0.5 Debye    <20% SMAPE     Weight by
                        RMSE        RMSE                          importance
```

---

## Summary

The loss function architecture is designed with specific purposes:

1. **Energy**: Dual loss for accuracy + robustness across wide range
2. **Force**: Physics-constrained via autodiff; normalized for size-invariance
3. **Dipole**: Phase-aware to handle vector orientation ambiguity
4. **NACR**: SMAPE for percentage-based error in tiny values
5. **Combination**: Weighted sum enables simultaneous multi-target training
6. **Regularization**: L2 prevents overfitting; 2e-5 coefficient empirically optimal

Each component is carefully designed for its specific physical meaning and typical value ranges.
