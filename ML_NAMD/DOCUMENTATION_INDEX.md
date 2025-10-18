# Complete Documentation Index

## Overview

This directory now contains comprehensive documentation for the `training.py` HIPPYNN molecular training script, with special emphasis on loss function design for each trainable property (energies, forces, dipoles, and NACRs).

---

## Documentation Files

### 1. **README.md** ⭐ PRIMARY REFERENCE
**Status**: ENHANCED (1103 → 1388 lines)  
**Purpose**: Complete script documentation with all details

**Contents**:
- Overview of HIPPYNN framework
- Script architecture and workflow
- Core training workflow diagram
- Detailed explanation of all four training targets
- **NEW: Loss Function Design for Each Property** (comprehensive section)
  - Energy loss (RMSE + MAE strategy)
  - Force loss (physics-constrained)
  - Dipole loss (phase-aware)
  - NACR loss (SMAPE for small values)
  - Loss aggregation strategies
  - L2 regularization details
- Network architecture specifications
- Database loading and preprocessing
- Training loop and optimization details
- Command-line interface documentation
- Default arguments reference table
- Usage examples (5 scenarios)
- Key design patterns

**When to Use**: Primary reference for understanding ANY aspect of the script

---

### 2. **LOSS_FUNCTION_GUIDE.md** 📚 QUICK REFERENCE
**Status**: NEW  
**Purpose**: Practical quick-reference guide for loss functions

**Contents**:
- Quick reference: All loss functions at a glance
- Loss function design philosophy
  - Dual loss strategy (RMSE + MAE)
  - Vector normalization rationale
  - Phase-aware loss functions
  - Percent-based loss for small values
- Multi-target loss weighting guide
  - Default behavior (all weights = 1.0)
  - Custom weighting strategies
- L2 regularization strategy deep dive
- Practical guidelines for loss selection
  - Scenario 1: Energy-only training
  - Scenario 2: Energy + Forces
  - Scenario 3: Energy + Dipole + Force
  - Scenario 4: Multi-State with NACRs
  - Scenario 5: Custom prioritization
- Debugging guide: Common loss issues
  - High energy loss
  - Force loss not improving
  - Dipole loss saturating
  - NACR loss very high
- Summary table of loss purposes

**When to Use**: 
- Quick lookup during training configuration
- Debugging training issues
- Choosing weights and targets
- Understanding design choices

---

### 3. **LOSS_ARCHITECTURE_VISUAL.md** 🎨 DIAGRAMS & FLOW
**Status**: NEW  
**Purpose**: Visual representation of loss function architecture

**Contents**:
- Loss function hierarchy diagram
- Detailed architecture maps for each loss:
  - Energy loss component breakdown
  - Force loss (with physics constraint explanation)
  - Dipole loss (with phase ambiguity visualization)
  - NACR loss (with SMAPE explanation)
  - L2 regularization term
- Loss computation flow during training
- Multi-target loss weighting example
- Loss function decision tree

**When to Use**:
- Understanding overall loss structure
- Visualizing component relationships
- Teaching others about the framework
- Debugging by understanding data flow

---

### 4. **ENHANCEMENT_SUMMARY.md** 📋 CHANGE LOG
**Status**: NEW  
**Purpose**: Summary of what was added to the documentation

**Contents**:
- Overview of enhancements
- Detailed breakdown of README.md additions
  - Energy loss design coverage
  - Force loss design coverage
  - Dipole loss design coverage
  - NACR loss design coverage
  - Loss aggregation details
  - L2 regularization strategy
- Key features of the enhancement
- File growth statistics
- How enhancements improve understanding
- Summary of new documents

**When to Use**: 
- Understanding what documentation was added
- Tracking changes
- Sharing updates with collaborators

---

### 5. **DOCUMENTATION_UPDATE.md** ✅ COMPLETION REPORT
**Status**: NEW  
**Purpose**: Summary of all documentation work completed

**Contents**:
- Summary of all additions
- List of files modified/created
- Key enhancements to README.md by property
- Content coverage checklist
- Quick reference comparison table
- Information density summary
- Verification checklist
- Next steps and usage recommendations

**When to Use**:
- Quick overview of what was done
- Confirmation that work is complete
- Next steps for using the documentation

---

## Quick Navigation Guide

### I need to understand...

**...what the script does?**  
→ Start with README.md [Overview](#overview)

**...how to train a model?**  
→ README.md [Usage Examples](#usage-examples)

**...about energy prediction?**  
→ README.md [Training Targets Explained](#training-targets-explained) → Energy section

**...about force prediction?**  
→ README.md [Loss Function Design](#loss-function-design-for-each-property) → Force section  
→ LOSS_FUNCTION_GUIDE.md → Force scenario

**...about dipole prediction?**  
→ README.md [Loss Function Design](#loss-function-design-for-each-property) → Dipole section  
→ LOSS_ARCHITECTURE_VISUAL.md → Dipole Loss detail map

**...about NACR prediction?**  
→ README.md [Loss Function Design](#loss-function-design-for-each-property) → NACR section  
→ LOSS_FUNCTION_GUIDE.md → NACR scenario

**...how loss functions work?**  
→ LOSS_FUNCTION_GUIDE.md [Loss Function Design Philosophy](#loss-function-design-philosophy)  
→ LOSS_ARCHITECTURE_VISUAL.md [Loss Function Detail Map](#loss-function-detail-map)

**...how to set training weights?**  
→ LOSS_FUNCTION_GUIDE.md [Multi-Target Loss Weighting](#multi-target-loss-weighting)  
→ README.md [Default Arguments Reference](#default-arguments-reference)

**...how to debug training issues?**  
→ LOSS_FUNCTION_GUIDE.md [Debugging Loss Issues](#debugging-loss-issues)

**...what the L2 regularization does?**  
→ README.md [L2 Regularization](#l2-regularization)  
→ LOSS_FUNCTION_GUIDE.md [L2 Regularization Strategy](#l2-regularization-strategy)

**...what all arguments do?**  
→ README.md [Default Arguments Reference](#default-arguments-reference) (comprehensive table)

**...how the network architecture works?**  
→ README.md [Network Architecture](#network-architecture)

**...what the full workflow is?**  
→ README.md [Core Workflow](#core-workflow) (with diagram)

---

## Loss Functions at a Glance

### Energy
- **Loss**: RMSE + MAE
- **Normalization**: 1.0
- **Why**: Dual strategy for accuracy + robustness
- **Typical Target**: < 0.05 eV

### Force
- **Loss**: RMSE + MAE
- **Normalization**: √(3n_atoms)
- **Why**: Physics-constrained via autodiff; scale-invariant
- **Typical Target**: < 0.1 eV/Å

### Dipole
- **Loss**: MSEPhaseLoss + MAEPhaseLoss
- **Normalization**: √3
- **Why**: Phase-aware for vector orientation
- **Typical Target**: < 0.5 Debye

### NACR
- **Loss**: SMAPEPhaseLoss
- **Normalization**: 1.0
- **Why**: Percentage error for tiny values (10⁻⁴)
- **Typical Target**: < 20% relative error

---

## Training Configuration Checklist

### Before Starting Training

- [ ] Read README.md overview and architecture
- [ ] Choose training targets: energy, force, dipole, NACR?
- [ ] Decide weights via LOSS_FUNCTION_GUIDE.md recommendations
- [ ] Check typical ranges for your properties
- [ ] Plan GPU memory (use --db-to-gpu?)
- [ ] Set up working directory structure
- [ ] Prepare dataset with proper naming (acn_Z.npy, acn_R.npy, etc.)

### During Training

- [ ] Monitor loss terms: check if decreasing as expected
- [ ] Compare individual property losses (RMSE, MAE, SMAPE)
- [ ] Watch per-state metrics if multi-state
- [ ] Use LOSS_FUNCTION_GUIDE.md debugging section if stuck

### After Training

- [ ] Check training_summary.json for final metrics
- [ ] Review plots in plots/ directory
- [ ] Compare validation vs test performance
- [ ] If retraining: use --reload with updated hyperparameters

---

## Property-Specific Deep Dives

### Energy Prediction
**Files**: README.md (Energy section), LOSS_ARCHITECTURE_VISUAL.md (Energy detail map)

**Key Points**:
- Standard RMSE + MAE loss
- Normalized for training across multiple states
- Per-state monitoring available
- Physics foundation: V(R) = E

### Force Prediction
**Files**: README.md (Force section), LOSS_ARCHITECTURE_VISUAL.md (Force detail map)

**Key Points**:
- Physics-constrained: F = -∇E
- Automatic differentiation ensures consistency
- Normalized by √(3n_atoms) for size-invariance
- Requires energy in training targets
- Critical for molecular dynamics accuracy

### Dipole Prediction
**Files**: README.md (Dipole section), LOSS_ARCHITECTURE_VISUAL.md (Dipole detail map)

**Key Points**:
- Two-step process: charges → dipole moment
- Phase-aware loss handles vector ambiguity
- MSEPhaseLoss and MAEPhaseLoss used
- Ground state often excluded (excited states more interesting)
- Charges reusable for NACR

### NACR Prediction
**Files**: README.md (NACR section), LOSS_ARCHITECTURE_VISUAL.md (NACR detail map)

**Key Points**:
- Non-adiabatic coupling vectors
- Very small values (10⁻⁴ to 10⁻³ Å⁻¹)
- SMAPE (percentage error) essential
- Phase-aware: V and -V equivalent
- Requires energy in targets
- Often challenging to predict accurately

---

## Mathematical Background

### Provided in Documentation

**README.md contains**:
- Energy: MSE, RMSE, MAE formulas
- Force: Vector force formulation
- Dipole: Phase-aware MSE/MAE formulas
- NACR: SMAPE formulation with epsilon
- L2 Regularization: Weight decay formula

**LOSS_ARCHITECTURE_VISUAL.md contains**:
- All formulas with proper notation
- Flow diagrams showing computation order
- Phase ambiguity mathematical explanation
- SMAPE detailed breakdown

---

## Troubleshooting Index

**Problem**: Training loss not decreasing  
→ LOSS_FUNCTION_GUIDE.md → Debugging: "Loss stuck at high value"

**Problem**: Energy RMSE high  
→ LOSS_FUNCTION_GUIDE.md → Debugging: "Energy loss stuck at high value"

**Problem**: Force loss not improving  
→ LOSS_FUNCTION_GUIDE.md → Debugging: "Force loss not improving"

**Problem**: Dipole loss saturating  
→ LOSS_FUNCTION_GUIDE.md → Debugging: "Dipole loss saturating"

**Problem**: NACR loss very high  
→ LOSS_FUNCTION_GUIDE.md → Debugging: "NACR loss very high"

**Problem**: Unsure about weights  
→ LOSS_FUNCTION_GUIDE.md → "Practical Guidelines for Loss Selection"

**Problem**: Need training example  
→ README.md → [Usage Examples](#usage-examples)

---

## Documentation Statistics

| Document | Lines | Purpose | Audience |
|----------|-------|---------|----------|
| README.md | 1388 | Complete reference | Everyone |
| LOSS_FUNCTION_GUIDE.md | 400+ | Quick reference | Practitioners |
| LOSS_ARCHITECTURE_VISUAL.md | 350+ | Visual learners | Visualizers |
| ENHANCEMENT_SUMMARY.md | 150+ | Change tracking | Managers |
| DOCUMENTATION_UPDATE.md | 200+ | Completion report | Project trackers |
| DOCUMENTATION_INDEX.md | 400+ | Navigation guide | First-time users |

**Total**: 2800+ lines of documentation  
**Coverage**: Complete script analysis + loss function deep dives

---

## Getting Started

### Absolute Beginner?
1. Read README.md [Overview](#overview) (5 min)
2. Read README.md [Core Workflow](#core-workflow) (10 min)
3. Run an example from [Usage Examples](#usage-examples) (with help)

### Familiar with ML?
1. Skim README.md [Training Targets Explained](#training-targets-explained) (10 min)
2. Deep dive: README.md [Loss Function Design](#loss-function-design-for-each-property) (20 min)
3. Reference LOSS_FUNCTION_GUIDE.md during training

### Modifying or Extending?
1. Complete README.md reading (60 min)
2. Study LOSS_ARCHITECTURE_VISUAL.md (30 min)
3. Understand Key Design Patterns in README.md (20 min)
4. Review specific property sections as needed

### Debugging Issues?
1. Describe your problem
2. Find it in LOSS_FUNCTION_GUIDE.md [Debugging](#debugging-loss-issues)
3. Follow suggested solutions
4. Refer to relevant README.md section for more details

---

## Key Insights Summary

✅ **Energy**: Uses RMSE + MAE for accuracy + robustness  
✅ **Force**: Physics-constrained via autodiff; no independent prediction needed  
✅ **Dipole**: Phase-aware loss prevents false penalties for direction flips  
✅ **NACR**: SMAPE handles percentage errors for tiny values (10⁻⁴)  
✅ **Multi-Task**: Weighted combination enables simultaneous training on all properties  
✅ **L2 Reg**: 2e-5 coefficient balances overfitting vs underfitting  
✅ **Normalization**: Vectors normalized by dimension (forces √(3n), dipoles √3)  

---

## Document Interconnections

```
README.md (Master Reference)
    ├─ → LOSS_FUNCTION_GUIDE.md (Quick lookup)
    ├─ → LOSS_ARCHITECTURE_VISUAL.md (Visual explanation)
    ├─ → ENHANCEMENT_SUMMARY.md (What changed)
    └─ → DOCUMENTATION_UPDATE.md (Completion status)

For Understanding:
    README.md → LOSS_ARCHITECTURE_VISUAL.md → LOSS_FUNCTION_GUIDE.md

For Quick Questions:
    LOSS_FUNCTION_GUIDE.md → README.md (for details)

For Debugging:
    LOSS_FUNCTION_GUIDE.md (find problem) → README.md (understand deeper)

For Teaching:
    LOSS_ARCHITECTURE_VISUAL.md (show diagrams) → README.md (explain details)
```

---

## Recommendations

### For Daily Use
- Keep README.md open for reference
- Bookmark LOSS_FUNCTION_GUIDE.md [Debugging](#debugging-loss-issues)
- Use LOSS_ARCHITECTURE_VISUAL.md for visual understanding

### For Collaboration
- Share README.md [Overview](#overview) with team
- Use LOSS_FUNCTION_GUIDE.md [Practical Guidelines](#practical-guidelines-for-loss-selection) for design decisions
- Reference ENHANCEMENT_SUMMARY.md to show what was documented

### For Learning
- Work through README.md systematically
- Study LOSS_ARCHITECTURE_VISUAL.md for visual learners
- Try examples from README.md [Usage Examples](#usage-examples)

### For Production
- Follow checklists in this index
- Reference LOSS_FUNCTION_GUIDE.md [Practical Guidelines](#practical-guidelines-for-loss-selection)
- Monitor metrics from README.md [Loss Computation](#loss-computation)

---

## Questions & Answers

**Q: Where do I start?**  
A: README.md [Overview](#overview) → [Core Workflow](#core-workflow)

**Q: How do I choose training targets?**  
A: LOSS_FUNCTION_GUIDE.md [Practical Guidelines](#practical-guidelines-for-loss-selection)

**Q: What should my weights be?**  
A: LOSS_FUNCTION_GUIDE.md [Multi-Target Loss Weighting](#multi-target-loss-weighting)

**Q: Why are forces special?**  
A: README.md → Force section, explains physics-constraint

**Q: What does phase-aware mean?**  
A: README.md → Dipole section, explains ambiguity

**Q: Why SMAPE for NACR?**  
A: LOSS_FUNCTION_GUIDE.md → NACR section

**Q: My training is stuck. Help?**  
A: LOSS_FUNCTION_GUIDE.md [Debugging Loss Issues](#debugging-loss-issues)

**Q: What's the difference between my training and validation loss?**  
A: README.md → Early Stopping section

**Q: Should I use --db-to-gpu?**  
A: README.md → GPU Memory Considerations

**Q: What if I need to resume training?**  
A: README.md [Usage Examples](#usage-examples) → Example 3

---

This index should help you navigate all available documentation efficiently!
