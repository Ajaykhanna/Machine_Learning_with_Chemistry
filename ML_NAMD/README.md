# HIPPYNN Molecular Training Script - Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [Script Architecture](#script-architecture)
3. [Core Workflow](#core-workflow)
4. [Training Targets Explained](#training-targets-explained)
5. [Input/Output Dimensions and Data Shapes](#inputoutput-dimensions-and-data-shapes)
6. [Network Architecture](#network-architecture)
7. [Loss Computation](#loss-computation)
8. [Command-Line Interface](#command-line-interface)
9. [Default Arguments Reference](#default-arguments-reference)
10. [Usage Examples](#usage-examples)
11. [Key Design Patterns](#key-design-patterns)

---

## Overview

This script (`training.py`) is a sophisticated machine learning training framework built on **HIPPYNN** (Hierarchical Interatomic Polyrnn for Python: Neural Networks) for training models on molecular properties. The script trains neural networks to predict:

- **Energies**: Ground state and excited state energies
- **Dipole Moments**: Electric dipole moments derived from atomic charges
- **Forces**: Atomic forces (negative gradients of energy with respect to positions)
- **NACRs**: Non-Adiabatic Coupling Vectors (coupling between electronic states)

The framework is designed for flexibility, supporting multi-target training with weighted combinations of different property predictions.

### Key Dependencies

- **hippynn**: Hierarchical neural network library for molecular predictions
- **torch**: PyTorch for deep learning and automatic differentiation
- **numpy**: Numerical computations
- **matplotlib**: Plotting (configured to use 'Agg' backend for non-interactive rendering)

---

## Script Architecture

The script is organized into the following logical components:

### 1. **Imports and Global Configuration** (Lines 1-31)

```python
# Recursion limit increased for complex graph structures
sys.setrecursionlimit(2000)

# Non-interactive plotting backend
matplotlib.use("Agg")

# PyTorch optimization settings for faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_default_dtype(torch.float32)
```

### 2. **Data Classes** (Lines 33-89)

The `ArgsList` dataclass serves as a type hint container for all CLI arguments, enabling IDE auto-completion. This is a workaround for the dynamic nature of argparse.

```python
@dataclass
class ArgsList:
    tag: str
    device: int
    interactive: bool
    ... [30+ more fields]
```

### 3. **Core Functions**

The script defines several functional layers:

| Function | Purpose |
|----------|---------|
| `build_network()` | Creates the HIPNN neural network architecture |
| `energy_target()` | Defines energy prediction nodes and loss functions |
| `force_training()` | Creates force prediction nodes using gradient of energy |
| `dipole_target()` | Creates dipole prediction via atomic charges |
| `nacr_target()` | Creates non-adiabatic coupling vectors |
| `build_output_layer()` | Orchestrates creation of all prediction targets |
| `build_loss()` | Combines individual losses into total training objective |
| `setup_plots()` | Configures visualization during training |
| `setup_experiment()` | Assembles training modules and optimizer |
| `load_database()` | Loads and preprocesses the molecular dataset |
| `reload_checkpoint()` | Resumes training from saved checkpoint |
| `main()` | Main training loop orchestrator |
| `path_handler()` | Manages experiment directory structure |
| `read_args()` | Argument parser with defaults |

---

## Core Workflow

### Execution Flow Diagram

```
START
  ↓
read_args() → Parse CLI arguments
  ↓
path_handler() → Check/create experiment directory
  ↓
main(params)
  ├─ If reload=True:
  │  └─ reload_checkpoint() → Load trained model and resume
  │
  └─ If reload=False (new training):
     ├─ build_network() → Create HIPNN architecture
     ├─ build_output_layer() → Create target prediction heads
     ├─ build_loss() → Combine losses
     ├─ setup_experiment() → Setup optimizer and scheduler
     └─ setup_plots() → Configure plotting
  ↓
load_database() → Load and split dataset
  ↓
train_model() → HIPPYNN's training loop
  ├─ Forward pass through network
  ├─ Compute loss for all targets
  ├─ Backward pass (autograd)
  ├─ Optimizer step
  ├─ Learning rate scheduling
  └─ Checkpoint saving
  ↓
Save training_summary.json with results
  ↓
END
```

### Step-by-Step Explanation

#### 1. **Network Construction** (`build_network`)

```python
def build_network(network_params: dict):
    # Input layer: atomic species (atomic numbers Z)
    species = inputs.SpeciesNode(db_name="Z")
    
    # Input layer: 3D atomic positions (R)
    positions = inputs.PositionsNode(db_name="R")
    positions.requires_grad = True  # Needed for force computation
    
    # Main HIPNN network
    network = networks.Hipnn(
        "hipnn_model", 
        (species, positions),  # Inputs
        module_kwargs=network_params  # Architecture parameters
    )
    
    return species, positions, network
```

**Key Points:**
- Positions require gradients because forces are computed as $\frac{\partial E}{\partial \mathbf{R}}$
- Species are atomic numbers (0 for padding, 1 for H, 6 for C, 7 for N, 8 for O)
- HIPNN is a hierarchical architecture with interaction layers and atom layers

---

## Training Targets Explained

### A. Energy Target (`energy_target`)

**Purpose**: Predict molecular energies for different electronic states

```python
def energy_target(
    n_states: int,
    network: networks.Hipnn,
    weight=1.0,
    multi_targets=False,
    include_ground_state=True,
):
```

**How it works:**

1. **State Handling**:
   - If `include_ground_state=True`: Training on S0, S1, S2, ... (ground + excited states)
   - If `include_ground_state=False`: Training only on S1, S2, ... (excited states)

2. **Single vs Multi-Target Mode**:
   - **Single-Target** (`multi_targets=False`):
     - Creates separate `HEnergyNode` for each state
     - Each node produces independent energy predictions
     - Nodes: S0, S1, S2, ... (n_states outputs)
   
   - **Multi-Target** (`multi_targets=True`):
     - Single `HEnergyNode` with internal head that predicts all states
     - More parameter sharing, reduced model size
     - Single node that outputs all states simultaneously

3. **Loss Functions**:
   - **RMSE** (Root Mean Squared Error): `MSELoss` with power 0.5
   - **MAE** (Mean Absolute Error): `MAELoss` with power 1
   - Both are computed per state and combined

**Output Structure**:
```python
{
    "loss": {
        "RMSE": (MSELoss, power=0.5, normalization=1),
        "MAE": (MAELoss, power=1, normalization=1),
    },
    "loss_weight": weight,  # Weighting for multi-task learning
    "outputs": [mol_energy_0, mol_energy_1, ...],  # Predicted energies
    "energy_nodes": [energy_node_0, energy_node_1, ...],  # For reuse in forces
}
```

---

### B. Force Training (`force_training`)

**Purpose**: Predict atomic forces from energy gradients

```python
def force_training(
    n_states: int,
    n_atoms: int,
    energy_nodes: List[targets.HEnergyNode],
    positions: inputs.PositionsNode,
    weight=1.0,
    sign=1,
    multi_targets=False,
):
```

**How it works:**

1. **Physics-Based Computation**:
   - Forces are NOT directly predicted by a separate network head
   - Instead, forces are computed via automatic differentiation:
   
   $$\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{R}_i}$$
   
   - Uses PyTorch's `autograd` through `physics.GradientNode`

2. **Sign Parameter**:
   - `sign=1`: Force = -∇E (standard convention)
   - `sign=-1`: Force = +∇E (alternative convention)

3. **State and Atom Dimensions**:
   - For each energy state, computes forces on all N atoms
   - Gradient is 3N-dimensional (3 coordinates per atom)
   - Force loss normalized by $\sqrt{3N}$ (number of coordinates)

4. **Loss Functions**:
   - **RMSE** with power 0.5
   - **MAE** with normalization factor $\sqrt{3N_{atoms}}$
   - Accounts for the fact that more atoms → larger absolute forces

**Output Structure**:
```python
{
    "loss": {
        "RMSE": (MSELoss, 0.5, 1),
        "MAE": (MAELoss, 1, sqrt(3*n_atoms)),
    },
    "loss_weight": weight,
    "outputs": [force_0, force_1, ...],  # Gradient nodes
    "force_nodes": [force_node_0, ...],
}
```

**Key Design Decision**: Forces are physics-constrained because they're derived from energy gradients. This ensures:
- Energy conservation (forces always gradient of energy)
- Reduced overfitting
- Better generalization to unseen geometries

---

### C. Dipole Target (`dipole_target`)

**Purpose**: Predict electric dipole moments from atomic partial charges

```python
def dipole_target(
    n_states: int,
    network: networks.Hipnn,
    positions: inputs.PositionsNode,
    weight=1.0,
    multi_targets=False,
    include_ground_state=False,  # Note: False by default for dipoles
):
```

**How it works:**

1. **Two-Step Process**:
   - **Step 1**: Network predicts partial atomic charges for each state
     - `HChargeNode` creates separate charge outputs per atom per state
     - Each atom gets a scalar charge $q_i$
   
   - **Step 2**: Dipole computed as charge-position product
     - `physics.DipoleNode` computes: $\mathbf{D} = \sum_i q_i \mathbf{R}_i$
     - Result is a 3D vector (x, y, z components)

2. **State Handling**:
   - By default, `include_ground_state=False` for dipoles
   - Rationale: Ground state dipole is usually less important
   - Can be enabled if needed

3. **Charge Nodes Reuse**:
   - Charge nodes can be **reused for NACR** computation
   - If dipole is in training targets, NACR uses same charges
   - If dipole is NOT in targets, separate charges created for NACR
   - Controlled by `--no-reuse-charges` flag

4. **Loss Functions**:
   - **RMSE** with `MSEPhaseLoss` (phase-aware loss for vectors)
   - **MAE** with `MAEPhaseLoss`
   - Normalization by $\sqrt{3}$ (3 components in dipole vector)
   - Phase-aware because dipole direction matters

**Output Structure**:
```python
{
    "loss": {
        "RMSE": (MSEPhaseLoss, 0.5, 1),  # Phase-aware MSE
        "MAE": (MAEPhaseLoss, 1, sqrt(3)),
    },
    "loss_weight": weight,
    "outputs": [dipole_0, dipole_1, ...],  # 3D vectors
    "charge_nodes": [charge_node_0, ...],  # For potential reuse
}
```

---

### D. NACR Target (`nacr_target`)

**Purpose**: Predict non-adiabatic coupling vectors between electronic states

**Non-Adiabatic Coupling Vectors (NACRs)** represent the coupling between different electronic states, important for:
- Charge transfer dynamics
- Multi-state simulations
- Photochemistry

```python
def nacr_target(
    training_targets: dict,
    n_states: int,
    n_atoms: int,
    network: networks.Hipnn,
    positions: inputs.PositionsNode,
    weight=1.0,
    multi_targets=False,
    no_reuse_charges=False,
):
```

**How it works:**

1. **Dependency Requirements**:
   - NACR requires at least **2 states** (coupling between states)
   - NACR requires **energies** to be in training targets
   - Energies used to compute energy gap between coupled states

2. **Charge Node Strategy**:
   - If `no_reuse_charges=False` (default):
     - Reuse charge nodes from dipole target
     - Efficient: shares network weights
   - If `no_reuse_charges=True`:
     - Create separate charge nodes for NACR only
     - More flexibility but more parameters

3. **NACR Pair Computation**:
   - For each pair of states (i, j) where i < j:
     - Compute coupling vector between states i and j
     - Uses: charges (q_i, q_j), positions, energy nodes (E_i, E_j)
     - Number of pairs: $\frac{n_{states}(n_{states}-1)}{2}$
   
   - **Example for 3 states (S0, S1, S2)**:
     - NACR(0,1), NACR(0,2), NACR(1,2) → 3 vectors

4. **Multi-Target Mode**:
   - Single `NACRMultiStateNode` predicts all couplings in one head
   - Shares representation across all pairs

5. **Single-Target Mode**:
   - Individual `NACRNode` for each state pair
   - Independent predictions

6. **Loss Function**:
   - `SMAPEPhaseLoss` (Symmetric Mean Absolute Percentage Error)
   - Phase-aware: handles vector orientation
   - Good for small values (important for NACRs)

---

## Output Layer Assembly (`build_output_layer`)

This orchestrates all targets:

```python
def build_output_layer(params, network, positions):
    training_targets = {}
    train_nacr = False
    train_force = False
    
    for i, t in enumerate(params.training_targets):
        weight = params.target_weights[i]
        
        if t == "energy":
            training_targets[t] = energy_target(...)
        elif t == "dipole":
            training_targets[t] = dipole_target(...)
        elif t == "force":
            train_force = True  # Deferred until energies created
        elif t == "nacr":
            train_nacr = True  # Deferred until energies created
    
    # Create force nodes (requires energy nodes)
    if train_force:
        training_targets["force"] = force_training(...)
    
    # Create NACR nodes (requires energy and optionally charge nodes)
    if train_nacr:
        training_targets = nacr_target(...)
    
    return training_targets
```

**Key Design Points:**
- **Dependency Order**: Energy must be created before forces or NACR
- **Charge Reuse**: Dipole charges can be reused for NACR
- **Validation**: Script exits if dependencies not satisfied
  - Can't train forces without energy
  - Can't train NACR without energy
  - Need ≥2 states for NACR

---

## Input/Output Dimensions and Data Shapes

### Network Inputs

The HIPPYNN network expects two inputs:

| Input | Key | Type | Shape | Description |
|-------|-----|------|-------|-------------|
| **Species** | "Z" | int64 | `(N_samples, N_atoms)` | Atomic numbers (0=padding, 1=H, 6=C, 7=N, 8=O) |
| **Positions** | "R" | float32 | `(N_samples, N_atoms, 3)` | Atomic coordinates in Ångströms (x, y, z) |

**Example for ACN (15 atoms, 1000 structures)**:
- Z shape: `(1000, 15)` - each row is atomic numbers for one molecule
- R shape: `(1000, 15, 3)` - each entry is (x,y,z) coordinate of an atom

### Network Outputs and Training Targets

#### 1. **Energy Outputs**

**Single-Target Mode** (`multi_targets=False`):
- Each state gets independent output node
- Number of nodes: `n_states` (or `n_states + 1` if including ground state)

| Property | Array Key | Output Shape | Notes |
|----------|-----------|--------------|-------|
| Ground State Energy (S0) | "S0" | `(N_samples,)` | Scalar energy per structure |
| Excited State 1 (S1) | "S1" | `(N_samples,)` | Scalar energy per structure |
| Excited State 2 (S2) | "S2" | `(N_samples,)` | Scalar energy per structure |

**Multi-Target Mode** (`multi_targets=True`):
- Single output node predicts all states
- Output shape: `(N_samples, n_states_output)` where `n_states_output = n_states + 1` (including ground)

**Example for 2 states with ground state**:
- Single-target: 3 outputs (S0, S1, S2), each shape `(1000,)`
- Multi-target: 1 output, shape `(1000, 3)`

#### 2. **Force Outputs**

**Always computed from energy gradient**:
- Automatically derived from energy via `physics.GradientNode`
- Number of outputs: same as energy outputs

| Property | Array Key | Output Shape | Notes |
|----------|-----------|--------------|-------|
| Ground State Forces (F0) | "F0" | `(N_samples, N_atoms*3)` | Flattened: 3 coordinates per atom |
| Excited State 1 Forces (F1) | "F1" | `(N_samples, N_atoms*3)` | Flattened: 3 coordinates per atom |
| Excited State 2 Forces (F2) | "F2" | `(N_samples, N_atoms*3)` | Flattened: 3 coordinates per atom |

**For ACN (15 atoms)**:
- Force output per state: `(N_samples, 45)` - 15 atoms × 3 coordinates
- Formula: $\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{R}_i}$

#### 3. **Dipole Outputs**

**Two-level computation**:
1. Atomic charges: `(N_samples, N_atoms)` for each state
2. Dipole moment: `(N_samples, 3)` - vector with x, y, z components

| Property | Array Key | Charges Shape | Dipole Shape | Notes |
|----------|-----------|----------------|--------------|-------|
| Excited State 1 Charges | "Q1" | `(N_samples, N_atoms)` | - | Partial atomic charges |
| Excited State 1 Dipole | "D1" | - | `(N_samples, 3)` | Total molecular dipole moment |
| Excited State 2 Dipole | "D2" | - | `(N_samples, 3)` | Total molecular dipole moment |

**Computation**:
$$\mathbf{D}_i = \sum_{j=1}^{N_{atoms}} q_j \cdot \mathbf{R}_j$$

where:
- $q_j$ = partial charge on atom j (from charge node)
- $\mathbf{R}_j$ = position of atom j
- Result is 3D vector (x, y, z dipole components)

**For ACN (15 atoms)**:
- Charges per state: `(N_samples, 15)` - one scalar charge per atom
- Dipole per state: `(N_samples, 3)` - x, y, z components

#### 4. **NACR Outputs** 

**Non-Adiabatic Coupling Vectors between electronic states**:
- Computed from charge gradients and energy differences
- Number of outputs: $\frac{n_{states}(n_{states}-1)}{2}$ (all state pairs)

| Property | Array Key | Output Shape | Notes |
|----------|-----------|--------------|-------|
| NACR (S1→S2) | "NACRdE_1_2" | `(N_samples, N_atoms*3)` | Flattened: 3 coords per atom |
| NACR (S1→S3) | "NACRdE_1_3" | `(N_samples, N_atoms*3)` | Flattened: 3 coords per atom |
| NACR (S2→S3) | "NACRdE_2_3" | `(N_samples, N_atoms*3)` | Flattened: 3 coords per atom |

**State Pair Counting**:
- 2 states: 1 NACR pair (1,2)
- 3 states: 3 NACR pairs (1,2), (1,3), (2,3)
- 4 states: 6 NACR pairs
- Formula: $\binom{n}{2} = \frac{n(n-1)}{2}$

**For ACN (15 atoms, 2 states)**:
- NACR output: `(N_samples, 45)` - 15 atoms × 3 coordinates
- Single NACR vector between the two states

### Complete Example: 2-State ACN Training

**Configuration**:
```python
n_states = 2
n_atoms = 15
training_targets = ["energy", "dipole", "force", "nacr"]
include_ground_state = True
```

**Input Data (from .npy files)**:
```
acn_Z.npy          → (N, 15)              # Species
acn_R.npy          → (N, 15, 3)           # Positions
```

**Output/Target Data (from .npy files)**:
```
Energy Targets:
  acn_S0.npy       → (N,)                 # Ground state
  acn_S1.npy       → (N,)                 # Excited state 1
  acn_S2.npy       → (N,)                 # Excited state 2

Force Targets (if training):
  acn_F0.npy       → (N, 45)              # Force on ground state
  acn_F1.npy       → (N, 45)              # Force on state 1
  acn_F2.npy       → (N, 45)              # Force on state 2

Dipole Targets (if training):
  acn_Q1.npy       → (N, 15)              # Charges for state 1 (internal)
  acn_Q2.npy       → (N, 15)              # Charges for state 2 (internal)
  acn_D1.npy       → (N, 3)               # Dipole moment state 1
  acn_D2.npy       → (N, 3)               # Dipole moment state 2

NACR Targets (if training):
  acn_NACRdE_1_2.npy → (N, 45)            # NACR between S1 and S2
```

**Network Forward Pass**:
```
Input: Z (N, 15), R (N, 15, 3)
  ↓
HIPNN Encoder
  ↓
Energy Heads (3 outputs):
  ├─ S0: (N,) → compare with acn_S0.npy
  ├─ S1: (N,) → compare with acn_S1.npy
  └─ S2: (N,) → compare with acn_S2.npy
  ↓
Force Nodes (autodiff from energies):
  ├─ F0: (N, 45) → compare with acn_F0.npy
  ├─ F1: (N, 45) → compare with acn_F1.npy
  └─ F2: (N, 45) → compare with acn_F2.npy
  ↓
Charge Heads (for dipoles):
  ├─ Q1: (N, 15) → internal (not compared directly)
  └─ Q2: (N, 15) → internal (not compared directly)
  ↓
Dipole Nodes (charge × position):
  ├─ D1: (N, 3) → compare with acn_D1.npy
  └─ D2: (N, 3) → compare with acn_D2.npy
  ↓
NACR Nodes (from charges and energies):
  └─ NACRdE_1_2: (N, 45) → compare with acn_NACRdE_1_2.npy
```

### Dataset Array Organization

**Standard naming convention**:
```
dataset_location/
├── {dataset_name}Z.npy              # Atomic species [ALWAYS REQUIRED]
├── {dataset_name}R.npy              # Atomic positions [ALWAYS REQUIRED]
├── {dataset_name}S0.npy             # Ground state energy [if training energy]
├── {dataset_name}S1.npy             # State 1 energy [if training energy]
├── {dataset_name}S2.npy             # State 2 energy [if training energy]
├── {dataset_name}F0.npy             # Ground state forces [if training force]
├── {dataset_name}F1.npy             # State 1 forces [if training force]
├── {dataset_name}F2.npy             # State 2 forces [if training force]
├── {dataset_name}Q1.npy             # State 1 charges [if training dipole]
├── {dataset_name}Q2.npy             # State 2 charges [if training dipole]
├── {dataset_name}D1.npy             # State 1 dipole [if training dipole]
├── {dataset_name}D2.npy             # State 2 dipole [if training dipole]
└── {dataset_name}NACRdE_1_2.npy     # NACR S1→S2 [if training nacr]
```

**Example for ACN dataset**:
```
/vast/home/akhanna2/data/ml_project/acn_data/RAW_DATA_Extraction/2025-Oct-17/
├── acn_Z.npy
├── acn_R.npy
├── acn_S0.npy
├── acn_S1.npy
├── acn_S2.npy
├── acn_F0.npy
├── acn_F1.npy
├── acn_F2.npy
├── acn_Q1.npy
├── acn_Q2.npy
├── acn_D1.npy
├── acn_D2.npy
└── acn_NACRdE_1_2.npy
```

### Batch Processing During Training

During training, HIPPYNN creates batches from the dataset:

```
Training Batch Size: B (e.g., 32)

Input Batch:
  Z_batch: (B, N_atoms)           e.g., (32, 15)
  R_batch: (B, N_atoms, 3)        e.g., (32, 15, 3)

Output Batch (for each property):
  Energy_batch: (B,)              e.g., (32,)
  Force_batch: (B, N_atoms*3)     e.g., (32, 45)
  Dipole_batch: (B, 3)            e.g., (32, 3)
  NACR_batch: (B, N_atoms*3)      e.g., (32, 45)

Loss Computation:
  loss = mean((output_batch - target_batch) ** 2)
```

### Multi-State Slicing Logic

When dataset has more states than requested, script automatically slices:

```python
# Example: Dataset has 5 states, but we want only 2
if multi_targets and n_dataset_columns > n_requested_states:
    # For energies: take first n_states columns
    arrays["E"] = arrays["E"][:, :n_requested_states+1]
    
    # For NACR: extract correct state pairs using triangular indexing
    # Original pairs: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
    # Keep pairs: (0,1), (0,2), (1,2)
```

### Memory Considerations

**RAM/VRAM usage for one batch**:

For B=32 samples, N=15 atoms:
```
Z_batch:          32 × 15 × int64      ≈ 3.8 KB
R_batch:          32 × 15 × 3 × float32 ≈ 57.6 KB
S_batch:          32 × 3 × float32      ≈ 384 B
F_batch:          32 × 45 × float32     ≈ 5.8 KB
D_batch:          32 × 3 × float32      ≈ 384 B
Q_batch:          32 × 15 × float32     ≈ 1.9 KB
NACR_batch:       32 × 45 × float32     ≈ 5.8 KB
───────────────────────────────────────
Per batch:        ≈ 75 KB (negligible)
```

**Full dataset transfer** (`--db-to-gpu`):

For N_samples=10000, N_atoms=15:
```
Z:     10000 × 15 × int64    ≈ 1.2 MB
R:     10000 × 15 × 3 × float ≈ 1.8 MB
All targets:                  ≈ 5-10 MB
───────────────────────────────
Total:                         ≈ 20-30 MB (easily fits on GPU)
```

---

## Loss Computation

### Loss Aggregation (`build_loss`)

```python
def build_loss(training_targets: dict, network: networks.Hipnn):
    validation_losses = {}
    
    for target_type, target_dict in training_targets.items():
        outputs = target_dict["outputs"]
        weight = target_dict["loss_weight"]
        loss_functions = target_dict["loss"]
        
        target_loss = 0
        
        for loss_name, (loss_node, power, normalization) in loss_functions.items():
            for output in outputs:
                # Compute loss for this output
                loss_value = loss_node.of_node(output)
                
                # Apply power (usually 0.5 for RMSE)
                if power != 1:
                    loss_value = loss_value ** power
                
                # Store individual loss
                validation_losses[f"{output.name}-{loss_name}"] = loss_value
            
            # Normalize (e.g., sqrt(3*n_atoms) for forces)
            if normalization != 1.0:
                loss_value /= normalization
            
            target_loss += loss_value
        
        # Weight different targets (multi-task learning)
        if weight != 1.0:
            target_loss *= weight
        
        total_loss += target_loss
    
    # L2 regularization
    l2_reg = loss.l2reg(network)
    loss_regularization = 2e-5 * l2_reg
    
    # Final loss
    validation_losses["Loss"] = total_loss + loss_regularization
```

### Loss Structure

**Example for training on energy + dipole + force**:

```
Loss (total) = λ_E * Loss_energy + λ_D * Loss_dipole + λ_F * Loss_force + 2e-5 * L2_reg
             = 1.0 * (RMSE_E + MAE_E) + 1.0 * (RMSE_D + MAE_D) + 1.0 * (RMSE_F + MAE_F) + 2e-5 * L2
```

**Key Features:**
- **Weighted Multi-Task Learning**: `target_weights` control contribution of each task
- **Per-Output Metrics**: Individual RMSE/MAE tracked for each state
- **Normalization**: Forces normalized by $\sqrt{3N}$, dipoles by $\sqrt{3}$
- **L2 Regularization**: 2e-5 coefficient to prevent overfitting

---

## Loss Function Design for Each Property

This section details the specific loss functions used for each trainable property and the rationale behind their design choices.

### A. Energy Loss Design

**Loss Functions**: MSE (squared error) and MAE (absolute error)

```python
"loss": {
    "RMSE": (loss.MSELoss, power=0.5, normalization=1),
    "MAE": (loss.MAELoss, power=1, normalization=1),
}
```

**Mathematical Formulation**:

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (E_{\text{pred}}^i - E_{\text{true}}^i)^2$$

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (E_{\text{pred}}^i - E_{\text{true}}^i)^2}$$

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |E_{\text{pred}}^i - E_{\text{true}}^i|$$

**Design Rationale**:

1. **Dual Loss Strategy (RMSE + MAE)**:
   - **RMSE** emphasizes larger errors (quadratic penalty)
   - **MAE** provides robustness to outliers (linear penalty)
   - Combined loss: $L_{\text{energy}} = \text{RMSE} + \text{MAE}$
   - Prevents overfitting to outlier energies while capturing overall accuracy

2. **Why Power=0.5 for RMSE**:
   - MSELoss computes mean squared error
   - Power 0.5 takes square root: $(\text{MSE})^{0.5} = \text{RMSE}$
   - RMSE is in same units as energy (easier interpretation)
   - Training uses RMSE directly, not MSE

3. **Normalization = 1**:
   - Energies are scalars (not vectors)
   - No per-coordinate normalization needed
   - Absolute energy scale preserved

4. **Per-State Tracking**:
   - For n_states=2: Separate RMSE/MAE for S0 and S1
   - Metrics: `S0-RMSE`, `S0-MAE`, `S1-RMSE`, `S1-MAE`
   - Allows monitoring accuracy per state
   - States with poor accuracy identified and addressed

**Typical Energy Scale**: 1-100 eV
**Typical RMSE Target**: < 0.05 eV for ground state

---

### B. Force Loss Design

**Loss Functions**: MSE (squared error) and MAE (absolute error)

```python
"loss": {
    "RMSE": (loss.MSELoss, power=0.5, normalization=1),
    "MAE": (loss.MAELoss, power=1, normalization=math.sqrt(n_atoms * 3)),
}
```

**Mathematical Formulation**:

$$\text{MSE}_{\text{force}} = \frac{1}{N} \sum_{i=1}^{N} ||\mathbf{F}_{\text{pred}}^i - \mathbf{F}_{\text{true}}^i||_2^2$$

$$\text{RMSE}_{\text{force}} = \sqrt{\text{MSE}_{\text{force}}}$$

$$\text{MAE}_{\text{force, normalized}} = \frac{1}{N} \sum_{i=1}^{N} ||\mathbf{F}_{\text{pred}}^i - \mathbf{F}_{\text{true}}^i||_1 / \sqrt{3n_{\text{atoms}}}$$

**Design Rationale**:

1. **Critical Normalization Factor** ($\sqrt{3n_{\text{atoms}}}$):
   - Forces are 3N-dimensional vectors (3 coordinates per atom)
   - Normalization equalizes loss across different molecule sizes
   - Ensures fair weighting regardless of molecular size
   - For n_atoms=15: normalization factor = $\sqrt{45} \approx 6.7$

2. **Why MSE + MAE**:
   - **RMSE** penalizes large errors in forces (important for dynamics)
   - **MAE** ensures smooth gradient flow and robustness
   - Combined: prevents overfitting to large force outliers

3. **Physics Constraint**:
   - Forces are **computed as gradients** of energy: $\mathbf{F} = -\nabla E$
   - NOT independently predicted by separate network head
   - Automatic differentiation ensures consistency
   - Network can only fit energies; forces follow naturally
   - Reduces parameters and improves generalization

4. **Gradient Sign Control**:
   - Parameter `--gradient-sign 1` (default): $\mathbf{F} = -\nabla E$
   - Parameter `--gradient-sign -1`: $\mathbf{F} = +\nabla E$
   - Allows different convention matching

**Typical Force Range**: -5 to +5 eV/Å
**Typical RMSE Target**: < 0.1 eV/Å

**Why Accurate Forces Matter**:
- Used in molecular dynamics simulations
- Small force errors → incorrect trajectories
- Energy-force consistency critical for dynamics

---

### C. Dipole Loss Design

**Loss Functions**: MSEPhaseLoss and MAEPhaseLoss (vector-aware losses)

```python
"loss": {
    "RMSE": (MSEPhaseLoss, power=0.5, normalization=1),
    "MAE": (MAEPhaseLoss, power=1, normalization=math.sqrt(3)),
}
```

**Mathematical Formulation**:

For a 3D dipole vector $\mathbf{D} = (D_x, D_y, D_z)$:

$$\text{MSEPhaseLoss} = \frac{1}{N} \sum_{i=1}^{N} \min(||\mathbf{D}_{\text{pred}}^i - \mathbf{D}_{\text{true}}^i||_2^2, ||\mathbf{D}_{\text{pred}}^i + \mathbf{D}_{\text{true}}^i||_2^2)$$

$$\text{MAEPhaseLoss} = \frac{1}{N} \sum_{i=1}^{N} \min(||\mathbf{D}_{\text{pred}}^i - \mathbf{D}_{\text{true}}^i||_1, ||\mathbf{D}_{\text{pred}}^i + \mathbf{D}_{\text{true}}^i||_1)$$

**Design Rationale**:

1. **Phase Ambiguity Problem**:
   - Dipole moment is a **vector with direction**
   - $\mathbf{D}$ and $-\mathbf{D}$ represent same physical property
   - Regular MSE/MAE would treat them as very different
   - Phase-aware loss handles this ambiguity

2. **How MSEPhaseLoss Works**:
   - Computes error for both $\mathbf{D}$ and $-\mathbf{D}$
   - Takes minimum: $\text{min}(||E_1||^2, ||E_2||^2)$
   - Automatically selects correct phase
   - Prevents unnecessary penalties for phase flips

3. **Why Normalization = $\sqrt{3}$**:
   - Dipole is a 3D vector
   - Normalizes loss per vector component
   - Accounts for dimensionality like forces do for coordinates

4. **Why Not Regular MSE/MAE**:
   - Dipole components are often small (0-10 Debye)
   - Phase flips cause large errors in standard loss
   - Would train network to avoid phase flips → poor generalization
   - Phase-aware loss allows natural exploration of solution space

5. **Per-State Dipole Tracking**:
   - Typically train on S1, S2, ... (excited states)
   - Ground state (S0) often excluded (`include_ground_state=False`)
   - Excited state dipoles usually more interesting chemically

**Typical Dipole Range**: 0-10 Debye (0-50 eÅ)
**Typical RMSE Target**: < 0.5 Debye

**Why Dipole Prediction Matters**:
- Characterizes charge distribution
- Important for interactions and spectroscopy
- Excited state dipoles reveal electronic redistribution

---

### D. NACR Loss Design

**Loss Function**: SMAPEPhaseLoss (specialized for coupling vectors)

```python
"loss": {
    "SMAPE": (SMAPEPhaseLoss, power=1, normalization=1),
}
```

**Mathematical Formulation**:

$$\text{SMAPE}_{\text{phase}} = \frac{1}{N} \sum_{i=1}^{N} \min\left(\text{SMAPE}(\mathbf{V}^i, \mathbf{V}_{\text{true}}^i), \text{SMAPE}(-\mathbf{V}^i, \mathbf{V}_{\text{true}}^i)\right)$$

where

$$\text{SMAPE}(\mathbf{a}, \mathbf{b}) = \frac{2}{3} \sum_{j=1}^{3} \frac{|a_j - b_j|}{|a_j| + |b_j| + \epsilon}$$

**Design Rationale**:

1. **Why SMAPE (Symmetric Mean Absolute Percentage Error)**:
   - NACR values are typically **very small** (10⁻⁴ to 10⁻³ Å⁻¹)
   - Absolute errors in standard MSE/MAE would be huge
   - Percentage error more appropriate for small-value predictions
   - Symmetric: treats over/under prediction equally

2. **Phase Ambiguity for NACRs**:
   - Like dipoles, NACR is a vector
   - $\mathbf{V}_{ij}$ and $-\mathbf{V}_{ij}$ represent same physics
   - SMAPEPhaseLoss automatically handles this
   - Prevents training from unnecessarily constraining phase

3. **Why NO Normalization (normalization=1)**:
   - SMAPE is already scale-invariant (percentage-based)
   - Additional normalization would be redundant
   - Vector dimension handled naturally by SMAPE formulation

4. **Epsilon in Denominator**:
   - Prevents division by zero for small values
   - Important since NACRs near zero are common
   - Allows smooth gradients throughout training

5. **Multiple NACR Pairs**:
   - For n_states=2: 1 pair (S0↔S1)
   - For n_states=3: 3 pairs (S0↔S1, S0↔S2, S1↔S2)
   - Each pair gets independent SMAPE loss
   - Total NACR loss = sum of individual pair losses

**Typical NACR Range**: 10⁻⁴ to 10⁻³ Å⁻¹
**Typical SMAPE Target**: < 20% (relative error)

**Why NACR Prediction Matters**:
- Essential for non-adiabatic dynamics simulations
- Controls transition probabilities between states
- Often most challenging property to predict accurately
- Small errors can significantly impact dynamics

---

### Loss Summary Table

| Property | Loss Functions | Normalization | Why This Choice |
|----------|---|---|---|
| **Energy** | RMSE + MAE | 1.0 | Dual strategy: RMSE for accuracy, MAE for robustness |
| **Force** | RMSE + MAE | $\sqrt{3n_{\text{atoms}}}$ | Scale-invariant across molecule sizes; physics-constrained |
| **Dipole** | MSEPhaseLoss + MAEPhaseLoss | $\sqrt{3}$ | Phase-aware for vector ambiguity; per-component normalization |
| **NACR** | SMAPEPhaseLoss | 1.0 | Scale-invariant for small values; phase-aware |

---

### Multi-Target Loss Aggregation

When training on multiple properties, the total loss is a weighted sum:

$$L_{\text{total}} = \sum_{k \in \{\text{energy, force, dipole, nacr}\}} \lambda_k \cdot L_k + \lambda_{\text{L2}} \cdot L_{\text{L2}}$$

where:
- $\lambda_k$ = `target_weights[k]` (default: 1.0 for each)
- $L_k$ = loss for property k
- $\lambda_{\text{L2}} = 2 \times 10^{-5}$ (L2 regularization coefficient)
- $L_{\text{L2}}$ = sum of squared network weights

**Example: Energy + Dipole + Force with weights [1.0, 0.5, 0.5]**

$$L_{\text{total}} = 1.0 \cdot L_{\text{energy}} + 0.5 \cdot L_{\text{dipole}} + 0.5 \cdot L_{\text{force}} + 2 \times 10^{-5} \cdot L_{\text{L2}}$$

**Weight Selection Strategy**:
- Default (1.0): Equal importance to all properties
- Larger weight: Prioritize this property's accuracy
- Smaller weight: Reduce impact of noisier properties
- Example: NACR is often noisier → use weight 0.5

---

### L2 Regularization

```python
l2_reg = loss.l2reg(network)
loss_regularization = 2e-5 * l2_reg
```

**Purpose**: Prevent overfitting by penalizing large weights

$$L_{\text{L2}} = 2 \times 10^{-5} \sum_{w \in \text{network}} w^2$$

**Design Choices**:
- **Coefficient = 2e-5**: Balances model complexity with data fit
  - Too large: Underfits, poor training accuracy
  - Too small: Overfits, poor validation accuracy
  - This value empirically works well for molecular properties

- **Why weight decay matters**:
  - Network has thousands of parameters
  - Without L2: Network learns noise in training data
  - With L2: Simpler solutions preferred
  - Improves validation performance

---

## Network Architecture

### HIPNN Network Parameters

```python
network_params = {
    "possible_species": [0, 1, 6, 7, 8],  # Atomic numbers to expect
    "n_features": 15,                      # Hidden layer dimensions
    "n_sensitivities": 20,                 # Number of radial basis functions
    "dist_soft_min": 0.8,                  # Soft minimum distance (Å)
    "dist_soft_max": 20.0,                 # Soft maximum distance (Å)
    "dist_hard_max": 24.0,                 # Hard cutoff distance (Å)
    "n_interaction_layers": 3,             # Message passing layers
    "n_atom_layers": 3,                    # Per-atom refinement layers
}
```

### Architecture Flow

```
Input Atoms (Z, R)
    ↓
Radial Basis Functions (n_sensitivities=20 RBFs)
    ↓
Edge Features (distances between atoms)
    ↓
[Interaction Layer × n_interaction_layers]
    ├─ Message passing between atoms
    ├─ Update node features
    └─ n_features dimensional representations
    ↓
[Atom Layer × n_atom_layers]
    ├─ Per-atom refinement
    └─ Prepares features for output layers
    ↓
Output Heads
    ├─ Energy nodes
    ├─ Charge nodes (for dipoles)
    └─ Other property-specific heads
```

**Key Architecture Details:**

| Parameter | Role | Typical Values |
|-----------|------|---|
| `n_features` | Hidden dimension size | 15-64 |
| `n_sensitivities` | RBF basis functions | 10-32 |
| `n_interaction_layers` | Message passing iterations | 2-5 |
| `n_atom_layers` | Per-atom MLP depth | 1-3 |
| `dist_soft_min/max` | RBF center range | 0.8-20 Å |
| `dist_hard_max` | Force zeros beyond cutoff | 20-30 Å |

---

## Database Loading and Preprocessing (`load_database`)

### Dataset Organization

```
dataset_location/
├── acn_Z.npy           # Atomic species
├── acn_R.npy           # Atomic positions
├── acn_S0.npy          # Ground state energy
├── acn_S1.npy          # First excited state energy
├── acn_S2.npy          # Second excited state energy
├── acn_F0.npy          # Ground state forces
├── acn_D1.npy          # Excited state dipole
├── acn_Q1.npy          # Charges for state 1
└── acn_NACRdE_1_2.npy  # NACR between states 1 and 2
```

### Multi-Target Slicing Logic

When `multi_targets=True`, the dataset may contain more states than requested:

```python
if n_columns > columns_expected:
    if "NACR" in key:
        # For NACR: extract correct state pairs
        m = int((np.sqrt(8*n_columns + 1) + 1) / 2)
        idx_orig = list(zip(*np.triu_indices(m, k=1)))
        idx_new = list(zip(*np.triu_indices(n_states, k=1)))
        slices = np.isin(idx_orig, idx_new).all(axis=1)
        arrays[key] = v[:, slices]
    elif key == "E":
        # For energies: take first n_states+1 (including ground)
        arrays[key] = v[..., :columns_expected]
    else:
        # For other properties: take first n_states columns
        arrays[key] = v[:, :columns_expected]
```

### Data Splitting

```python
database.make_trainvalidtest_split(
    test_size=split_ratio[0],      # e.g., 0.1 → 10% test
    valid_size=split_ratio[1]      # e.g., 0.2 → 20% validation
)
# Results in: 70% train, 20% valid, 10% test
```

### GPU Transfer

```python
if params.db_to_gpu and torch.cuda.is_available():
    database.send_to_device(params.device)
```

---

## Training Loop and Optimization

### Optimizer Setup

```python
optimizer = torch.optim.AdamW(
    training_modules.model.parameters(),
    lr=params.init_learning_rate  # Default: 1e-3
)
```

**AdamW Features:**
- Decoupled weight decay (better for regularization)
- Adaptive learning rates per parameter
- Momentum and second-moment estimates

### Learning Rate Scheduler

```python
scheduler = RaiseBatchSizeOnPlateau(
    optimizer=optimizer,
    max_batch_size=params.max_batch_size,
    patience=params.raise_batch_patience,  # Default: 96 epochs
    factor=0.5,
)
```

**Strategy**:
1. If validation loss plateaus for `raise_batch_patience` epochs
2. Instead of lowering learning rate, **increase batch size** by factor 0.5
3. Larger batch → noisier gradients stabilize, training progresses
4. Scales to `max_batch_size`

### Early Stopping Controller

```python
controller = PatienceController(
    optimizer=optimizer,
    scheduler=scheduler,
    batch_size=params.init_batch_size,        # Start: 32
    eval_batch_size=params.max_batch_size,    # Use full batch for eval
    max_epochs=params.max_epochs,             # Max: 3000
    stopping_key=params.stopping_key,         # "Loss"
    fraction_train_eval=0.1,                  # 10% train used for validation
    termination_patience=params.termination_patience,  # Default: 500
)
```

**Early Stopping**:
- Monitors validation loss metric specified by `stopping_key`
- Stops if no improvement for `termination_patience` epochs
- Saves best model checkpoint

---

## Path Handling (`path_handler`)

### Experiment Directory Structure

```
work_dir/
└── {tag}_{n_features}_{n_sensitivities}_{lower_cutoff}_{upper_cutoff}_{cutoff_distance}_{n_interactions}_{n_atom_layers}/
    ├── best_model.pt                 # Best model weights
    ├── best_checkpoint.pt            # Best + optimizer state
    ├── experiment_structure.pt       # Network architecture
    ├── training_metrics.pt           # Collected metrics
    ├── training_log.txt              # Console output
    ├── training_summary.json         # Final results
    └── plots/
        ├── epochs/
        │   ├── epoch0/
        │   ├── epoch10/
        │   └── ...
        ├── FinalTraining/
        │   ├── test/
        │   ├── train/
        │   └── valid/
        └── over_time/
```

### Directory Handling Logic

```python
if os.path.exists(dir_name):
    if params.retrain:
        # Force retrain: remove and recreate
        shutil.rmtree(dir_name)
        os.mkdir(dir_name)
    else:
        # Check if training complete
        if os.path.exists(f"{dir_name}/training_summary.json"):
            summary = json.load(...)
            if len(summary) >= 7:  # Complete summary
                print(f"{dir_name} already finished")
                return summary
        # Incomplete: reset directory
        shutil.rmtree(dir_name)
        os.mkdir(dir_name)
```

---

## Command-Line Interface

### Argument Parsing

The script uses Python's `argparse` with custom formatter:

```python
def read_args(
    tag="test",
    device=0,
    ... [60+ parameters with defaults]
):
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter  # Shows defaults + preserves formatting
    )
    
    # Add 60+ arguments
    parser.add_argument("--tag", type=str, default=tag, help="...")
    ...
    
    args = ArgsList(**vars(parser.parse_args()))
    return args
```

### Argument Categories

Arguments are grouped by functionality:

1. **Task Configuration**: tag, device, interactive, noprogress
2. **Path Handling**: work-dir, handle-work-dir, reload, retrain
3. **Optimization**: map-devices, db-to-gpu, custom-kernel
4. **Model Selection**: n-states, n-atoms, training-targets, target-weights
5. **NACR Options**: no-reuse-charges, multi-targets, gradient-sign
6. **Network Architecture**: n-interactions, n-atom-layers, n-features, n-sensitivities
7. **Distance Functions**: lower-cutoff, upper-cutoff, cutoff-distance
8. **Dataset**: dataset-location, dataset-name, split-ratio, seed, n-workers
9. **Training**: init-batch-size, max-batch-size, init-learning-rate
10. **Scheduling**: raise-batch-patience, termination-patience, max-epochs, stopping-key

---

## Default Arguments Reference

### Global Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tag` | str | "test" | Identifier for this experiment run |
| `--device` | int | 0 | CUDA device index (0 = first GPU) |
| `-i, --interactive` | bool | False | Print output to both console and log file |
| `-P, --noprogress` | bool | False | Suppress progress bars |
| `--custom-kernel` | bool | False | Enable HIPPYNN custom CUDA kernels |
| `--log-filename` | str | "training_log.txt" | Output log file name |

### Path Management

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--work-dir` | str | "test" | Root directory for experiments |
| `--handle-work-dir` | bool | False | Auto-create experiment subdirectories |
| `--reload` | bool | False | Resume from checkpoint (must use with --work-dir) |
| `--retrain` | bool | True | Force retrain if directory exists |
| `--map-devices` | bool | False | Enable cross-GPU device mapping on reload |
| `--update-parameters` | bool | False | Update batch size/patience on reload |
| `--db-to-gpu` | bool | False | Transfer entire dataset to GPU memory |

### Model Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n-states` | int | 2 | Number of electronic states to include |
| `--n-atoms` | int | 15 | Number of atoms in molecule |
| `--training-targets` | list | ["energy", "dipole"] | Properties to predict (comma-separated) |
| `--target-weights` | list | None | Weights for each target in loss (auto=1.0 each) |
| `--multi-targets` | bool | False | Use single output head for all states |
| `--no-reuse-charges` | bool | False | Don't reuse dipole charges for NACR |
| `--gradient-sign` | int | 1 | Sign for force calculation (-∇E or +∇E) |

### Network Architecture

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n-interactions` | int | 3 | Number of interaction layers (message passing) |
| `--n-atom-layers` | int | 3 | Number of per-atom refinement layers |
| `--n-features` | int | 15 | Hidden dimension size |
| `--n-sensitivities` | int | 20 | Number of radial basis functions |
| `--possible-species` | list | [0,1,6,7,8] | Atomic numbers in dataset (padding,H,C,N,O) |

### Distance Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lower-cutoff` | float | 0.8 | Minimum distance for RBF (Ångströms) |
| `--upper-cutoff` | float | 20.0 | Maximum distance for RBF (Ångströms) |
| `--cutoff-distance` | float | 24.0 | Hard cutoff beyond which interactions zero |

### Dataset Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset-location` | str | "{RAW_DATA_PATH}" | Path to folder with .npy files |
| `--dataset-name` | str | "acn_" | Prefix for dataset numpy arrays |
| `--split-ratio` | list | [0.07, 0.01] | [test_frac, valid_frac] |
| `--seed` | int | 7777 | Random seed for train/valid/test split |
| `--n-workers` | int | 1 | DataLoader workers (0=main thread) |

### Training Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--init-batch-size` | int | 32 | Initial batch size for first epoch |
| `--max-batch-size` | int | 512 | Maximum batch size scheduler can reach |
| `--init-learning-rate` | float | 1e-3 | Initial learning rate for AdamW |
| `--raise-batch-patience` | int | 96 | Epochs of plateau before increasing batch |
| `--max-epochs` | int | 3000 | Maximum training epochs |
| `--termination-patience` | int | 500 | Epochs of plateau before early stopping |
| `--stopping-key` | str | "Loss" | Metric name to monitor for early stopping |
| `--plot-frequency` | int | 100 | Save plots every N epochs (0=disable) |

---

## Usage Examples

### Example 1: Basic Training (Energies and Dipoles)

```bash
python training.py \
    --tag my_acn_model \
    --device 0 \
    --n-states 2 \
    --n-atoms 15 \
    --training-targets energy,dipole \
    --work-dir /path/to/models \
    --handle-work-dir \
    --dataset-location /path/to/data
```

### Example 2: Include Forces and NACR

```bash
python training.py \
    --tag acn_full \
    --device 0 \
    --n-states 2 \
    --training-targets energy,dipole,force,nacr \
    --target-weights 1.0,1.0,0.5,0.5 \
    --multi-targets \
    --work-dir /path/to/models \
    --handle-work-dir
```

**Key Points:**
- Forces and NACR require energy to be in targets
- Weights balance contribution of each property
- Multi-targets uses single output head (fewer params)

### Example 3: Resume Training with Updated Hyperparameters

```bash
python training.py \
    --tag acn_full \
    --reload \
    --work-dir /path/to/models \
    --map-devices \
    --update-parameters \
    --max-epochs 5000 \
    --termination-patience 1000
```

**Key Points:**
- `--reload` loads checkpoint from last training
- `--map-devices` enables cross-GPU loading
- `--update-parameters` refreshes batch size and patience schedules
- New max-epochs and patience override old values

### Example 4: Larger Network with Custom Architecture

```bash
python training.py \
    --tag acn_large \
    --n-features 32 \
    --n-sensitivities 32 \
    --n-interactions 5 \
    --n-atom-layers 4 \
    --init-batch-size 64 \
    --max-batch-size 1024 \
    --init-learning-rate 5e-4 \
    --work-dir /path/to/models \
    --handle-work-dir \
    --db-to-gpu
```

**Considerations:**
- Larger network: more parameters, longer training
- Smaller learning rate: more stable but slower convergence
- Larger batch sizes: more GPU memory needed
- `--db-to-gpu`: loads full dataset to GPU (requires sufficient VRAM)

### Example 5: Multi-State Training (3 States)

```bash
python training.py \
    --tag acn_3states \
    --n-states 3 \
    --training-targets energy,dipole,nacr \
    --work-dir /path/to/models \
    --handle-work-dir
```

**NACR Output Count**:
- For 3 states: 3×2/2 = 3 NACR vectors
- (S0→S1, S0→S2, S1→S2)

---

## Key Design Patterns

### 1. **Modular Target Creation**

```python
# Each target is independent function
targets_dict = {
    "energy": energy_target(...),
    "dipole": dipole_target(...),
    "force": force_training(...),
    "nacr": nacr_target(...)
}

# Can be composed in any combination
```

**Benefit**: Easy to add new property types without modifying core loop

### 2. **Physics-Constrained Force Prediction**

```python
# Forces derived from energy gradients
force = physics.GradientNode(energy_node, positions)

# NOT a separate network prediction
```

**Benefit**: 
- Ensures F = -∇E automatically
- Reduces parameters
- Improves generalization

### 3. **Charge Node Reuse**

```python
# Option 1: Reuse charges
dipole_charges = ...
nacr_charges = dipole_charges  # Same network heads

# Option 2: Separate charges
nacr_charges = targets.HChargeNode(...)  # New heads
```

**Benefit**: Flexibility to share or separate parameters

### 4. **Weighted Multi-Task Learning**

```python
total_loss = sum(
    weight_i * loss_i 
    for i in range(num_targets)
)
```

**Benefit**: Prioritize certain properties over others

### 5. **Configurable State Handling**

```python
# Single-target mode: separate output per state
if not multi_targets:
    outputs = [node_S0, node_S1, node_S2]

# Multi-target mode: single output with heads
else:
    outputs = [combined_node]
```

**Benefit**: Trade-off between model size and output flexibility

### 6. **Checkpoint-Based Resumption**

```python
if reload:
    checkpoint = load_checkpoint_from_cwd()
    # Resume with potentially different hyperparameters
    controller.max_epochs = new_max_epochs
```

**Benefit**: Continue training without restarting from scratch

### 7. **Automatic Dataset Slicing**

```python
# Dataset may have more states than requested
# Script automatically slices to requested states
if n_dataset_columns > n_requested_states:
    array = array[:, :n_requested_states]
```

**Benefit**: Use same dataset for different state counts

### 8. **Loss Aggregation Pipeline**

```
Per-node losses (RMSE, MAE)
    ↓
Target-level losses (sum over outputs)
    ↓
Weighted multi-task loss (sum over targets)
    ↓
L2 regularization
    ↓
Total training loss
```

**Benefit**: Transparent loss computation with monitoring at each level

---

## Important Notes

### GPU Memory Considerations

- **Without `--db-to-gpu`**: Dataset stays in CPU RAM, transferred per batch
  - Lower memory footprint
  - Slower for repeated epochs
  - Recommended for large datasets

- **With `--db-to-gpu`**: Entire dataset transferred to GPU memory
  - Requires sufficient VRAM
  - Much faster training
  - Good for datasets < 4 GB

### Logging and Output

- **Without `--interactive`**: Output redirected to `training_log.txt`
  - Needed for SLURM jobs to avoid huge log files
  - Use with `tail -f training_log.txt` to monitor

- **With `--interactive`**: Output to both console and log file
  - Useful for debugging on login nodes

### Plotting

- Frequency controlled by `--plot-frequency`
- Plots saved to `plots/` subdirectory
- **Sensitivity Plots**: One per interaction layer
- **Prediction vs Target Histograms**: One per prediction node

### Early Stopping Strategy

1. Training starts with `init_batch_size` and `init_learning_rate`
2. If validation loss plateaus for `raise_batch_patience` epochs:
   - Batch size increased (learning becomes more stable)
3. If still plateau for `termination_patience` epochs:
   - Training terminates (early stopping)

**Default**:
- `raise_batch_patience = 96`: Switch batch size after 96 epochs of plateau
- `termination_patience = 500`: Stop after 500 total epochs of plateau

### Checkpoint System

Three checkpoint files saved during training:

1. **best_model.pt**: Just model weights
2. **best_checkpoint.pt**: Model + optimizer state (for resuming)
3. **experiment_structure.pt**: Network architecture (for inference)

Plus continuous saves to allow resuming from interruptions.

---

## Summary

This training script is a comprehensive framework for molecular property prediction using HIPPYNN. Its key strengths are:

1. **Flexibility**: Train on any combination of energies, forces, dipoles, and NACRs
2. **Physics-Aware**: Forces derived from energy gradients
3. **Multi-Task Learning**: Weighted combination of multiple objectives
4. **Scalability**: Configurable architecture from small to large networks
5. **Robustness**: Checkpoint system and resumption capabilities
6. **Monitoring**: Detailed loss tracking and visualization

The script handles the full workflow from data loading through training to saving results, with extensive CLI options for experiment customization.
