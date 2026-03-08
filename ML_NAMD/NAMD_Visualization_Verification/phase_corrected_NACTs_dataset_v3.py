"""
NACT Phase Correction — Order-Independent Iterative Algorithm (v3)

Corrects the wavefunction sign ambiguity in raw NACTs, then recomputes
scaled dENACTs from the corrected raw NACTs and energy gaps.

Why correct raw NACTs first (not dENACTs directly):
  dENACT = NACT × ΔE, where ΔE = E_j - E_i can itself be negative when
  states cross. If you phase-correct dENACTs, the algorithm conflates two
  sources of sign change: physical (from ΔE) and spurious (from phase
  ambiguity). Correcting raw NACTs isolates the phase ambiguity cleanly.

Algorithm:
  1. Load raw NACT pairs for all configurations
  2. Compute global reference vector (mean of all NACTs with current signs)
  3. For each configuration, try all 2^(n_states-1) valid sign combos,
     pick the one closest to the reference
  4. Recompute reference from corrected data
  5. Repeat until convergence (no more flips)
  6. Apply same sign corrections to raw NACTs
  7. Recompute dENACT = corrected_NACT × ΔE from energy data

Outputs:
  - Phase-corrected acn_NACT{i}{j}.npy files
  - Recomputed acn_dENACT{i}{j}.npy files
  - Symlinks for non-NACT files (Z, R, E, etc.)
  - Diagnostic PDF and metadata JSON
"""

import os
import json
import numpy as np
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = (
    "/vast/home/akhanna2/scratch/nexmd/acetylacetone_diketone/gsoptd_md/50K_nact_data"
)
DATA_PREFIX = "acn_"
N_STATES_TO_USE = 3
INCLUDE_GROUND_STATE = False

OUTPUT_DIR = os.path.join(os.path.dirname(DATA_DIR), "50K_phase_corrected")

MAX_ITERATIONS = 50  # Usually converges in 3-5

# ============================================================================
# Setup
# ============================================================================
state_start = 0 if INCLUDE_GROUND_STATE else 1
state_end = N_STATES_TO_USE + (0 if INCLUDE_GROUND_STATE else 1)
state_indices = list(range(state_start, state_end))
n_active_states = len(state_indices)

nact_pairs = list(combinations(state_indices, 2))
n_pairs = len(nact_pairs)

pair_indices_relative = [(i - state_start, j - state_start) for (i, j) in nact_pairs]

# Build sign matrix: 2^(n_states-1) valid sign assignments
n_free = n_active_states - 1
n_combos = 2**n_free

sign_matrix = []  # (n_combos, n_pairs)
for c in range(n_combos):
    state_signs = [1]  # Fix first state
    for bit in range(n_free):
        state_signs.append(1 if (c >> bit) & 1 == 0 else -1)
    pair_signs = [state_signs[i] * state_signs[j] for (i, j) in pair_indices_relative]
    sign_matrix.append(pair_signs)

sign_matrix = np.array(sign_matrix, dtype=np.float64)  # (n_combos, n_pairs)

print("=" * 80)
print("NACT Phase Correction v3 — Raw NACTs + dENACT Recomputation")
print("=" * 80)
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"States: {state_indices}")
print(f"Pairs: {nact_pairs}")
print(f"Valid sign combos: {n_combos}")
for c in range(n_combos):
    signs_str = "  ".join(
        f"NACT{nact_pairs[p][0]}{nact_pairs[p][1]}={'+'if sign_matrix[c,p]>0 else '-'}"
        for p in range(n_pairs)
    )
    print(f"  Combo {c}: {signs_str}")
print()

# ============================================================================
# Load raw NACT data (NOT dENACT — phase correction on raw NACTs only)
# ============================================================================
print("Loading RAW NACT data (not scaled)...")
n_configs = None
pair_arrays = []

for i, j in nact_pairs:
    pair_name = f"NACT{i}{j}"
    filepath = os.path.join(DATA_DIR, f"{DATA_PREFIX}{pair_name}.npy")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Missing raw NACT file: {filepath}")

    data = np.load(filepath).astype(np.float64)

    if n_configs is None:
        n_configs = len(data)
    else:
        assert len(data) == n_configs, f"Shape mismatch: {pair_name}"

    pair_arrays.append(data)
    print(
        f"  {pair_name}: {len(data)} configs, "
        f"range [{data.min():.6f}, {data.max():.6f}], "
        f"mean={data.mean():.6f}, std={data.std():.6f}"
    )

# (n_configs, n_pairs)
nact_array = np.column_stack(pair_arrays)
print(f"\nRaw NACT array shape: {nact_array.shape}")

# ============================================================================
# Load energies for dENACT recomputation
# ============================================================================
print("\nLoading energy data...")
energy_file = os.path.join(DATA_DIR, f"{DATA_PREFIX}E.npy")
if not os.path.isfile(energy_file):
    raise FileNotFoundError(
        f"Missing energy file: {energy_file}\n"
        f"Energy data is required to recompute dENACTs after phase correction."
    )

energies = np.load(energy_file).astype(np.float64)
print(f"  Energies shape: {energies.shape}")
assert (
    energies.shape[0] == n_configs
), f"Energy configs ({energies.shape[0]}) != NACT configs ({n_configs})"

# Precompute energy gaps for each pair
# energies[:, 0] = S0, energies[:, i] = Si
energy_gaps = {}
for i, j in nact_pairs:
    delta_E = energies[:, j] - energies[:, i]  # (n_configs,)
    energy_gaps[f"{i}_{j}"] = delta_E
    n_negative = np.sum(delta_E < 0)
    print(
        f"  ΔE(S{i}-S{j}): mean={delta_E.mean():.4f}, "
        f"range=[{delta_E.min():.4f}, {delta_E.max():.4f}], "
        f"negative: {n_negative} ({100*n_negative/n_configs:.1f}%)"
    )

# ============================================================================
# Iterative phase correction (on raw NACTs)
# ============================================================================
print(f"\nRunning iterative phase correction (max {MAX_ITERATIONS} iterations)...")

current_combos = np.zeros(n_configs, dtype=np.int32)
corrected = nact_array.copy()

for iteration in range(MAX_ITERATIONS):
    # Step 1: Compute global reference from currently corrected data
    reference = np.mean(corrected, axis=0)  # (n_pairs,)

    # Step 2: For each config, find best sign combo
    n_flips = 0
    new_combos = np.zeros(n_configs, dtype=np.int32)

    # Process in chunks for memory efficiency
    chunk_size = 5000
    for start in range(0, n_configs, chunk_size):
        end = min(start + chunk_size, n_configs)
        chunk = nact_array[start:end]  # (chunk, n_pairs)

        # signed_chunk: (n_combos, chunk, n_pairs)
        signed_chunk = sign_matrix[:, np.newaxis, :] * chunk[np.newaxis, :, :]

        # cost: (n_combos, chunk)
        diff = signed_chunk - reference[np.newaxis, np.newaxis, :]
        cost = np.sum(diff**2, axis=2)

        # Best combo per config in this chunk
        best = np.argmin(cost, axis=0)  # (chunk,)
        new_combos[start:end] = best

    # Count flips
    n_flips = np.sum(new_combos != current_combos)
    current_combos = new_combos

    # Step 3: Apply corrections
    for k in range(n_configs):
        corrected[k] = sign_matrix[current_combos[k]] * nact_array[k]

    # Stats
    combo_counts = np.bincount(current_combos, minlength=n_combos)
    print(
        f"  Iteration {iteration + 1}: {n_flips} flips | "
        f"combo distribution: {dict(enumerate(combo_counts.tolist()))}"
    )

    if n_flips == 0:
        print(f"  Converged after {iteration + 1} iterations!")
        break
else:
    print(
        f"  Did not fully converge after {MAX_ITERATIONS} iterations "
        f"(last iteration had {n_flips} flips)"
    )

# ============================================================================
# Recompute dENACT from corrected raw NACTs
# ============================================================================
print("\nRecomputing scaled dENACT = corrected_NACT × ΔE ...")

denact_corrected = np.zeros_like(corrected)
for p, (i, j) in enumerate(nact_pairs):
    delta_E = energy_gaps[f"{i}_{j}"]
    denact_corrected[:, p] = corrected[:, p] * delta_E

    pair_name = f"dENACT{i}{j}"
    print(
        f"  {pair_name}: mean={denact_corrected[:, p].mean():.6f}, "
        f"std={denact_corrected[:, p].std():.6f}"
    )

# ============================================================================
# Final statistics
# ============================================================================
print("\n" + "=" * 80)
print("Final Statistics")
print("=" * 80)

combo_counts = np.bincount(current_combos, minlength=n_combos)
for c in range(n_combos):
    pct = 100.0 * combo_counts[c] / n_configs
    signs_str = "".join("+" if s > 0 else "-" for s in sign_matrix[c])
    print(f"  Combo {c} [{signs_str}]: {combo_counts[c]:6d} configs ({pct:.1f}%)")

identity_count = combo_counts[0]
flipped_count = n_configs - identity_count
print(
    f"\nTotal requiring correction: {flipped_count} / {n_configs} "
    f"({100.0 * flipped_count / n_configs:.1f}%)"
)

print("\nRaw NACT statistics (before → after):")
for p, (i, j) in enumerate(nact_pairs):
    pair_name = f"NACT{i}{j}"
    raw_mean = np.mean(nact_array[:, p])
    raw_std = np.std(nact_array[:, p])
    cor_mean = np.mean(corrected[:, p])
    cor_std = np.std(corrected[:, p])
    print(f"  {pair_name}:")
    print(f"    Mean:  {raw_mean:10.6f} → {cor_mean:10.6f}")
    print(f"    Std:   {raw_std:10.6f} → {cor_std:10.6f}")

print("\nScaled dENACT statistics (original → recomputed from corrected NACTs):")
for p, (i, j) in enumerate(nact_pairs):
    pair_name = f"dENACT{i}{j}"
    # Load original dENACT for comparison
    orig_file = os.path.join(DATA_DIR, f"{DATA_PREFIX}{pair_name}.npy")
    if os.path.isfile(orig_file):
        orig = np.load(orig_file).astype(np.float64)
        print(f"  {pair_name}:")
        print(f"    Original:  mean={orig.mean():10.6f}, std={orig.std():10.6f}")
        print(
            f"    Corrected: mean={denact_corrected[:, p].mean():10.6f}, "
            f"std={denact_corrected[:, p].std():10.6f}"
        )
    else:
        print(f"  {pair_name}: (no original file for comparison)")
        print(
            f"    Corrected: mean={denact_corrected[:, p].mean():10.6f}, "
            f"std={denact_corrected[:, p].std():10.6f}"
        )

# ============================================================================
# Phase consistency check
# ============================================================================
if n_pairs >= 3:
    print("\nPhase consistency check (sign(ij) × sign(ik) == sign(jk)):")
    n_violations = 0
    for k in range(n_configs):
        signs = sign_matrix[current_combos[k]]
        for a in range(n_active_states):
            for b in range(a + 1, n_active_states):
                for c_state in range(b + 1, n_active_states):
                    idx_ab = pair_indices_relative.index((a, b))
                    idx_ac = pair_indices_relative.index((a, c_state))
                    idx_bc = pair_indices_relative.index((b, c_state))
                    if signs[idx_ab] * signs[idx_ac] != signs[idx_bc]:
                        n_violations += 1
    print(f"  Constraint violations: {n_violations} (should be 0)")

# ============================================================================
# Save corrected data
# ============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nSaving corrected data to: {OUTPUT_DIR}")

# Save corrected raw NACTs
for p, (i, j) in enumerate(nact_pairs):
    pair_name = f"NACT{i}{j}"
    out_path = os.path.join(OUTPUT_DIR, f"{DATA_PREFIX}{pair_name}.npy")
    np.save(out_path, corrected[:, p].astype(np.float32))
    print(f"  Saved corrected {pair_name}")

# Save recomputed dENACTs
for p, (i, j) in enumerate(nact_pairs):
    pair_name = f"dENACT{i}{j}"
    out_path = os.path.join(OUTPUT_DIR, f"{DATA_PREFIX}{pair_name}.npy")
    np.save(out_path, denact_corrected[:, p].astype(np.float32))
    print(f"  Saved recomputed {pair_name}")

# Symlink non-NACT/dENACT files (Z, R, E, etc.)
nact_filenames = set()
for i, j in nact_pairs:
    nact_filenames.add(f"{DATA_PREFIX}NACT{i}{j}.npy")
    nact_filenames.add(f"{DATA_PREFIX}dENACT{i}{j}.npy")

n_symlinked = 0
for fname in os.listdir(DATA_DIR):
    if fname.endswith(".npy") and fname not in nact_filenames:
        src_path = os.path.join(DATA_DIR, fname)
        dst_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.islink(dst_path) or os.path.isfile(dst_path):
            os.remove(dst_path)
        os.symlink(os.path.abspath(src_path), dst_path)
        n_symlinked += 1

print(f"  Symlinked {n_symlinked} non-NACT files (Z, R, E, etc.)")

# Save the sign assignment per configuration (for reproducibility)
combo_path = os.path.join(OUTPUT_DIR, "phase_assignments.npy")
np.save(combo_path, current_combos)
print(f"  Saved phase assignments: {combo_path}")

# Save metadata
metadata = {
    "algorithm": "iterative_kmeans_sign_correction_v3",
    "source_data_dir": DATA_DIR,
    "n_states_to_use": N_STATES_TO_USE,
    "include_ground_state": INCLUDE_GROUND_STATE,
    "state_indices": state_indices,
    "nact_pairs": [list(p) for p in nact_pairs],
    "n_configs": int(n_configs),
    "max_iterations": MAX_ITERATIONS,
    "combo_counts": {str(c): int(combo_counts[c]) for c in range(n_combos)},
    "total_corrected": int(flipped_count),
    "pct_corrected": float(100.0 * flipped_count / n_configs),
    "final_reference": np.mean(corrected, axis=0).tolist(),
    "correction_target": "raw_NACT (dENACT recomputed from corrected NACT × ΔE)",
    "energy_file_used": os.path.basename(energy_file),
    "energy_gap_stats": {
        f"dE_S{i}_S{j}": {
            "mean": float(energy_gaps[f"{i}_{j}"].mean()),
            "std": float(energy_gaps[f"{i}_{j}"].std()),
            "min": float(energy_gaps[f"{i}_{j}"].min()),
            "max": float(energy_gaps[f"{i}_{j}"].max()),
            "n_negative": int(np.sum(energy_gaps[f"{i}_{j}"] < 0)),
        }
        for (i, j) in nact_pairs
    },
}

meta_path = os.path.join(OUTPUT_DIR, "phase_correction_metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=4)
print(f"  Saved metadata: {meta_path}")

# ============================================================================
# Diagnostic plots
# ============================================================================
pdf_path = os.path.join(OUTPUT_DIR, "phase_correction_diagnostic.pdf")
print(f"\nGenerating diagnostic plots -> {pdf_path}")

with PdfPages(pdf_path) as pdf:

    # --- Page 1: Raw NACT before/after time series ---
    n_show = min(3000, n_configs)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(16, 4 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, 2)

    for p, (i, j) in enumerate(nact_pairs):
        pair_name = f"NACT{i}{j}"
        axes[p, 0].plot(nact_array[:n_show, p], color="orange", alpha=0.7, lw=0.5)
        axes[p, 0].set_title(f"{pair_name} — RAW (before)")
        axes[p, 0].set_ylabel(pair_name)

        axes[p, 1].plot(corrected[:n_show, p], color="steelblue", alpha=0.7, lw=0.5)
        axes[p, 1].set_title(f"{pair_name} — CORRECTED (after)")
        axes[p, 1].set_ylabel(pair_name)

    axes[-1, 0].set_xlabel("Configuration index")
    axes[-1, 1].set_xlabel("Configuration index")
    fig.suptitle("Raw NACT Phase Correction", fontsize=14, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 2: dENACT before/after time series ---
    fig, axes = plt.subplots(n_pairs, 2, figsize=(16, 4 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, 2)

    for p, (i, j) in enumerate(nact_pairs):
        pair_name = f"dENACT{i}{j}"

        # Load original dENACT for comparison
        orig_file = os.path.join(DATA_DIR, f"{DATA_PREFIX}{pair_name}.npy")
        if os.path.isfile(orig_file):
            orig = np.load(orig_file).astype(np.float64)
            axes[p, 0].plot(orig[:n_show], color="orange", alpha=0.7, lw=0.5)
            axes[p, 0].set_title(f"{pair_name} — ORIGINAL")
        else:
            axes[p, 0].set_title(f"{pair_name} — (no original)")

        axes[p, 0].set_ylabel(pair_name)
        axes[p, 1].plot(
            denact_corrected[:n_show, p], color="steelblue", alpha=0.7, lw=0.5
        )
        axes[p, 1].set_title(f"{pair_name} — RECOMPUTED (from corrected NACT)")
        axes[p, 1].set_ylabel(pair_name)

    axes[-1, 0].set_xlabel("Configuration index")
    axes[-1, 1].set_xlabel("Configuration index")
    fig.suptitle("Scaled dENACT: Original vs Recomputed", fontsize=14, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 3: Histograms (raw NACT) ---
    fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 4 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, 2)

    for p, (i, j) in enumerate(nact_pairs):
        pair_name = f"NACT{i}{j}"
        axes[p, 0].hist(nact_array[:, p], bins=200, color="orange", alpha=0.7)
        axes[p, 0].set_title(f"{pair_name} — BEFORE")
        axes[p, 0].axvline(0, color="red", ls="--", alpha=0.5)

        axes[p, 1].hist(corrected[:, p], bins=200, color="steelblue", alpha=0.7)
        axes[p, 1].set_title(f"{pair_name} — AFTER")
        axes[p, 1].axvline(0, color="red", ls="--", alpha=0.5)

    fig.suptitle("Raw NACT Histograms", fontsize=14, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 4: Histograms (dENACT) ---
    fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 4 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, 2)

    for p, (i, j) in enumerate(nact_pairs):
        pair_name = f"dENACT{i}{j}"
        orig_file = os.path.join(DATA_DIR, f"{DATA_PREFIX}{pair_name}.npy")
        if os.path.isfile(orig_file):
            orig = np.load(orig_file).astype(np.float64)
            axes[p, 0].hist(orig, bins=200, color="orange", alpha=0.7)
        axes[p, 0].set_title(f"{pair_name} — ORIGINAL")
        axes[p, 0].axvline(0, color="red", ls="--", alpha=0.5)

        axes[p, 1].hist(denact_corrected[:, p], bins=200, color="steelblue", alpha=0.7)
        axes[p, 1].set_title(f"{pair_name} — RECOMPUTED")
        axes[p, 1].axvline(0, color="red", ls="--", alpha=0.5)

    fig.suptitle("Scaled dENACT Histograms", fontsize=14, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 5: Phase assignment scatter ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(range(n_configs), current_combos, s=0.5, alpha=0.3, c="darkred")
    ax.set_xlabel("Configuration index")
    ax.set_ylabel("Chosen sign combo")
    ax.set_title("Phase Assignment Over Configurations")
    ax.set_yticks(range(n_combos))
    labels = [
        "".join("+" if s > 0 else "-" for s in sign_matrix[c]) for c in range(n_combos)
    ]
    ax.set_yticklabels(labels)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 6: Pair-pair scatter (corrected NACTs) ---
    if n_pairs >= 2:
        pair_combos = list(combinations(range(n_pairs), 2))
        n_scatter = min(len(pair_combos), 3)
        fig, axes = plt.subplots(1, n_scatter, figsize=(6 * n_scatter, 5))
        if n_scatter == 1:
            axes = [axes]

        for ax_idx, (p1, p2) in enumerate(pair_combos[:n_scatter]):
            name1 = f"NACT{nact_pairs[p1][0]}{nact_pairs[p1][1]}"
            name2 = f"NACT{nact_pairs[p2][0]}{nact_pairs[p2][1]}"
            ax = axes[ax_idx]
            ax.scatter(
                corrected[:, p1], corrected[:, p2], s=0.5, alpha=0.1, c="steelblue"
            )
            ax.set_xlabel(name1)
            ax.set_ylabel(name2)
            ax.set_title(f"{name1} vs {name2} (corrected)")

        fig.suptitle("Corrected NACT Pair Correlations", fontsize=14, y=1.01)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print("\n" + "=" * 80)
print("Phase correction complete!")
print("=" * 80)
print(f"Corrected data saved to: {OUTPUT_DIR}")
print(f"  - {n_pairs} corrected raw NACT files (acn_NACT*.npy)")
print(f"  - {n_pairs} recomputed dENACT files (acn_dENACT*.npy)")
print(f"  - {n_symlinked} symlinked files (Z, R, E, etc.)")
print(f"\nFor training, set:")
print(f'  DATA_DIR = "{OUTPUT_DIR}"')
