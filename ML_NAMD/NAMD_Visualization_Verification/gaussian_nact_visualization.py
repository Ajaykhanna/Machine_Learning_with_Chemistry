import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def generate_ref_pairs(ref, total):
    pairs = []
    if ref > 1:
        pairs.append(tuple(sorted((ref, ref - 1))))
    curr_upper = ref + 1
    while len(pairs) < total:
        pairs.append(tuple(sorted((ref, curr_upper))))
        curr_upper += 1
    return sorted(pairs)


def plot_density_analysis(args):
    setup_logging()

    tracked_pairs = generate_ref_pairs(args.ref_state, args.total_pairs)
    tracked_states = sorted(list(set([s for p in tracked_pairs for s in p])))

    frames, nact_history, pes_history = (
        [],
        {p: [] for p in tracked_pairs},
        {s: [] for s in tracked_states},
    )

    logging.info(
        f"Processing {args.end - args.start} frames. Using Density Heatmap for high-res data."
    )

    for frame_idx in tqdm(
        range(args.start, args.end + 1, args.step), desc="Loading Data"
    ):
        n_path, p_path = os.path.join(
            args.dir, f"frame_{frame_idx}", "nact.out"
        ), os.path.join(args.dir, f"frame_{frame_idx}", "pes.out")
        if not (os.path.exists(n_path) and os.path.exists(p_path)):
            continue

        try:
            n_data, p_data = np.loadtxt(n_path, ndmin=2), np.loadtxt(p_path, ndmin=2)
            n_states = int(np.sqrt(n_data.shape[1] - 1))
            n_matrix = n_data[0, 1:].reshape(n_states, n_states)

            for s1, s2 in tracked_pairs:
                # Store the sign * magnitude (or just sign) for polarization
                val = n_matrix[s1, s2]
                nact_history[(s1, s2)].append(
                    np.sign(val) if abs(val) > args.threshold else 0
                )

            for s in tracked_states:
                pes_history[s].append(p_data[0, s + 1])
            frames.append(frame_idx)
        except:
            continue

    if not frames:
        return

    # Visualization: Heatmap for 100K frames
    n_pairs = len(tracked_pairs)
    fig, (ax_pes, ax_heat) = plt.subplots(
        2, 1, figsize=(15, 8), gridspec_kw={"height_ratios": [1, 2]}, sharex=True
    )

    # 1. PES Plot (Standard line plot works fine for PES at 100K)
    cmap_pes = plt.colormaps.get_cmap("tab10")
    for i, s in enumerate(tracked_states):
        ax_pes.plot(
            frames,
            pes_history[s],
            label=f"S{s}",
            lw=1,
            alpha=0.8,
            color=cmap_pes(i % 10),
        )
    ax_pes.set_ylabel("Energy (eV)", fontweight="bold")
    ax_pes.set_title(
        f"100K Frame Analysis: PES and Phase Polarization (Sigma={args.sigma})",
        fontsize=14,
    )
    ax_pes.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # 2. Phase Heatmap
    heatmap_data = []
    for p in tracked_pairs:
        # Apply Gaussian smoothing to the sign vector to see "dominant" phase
        smoothed_phase = gaussian_filter1d(
            np.array(nact_history[p], dtype=float), sigma=args.sigma
        )
        heatmap_data.append(smoothed_phase)

    heatmap_matrix = np.array(heatmap_data)
    im = ax_heat.imshow(
        heatmap_matrix,
        aspect="auto",
        cmap="RdBu",
        interpolation="none",
        extent=[min(frames), max(frames), n_pairs - 0.5, -0.5],
        vmin=-1,
        vmax=1,
    )

    ax_heat.set_yticks(range(n_pairs))
    ax_heat.set_yticklabels(
        [f"S{p[0]}-S{p[1]}" for p in tracked_pairs], fontweight="bold"
    )
    ax_heat.set_xlabel("Configuration Number", fontweight="bold")
    ax_heat.set_ylabel("State Pairs", fontweight="bold")

    # Colorbar logic
    cbar = fig.colorbar(im, ax=ax_heat, orientation="horizontal", pad=0.15, aspect=50)
    cbar.set_label(
        "Phase Polarization (Solid Blue = Stable +, Solid Red = Stable -, White = Rapid Flips/Noise)",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    logging.info(f"Saved High-Density Analysis to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100000)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--ref_state", type=int, default=1)
    parser.add_argument("--total_pairs", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument(
        "--sigma",
        type=float,
        default=50.0,
        help="Smoothing radius. Increase for 100K frames.",
    )
    parser.add_argument("--out", type=str, default="nact_100k_density.png")
    plot_density_analysis(parser.parse_args())
