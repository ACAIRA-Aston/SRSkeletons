import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_difficulty_grids(parquet_path, difficulty_name):
    # Load curated data
    df = pd.read_parquet(parquet_path)
    print(f"\nGenerating grids for {difficulty_name.upper()} dataset...")

    # Define slice intervals and grid specifications
    plots_config = [
        {"range": (0, 20), "grid": (5, 4), "suffix": "1_to_20"},
        {"range": (20, 40), "grid": (5, 4), "suffix": "21_to_40"},
        {"range": (40, 50), "grid": (4, 3), "suffix": "41_to_50"}
    ]

    for p_cfg in plots_config:
        start_idx, end_idx = p_cfg["range"]
        rows, cols = p_cfg["grid"]

        subset_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        num_items = len(subset_df)

        if num_items == 0:
            continue

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 3.5 * rows), dpi=200)
        axes_flat = axes.flatten()

        for idx in range(rows * cols):
            ax = axes_flat[idx]

            # If we run out of equations before filling the grid (specifically Plot 3)
            if p_cfg['grid'] == (4, 3):
                if idx == num_items - 1 or idx == num_items + 1:
                    ax.axis('off')  # Hide unused grid panels cleanly
                    continue
                elif idx == num_items:
                    idx = num_items - 1

            row = subset_df.iloc[idx]
            equation_id = f"{difficulty_name[0].upper()}{start_idx + idx + 1}"

            # Reconstruct numpy arrays from Parquet nested lists
            x_vals = np.array(row['X'])
            y_vals = np.array(row['Y'])

            # Plot properties
            ax.scatter(x_vals, y_vals, color='royalblue', alpha=0.65, edgecolors='none', s=10)
            ax.set_title(f"{equation_id}", fontsize=10, pad=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()

        # Save output high-res figures
        filename = f"../plots/grid_{difficulty_name}_{p_cfg['suffix']}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved: {filename}")


# Run generation on all three splits
import os

os.makedirs("../plots", exist_ok=True)

generate_difficulty_grids("../data/curated_datasets_easy.parquet", "easy")
generate_difficulty_grids("../data/curated_datasets_medium.parquet", "medium")
generate_difficulty_grids("../data/curated_datasets_hard.parquet", "hard")