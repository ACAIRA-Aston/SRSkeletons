import os
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
            if p_cfg['grid'] == (4, 3):
                if idx == num_items - 1 or idx == num_items + 1:
                    ax.axis('off')  # Hide unused grid panels cleanly
                    continue
                elif idx == num_items:
                    idx = num_items - 1

            row = subset_df.iloc[idx]
            equation_id = f"{difficulty_name[0].upper()}{start_idx + idx + 1}"
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


def create_split_datasets_and_grids(parquet_path, difficulty_name):
    """
    Splits each dataset adaptively using percentiles of X:
      - Extrapolation (10%): bottom 5% (left tail) + top 5% (right tail)
      - Interpolation (90%): middle 80% (between 5th and 95th percentiles)
    Saves the split data to separate Parquet files and plots the curves with
    extrapolation in orange and interpolation in blue.
    """
    df = pd.read_parquet(parquet_path)
    print(f"\n=======================================================")
    print(f"  Splitting & Plotting (90/10 Percentile): {difficulty_name.upper()} Dataset")
    print(f"=======================================================")

    interp_records = []
    extrap_records = []

    # Filter each row using curve-specific 10th and 90th percentiles of X
    for _, row in df.iterrows():
        x_vals = np.array(row['X'])
        y_vals = np.array(row['Y'])

        # Compute dynamic percentile thresholds for the current curve's X range
        p10, p90 = np.percentile(x_vals, [5, 95])

        # Extrapolation mask: bottom 10% (left) or top 10% (right)
        extrap_mask = (x_vals < p10) | (x_vals > p90)
        interp_mask = ~extrap_mask

        interp_records.append({
            'skeleton': row['skeleton'],
            'equation': row['equation'],
            'X': x_vals[interp_mask].tolist(),
            'Y': y_vals[interp_mask].tolist()
        })

        extrap_records.append({
            'skeleton': row['skeleton'],
            'equation': row['equation'],
            'X': x_vals[extrap_mask].tolist(),
            'Y': y_vals[extrap_mask].tolist()
        })

    # Save split Parquet files
    data_dir = os.path.dirname(parquet_path)
    interp_parquet_path = os.path.join(data_dir, f"interpolation_datasets_{difficulty_name}.parquet")
    extrap_parquet_path = os.path.join(data_dir, f"extrapolation_datasets_{difficulty_name}.parquet")

    pd.DataFrame(interp_records).to_parquet(interp_parquet_path, engine='pyarrow', compression='snappy')
    pd.DataFrame(extrap_records).to_parquet(extrap_parquet_path, engine='pyarrow', compression='snappy')
    print(f"✓ Saved percentile-split datasets:\n  - {interp_parquet_path}\n  - {extrap_parquet_path}")

    # Plot the combined grids
    plots_config = [
        {"range": (0, 20), "grid": (5, 4), "suffix": "1_to_20"},
        {"range": (20, 40), "grid": (5, 4), "suffix": "21_to_40"},
        {"range": (40, 50), "grid": (4, 3), "suffix": "41_to_50"}
    ]

    df_interp = pd.DataFrame(interp_records)
    df_extrap = pd.DataFrame(extrap_records)

    for p_cfg in plots_config:
        start_idx, end_idx = p_cfg["range"]
        rows, cols = p_cfg["grid"]

        sub_interp = df_interp.iloc[start_idx:end_idx].reset_index(drop=True)
        sub_extrap = df_extrap.iloc[start_idx:end_idx].reset_index(drop=True)
        num_items = len(sub_interp)

        if num_items == 0:
            continue

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 3.5 * rows), dpi=200)
        axes_flat = axes.flatten()

        for idx in range(rows * cols):
            ax = axes_flat[idx]

            # Hide extra grid boxes
            if idx >= num_items:
                ax.axis('off')
                continue

            row_int = sub_interp.iloc[idx]
            row_ext = sub_extrap.iloc[idx]
            equation_id = f"{difficulty_name[0].upper()}{start_idx + idx + 1}"

            x_int, y_int = np.array(row_int['X']), np.array(row_int['Y'])
            x_ext, y_ext = np.array(row_ext['X']), np.array(row_ext['Y'])

            # Plot extrapolation first (orange), then interpolation (blue)
            ax.scatter(x_ext, y_ext, color='darkorange', alpha=0.7, edgecolors='none', s=10, label='Extrapolation')
            ax.scatter(x_int, y_int, color='royalblue', alpha=0.7, edgecolors='none', s=10, label='Interpolation')

            ax.set_title(f"{equation_id}", fontsize=10, pad=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()

        filename = f"../plots/split_grid_{difficulty_name}_{p_cfg['suffix']}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved plot grid: {filename}")


# Run generation on all three splits
if __name__ == "__main__":
    os.makedirs("../plots", exist_ok=True)

    difficulties = ["easy", "medium", "hard"]

    for diff in difficulties:
        curated_file = f"../data/curated_datasets_{diff}.parquet"
        if os.path.exists(curated_file):
            # Option 1: Plot original standard grid
            # generate_difficulty_grids(curated_file, diff)

            # Option 2: Split into interpolation/extrapolation datasets and plot combined
            create_split_datasets_and_grids(curated_file, diff)
