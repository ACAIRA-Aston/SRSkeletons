import os
import numpy as np
import pandas as pd
import sympy as sp
from tqdm import tqdm


def sample_extrapolation_domain(n_samples=5000):
    """
    Samples points uniformly across the domain [-12.5, -10) U (10, 12.5].
    Splits samples equally between the left and right intervals and returns them sorted.
    """
    half_samples = n_samples // 2

    # Sample from [-12.5, -10.0)
    left_interval = np.random.uniform(-12.5, -10.0, size=half_samples)
    # Sample from (10.0, 12.5]
    right_interval = np.random.uniform(10.0, 12.5, size=n_samples - half_samples)

    # Combine and sort for clean order
    x_ext = np.sort(np.concatenate([left_interval, right_interval]))
    return x_ext


def evaluate_equation(eq_str, x_vals):
    """
    Parses a string equation using SymPy and evaluates it over a NumPy vector.
    """
    x = sp.Symbol('x')
    expr = sp.sympify(eq_str)

    # result = modify_constants_avoidNaNs(new_expr, support, bounded_operations(), curr_p, Xbounds,
    #                                     variable=varX, extrapolate=extrapolate)
    # support, new_expr, _, _ = result

    func = sp.lambdify(x, expr, modules=['numpy'])

    with np.errstate(all='ignore'):
        y_vals = func(x_vals)
        if np.isscalar(y_vals):
            y_vals = np.full_like(x_vals, y_vals, dtype=float)

    return y_vals


def generate_extrapolation_datasets(
        difficulties=('easy', 'medium', 'hard'),
        data_dir='../data',
        n_samples=5000
):
    """
    Loads curated parquet files per difficulty, generates extrapolation data (5000 samples),
    and saves the complementary extrapolation dataset.
    """
    os.makedirs(data_dir, exist_ok=True)

    for diff in difficulties:
        input_path = os.path.join(data_dir, f'curated_datasets_{diff}.parquet')
        output_path = os.path.join(data_dir, f'extrapolation_datasets_{diff}.parquet')

        if not os.path.exists(input_path):
            print(f"⚠ Warning: File not found at {input_path}. Skipping '{diff}' split.")
            continue

        print(f"\n=======================================================")
        print(f"  Generating Extrapolation Dataset: {diff.upper()} Split")
        print(f"=======================================================")

        df_curated = pd.read_parquet(input_path)
        extrapolation_records = []

        for idx, row in tqdm(df_curated.iterrows(), total=len(df_curated), desc=f"Processing {diff}"):
            skeleton = row['skeleton']
            eq_str = row['equation']

            # Generate domain points
            X_ext = sample_extrapolation_domain(n_samples=n_samples)

            try:
                # Evaluate Y values over domain
                Y_ext = evaluate_equation(eq_str, X_ext)

                extrapolation_records.append({
                    'skeleton': skeleton,
                    'equation': eq_str,
                    'X': X_ext.tolist(),
                    'Y': Y_ext.tolist()
                })
            except Exception as e:
                print(f"\n❌ Error evaluating skeleton '{skeleton}' with equation '{eq_str}': {e}")

        # Save to parquet
        df_extrapolation = pd.DataFrame(extrapolation_records)
        df_extrapolation.to_parquet(output_path, engine='pyarrow', compression='snappy')
        print(f"✓ Saved {len(df_extrapolation)} records to: {output_path}")


if __name__ == '__main__':
    # Set seed for reproducibility across extrapolation splits if needed
    np.random.seed(42)

    # Run extrapolation generator for all difficulties
    generate_extrapolation_datasets(difficulties=['easy', 'medium', 'hard'])