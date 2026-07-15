import pandas as pd


def create_difficulty_datasets(input_path="../data/canonical_skeletons.parquet", output_dir="../data/", target_size=50,
                               shortest_count=25, random_state=42):
    # Load and ensure it is sorted by skeleton length
    df = pd.read_parquet(input_path)
    if 'skeleton_length' not in df.columns:
        df['skeleton_length'] = df['skeleton'].str.len()

    df_sorted = df.sort_values(by=["skeleton_length"]).reset_index(drop=True)
    n = len(df_sorted)

    # Split into 3 equal complexity buckets based on string length
    splits = {
        'easy': df_sorted.iloc[:n // 3].copy(),
        'medium': df_sorted.iloc[n // 3: 2 * (n // 3)].copy(),
        'hard': df_sorted.iloc[2 * (n // 3):].copy()
    }

    for difficulty, split_df in splits.items():
        # Make sure the split itself is sorted by length to easily pull the shortest
        split_sorted = split_df.sort_values(by=["skeleton_length"])

        # Determine slice boundaries safely (in case a partition is unexpectedly small)
        n_shortest = min(shortest_count, len(split_sorted))
        n_random = min(target_size - n_shortest, len(split_sorted) - n_shortest)

        # 1. Take the absolute shortest equations from this specific partition
        shortest_df = split_sorted.iloc[:n_shortest]

        # 2. Sample the remaining equations from the rest of the pool (no duplicates)
        remaining_df = split_sorted.iloc[n_shortest:]
        random_df = remaining_df.sample(n=n_random, random_state=random_state)

        # 3. Combine them and shuffle so the 10 shortest are dispersed naturally
        combined_df = pd.concat([shortest_df, random_df]).sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Save to parquet
        output_path = f"{output_dir}canonical_skeletons_{difficulty}.parquet"
        combined_df.to_parquet(output_path, engine='pyarrow', compression='snappy')

        print(f"Saved {len(combined_df)} '{difficulty}' skeletons (25 shortest + {len(random_df)} random) to {output_path}")


# Run the updated partition logic
create_difficulty_datasets()
