import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import time
from src.grammar.cfg import StructuralGrammar
from src.pipeline import generate_exhaustive_skeletons
import config


def main():
    print("--- Starting Exhaustive Skeleton Generation ---")
    print(f"Constraints: N_U <= {config.N_U}, N_D <= {config.N_D}")
    print(f"Operators: {len(config.UNARY_OPERATORS)} unary types loaded.")

    # Initialize the Grammar with configured operators
    grammar = StructuralGrammar(unary_operators=config.UNARY_OPERATORS)

    # Execute the Generation Pipeline
    start_time = time.time()
    unique_skeletons = generate_exhaustive_skeletons(grammar=grammar, n_u=config.N_U, n_d=config.N_D, unary_ops=config.UNARY_OPERATORS)
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nGeneration complete!")
    print(f"Total Unique Skeletons: {len(unique_skeletons):,}")
    print(f"Time Elapsed: {duration:.2f} seconds")

    print(f"Saving to {config.OUTPUT_PATH}...")
    df = pd.DataFrame(list(unique_skeletons), columns=['skeleton'])
    df['skeleton_length'] = df['skeleton'].str.len()
    df.sort_values(by=["skeleton_length"])
    df.to_parquet(config.OUTPUT_PATH, engine='pyarrow', compression='snappy')

    print("Success!")


if __name__ == "__main__":
    main()