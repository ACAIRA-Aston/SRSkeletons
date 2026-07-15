import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import re
import time
import config
import pandas as pd
from src.grammar.cfg import StructuralGrammar
from src.pipeline import generate_exhaustive_skeletons
from src.symbolic.reducer import inv, cube_func, square_func
from sympy import sympify, symbols, parse_expr, Function, Pow, exp


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

    df = pd.DataFrame(list(unique_skeletons), columns=['skeleton'])
    df['skeleton_length'] = df['skeleton'].str.len()
    df.sort_values(by=["skeleton_length"])
    df.to_parquet(config.OUTPUT_PATH, engine='pyarrow', compression='snappy')

    # Clean dataset
    print("Cleaning dataset")
    LOCAL_DICT = {'inv': inv, 'powu': Function('powu'), 'cube': cube_func, 'square': square_func}
    unique_skeletons = []

    skeletons = pd.read_parquet('../data/canonical_skeletons.parquet')

    for sk in range(skeletons.shape[0]):
        skeleton = skeletons.iloc[sk].skeleton
        skeleton = skeleton.replace('logg', 'log')
        skeleton = skeleton.replace('sqrt2', 'sqrt')
        expr = parse_expr(skeleton, local_dict=LOCAL_DICT, evaluate=False)
        flag = False
        powu_terms = expr.atoms(Function('powu'))
        for term in powu_terms:
            arg = term.args[0]
            arg_str = str(arg)
            if re.search(r'\bP\b', arg_str) and (not re.search(r'\(P\)', arg_str)) and (not re.search(r'\/P', arg_str)):
                flag = True
                break

        if not flag:
            unique_skeletons.append(skeleton)
        else:
            pass

    print(f"Saving to {config.OUTPUT_PATH}...")
    df = pd.DataFrame(unique_skeletons, columns=['skeleton'])
    df['skeleton_length'] = df['skeleton'].str.len()
    df.sort_values(by=["skeleton_length"])
    df.to_parquet(config.OUTPUT_PATH, engine='pyarrow', compression='snappy')

    print("Success!")


if __name__ == "__main__":
    main()