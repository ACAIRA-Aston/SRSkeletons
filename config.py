"""
Global configuration for the Symbolic Skeleton Generator.
N_U and N_D could be adjusted to control search space explosion.
"""
import os

# Constraints
N_U = 3        # Total number of unary operators allowed in a skeleton
N_D = 2        # Maximum nesting depth of unary operators

# Unary Operators
# Includes standard, trigonometric, hyperbolic, and specialized functions
UNARY_OPERATORS = [
    'square', 'sin' #'sin', 'tan', 'exp', 'logg', 'sqrt', 'abs', 'sinh', 'cosh', 'tanh', 'asin', 'atan', 'asinh', 'acosh', 'atanh',
     # 'square', 'cube', 'powu', 'inv'
]

# --- Storage Settings ---
OUTPUT_DIR = "results"
OUTPUT_FILENAME = "canonical_skeletons.parquet"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# Create directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
