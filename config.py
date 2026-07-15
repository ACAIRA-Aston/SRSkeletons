"""
Global configuration for the Symbolic Skeleton Generator.
N_U and N_D could be adjusted to control search space explosion.
"""
import os

# Constraints
N_U = 2        # Total number of unary operators allowed in a skeleton
N_D = 2        # Maximum nesting depth of unary operators

# Unary Operators
# Includes standard, trigonometric, hyperbolic, and specialized functions
UNARY_OPERATORS = [
    'sin', 'tan', 'exp', 'logg', 'sqrt2', 'abs', 'sinh', 'cosh', 'tanh', 'asin', 'atan', 'asinh', 'acosh', 'atanh',
     'square', 'cube', 'powu', 'inv'
]

# --- Storage Settings ---
OUTPUT_DIR = "src/data"
OUTPUT_FILENAME = "canonical_skeletons.parquet"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
OUTPUT_PATH2 = os.path.join(OUTPUT_DIR, "canonical_skeletons.parquet")

# Create directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# --- SAMPLING ---
multiplicative_min = -5
multiplicative_max = 5
additive_min = -5
additive_max = 5
max_number_of_points = 10000
fun_support_min = -10
fun_support_max = 10
number_of_sets = 10