import sympy as sp
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from config import UNARY_OPERATORS
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import os
from src.symbolic.reducer import absorb_constants

# Set professional academic plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Define global canonical symbols
C, L, P = sp.symbols('C L P')

# ==========================================
# ALLOWED OPERATORS WHITELIST & HELPER
# ==========================================
ALLOWED_OPERATORS = {
    'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'sinh', 'cosh',
    'tanh', 'asin', 'atan', 'asinh', 'acosh', 'atanh', 'square',
    'cube', 'pow', 'powu'
}


def get_clean_op_name(expr):
    """
    Returns standardized operator name if expr.func is in ALLOWED_OPERATORS.
    Returns None for binary ops (Add, Mul), constants, symbols, and excluded ops like 'inv'.
    """
    if not expr or not hasattr(expr, 'func'):
        return None
    if expr.func in (sp.Symbol, sp.Number, sp.Add, sp.Mul):
        return None

    func_str = getattr(expr.func, '__name__', str(expr.func)).lower()

    # Standardize common aliases
    if func_str in ('abs', 'absolute'):
        return 'abs'
    if func_str in ('powu', 'pow', 'power'):
        return 'pow'

    if func_str in ALLOWED_OPERATORS:
        return func_str

    return None


# ==========================================
# 1. STRUCTURAL METRICS EXTRACTION
# ==========================================

def get_unary_nesting_depth(expr):
    """
    Computes maximum nesting depth considering ONLY allowed operators.
    """
    if not expr or not expr.args:
        return 0

    op_name = get_clean_op_name(expr)
    is_allowed = (op_name is not None)

    child_depths = [get_unary_nesting_depth(arg) for arg in expr.args]
    max_child_depth = max(child_depths) if child_depths else 0

    return (1 + max_child_depth) if is_allowed else max_child_depth


def count_operators(expr):
    """
    Counts total occurrences of ONLY allowed operators in the expression tree.
    """
    if not expr or not expr.args:
        return 0

    op_name = get_clean_op_name(expr)
    current_count = 1 if op_name else 0

    return current_count + sum(count_operators(arg) for arg in expr.args)


def extract_operators_list(expr):
    """
    Extracts the unique set of allowed operators present in the expression tree.
    """
    ops = set()
    op_name = get_clean_op_name(expr)
    if op_name:
        ops.add(op_name)
    for arg in expr.args:
        ops.update(extract_operators_list(arg))
    return ops


# ==========================================
# 2. DATASET LOADING & WORKER HELPERS
# ==========================================

def load_skeletons_from_source(path):
    """Loads skeletons from either a .parquet file or a plain text (.txt) file."""
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
        if 'skeleton' not in df.columns:
            raise ValueError(f"Parquet file {path} must contain a 'skeleton' column.")
        return df['skeleton'].tolist()
    else:
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]


def _process_lines_chunk(lines_chunk):
    """Processes a batch of expression strings in a separate worker process."""
    local_freqs = Counter()
    local_depths = []
    local_ops = []
    local_vocab = set()
    local_unary = Counter()

    local_dict = {
        'C': C, 'L': L, 'P': P,
        'powu': sp.Function('powu'),
        'square': sp.Function('square'),
        'cube': sp.Function('cube'),
        'inv': sp.Function('inv'),
        'Abs': sp.Abs
    }

    for line in lines_chunk:
        try:
            expr = sp.parse_expr(line, local_dict=local_dict, evaluate=False)
        except Exception:
            continue

        expr_str = str(expr)
        local_freqs[expr_str] += 1

        local_depths.append(get_unary_nesting_depth(expr))
        op_counts = count_operators(expr)
        if op_counts > 100:
            print(expr)
            print(absorb_constants(sp.simplify(expr)))
        local_ops.append(op_counts)

        expr_ops = extract_operators_list(expr)
        local_vocab.update(expr_ops)
        for op in expr_ops:
            local_unary[op] += 1

    return local_freqs, local_depths, local_ops, local_vocab, local_unary


# ==========================================
# 3. PARALLELIZED DATASET ANALYSIS WITH CACHING
# ==========================================

def analyze_dataset(name, path, chunk_size=5000, num_workers=None):
    """
    Parses a dataset file in parallel. Uses a cache file (.pkl) if available
    to avoid re-running expensive parsing on subsequent calls.
    """
    cache_path = f"{name}.analysis_cache.pkl"
    if os.path.exists(cache_path):
        print(f"-> Loading cached analysis for {name} from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    raw_lines = load_skeletons_from_source(path)
    total_expressions = len(raw_lines)

    if total_expressions == 0:
        empty_res = ({}, Counter(), [], [], Counter())
        return empty_res

    chunks = [raw_lines[i:i + chunk_size] for i in range(0, total_expressions, chunk_size)]

    frequencies = Counter()
    depths = []
    op_counts = []
    all_vocabulary = set()
    unary_operator_counter = Counter()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_lines_chunk, chunk): len(chunk) for chunk in chunks}

        with tqdm(total=total_expressions, desc=f"Analyzing {path.split('/')[-1]}") as pbar:
            for future in as_completed(futures):
                chunk_len = futures[future]
                try:
                    loc_freqs, loc_depths, loc_ops, loc_vocab, loc_unary = future.result()
                    frequencies.update(loc_freqs)
                    depths.extend(loc_depths)
                    op_counts.extend(loc_ops)
                    all_vocabulary.update(loc_vocab)
                    unary_operator_counter.update(loc_unary)
                except Exception as e:
                    print(f"\nError processing chunk: {e}")
                pbar.update(chunk_len)

    total_parsed = sum(frequencies.values())

    stats = {
        "Total Expressions": total_expressions,
        "Unique Skeletons": len(frequencies),
        "Duplication Rate (%)": (1 - (len(frequencies) / total_parsed)) * 100 if total_parsed > 0 else 0,
        "Dictionary Size (Unique Ops)": len(all_vocabulary),
        "Mean Unary Nesting Depth": np.mean(depths) if depths else 0,
        "Max Unary Nesting Depth": np.max(depths) if depths else 0,
        "Mean Operators per Expression": np.mean(op_counts) if op_counts else 0,
        "Max Operators per Expression": np.max(op_counts) if op_counts else 0
    }

    result = (stats, frequencies, depths, op_counts, unary_operator_counter)

    # Save to cache
    print(f"-> Saving analysis cache to {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)

    return result


# ==========================================
# 4. MULTI-DATASET COMPARATIVE PLOTTING
# ==========================================

def compare_datasets(*dataset_tuples):
    """
    Compares one or more datasets passed as tuples of (name, path).
    """
    results = {}
    for name, path in dataset_tuples:
        print(f"\nProcessing '{name}' from {path}...")
        stats, freqs, depths, ops, unary = analyze_dataset(name, path)
        results[name] = {
            'stats': stats,
            'freqs': freqs,
            'depths': depths,
            'ops': ops,
            'unary': unary
        }

    # Print Quantitative Summary Table
    print("\n" + "=" * 80)
    print("QUANTITATIVE METRICS COMPARISON SUMMARY")
    print("=" * 80)
    metrics_keys = list(next(iter(results.values()))['stats'].keys())
    header = f"{'Metric':<35} | " + " | ".join([f"{name[:15]:<15}" for name, _ in dataset_tuples])
    print(header)
    print("-" * len(header))

    for key in metrics_keys:
        row_str = f"{key:<35} | "
        vals = []
        for name, _ in dataset_tuples:
            val = results[name]['stats'][key]
            if isinstance(val, float):
                vals.append(f"{val:<15.2f}")
            else:
                vals.append(f"{val:<15}")
        print(row_str + " | ".join(vals))
    print("=" * 80 + "\n")

    # Generate 2-Column Multi-Dataset Comparison Plot with a narrower left column and reduced column gap
    fig = plt.figure(figsize=(6.7, 4.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.75, 1.25], wspace=0.15, hspace=0.35)

    axes = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ])

    palette = sns.color_palette("tab10", len(dataset_tuples))

    # --- Panel A: Zipfian Frequency Skew ---
    ax1 = axes[0, 0]
    for idx, (name, _) in enumerate(dataset_tuples):
        sorted_counts = sorted(results[name]['freqs'].values(), reverse=True)
        ax1.plot(range(1, len(sorted_counts) + 1), sorted_counts, label=name, color=palette[idx], lw=1.5)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_title('(a) Skeleton Frequency Rank', fontsize=9, fontweight='bold', pad=2)
    ax1.set_xlabel('Unique Rank (Log)', fontsize=8)
    ax1.set_ylabel('Freq. (Log)', fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.legend(frameon=True, fontsize=7, loc='upper right')

    # --- Panel B: Unary Nesting Depth Distribution ---
    ax2 = axes[0, 1]
    all_depths = [d for data in results.values() for d in data['depths']]
    max_d = max(all_depths, default=0)
    bins_d = np.arange(-0.5, max_d + 1.5, 1)
    for idx, (name, _) in enumerate(dataset_tuples):
        ax2.hist(results[name]['depths'], bins=bins_d, density=True, alpha=0.5, label=name, color=palette[idx],
                 edgecolor='black', linewidth=0.6)
    ax2.set_title('(b) Unary Nesting Depth', fontsize=9, fontweight='bold', pad=2)
    ax2.set_xlabel('Nesting Depth', fontsize=8)
    ax2.set_ylabel('Density', fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.legend(frameon=True, fontsize=7)

    # --- Panel C: Operators per Expression Distribution (Capped to dense region) ---
    ax3 = axes[1, 0]
    all_ops = [o for data in results.values() for o in data['ops']]
    if all_ops:
        upper_limit = int(np.percentile(all_ops, 99.5))
        upper_limit = max(upper_limit, 10)  # Tightly bounded for accurate 0-10 range
        bins_o = np.arange(-0.5, upper_limit + 1.5, 1)

        for idx, (name, _) in enumerate(dataset_tuples):
            ax3.hist(results[name]['ops'], bins=bins_o, density=True, alpha=0.5, label=name, color=palette[idx],
                     edgecolor='black', linewidth=0.6)
        ax3.set_xlim(-0.5, upper_limit + 0.5)

    ax3.set_title('(c) Operators per Expression', fontsize=9, fontweight='bold', pad=2)
    ax3.set_xlabel('Operator Count', fontsize=8)
    ax3.set_ylabel('Density', fontsize=8)
    ax3.tick_params(labelsize=7)
    ax3.legend(frameon=True, fontsize=7)

    # --- Panel D: Unary Operator Proportions ---
    ax4 = axes[1, 1]

    all_unary_operators = sorted(list(set().union(*(data['unary'].keys() for data in results.values()))))
    x = np.arange(len(all_unary_operators))
    width = 0.8 / len(dataset_tuples)

    for idx, (name, _) in enumerate(dataset_tuples):
        unary_counts = results[name]['unary']
        total = sum(unary_counts.values()) or 1

        props = [unary_counts.get(op, 0) / total for op in all_unary_operators]
        ax4.bar(x + (idx - len(dataset_tuples) / 2 + 0.5) * width, props, width, label=name, color=palette[idx])

    ax4.set_title('(d) Unary Operator Proportions', fontsize=9, fontweight='bold', pad=2)
    ax4.set_xlabel('Operator Type', fontsize=8)
    ax4.set_ylabel('Proportion', fontsize=8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(all_unary_operators, rotation=45, ha='right', fontsize=7)
    ax4.tick_params(labelsize=7)
    ax4.legend(frameon=True, fontsize=7)
    # Save figure as high-resolution JPEG
    plt.savefig('multi_dataset_comparison.jpg', format='jpg', dpi=300, bbox_inches='tight')
    print("\nComparison plots successfully saved as 'multi_dataset_comparison.jpg'.")
    plt.show()


# --- Example Usage ---
if __name__ == '__main__':
    dataset_path = '../data/canonical_skeletons.parquet'
    dataset_name = 'Ours'
    dataset2_path = 'SeTGAP_skeletons_canonicalized.txt'
    dataset2_name = 'SeTGAP'
    dataset3_path = 'NeSymRes_skeletons_canonicalized.txt'
    dataset3_name = 'NeSymRes'

    compare_datasets((dataset_name, dataset_path), (dataset2_name, dataset2_path), (dataset3_name, dataset3_path))