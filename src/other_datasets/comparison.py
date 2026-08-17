import sympy as sp
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import UNARY_OPERATORS

# Set professional academic plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Define global canonical symbols and unary operator vocabulary
C, L, P = sp.symbols('C L P')

# ==========================================
# 1. STRUCTURAL METRICS EXTRACTION
# ==========================================

def get_unary_nesting_depth(expr):
    """
    Computes the maximum nesting depth of unary operators in the expression tree.
    Unary operators (functions like sin, log, exp, square, etc.) increment depth,
    while binary/n-ary operations (Add, Mul, Pow) pass through the maximum child depth.
    """
    if not expr or not expr.args:
        return 0

    is_unary = False
    if expr.func not in (sp.Symbol, sp.Number, sp.Add, sp.Mul, sp.Pow):
        func_name = str(expr.func)
        if func_name in UNARY_OPERATORS or not getattr(expr.func, 'is_commutative', True):
            is_unary = True

    child_depths = [get_unary_nesting_depth(arg) for arg in expr.args]
    max_child_depth = max(child_depths) if child_depths else 0

    return (1 + max_child_depth) if is_unary else max_child_depth


def count_operators(expr):
    if not expr.args:
        return 0
    op_count = 1 if not isinstance(expr, (sp.Symbol, sp.Number)) else 0
    return op_count + sum(count_operators(arg) for arg in expr.args)


def extract_operators_list(expr):
    ops = set()
    if expr.func not in (sp.Symbol, sp.Number, sp.Add, sp.Mul, sp.Pow):
        ops.add(str(expr.func))
    for arg in expr.args:
        ops.update(extract_operators_list(arg))
    return ops


# ==========================================
# 2. DATASET PARSING & ANALYSIS
# ==========================================

def load_skeletons_from_source(path):
    """Loads skeletons from either a .parquet file or a plain text (.txt) file."""
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
        if 'skeleton' not in df.columns:
            raise ValueError(f"Parquet file {path} must contain a 'skeletons' column.")
        return df['skeleton'].tolist()
    else:
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]


def analyze_dataset(path):
    """
    Parses a dataset file (parquet or txt) and extracts frequencies, depths, and operator stats.
    """
    raw_lines = load_skeletons_from_source(path)

    total_expressions = 0
    frequencies = Counter()
    depths = []
    op_counts = []
    all_vocabulary = set()
    unary_operator_counter = Counter()

    local_dict = {
        'C': C, 'L': L, 'P': P,
        'powu': sp.Function('powu'),
        'square': sp.Function('square'),
        'cube': sp.Function('cube'),
        'inv': sp.Function('inv'),
        'Abs': sp.Abs
    }

    for line in raw_lines:
        total_expressions += 1
        try:
            expr = sp.parse_expr(line, local_dict=local_dict, evaluate=False)
        except Exception:
            continue

        expr_str = str(expr)
        frequencies[expr_str] += 1

        depths.append(get_unary_nesting_depth(expr))
        op_counts.append(count_operators(expr))

        expr_ops = extract_operators_list(expr)
        all_vocabulary.update(expr_ops)
        for op in expr_ops:
            unary_operator_counter[op] += 1

    stats = {
        "Total Expressions": total_expressions,
        "Unique Skeletons": len(frequencies),
        "Duplication Rate (%)": (1 - (len(frequencies) / total_expressions)) * 100 if total_expressions > 0 else 0,
        "Dictionary Size (Unique Ops)": len(all_vocabulary),
        "Mean Unary Nesting Depth": np.mean(depths) if depths else 0,
        "Max Unary Nesting Depth": np.max(depths) if depths else 0,
        "Mean Operators per Expression": np.mean(op_counts) if op_counts else 0,
        "Max Operators per Expression": np.max(op_counts) if op_counts else 0
    }

    return stats, frequencies, depths, op_counts, unary_operator_counter


# ==========================================
# 3. MULTI-DATASET COMPARATIVE PLOTTING
# ==========================================

def compare_datasets(*dataset_tuples):
    """
    Compares one or more datasets passed as tuples of (name, path).
    Example: compare_datasets(('SeTGAP', 'path/to/setgap.txt'), ('Ours', 'path/to/ours.parquet'))
    """
    results = {}
    for name, path in dataset_tuples:
        print(f"Analyzing '{name}' from {path}...")
        stats, freqs, depths, ops, unary = analyze_dataset(path)
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

    # Generate 4-Panel Multi-Dataset Comparison Plot
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    palette = sns.color_palette("tab10", len(dataset_tuples))

    # --- Panel A: Zipfian Frequency Skew ---
    ax1 = axes[0, 0]
    for idx, (name, _) in enumerate(dataset_tuples):
        sorted_counts = sorted(results[name]['freqs'].values(), reverse=True)
        ax1.plot(range(1, len(sorted_counts) + 1), sorted_counts, label=name, color=palette[idx], lw=2.5)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_title('(a) Skeleton Frequency Rank (Zipfian Skew)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Unique Skeleton Rank (Log)')
    ax1.set_ylabel('Generation Frequency (Log)')
    ax1.legend(frameon=True)

    # --- Panel B: Unary Nesting Depth Distribution ---
    ax2 = axes[0, 1]
    all_depths = [d for data in results.values() for d in data['depths']]
    max_d = max(all_depths, default=0)
    bins_d = np.arange(-0.5, max_d + 1.5, 1)
    for idx, (name, _) in enumerate(dataset_tuples):
        ax2.hist(results[name]['depths'], bins=bins_d, density=True, alpha=0.5, label=name, color=palette[idx],
                 edgecolor='black')
    ax2.set_title('(b) Distribution of Unary Nesting Depth', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Unary Nesting Depth')
    ax2.set_ylabel('Normalized Density')
    ax2.legend(frameon=True)

    # --- Panel C: Operators per Expression Distribution ---
    ax3 = axes[1, 0]
    all_ops = [o for data in results.values() for o in data['ops']]
    max_o = max(all_ops, default=0)
    bins_o = np.arange(-0.5, max_o + 1.5, 1)
    for idx, (name, _) in enumerate(dataset_tuples):
        ax3.hist(results[name]['ops'], bins=bins_o, density=True, alpha=0.5, label=name, color=palette[idx],
                 edgecolor='black')
    ax3.set_title('(c) Distribution of Operators per Expression', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Total Operator Count')
    ax3.set_ylabel('Normalized Density')
    ax3.legend(frameon=True)

    # --- Panel D: Unary Operator Proportions ---
    ax4 = axes[1, 1]
    all_UNARY_OPERATORS = sorted(list(set().union(*(data['unary'].keys() for data in results.values()))))
    x = np.arange(len(all_UNARY_OPERATORS))
    width = 0.8 / len(dataset_tuples)

    for idx, (name, _) in enumerate(dataset_tuples):
        unary_counts = results[name]['unary']
        total = sum(unary_counts.values()) or 1
        props = [unary_counts.get(op, 0) / total for op in all_UNARY_OPERATORS]
        ax4.bar(x + (idx - len(dataset_tuples) / 2 + 0.5) * width, props, width, label=name, color=palette[idx])

    ax4.set_title('(d) Unary Operator Proportions', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Unary Operator Type')
    ax4.set_ylabel('Relative Proportion')
    ax4.set_xticks(x)
    ax4.set_xticklabels(all_UNARY_OPERATORS, rotation=45, ha='right')
    ax4.legend(frameon=True)

    plt.tight_layout()
    plt.savefig('multi_dataset_comparison.pdf', format='pdf', bbox_inches='tight')
    print("Comparison plots successfully saved as 'multi_dataset_comparison.pdf'.")
    plt.show()


# --- Example Usage ---
if __name__ == '__main__':
    dataset1_path = '../data/canonical_skeletons.parquet'
    dataset1_name = 'Ours'
    dataset2_path = 'SeTGAP_skeletons_canonicalized.txt'
    dataset2_name = 'SeTGAP'

    compare_datasets((dataset1_name, dataset1_path), (dataset2_name, dataset2_path))
