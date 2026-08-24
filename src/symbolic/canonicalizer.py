import sympy as sp
from functools import partial
import multiprocessing as mp
from collections import defaultdict
from src.symbolic.normalizer import canonical_sort
from src.utils.skeletons import get_skeleton_var

# Define canonical symbols
C, L, P = sp.symbols('C L P')


# ==========================================
# 1. POLYNOMIAL ALGEBRA & DEGREE HELPERS
# ==========================================

def get_poly_degree(node):
    """
    Returns the integer polynomial degree of `node` if it is a pure polynomial in x,
    or None if it contains non-polynomial functions/operations.
    """
    if not node.args:
        node_str = str(node)
        if node_str.startswith(('ca', 'cm')) or node.is_Number or node == C:
            return 0
        elif node_str.startswith('x') or node == L:
            return 1
        elif node == P:
            return 2  # P represents degree >= 2
        return None

    if isinstance(node, sp.Add):
        degrees = [get_poly_degree(arg) for arg in node.args]
        if any(d is None for d in degrees):
            return None
        return max(degrees)

    if isinstance(node, sp.Mul):
        degrees = [get_poly_degree(arg) for arg in node.args]
        if any(d is None for d in degrees):
            return None
        return sum(degrees)

    if isinstance(node, sp.Pow):
        base, exp = node.args
        base_deg = get_poly_degree(base)
        if base_deg is None:
            return None
        if exp.is_Integer and exp >= 0:
            return base_deg * int(exp)
        return None

    return None


def poly_to_deg(p):
    """Maps canonical polynomial terms to integer degrees."""
    if p == C:
        return 0
    elif p == L:
        return 1
    elif p == P:
        return 2
    elif p.is_Number or p == sp.Integer(1):
        return 0
    return 0


def add_poly_terms(poly_list):
    """
    Combines polynomial terms under addition following closure rules:
    C + C -> C | C + L -> L | L + L -> L | L + P -> P | P + P -> P
    """
    if not poly_list:
        return C

    degs = [poly_to_deg(p) for p in poly_list]
    max_deg = max(degs)

    if max_deg == 0:
        if any(p == C for p in poly_list) or len(poly_list) > 1:
            return C
        return sp.Integer(1)
    elif max_deg == 1:
        return L
    else:
        return P


def multiply_poly_terms(poly_list):
    """
    Combines polynomial terms under multiplication following closure rules:
    C * C -> C | C * L -> L | L * L -> P | C * P -> P
    """
    if not poly_list:
        return sp.Integer(1)

    degs = [poly_to_deg(p) for p in poly_list]
    total_deg = sum(degs)

    if total_deg == 0:
        if any(p == C for p in poly_list):
            return C
        return sp.Integer(1)
    elif total_deg == 1:
        return L
    else:
        return P


def extract_poly_and_nonpoly(node):
    """
    Decomposes a canonicalized node into (poly_factor, non_poly_factor).
    poly_factor is one of: 1, C, L, P.
    non_poly_factor is either 1 or a non-polynomial SymPy expression.
    """
    if node in (C, L, P):
        return node, sp.Integer(1)

    if node.is_Number or node == sp.Integer(1):
        return sp.Integer(1), sp.Integer(1)

    if isinstance(node, sp.Mul):
        poly_list = []
        non_poly_list = []
        for arg in node.args:
            p, np_factor = extract_poly_and_nonpoly(arg)
            if p != 1:
                poly_list.append(p)
            if np_factor != 1:
                if isinstance(np_factor, sp.Mul):
                    non_poly_list.extend(np_factor.args)
                else:
                    non_poly_list.append(np_factor)

        combined_poly = multiply_poly_terms(poly_list)
        if not non_poly_list:
            combined_non_poly = sp.Integer(1)
        elif len(non_poly_list) == 1:
            combined_non_poly = non_poly_list[0]
        else:
            combined_non_poly = sp.Mul(*non_poly_list, evaluate=False)

        return combined_poly, combined_non_poly

    deg = get_poly_degree(node)
    if deg is not None:
        poly_val = C if deg == 0 else (L if deg == 1 else P)
        return poly_val, sp.Integer(1)
    else:
        return sp.Integer(1), node


# ==========================================
# 2. CANONICAL COMBINERS
# ==========================================

def _combine_add(args):
    """
    Flattens additions, groups terms by non-polynomial factors, and applies
    polynomial closure rules to the grouped coefficients.
    """
    flat_args = []
    for arg in args:
        if isinstance(arg, sp.Add):
            flat_args.extend(arg.args)
        else:
            flat_args.append(arg)

    groups = defaultdict(list)
    for arg in flat_args:
        p, np_factor = extract_poly_and_nonpoly(arg)
        groups[np_factor].append(p)

    final_terms = []
    for np_factor, poly_list in groups.items():
        combined_poly = add_poly_terms(poly_list)

        if np_factor == 1:
            final_terms.append(combined_poly)
        elif combined_poly == 1:
            final_terms.append(np_factor)
        else:
            final_terms.append(sp.Mul(combined_poly, np_factor, evaluate=False))

    if not final_terms:
        return C
    if len(final_terms) == 1:
        return final_terms[0]
    return sp.Add(*final_terms, evaluate=False)


def _combine_mul(args):
    """
    Flattens multiplications and combines polynomial factors.
    """
    flat_args = []
    for arg in args:
        if isinstance(arg, sp.Mul):
            flat_args.extend(arg.args)
        else:
            flat_args.append(arg)

    poly_list = []
    non_poly_list = []
    for arg in flat_args:
        p, np_factor = extract_poly_and_nonpoly(arg)
        if p != 1:
            poly_list.append(p)
        if np_factor != 1:
            if isinstance(np_factor, sp.Mul):
                non_poly_list.extend(np_factor.args)
            else:
                non_poly_list.append(np_factor)

    combined_poly = multiply_poly_terms(poly_list)

    final_factors = []
    if combined_poly != 1:
        final_factors.append(combined_poly)
    final_factors.extend(non_poly_list)

    if not final_factors:
        return sp.Integer(1)
    if len(final_factors) == 1:
        return final_factors[0]
    return sp.Mul(*final_factors, evaluate=False)


# ==========================================
# 3. MAIN CANONICALIZER
# ==========================================

def canonicalize_skeleton(expr_input):
    """
    Transforms concrete skeleton expressions into canonical abstract forms (C, L, P, powu, square, cube).
    """
    if isinstance(expr_input, str):
        local_dict = {
            'C': C, 'L': L, 'P': P,
            'inv': sp.Function('inv'),
            'powu': sp.Function('powu'),
            'square': sp.Function('square'),
            'cube': sp.Function('cube'),
            'Abs': sp.Abs
        }
        expr = sp.parse_expr(expr_input, local_dict=local_dict, evaluate=False)
    else:
        expr = expr_input

    if str(expr) == 'c':
        return C

    canon_expr = _canonicalize_node(expr)
    return canonical_sort(canon_expr)


def _canonicalize_node(node):
    # 1. Check if the current subtree is a pure polynomial in x
    deg = get_poly_degree(node)
    if deg is not None:
        if deg == 0:
            return C
        elif deg == 1:
            return L
        else:
            return P

    # 2. Handle non-polynomial subexpressions
    if isinstance(node, sp.Add):
        canon_args = [_canonicalize_node(arg) for arg in node.args]
        return _combine_add(canon_args)

    elif isinstance(node, sp.Mul):
        canon_args = [_canonicalize_node(arg) for arg in node.args]
        return _combine_mul(canon_args)

    elif isinstance(node, sp.Pow):
        base, exp = node.args
        canon_base = _canonicalize_node(base)

        if exp == 2:
            return P if canon_base in (L, P) else sp.Function('square')(canon_base)
        elif exp == 3:
            return P if canon_base in (L, P) else sp.Function('cube')(canon_base)
        elif exp == -1:
            return sp.Mul(1, sp.Pow(canon_base, -1, evaluate=False), evaluate=False)
        else:
            return P if (canon_base in (L, P) and exp.is_Integer and exp >= 1) else sp.Function('powu')(canon_base)

    else:
        canon_args = [_canonicalize_node(arg) for arg in node.args]
        func_name = str(node.func)

        if func_name in ('powu', 'square', 'cube'):
            arg = canon_args[0]
            return P if arg in (L, P) else sp.Function(func_name)(arg)
        elif func_name == 'inv':
            arg = canon_args[0]
            return sp.Mul(1, sp.Pow(arg, -1, evaluate=False), evaluate=False)
        else:
            return node.func(*canon_args, evaluate=False)


# ==========================================
# 4. FILE PROCESSING & EXECUTION
# ==========================================

def process_single_line(line, is_nesymres):
    line = line.strip()
    if not line:
        return None

    if is_nesymres:
        line = get_skeleton_var(
            expr=sp.sympify(line),
            var='x_1',
            var_names=['x_1', 'x_2', 'x_3'],
            expand=False
        )
        line = str(line)

    canonical_expr = canonicalize_skeleton(line)
    canonical_expr = canonical_sort(canonical_expr)
    return str(canonical_expr)


def process_skeleton_file(input_filepath, output_filepath, num_workers=None):
    from tqdm import tqdm
    from itertools import islice
    if num_workers is None:
        num_workers = mp.cpu_count()

    is_nesymres = 'NeSymRes' in input_filepath
    worker_func = partial(process_single_line, is_nesymres=is_nesymres)
    max_lines = 10000000
    with open(input_filepath, 'r') as infile, open(output_filepath, 'w', buffering=1) as outfile:
        lines_stream = islice(infile, max_lines) if max_lines is not None else infile
        chunksize = 1000 if max_lines is None else max(1, min(1000, max_lines // (num_workers * 2)))
        with mp.Pool(processes=num_workers) as pool:
            mapped_results = pool.imap(worker_func, lines_stream, chunksize=chunksize)
            for result in tqdm(mapped_results, total=max_lines, desc=f"Processing on {num_workers} cores"):
                if result is not None:
                    outfile.write(f"{result}\n")


if __name__ == '__main__':
    # process_skeleton_file('C:\\Users\\moralesg\\Documents\\GitHub\\SRSkeletons\\src\\other_datasets\\NeSymRes_skeletons.txt',
    #                       'C:\\Users\\moralesg\\Documents\\GitHub\\SRSkeletons\\src\\other_datasets\\NeSymRes_skeletons_canonicalized.txt')

    process_skeleton_file('C:\\Users\\gsjj1\\PycharmProjects\\SRSkeletons\\src\\other_datasets\\NeSymRes_skeletons.txt',
                          'C:\\Users\\gsjj1\\PycharmProjects\\SRSkeletons\\src\\other_datasets\\NeSymRes_skeletons_canonicalized.txt')
