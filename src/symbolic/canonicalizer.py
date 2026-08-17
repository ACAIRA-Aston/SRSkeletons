import sympy as sp

import src
from src.symbolic.normalizer import canonical_sort

# Define canonical symbols
C, L, P = sp.symbols('C L P')


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


def _combine_add(args):
    poly_args = [arg for arg in args if get_poly_degree(arg) is not None]
    non_poly_args = [arg for arg in args if get_poly_degree(arg) is None]

    final_terms = []
    if poly_args:
        max_deg = max(get_poly_degree(arg) for arg in poly_args)
        collapsed = C if max_deg == 0 else (L if max_deg == 1 else P)
        final_terms.append(collapsed)

    final_terms.extend(non_poly_args)
    if len(final_terms) == 1:
        return final_terms[0]
    return sp.Add(*final_terms, evaluate=False)


def _combine_mul(args):
    poly_args = [arg for arg in args if get_poly_degree(arg) is not None]
    non_poly_args = [arg for arg in args if get_poly_degree(arg) is None]

    final_terms = []
    if poly_args:
        total_deg = sum(get_poly_degree(arg) for arg in poly_args)
        collapsed = C if total_deg == 0 else (L if total_deg == 1 else P)
        # Always retain the collapsed polynomial factor (C, L, or P)
        final_terms.append(collapsed)

    final_terms.extend(non_poly_args)
    if len(final_terms) == 1:
        return final_terms[0]
    return sp.Mul(*final_terms, evaluate=False)


def process_skeleton_file(input_filepath, output_filepath):
    from tqdm import trange
    unique_canonical_skeletons = []

    with open(input_filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    with trange(len(lines)) as t:
        for iline in t:
            line = lines[iline]

            # Canonicalize to C, L, P
            canonical_expr = canonicalize_skeleton(line)
            canonical_expr = canonical_sort(canonical_expr)
            unique_canonical_skeletons.append(str(canonical_expr))
            # print(f"Input:  {line}")
            # print(f"Output: {str(canonical_expr)}\n")

    # Write deduplicated canonical skeletons to file
    with open(output_filepath, 'w') as f:
        for skel in sorted(unique_canonical_skeletons):
            f.write(f"{skel}\n")
    # print(
    #     f"Processed {input_filepath} -> {len(unique_canonical_skeletons)} unique canonical forms written to {output_filepath}")


if __name__ == '__main__':
    # test_cases = [
    #     "ca_1 + cm_1*(ca_2 + sin(ca_3 + cm_2*x_1))**4",
    #     "ca_1 + cm_1*(ca_2 + sin(ca_3 + cm_2*x_1))**2",
    #     "ca_1 + cm_1*(ca_2 + sin(ca_3 + cm_2*x_1))**3",
    #     "cm_1*x_1**5 + (cm_2*x_1 + ca_1)**2 + cm_3*x_1**3 + ca_2",
    #     "ca_1 + cm_1/(ca_2 + (ca_3 + x_1)**2)",
    #     "ca_1 + cm_1*(ca_2 + x_1)*sinh(ca_3 + cm_2*x_1)"
    # ]
    #
    # for skel in test_cases:
    #     print(f"Input:  {skel}")
    #     print(f"Output: {canonicalize_skeleton(skel)}\n")

    process_skeleton_file('C:\\Users\\gsjj1\\PycharmProjects\\SRSkeletons\\src\\other_datasets\\SeTGAP_skeletons.txt',
                          'C:\\Users\\gsjj1\\PycharmProjects\\SRSkeletons\\src\\other_datasets\\SeTGAP_skeletons_canonicalized.txt')
