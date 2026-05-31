import sympy as sp


def get_complexity_metrics(expr):
    """ Computes a sort key for a SymPy expression """
    ops = sp.count_ops(expr)
    def get_depth(e):  # Calculate nesting depth
        if not e.args:
            return 0
        return 1 + max(get_depth(arg) for arg in e.args)

    depth = get_depth(expr)
    return ops, depth, str(expr)  # Use string representation as the ultimate tie-breaker


def canonical_sort(expr):
    """ Recursively sorts commutative SymPy operations to ensure a unique canonical string representation of a functional skeleton. """
    if not expr.args:
        return expr

    sorted_args = [canonical_sort(arg) for arg in expr.args]  # Recursively sort all child arguments first

    if isinstance(expr, (sp.Add, sp.Mul)):  # If the operation is commutative, sort the arguments based on complexity
        sorted_args = sorted(sorted_args, key=get_complexity_metrics)

    # Reconstruct the expression with the new argument order
    new_expr = expr.func(*sorted_args, evaluate=False)  # evaluate=False prevents SymPy from re-ordering them back

    return new_expr
