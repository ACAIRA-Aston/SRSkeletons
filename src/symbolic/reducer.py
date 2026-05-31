import sympy as sp
import config
from collections import Counter
from src.utils.sympy_utils import absorb_constants
from sympy import (Symbol, Abs, conjugate, Pow, log, exp, sqrt, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
                   asinh, acosh, atanh, Rational)

# Define standard placeholders as global SymPy symbols
C, L, P = sp.symbols('C L P')
# A local dictionary ensuring SymPy cleanly maps tokens to our designated symbols
LOCAL_DICT = {'C': C, 'L': L, 'P': P}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _node_to_string(node) -> str:
    """ Flattens a grammar Node tree into a raw mathematical string """
    if not node.children:
        return str(node.symbol)
    return "".join(_node_to_string(child) for child in node.children)


def _build_inverse_map() -> dict:
    """
    Discover inverse function pairs by calling .inverse() on a dummy applied expression, then supplement the handful SymPy doesn't expose.
    """
    _probe_sym = Symbol("_probe_", real=True)
    candidates = [ tan, asin, acos, atan, sinh, tanh, asinh, acosh, atanh, exp, log, conjugate, ]  # TODO: Are there more?
    imap: dict = {}
    for f in candidates:
        try:
            inv = f(_probe_sym).inverse()
            imap[f] = inv
            imap[inv] = f
        except Exception:
            pass

    # sin, cos, cosh don't expose .inverse() in all versions
    for f, g in [(sin, asin), (cos, acos), (cosh, acosh)]:
        imap.setdefault(f, g)
        imap.setdefault(g, f)

    imap[conjugate] = conjugate  # conjugate is its own inverse
    return imap

INVERSE_MAP: dict = _build_inverse_map()

_NEEDS_POSITIVE = {log, sqrt, acosh}
_NEEDS_REAL = { exp, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, atanh, conjugate, }  # TODO: Check again when adding more unary ops.


def _assumptions_for(func) -> dict:
    """Minimal assumptions for ``func`` to be well-defined and invertible."""
    if func in _NEEDS_POSITIVE:
        return {"positive": True}  # positive implies real
    if func in _NEEDS_REAL or func in INVERSE_MAP:
        return {"real": True}
    return {}


def _simplify_pow_pow(expr: sp.Expr):
    """Return simplified form of (x**a)**b when a*b==1, else None."""
    if not isinstance(expr, Pow):
        return None
    base, outer_exp = expr.args
    if not isinstance(base, Pow):
        return None
    inner_base, inner_exp = base.args

    if sp.simplify(outer_exp * inner_exp) == 1:
        # Even integer inner exponent → Abs is necessary for real x
        if inner_exp.is_integer and inner_exp.is_even:
            return Abs(inner_base)
        return inner_base
    return None


def _walk(expr):
    """Tree walker (bottom-up)"""
    if not expr.args:
        return expr

    # Recurse first
    new_args = tuple(_walk(a) for a in expr.args)
    if new_args != expr.args:
        expr = expr.func(*new_args)

    # (x**a)**b  →  x  (or Abs(x))
    result = _simplify_pow_pow(expr)
    if result is not None:
        return _walk(result)

    # f(g(x))  →  x
    inner = expr.args[0]
    if inner.args and (expr.func in INVERSE_MAP) and (INVERSE_MAP[expr.func] is inner.func):
        return _walk(inner.args[0])

    return expr


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------

def cancel_inverses(expr, force_assumptions: bool = True) -> sp.Expr:
    """
    Simplify an expression by cancelling redundant inverse operator pairs.
    Assumptions are inferred automatically from the outer functions so that the cancellations are mathematically valid:
      - log, sqrt, acosh        ->  argument assumed positive (> 0)
      - exp, trig, hyp          ->  argument assumed real
      - algebraic / conjugate   ->  always valid
    """
    if not force_assumptions:
        return sp.simplify(_walk(expr))

    # Infer assumptions per free symbol
    assumptions_map: dict = {}

    def _collect(e) -> None:
        if not e.args:
            return
        needed = _assumptions_for(e.func)
        if needed:
            for sym in e.free_symbols:
                current = assumptions_map.get(sym, {})
                if current.get("positive") or needed.get("positive"):
                    assumptions_map[sym] = {"positive": True}
                else:
                    assumptions_map[sym] = {**current, **needed}
        for a in e.args:
            _collect(a)

    _collect(expr)

    for sym in expr.free_symbols:  # Default: real
        assumptions_map.setdefault(sym, {"real": True})

    # Rebuild symbols
    sub_map = {sym: Symbol(sym.name, **assumptions_map[sym]) for sym in expr.free_symbols}
    expr = expr.subs(sub_map)

    # SymPy built-in passes (catch log/exp, powsimp, etc.)
    expr = sp.powsimp(sp.expand_log(sp.logcombine(expr, force=True), force=True))
    expr = _walk(expr)

    return sp.simplify(expr)


def simplify(expr_node):
    """
    Executes standard algebraic simplifications, and applies constant absorption rules.
    """
    # Map the node structure to a flat string representation and parse it into SymPy
    if not isinstance(expr_node, sp.Expr):
        if isinstance(expr_node, str):
            parsed_expr = sp.sympify(expr_node)
        else:
            raw_str = _node_to_string(expr_node)
            parsed_expr = sp.parse_expr(raw_str, local_dict=LOCAL_DICT)
    else:
        parsed_expr = expr_node

    # Apply standard algebraic reduction
    simplified_expr = sp.expand(sp.simplify(sp.expand(cancel_inverses(parsed_expr)), evaluate=False), evaluate=False)
    return absorb_constants(simplified_expr)


# def is_less_complex(s_prime: sp.Expr, s_node) -> bool:
#     """
#     Compares the structural complexity of a simplified SymPy expression against the unsimplified candidate grammar node tree.
#     Returns True if the simplified version yields a strictly smaller operation footprint.
#     """
#     if not isinstance(s_node, sp.Expr):
#         raw_str = _node_to_string(s_node)
#         s_node = sp.parse_expr(raw_str, local_dict=LOCAL_DICT)
#     return sp.count_ops(s_prime) < sp.count_ops(s_node)

def is_less_complex(s_prime: sp.Expr, s_node_str: str) -> bool:
    """
    Compares the structural complexity of a simplified SymPy expression
    against the unsimplified candidate grammar string.

    Returns True only if the simplified version has fewer total operations
    AND does not introduce new algebraic structures (like factorization powers).
    """
    # 1. Parse the reference string safely into SymPy without auto-evaluating
    parsed_node = sp.parse_expr(s_node_str, local_dict=LOCAL_DICT, evaluate=False)

    # Helper to gather all operational functions/classes used in the tree
    def get_operator_types(expr: sp.Expr) -> set:
        # sub_expr.args ensures we ignore raw leaves like symbols 'C' or 'L'
        return {sub_expr.func for sub_expr in sp.preorder_traversal(expr) if sub_expr.args}

    # 2. Extract operational footprints
    orig_op_types = get_operator_types(parsed_node)
    prime_op_types = get_operator_types(s_prime)

    # 3. Check if SymPy sneaked in a new type of operation (e.g., sp.Pow)
    introduced_ops = prime_op_types - orig_op_types
    if introduced_ops:
        # A new structural type was introduced (likely due to factorization),
        # so it's not a pure reduction.
        return True

    # 4. Fall back to a strict operator count if the operator profiles match
    return sp.count_ops(s_prime) < sp.count_ops(parsed_node)


def prune_linear_constants(node):
    """
    Recursively scans the tree. If an inner node with the symbol '+' or 'Add' represents a sub-expression algebraically equal to
    'C + C*L' or 'C + C*P', its children are wiped and it collapses into a terminal 'L' or 'P' node.
    """
    if not node:
        return

    if node.children:
        subtree_str = _node_to_string(node)
        parsed_expr = sp.simplify(sp.parse_expr(subtree_str))
        C, L, P = sp.symbols('C L P')
        target_L = C * (L + 1)
        target_P = C * (P + 1)
        if parsed_expr == target_L:
            node.children = []
            node.symbol = 'L'
        elif parsed_expr == target_P:
            node.children = []
            node.symbol = 'P'
        else:
            for i, child in enumerate(node.children):
                node.children[i] = prune_linear_constants(child)
    return node




# if __name__ == "__main__":
#     class MockNode:
#         def __init__(self, symbol, children=None):
#             self.symbol = symbol
#             self.children = children or []
#
#     # 1. Build Original Node Branch: C + C * sin(L) * sin(L)
#     sin1 = MockNode('sin', [MockNode('('), MockNode('L'), MockNode(')')])
#     sin2 = MockNode('sin', [MockNode('('), MockNode('L'), MockNode(')')])
#     prod = MockNode('*', [MockNode('C'), MockNode('*'), sin1, MockNode('*'), sin2])
#     orig_tree = MockNode('+', [MockNode('C'), MockNode('+'), prod])
#
#     # 2. Define the SymPy equivalent after simplification/expansion: C * sin(L)**2 + C
#     C, L = sp.symbols('C L')
#     sympy_expr = C * sp.sin(L)**2 + C
#
#     # Naive count analysis
#     naive_node_ops = sum(1 for k, v in get_node_profile(orig_tree).items() if k in ('+', '*', 'sin', '**'))
#     # Wait, let's look at the absolute profiles to see the contrast:
#     p_node = get_node_profile(orig_tree)
#     p_sympy = get_sympy_profile(sympy_expr)
#
#     print("--- Profile Breakdown ---")
#     print(f"Original Node Profile:  {dict(p_node)}")
#     print(f"Simplified SymPy Profile: {dict(p_sympy)}")
#     print("-" * 30)
#
#     mutation_detected = detect_structural_mutation(orig_tree, sympy_expr)
#     print(f"Was a factorization/structural rewrite detected? {mutation_detected}")
