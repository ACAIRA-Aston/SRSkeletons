import re
# from __future__ import annotations
import sympy as sp
import numpy as np
from sympy import Add, Mul, Pow
from numpy import iscomplex, isnan, isinf, abs


def _avoid_operations_between_constants(xp):
    # Functions adapted from my previous work https://github.com/NISL-MSU/MultiSetSR
    """ Simplify exponent of constants inside symbolic expression and constant multiplied by constants.
    E.g., constant ** 3 = constant or cons1 * cons2 * cons3 = cons1. This will be used when skeletons are already generated
    :param xp: Symbolic expression"""
    args = xp.args
    t_args = args
    new_args = []

    if isinstance(xp, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
        args1 = args[0]
        args2 = args[1]
        if (args1.is_number or (isinstance(args1, sp.Symbol) and ("cm" in str(args1) or "ca" in str(args1) or "c" in str(args1) or "C" in str(args1)))) and \
                (args2.is_number or (isinstance(args2, sp.Symbol) and ("cm" in str(args2) or "ca" in str(args2) or "c" in str(args2) or "C" in str(args2)))):
            va = None
            if isinstance(args1, sp.Symbol) and ("cm" in str(args1) or "ca" in str(args1) or "c" in str(args1) or "C" in str(args1)):
                va = args1
            elif isinstance(args2, sp.Symbol) and ("cm" in str(args2) or "ca" in str(args2) or "c" in str(args2) or "C" in str(args2)):
                va = args2
            t_args = [va, sp.sympify("1")]

    if isinstance(xp, sp.exp):  # If it's an exp function, ignore the power and focus only on the base
        args1 = args[0]
        if args1.is_number or (isinstance(args1, sp.Symbol) and ("cm" in str(args1) or "ca" in str(args1) or "c" in str(args1) or "C" in str(args1))):
            t_args = tuple([sp.sympify("0")])

    if isinstance(xp, sp.Mul) or isinstance(xp, sp.Add):
        t_args = []
        # Check if two or more of the arguments are constants and replace them for just one constant
        flag = False  # This flag turns to True when a constant was already found
        for ia, arg in enumerate(args):
            if arg.is_number and any([(isinstance(co, sp.Symbol) and ('c' in str(co) or 'C' in str(co))) for co in (args[0:ia] + args[ia + 1:])]):
                # If the current number is summed or multiplied by other constants, replace it by 0 or one
                if isinstance(xp, sp.Mul):
                    arg = sp.sympify("1")
                else:
                    arg = sp.sympify("0")
            if isinstance(arg, sp.Symbol) and ("cm" in str(arg) or "ca" in str(arg) or "c" in str(arg) or "C" in str(arg)):
                if not flag:
                    flag = True
                else:
                    arg = sp.sympify("1")
            t_args.append(arg)

    for arg in t_args:
        if arg.is_number:  # If it's a number, add it to the list
            new_args.append(arg)
        elif isinstance(arg, sp.Symbol):
            new_args.append(arg)
        else:  # If it's composed, explore a lower level of the tree
            new_args.append(_avoid_operations_between_constants(arg))

    if len(new_args) > 0:
        new_xp = xp.func(*new_args)
    else:
        new_xp = xp

    if len(new_args) == 1 and 'c' == str(new_args[0]):  # If it's a unary operation and it's argument is a constant
        new_xp = sp.sympify('c')

    return new_xp


def avoid_operations_between_constants(xp):
    """Repeat the _avoid_operations_between_constants function until no change is possible"""
    xp2 = None
    while xp != xp2:
        xp2 = xp
        xp = _avoid_operations_between_constants(xp)
    return xp


"""
Canonical placeholder absorption for symbolic expressions.
Placeholder tiers (by symbol name convention)
----------------------------------------------
  C – constant:    symbols with 'c', 'C', 'cm', or 'ca' in the name;
                   also numeric literals.
  L – linear:      symbols with 'L' in the name.
  P – polynomial:  symbols with 'P' in the name.

Canonical equivalence rules applied  (see Definition in paper)
--------------------------------------------------------------
  Constant:   C+c ~ C,   C*c ~ C,   n*C ~ C,   C**n ~ C,   f(C) ~ C
  Linear:     x+C ~ L,   x*C ~ L,   L+C ~ L,   L+x ~ L,   L*C ~ L,   n*L ~ L
  Polynomial: L*x ~ P,   P+c ~ P,   P+x ~ P,   P*x ~ P,
              P+L ~ P,   P*L ~ P,   L*L ~ P

In every Add / Mul, when multiple args share the same placeholder tier, only one representative is kept (the rest collapse to 0 or 1 respectively).
"""

TYPE_RANK: dict[str, int] = {'C': 0, 'L': 1, 'P': 2}


def _sym_type(sym: sp.Symbol) -> str:
    """Return the placeholder tier of a Symbol, or 'var' for real variables."""
    s = str(sym)
    if 'cm' in s or 'ca' in s:   # these are multiplicative or additive constant placeholders
        return 'C'
    if 'P' in s:
        return 'P'
    if 'L' in s:
        return 'L'
    if 'c' in s or 'C' in s:
        return 'C'
    return 'var'


def _expr_type(expr) -> str | None:
    """ Return the tier of expr ('C', 'L', or 'P') if the whole expression collapses to a pure placeholder type, otherwise None.
    None means the expression contains non-placeholder sub-structure (e.g. log(P), sin(L+…)) that cannot itself be absorbed.
    """
    if expr.is_number:
        return 'C'

    if isinstance(expr, sp.Symbol):
        return _sym_type(expr)

    if isinstance(expr, Add):
        types = [_expr_type(a) for a in expr.args]
        ph = [t for t in types if t in TYPE_RANK]
        if not ph or any(t not in TYPE_RANK for t in types):  # Mixed with var/None -> not a pure placeholder Add
            return None
        return max(ph, key=lambda t: TYPE_RANK[t])  # Returns the one with the max hierarchy

    if isinstance(expr, Mul):
        # If a multiplier is Pow (i.e., if it's a division), then no simplification will be made
        if any(isinstance(arg, Pow) for arg in expr.args):
            return None
        types = [_expr_type(a) for a in expr.args]
        has_var  = any(t == 'var'  for t in types)
        has_none = any(t is  None  for t in types)
        ph = [t for t in types if t in TYPE_RANK]
        if has_none:
            return None
        dominant = max(ph, key=lambda t: TYPE_RANK[t]) if ph else None
        if has_var:
            if dominant == 'C':   return 'L'        # x*C ~ L
            if dominant in ('L', 'P'):  return 'P'  # x*L ~ P, x*P ~ P
            return 'var'
        return dominant   # may be None if no ph found

    if isinstance(expr, Pow):
        b_type = _expr_type(expr.args[0])
        e_type = _expr_type(expr.args[1])
        if b_type in TYPE_RANK and (e_type in TYPE_RANK or expr.args[1].is_number):
            return b_type   # C**n ~ C, L**n ~ L, P**n ~ P
        if b_type == 'var' and e_type in TYPE_RANK:
            return 'P'      # x**C ~ P
        return None

    if len(expr.args) == 1:  # Unary function: f(C) ~ C
        arg_t = _expr_type(expr.args[0])
        if arg_t == 'C':
            return 'C'
        return None

    return None


def _absorb_once(expr):
    """One bottom-up pass: apply all canonical absorption rules once."""

    if expr.is_number or isinstance(expr, sp.Symbol):  # Leaves
        return expr

    new_args = tuple(_absorb_once(a) for a in expr.args)  # Recurse into children first
    if new_args != expr.args:
        expr = expr.func(*new_args)

    if not isinstance(expr, (Add, Mul, Pow)) and len(expr.args) == 1:  # f(C) -> C
        arg = expr.args[0]
        if _expr_type(arg) == 'C':
            return arg if isinstance(arg, sp.Symbol) else sp.Symbol('C')

    #  Pow
    if isinstance(expr, Pow):  # Pow: C**n -> C,  placeholder**placeholder -> dominant
        b, e = expr.args
        if e > 0:
            expr_type = _expr_type(b)
            b_type = expr_type
            # Any placeholder base to a constant/placeholder power → keep base
            if b_type == 'C':
                return sp.Symbol('C')
            elif b_type == 'L':
                return sp.Symbol('P') if (isinstance(b, sp.Symbol) and e.is_integer) else expr
            elif b_type == 'P':
                return b if (isinstance(b, sp.Symbol) and e.is_integer) else expr
        return expr

    #  Mul
    if isinstance(expr, Mul):
        args      = list(expr.args)
        # If a multiplier is Pow (i.e., if it's a division), then no simplification will be made
        if any(isinstance(arg, Pow) for arg in args):
            return expr

        arg_types = [_expr_type(a) for a in args]
        ph_types  = [t for t in arg_types if t in TYPE_RANK]
        if not ph_types:
            return expr

        dom = max(ph_types, key=lambda t: TYPE_RANK[t])

        # Special case: numeric_coefficient * single_placeholder_symbol → symbol (n*X ~ X for any tier X)
        nums    = [(i, a) for i, a in enumerate(args) if a.is_number]
        ph_syms = [(i, a) for i, a in enumerate(args)
                   if isinstance(a, sp.Symbol) and _sym_type(a) in TYPE_RANK]
        if nums and len(args) == len(nums) + 1 and len(ph_syms) == 1:
            return ph_syms[0][1]

        # General: replace lower-rank pure-placeholder args with 1, and keep only ONE arg of each tier
        new_args = _replace(args, arg_types, dom, r=sp.S.One)
        return expr.func(*new_args)

    # Add
    if isinstance(expr, Add):
        args      = list(expr.args)
        arg_types = [_expr_type(a) for a in args]
        ph_types  = [t for t in arg_types if t in TYPE_RANK]
        if not ph_types:
            return expr

        dom = max(ph_types, key=lambda t: TYPE_RANK[t])
        # Replace lower-rank pure-placeholder args with 0, and keep only ONE arg of each tier
        new_args = _replace(args, arg_types, dom, r=sp.S.Zero)
        return expr.func(*new_args)

    return expr

def _replace(args, arg_types, dom, r=sp.S.One):
    """Helper function to replace summands or multiplicands"""
    seen_tiers: set[str] = set()
    new_args = []
    for a, t in zip(args, arg_types):
        if t in TYPE_RANK:
            if TYPE_RANK[t] < TYPE_RANK[dom]:
                new_args.append(r)  # absorb lower tier
            elif t in seen_tiers:
                new_args.append(r)  # absorb duplicate same-tier
            else:
                seen_tiers.add(t)
                new_args.append(a)
        else:
            new_args.append(a)
    return new_args


def absorb_constants(expr: sp.Expr, max_iter: int = 30) -> sp.Expr:
    """ Apply canonical placeholder absorption rules until the expression stabilises (fixed point). """
    prev = None
    for _ in range(max_iter):
        if expr == prev:
            break
        prev = expr
        expr = _absorb_once(expr)
        expr = sp.sympify(expr)   # let SymPy cancel 0-terms, merge coefficients
    return expr


def add_constant_identifier(sk, cm_counter=1, ca_counter=1):
    """Label each coefficient of a skeleton expression"""
    args = sk.args
    new_args = []

    for arg in args:
        if arg.is_number:  # If it's a number, add it to the list
            new_args.append(arg)
        elif isinstance(arg, sp.Symbol) and 'C' in str(arg):
            if isinstance(sk, sp.Add):
                new_args.append(sp.sympify('ca_' + str(ca_counter)))
                ca_counter += 1
            else:
                new_args.append(sp.sympify('cm_' + str(cm_counter)))
                cm_counter += 1
        elif isinstance(arg, sp.Symbol):
            new_args.append(arg)
        else:  # If it's composed, explore a lower level of the tree
            deep_xp, cm_counter, ca_counter = add_constant_identifier(arg, cm_counter=cm_counter, ca_counter=ca_counter)
            new_args.append(deep_xp)

    new_xp = sk.func(*new_args)
    return new_xp, cm_counter, ca_counter


def get_op(xp, Targ):
    """Extract the operatorS in which the n-th argument is inside of
    :param xp: Symbolic expression
    :param Targ: The target argument"""
    args = xp.args

    if isinstance(xp, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
        args = args[0]
        if args.is_number or isinstance(args, sp.Symbol) or (isinstance(args, sp.Symbol) and 'c' in str(args)):
            args = [args]
        else:
            args = args.args

    ops = []
    for arg in args:
        if arg.is_number or (isinstance(arg, sp.Symbol) and 'c' in str(arg)):  # If it's a number, add it to the list
            if str(arg) == Targ:
                return str(xp.func)
        else:  # If it's composed, explore a lower level of the tree
            op = get_op(arg, Targ)
            if len(op) == 3:
                ops = op
            else:
                if isinstance(op, list):
                    ops = [str(xp.func)] + op
                else:
                    ops = [str(xp.func)] + [op]
    return ops


def get_op_constant(xp, c):
    """Extract the operator in which the specified constant name is inside (ignoring mult or add)
    :param xp: Symbolic expression
    :param c: Constant name"""
    args = xp.args

    for arg in args:
        if c in str(arg):
            if arg.is_Symbol and str(arg) == c:
                return True  # str(xp.func)
            else:  # If it's composed, explore a lower level of the tree
                op = get_op_constant(arg, c)
                if op is True:
                    if len(arg.args) == 1:  # It will return the first unary operation the constant is in
                        return str(arg.func)
                    elif isinstance(arg, sp.Pow):  # In case it's Pow, return the exponent too
                        return str(arg.func), arg.args[1]
                    else:
                        return op
                else:
                    return op
        else:
            continue


def bounded_operations():
    """Returns the bounded and double-bounded unary operations.
    Single-bounded operations are assigned their minimum or maximum admissible value.
    Double-bounded operations are assigned their minimum and maximum admissible values."""

    bounded = {'sqrt': ('min', 0.01),
               'log': ('min', 0.001),
               'acosh': ('min', 1.001),
               'exp': ('max', 10),
               'sinh': ('max', 8),
               'cosh': ('max', 8),
               'tanh': ('max', 8)}
    double_bounded = {'asin': [-0.999, 0.999],
                      'acos': [-0.999, 0.999],
                      'atanh': [-0.999, 0.999],}
    trig_functions = ['sin', 'cos', 'tan']
    op_with_singularities = {'tan': 50, 'div': 50}

    return bounded, double_bounded, op_with_singularities, trig_functions


def get_args(xp, return_symbols=False):
    """Extract all numeric arguments and coefficients of an expression
    :param xp: Symbolic expression
    :param return_symbols: If True, it also returns the symbols in the expression"""
    args = xp.args
    num_args = []

    if isinstance(xp, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
        args = args[0]
        if args.is_number or isinstance(args, sp.Symbol):
            args = [args]
        else:
            args = args.args

    for arg in args:
        if arg.is_number or (isinstance(arg, sp.Symbol) and 'c' in str(arg)):  # If it's a number or coeff, add it
            num_args.append(arg)
        elif isinstance(arg, sp.Symbol) and return_symbols:
            num_args.append(arg)
        else:  # If it's composed, explore a lower level of the tree
            num_args = num_args + get_args(arg, return_symbols=return_symbols)
    return num_args


def set_args(xp, target_args, return_symbols=False):
    """Set all numeric arguments of an expression"""
    args = xp.args
    new_args = []

    if isinstance(xp, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
        if xp.args[0].is_number or (
                isinstance(xp.args[0], sp.Symbol) and str(xp.args[0]) == 'c'):  # If it's a number, add it to the list
            new_args.append(target_args.pop(0))
        elif isinstance(xp.args[0], sp.Symbol):
            new_args.append(xp.args[0])
        else:  # If it's composed, explore a lower level of the tree
            new_args.append(set_args(xp.args[0], target_args))
        new_args.append(xp.args[1])
    else:
        for arg in args:
            if arg.is_number or (
                    isinstance(arg, sp.Symbol) and 'c' in str(arg)):  # If it's a number or coeff, add it to the list
                new_args.append(target_args.pop(0))
            elif isinstance(arg, sp.Symbol) and not return_symbols:
                new_args.append(arg)
            elif isinstance(arg, sp.Symbol) and return_symbols:
                new_args.append(target_args.pop(0))
            else:  # If it's composed, explore a lower level of the tree
                new_args.append(set_args(arg, target_args, return_symbols=return_symbols))
    new_xp = xp.func(*new_args)
    return new_xp


def is_nan(x, bound=None):
    """Encompasses NaN, Inf values, and outliers"""
    if bound is None:
        return iscomplex(x) + isnan(x) + isinf(x) + (abs(x) > 100000)  # + array(outliers)
    else:
        return iscomplex(x) + isnan(x) + isinf(x) + (abs(x) > 100000) + (abs(x) > bound)  # + array(outliers)


def is_p_inside_unary_op(exp):
    """
    Check if a given "P" in an expression string is inside a unary operator
    """
    ops = {'sin', 'tan', 'exp', 'logg', 'sqrt2', 'sinh', 'cosh', 'tanh', 'asin', 'atan', 'asinh', 'acosh', 'atanh', 'square', 'cube'}
    p_match = re.search(r'\bP\b', exp)
    if not p_match:
        return False
    idx = p_match.start()
    prefix = exp[:idx]

    # Walk the prefix tracking a stack of '(' positions
    stack = []
    for i, ch in enumerate(prefix):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                stack.pop()

    if not stack:
        return False  # P isn't inside any parens at all

    innermost_open = stack[-1]
    # Grab the identifier immediately preceding that '('
    name_match = re.search(r'([a-zA-Z_]\w*)\s*$', prefix[:innermost_open])
    if not name_match:
        return False

    func_name = name_match.group(1)
    return func_name in ops


def safe_pow_constant(expr):
    # Automatically forces real roots: sign(base) * abs(base)**exp
    return expr.replace(sp.Pow, lambda b, e: sp.Integer(1) if b.is_Number else b**e)


def safe_pow(expr):
    # Automatically forces real roots: sign(base) * abs(base)**exp
    return expr.replace(sp.Pow, lambda b, e: sp.sign(b) * sp.Abs(b)**e if not e.is_Integer else b**e)


def safe_evaluation(expr, support, varX):
    """Evaluates an expression on a given support vector safely"""
    expr = safe_pow(expr)
    function = sp.lambdify(sp.flatten([varX]), expr, 'numpy')
    # Evaluate values from the support vector into the new expression
    return np.array(function(support))



if __name__ == '__main__':
    C, L, P, x = sp.symbols('C L P x')
    c1, c2, c3 = sp.symbols('c1 c2 c3')

    cases = [
        # ── Constant tier ────────────────────────────────────────────────
        ('C + C',                          C + C,                                        C),
        ('C * C',                          C * C,                                        C),
        ('3 * C',                          3 * C,                                        C),
        ('c1 + c2',                        c1 + c2,                                      c1),
        ('c1 * c2 * c3',                   c1 * c2 * c3,                                 c1),
        ('sin(C)',                          sp.sin(C),                                    C),
        ('exp(C)',                          sp.exp(C),                                    C),
        ('log(C)',                          sp.log(C),                                    C),
        ('P ** 3',                          P ** 3,                                       P),
        # ── Linear tier ──────────────────────────────────────────────────
        ('C + L',                          C + L,                                        L),
        ('C * L',                          C * L,                                        L),
        ('3 * L',                          3 * L,                                        L),
        ('x + C',                          x + C,                                        x + C),  # stays (var+C = L-type structure but kept)
        ('x * C',                          x * C,                                        C * x),
        # # ── Polynomial tier ───────────────────────────────────────────────
        ('L * L',                          L * L,                                    P),
        ('C + L + P',                      C + L + P,                                    P),
        ('P + C',                          P + C,                                        P),
        ('P + L',                          P + L,                                        P),
        ('P * C',                          P * C,                                        P),
        ('P * L',                          P * L,                                        P),
        ('3 * P',                          3 * P,                                        P),
        # # ── Nested / mixed ────────────────────────────────────────────────
        ('C*sin(C)',                        C * sp.sin(C),                                C),
        ('L + C*sin(C)',                    L + C * sp.sin(C),                            L),
        ('C+L+P+C*sin(C+L+C*log(C*P))',
         C + L + P + C * sp.sin(C + L + C * sp.log(C * P)),
         P + C * sp.sin(L + C * sp.log(P))),
        # sin(L) is not simplified (no rule), but L*sin(L) + P → still P + L*sin(L)
        ('P + L*sin(L)',                    P + sp.sin(L) * L,                            P + L * sp.sin(L)),
    ]

    passed = 0
    w = (46, 38, 38)
    print(f"{'Expression':<{w[0]}} {'Expected':<{w[1]}} {'Got':<{w[2]}} OK")
    print('─' * (sum(w) + 6))
    for label, expr_in, expected in cases:
        result   = absorb_constants(expr_in)
        match    = (sp.simplify(result - expected) == 0
                    or str(result) == str(expected))
        passed  += match
        ok       = '✓' if match else '✗'
        print(f"{label:<{w[0]}} {str(expected):<{w[1]}} {str(result):<{w[2]}} {ok}")

    print(f"\n{passed}/{len(cases)} passed")