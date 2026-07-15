import re
import config
import random
import numpy as np
from fractions import Fraction
from src.symbolic.reducer import inv, cube_func, square_func
from src.sampler.avoidNaNs import modify_constants_avoidNaNs
from sympy import sympify, symbols, parse_expr, Function, Pow, Integer, simplify, nsimplify, exp as sp_exp
from src.utils.sympy_utils import (add_constant_identifier, get_op_constant, bounded_operations, get_args, set_args,
                                   is_nan, is_p_inside_unary_op, safe_pow_constant, safe_evaluation, avoid_operations_between_constants)

varX = symbols('x', real=True)

def random_exponent(int_prob=0.7, max_denom=6, low=0, high=7):
    """
    Random rational in (low, high), excluding EXCLUDED values.
    Biased toward integers (other than the excluded ones).
    """
    EXCLUDED = {Fraction(1, 2), Fraction(1), Fraction(2), Fraction(3)}
    allowed_ints = [n for n in range(low, high + 1) if low < n < high and Fraction(n) not in EXCLUDED]

    while True:
        if random.random() < int_prob and allowed_ints:
            return Integer(random.choice(allowed_ints))
        else:
            denom = random.randint(2, max_denom)
            numer = random.randint(low * denom + 1, high * denom - 1)  # strictly inside (low, high)
            frac = Fraction(numer, denom)
            if frac in EXCLUDED:
                continue  # resample
            return nsimplify(frac)  # returns sp.Integer or sp.Rational as appropriate


def sample_support(curr_p, Xbounds, extrapolate=False):
    if not extrapolate:
        curr = np.random.uniform(Xbounds[0], Xbounds[1], size=int(curr_p))
    else:
        # Extrapolation should verify bounded operations
        curr = np.random.uniform(Xbounds[0] * 2, Xbounds[1] * 2, size=int(curr_p))
    return curr


def sample_constants(exp, p_sparse=0.2):
    """ Samples random values for constants with a bias towards sparsification.
        :param exp: The string or SymPy expression containing constant placeholders.
        :param p_sparse: Probability (0.0 to 1.0) that a constant will be sparsified (set to 1 for 'cm' or 0 for 'ca').
    """
    LOCAL_DICT = {'inv': inv, 'powu': Function('powu'), 'cube': cube_func, 'square': square_func, 'x': varX}

    # Determine if P sits inside an unclosed parenthesis (i.e. is an argument of some unary op / grouping)
    p_nested = is_p_inside_unary_op(exp)

    # Update operator names
    exp = exp.replace('logg', 'log')
    exp = exp.replace('sqrt2', 'sqrt')

    exp = parse_expr(exp, local_dict=LOCAL_DICT, evaluate=False)

    # Replace every powu(arg) with arg**random_exponent
    def replace_powu(expr):
        # Find all powu(...) instances
        powu_terms = expr.atoms(Function('powu'))
        subs_dict = {}
        for term in powu_terms:
            arg = term.args[0]
            exponent = random_exponent() # random.randint(4, 7)
            subs_dict[term] = Pow(arg, exponent)
        return expr.subs(subs_dict)
    exp = replace_powu(exp)

    # If P appears inside an exponential, constrain it
    expo_terms = exp.atoms(sp_exp)
    for term in expo_terms:
        arg = term.args[0]
        arg_str = str(arg)
        if re.search(r'\bP\b', arg_str):
            p_nested = True
            break
    pow_terms = exp.atoms(Pow)
    for term in pow_terms:
        arg = term.args[0]
        arg_str = str(arg)
        if re.search(r'\bP\b', arg_str) and term.args[1] > 1:
            p_nested = True
            break

    # Check Ls
    exp = exp.subs(sympify('L'), sympify('C*x + C', locals={'x': varX}))

    # Check Ps
    if p_nested:
        degree = random.randint(2, 3)
    else:
        degree = random.randint(2, 6)  # Decide randomly the degree of the polynomial expression P
    C_coeffs = symbols(f'C0:{degree + 1}')
    poly_expr = sum(C_coeffs[i] * varX ** i for i in range(degree + 1))
    exp = exp.subs(sympify('P'), poly_expr)

    # Add constant identifiers
    exp, _, _ = add_constant_identifier(exp)

    # Start sampling
    all_symbols = [str(s) for s in exp.free_symbols]
    ca_symbols = [s for s in all_symbols if s.startswith("ca")]
    cm_symbols = [s for s in all_symbols if s.startswith("cm")]

    # ==========================================
    # LOOP 1: Additive Constants (ca)
    # ==========================================
    for c in ca_symbols:
        # Safety check: see if a previous substitution eliminated this term entirely
        if c not in str(exp):
            continue

        # Sparsification check: force to identity value 0
        if np.random.rand() < p_sparse:
            exp = exp.subs(c, 0)
            continue

        # Check if constant is inside an exponential or hyperbolic function
        op = get_op_constant(exp, c)
        if not isinstance(op, bool) and op is not None:
            if ('exp' in op) or ('sinh' in op) or ('cosh' in op) or ('tanh' in op):
                exp = exp.subs(c, np.round(np.random.uniform(low=-1, high=1), 3))
            else:
                exp = exp.subs(c, np.round(np.random.uniform(low=config.additive_min,
                                                             high=config.additive_max), 3))
        else:
            exp = exp.subs(c, np.round(np.random.uniform(low=config.additive_min,
                                                         high=config.additive_max), 3))

    exp = avoid_operations_between_constants(exp)
    # ==========================================
    # LOOP 2: Multiplicative Constants (cm)
    # ==========================================
    for c in cm_symbols:
        # Kept your original string presence check (crucial here if ca loop zeroed out terms)
        if c in str(exp):
            # Sparsification check: force to identity value 1
            if np.random.rand() < p_sparse:
                exp = exp.subs(c, 1)
                continue

            op = get_op_constant(exp, c)
            if not isinstance(op, bool) and op is not None:
                if ('exp' in op) or ('sinh' in op) or ('cosh' in op) or ('tanh' in op):
                    val = np.random.uniform(low=-3, high=3)
                    while np.abs(val) < 0.05:  # Avoid too small values
                        val = np.random.uniform(low=-3, high=3)
                    exp = exp.subs(c, np.round(val, 3))
                elif ('sin' in op) or ('cos' in op) or ('tan' in op):
                    val = np.random.uniform(low=-5, high=5)
                    while np.abs(val) < 0.1:  # Avoid too small values
                        val = np.random.uniform(low=-5, high=5)
                    exp = exp.subs(c, np.round(val, 3))
                elif isinstance(op, tuple):
                    if 'Pow' in op[0] and op[1] == 4:
                        val = np.random.uniform(low=-3, high=3)
                        exp = exp.subs(c, np.round(val, 3))
                    else:
                        exp = exp.subs(c, np.round(np.random.uniform(low=config.multiplicative_min,
                                                                     high=config.multiplicative_max), 3))
                else:
                    exp = exp.subs(c, np.round(np.random.uniform(low=config.multiplicative_min,
                                                                 high=config.multiplicative_max), 3))
            else:
                exp = exp.subs(c, np.round(np.random.uniform(low=config.multiplicative_min,
                                                             high=config.multiplicative_max), 3))

    return safe_pow_constant(exp)


def rescale_expression(expr, x, old_support, new_range):
    a, b = (np.min(old_support), np.max(old_support))
    c, d = new_range
    scale = (b - a) / (d - c)
    x_old_in_terms_of_new = a + (x - c) * scale
    new_expr = expr.subs(x, x_old_in_terms_of_new)
    new_expr = simplify(new_expr)
    return c + (old_support - a) * ((d - c) / (b - a)), new_expr


def generate_data(skeleton, extrapolate=False, n_sets=None, xmin=None, xmax=None):
    """
    Given a list of skeleton equations, sample their inputs and corresponding constants and evaluate their data
    :param eq: List of skeleton equations
    :param return_exprs: If True, return the expression that were sampled during the process
    :param extrapolate: If True, sample support values that go beyond the training domain
    :param n_sets: If not None, it explicitly specifies the number of sets to be generated
    :param xmin: If not None, it explicitly specifies the minimum support value to be used for generation
    :param xmax: If not None, it explicitly specifies the minimum support value to be used for generation
    """
    curr_p = config.max_number_of_points
    # # Uncomment the code below if you have a specific skeleton from which you want to sample data as an example
    # sk = sympy.sympify('c*log(c*cos(c*x0) + c) + c')
    # sk, _, _ = add_constant_identifier(sk)
    # coeff_dict = dict()
    # var = None
    # for constant in sk.free_symbols:
    #     if 'c' in str(constant):
    #         coeff_dict[str(constant)] = constant
    #     if 'x' in str(constant):
    #         var = constant
    # eq = SimpleEquation(expr=sk, coeff_dict=coeff_dict, variables=[var])
    # exprs = eq.expr

    # Randomly stretch or shrink input domain. E.g., from [-10, 10] to [-5, 5] or [-2, 2]
    divider = 1  # np.random.randint(2, 10) / 2
    if xmin is None and xmax is None:
        minX, maxX = config.fun_support_min / divider, config.fun_support_max / divider
        Xbounds = [minX, maxX]
    else:
        minX, maxX = xmin, xmax
        Xbounds = [minX, maxX]

    # Check if there's any bounded operation in the expression
    bounded, double_bounded, op_with_singularities, trig_functions = bounded_operations()
    is_bounded = False
    if any([b in skeleton for b in list(bounded.keys())]) or any([b in skeleton for b in trig_functions]) or \
            any([b in skeleton for b in list(double_bounded.keys())]) or 'inv' in skeleton or \
            any([b in skeleton for b in list(op_with_singularities)]) or '**0.5' in skeleton or '**(-' in skeleton:
        is_bounded = True

    # If the number of sets is explicitly specified
    Ns = config.number_of_sets
    if n_sets is not None:
        Ns = n_sets

    # Create "Ns" vectors to store the evaluations
    X = np.zeros((curr_p, Ns), dtype=np.float32)
    Y = np.zeros((curr_p, Ns), dtype=np.float32)
    # Sample constants "Ns" times
    support = sample_support(curr_p, Xbounds, extrapolate)
    n_set = 0
    sampled_exprs = [''] * Ns

    ###########################################################
    # Start Ns iterations
    ###########################################################
    ctr = 0
    while n_set < Ns:  # for loop but using while because it may restart
        ctr += 1
        if ctr >= 60:  # Just in case, cap the number of times we keep trying
            return None

        restart = True
        while restart:
            new_expr = sample_constants(skeleton)
            if new_expr.is_Number or 'x' not in str(new_expr):
                continue
            # print(new_expr)
            args = get_args(new_expr)
            # If there's a coefficient that is too big, clip it to 50
            if any(np.abs(args) > 50):
                args = [arg if np.abs(arg) <= 50 else 50 * np.sign(arg) for arg in args]
                new_expr = set_args(new_expr, args)
            # new_expr = safe_pow(new_expr)

            # If the expression contains a bounded operation, modify the constants to avoid NaN values
            # new_expr = sympify('1/(x+4)**3+exp(1/(x-4))', locals={'x': varX})
            # new_expr = sympify('-1/(x)', locals={'x': varX})
            # new_expr = sympify('-1/(x**5)', locals={'x': varX})
            # new_expr = sympify('-3.12*x/sin(1.45*x)-2.2', locals={'x': varX})
            # new_expr = sympify('1/(sin(6.3335*exp(x))-1.5)', locals={'x': varX})
            # new_expr = sympify('4.024 + 0.394*Abs((2.842*x + 4.628)*Abs(x) + 3.279)**(11/3)*sign((-2.842*x - 4.628)*Abs(x) - 3.279)/(4.94*Abs(2.967*x**6 + 0.908*x**5 + 0.311*x**4 - 4.433*x**3 + 1.552*x**2 + 1.885*x) - 1.819)', locals={'x': varX})
            # new_expr = sympify('-1.041*acosh(-0.953 + 2.123/(x + 3.181))', locals={'x': varX})
            if is_bounded:
                # try:
                result = modify_constants_avoidNaNs(new_expr, support, bounded_operations(), curr_p, Xbounds,
                                                    variable=varX, extrapolate=extrapolate)
                # except Exception:
                #     continue
                if result is None:
                    n_set = 0
                    continue
                support, new_expr, _, _ = result
                if str(new_expr) == 'nan':
                    continue
                # If the new expression needs coefficients that are too large, try again
                args = get_args(new_expr)
                if any(np.abs(args) > 100) or any([not (aa.is_Float or aa.is_Integer) for aa in args]) or args is None:
                    continue

            # If there's a coefficient that is too small, replace it by zero
            if any(np.abs(args) < 0.000001):
                args = [arg if np.abs(arg) >= 0.000001 else 0 for arg in args]
                new_expr2 = set_args(new_expr, args)
                # If after simplification of the new expression, the variable is removed, do not consider this equation
                args = get_args(new_expr2)
                if 'x' not in str(new_expr2):
                    continue
                elif any([not (aa.is_Float or aa.is_Integer) for aa in args]) or args is None:
                    continue
                else:
                    new_expr = new_expr2
            restart = False

        vals= safe_evaluation(new_expr, support, varX)
        try:
            if len(vals) != 10000:
                print("Failed")
                continue
        except Exception:
            pass
        means, std = np.mean(vals, axis=0), np.std(vals, axis=0)
        vals = (vals - means) / std

        X[:, n_set] = support
        Y[:, n_set] = vals
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X[:, n_set] = X[indices, n_set]
        Y[:, n_set] = Y[indices, n_set]
        sampled_exprs[n_set] = (new_expr - means) / std

        if any(is_nan(vals)):  # If there's an undefined value, try again
            continue
        elif std == 0 or 'x' not in str(new_expr) or 'x' not in str(new_expr):
            continue

        if std > 0.01: #  and not any(is_nan(vals)):  # If the result is too flat, try again
            if np.isnan(vals).any() or np.isinf(vals).any() or np.isnan(means) or \
                    np.isinf(means) or np.isnan(std) or np.isinf(std):
                continue

            # If there's not enough samples between -4 and 4, try again (it avoids having functions with big gaps)
            selected_indices = np.where((support >= -4) & (support <= 4))[0]
            dt = (maxX - minX) * 0.7  # If at least 3/4 of the domain space is not covered, try again
            if len(selected_indices) < 2500 or np.std(support) < 3 or np.max(support) - np.min(support) < dt:
                if (maxX - minX) * 0.5 < np.max(support) - np.min(support):
                    support, sampled_exprs[n_set] = rescale_expression(sampled_exprs[n_set], varX, support, (-10, 10))
                else:
                    continue

            n_set += 1
        else:
            continue

    return X, Y, sampled_exprs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    expr = 'C*log(L) + C + L*sinh(C*inv(P) + C)'  # 'C*powu(C*sin(L) + C) + C'
    ct = 0
    # while ct < 50:
    #     print(sample_constants(expr))
    #     ct += 1

    Ns = 10
    xx = generate_data(expr, n_sets=Ns)

    n_cols = int(np.ceil(Ns / 2))
    fig, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(4 * n_cols, 6))
    axes_flat = axes.flatten()
    for i in range(Ns):
        ax = axes_flat[i]
        ax.scatter(xx[0][:, i], xx[1][:, i], alpha=0.6, edgecolors='none', s=10)
        ax.set_title(f'Set {i + 1}')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.autoscale(enable=True, axis='both', tight=False)
    plt.tight_layout()
    plt.show()
    pass