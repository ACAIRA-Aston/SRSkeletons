import sympy as sp


def get_skeleton_var(expr, var, var_names, expand=True):
    skeleton = numeric_to_placeholder(expr, var=var)
    for v in var_names:
        if v != var:
            skeleton = skeleton.subs(sp.sympify(v), sp.sympify('c'))
    skeleton2 = None
    if expand:
        skeleton = avoid_operations_between_constants(sp.expand(sp.expand(sp.sympify('c') * skeleton)))
    while skeleton != skeleton2:
        skeleton2 = skeleton
        skeleton = avoid_operations_between_constants(sp.expand(sp.sympify('c') * skeleton))
        skeleton = numeric_to_placeholder(skeleton, var=var)
    if 'c' not in str(skeleton):
        skeleton = skeleton * sp.sympify('c')
    if 'x' in str(skeleton):  # If there are no variables in the expression, skip adding identifiers
        skeleton = add_constant_identifier(skeleton)[0]
    return skeleton


def numeric_to_placeholder(expr, var=None):
    """Given an expression with numeric constants, replace all constants with placeholder 'c'
   If var is not None, consider any variable different from 'var' a numeric constant"""
    if not expr.args:
        if expr.is_number or (var is not None and expr.is_Symbol and str(expr) != var):
            return sp.sympify('c')
        else:
            return expr

    new_args = []
    for iarg, sub_expr in enumerate(expr.args):
        if iarg == len(expr.args) - 1:
            if isinstance(expr, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
                continue
        new_args.append(numeric_to_placeholder(sub_expr, var=var))

    if isinstance(expr, sp.Pow):
        new_args.append(expr.args[1])

    new_xp = expr.func(*new_args)
    return new_xp


def _avoid_operations_between_constants(xp):
    """ Simplify exponent of constants inside symbolic expression and constant multiplied by constants.
    E.g., constant ** 3 = constant or cons1 * cons2 * cons3 = cons1
    :param xp: Symbolic expression"""
    args = xp.args
    t_args = args
    new_args = []

    if isinstance(xp, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
        args1 = args[0]
        args2 = args[1]
        if (args1.is_number or (
                isinstance(args1, sp.Symbol) and ("cm" in str(args1) or "ca" in str(args1) or "c" in str(args1)))) and \
                (args2.is_number or (isinstance(args2, sp.Symbol) and (
                        "cm" in str(args2) or "ca" in str(args2) or "c" in str(args2)))):
            va = None
            if isinstance(args1, sp.Symbol) and ("cm" in str(args1) or "ca" in str(args1) or "c" in str(args1)):
                va = args1
            elif isinstance(args2, sp.Symbol) and ("cm" in str(args2) or "ca" in str(args2) or "c" in str(args2)):
                va = args2
            t_args = [va, sp.sympify("1")]

    if isinstance(xp, sp.exp):  # If it's an exp function, ignore the power and focus only on the base
        args1 = args[0]
        if (args1.is_number or (
                isinstance(args1, sp.Symbol) and ("cm" in str(args1) or "ca" in str(args1) or "c" in str(args1)))):
            t_args = tuple([sp.sympify("0")])

    if isinstance(xp, sp.Mul) or isinstance(xp, sp.Add):
        t_args = []
        # Check if two or more of the arguments are constants and replace them for just one constant
        flag = False  # This flag turns to True when a constant was already found
        for ia, arg in enumerate(args):
            if arg.is_number and any([(isinstance(co, sp.Symbol) and 'c' in str(co)) for co in (args[0:ia] + args[ia + 1:])]):
                # If the current number is summed or multiplied by other constants, replace it by 0 or one
                if isinstance(xp, sp.Mul):
                    arg = sp.sympify("1")
                else:
                    arg = sp.sympify("0")
            if isinstance(arg, sp.Symbol) and ("cm" in str(arg) or "ca" in str(arg) or "c" in str(arg)):
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
    """Repeat the _avoid_operations_between_constants function until no change is found"""
    xp2 = None
    while xp != xp2:
        xp2 = xp
        xp = _avoid_operations_between_constants(xp)
    return xp

def add_constant_identifier(sk, cm_counter=1, ca_counter=1):
    """Label each coefficient of a skeleton expression"""
    args = sk.args
    new_args = []

    for arg in args:
        if arg.is_number:  # If it's a number, add it to the list
            new_args.append(arg)
        elif isinstance(arg, sp.Symbol) and 'c' in str(arg):
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

