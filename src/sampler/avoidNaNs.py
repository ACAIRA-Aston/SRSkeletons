import copy
import numpy as np
from src.utils.sympy_utils import is_nan
from sympy import sympify, diff, lambdify, Abs
from src.utils.sympy_utils import safe_evaluation


def modify_constants_avoidNaNs(expr, x, bounded_ops, npoints, Xbounds, variable=sympify('x'), extrapolate=False,
                               excl_intervals=None, alpha=20, in_trig=False):
    """Modifies the constants inside unary bounded operations to avoid generating NaN values
    :param expr: Original expression
    :param x: Support vector of variable x values
    :param bounded_ops: Dictionary including the admissible values inside each bounded operation
    :param npoints: Number of points used for support
    :param Xbounds: Minimum and maximum support values
    :param variable: Name of the Sympy variable contained in the expression. Default: 'x'
    :param extrapolate: If True, sample support values that go beyond the training domain
    :param excl_intervals: List of continuous tuples (start, end) defining forbidden domains
    :param alpha: Hyperparameter threshold for relative gradient evaluation (|f'/f| >= alpha)
    :param in_trig: If True, indicates that the current analyzed expression is inside a trig. function
    """
    if excl_intervals is None:
        excl_intervals = []

    if extrapolate:
        minb, maxb = Xbounds[0] * 2, Xbounds[1] * 2
    else:
        minb, maxb = Xbounds[0], Xbounds[1]

    args = expr.args
    new_args = []

    contain_singularity = False
    for arg in args:
        # Check if there's any unary and bounded function inside this argument
        contains_unary_bounded = False
        if (any([f in str(arg) for f in list(bounded_ops[0].keys())]) or any([f in str(arg) for f in bounded_ops[-1]]) or
            any([f in str(arg) for f in list(bounded_ops[1].keys())]) or '/' in str(arg) or
            any([f in str(arg) for f in list(bounded_ops[2].keys())]) or '**0.5' in str(arg) or
            '**(-' in str(arg)) and not arg.is_number:
            contains_unary_bounded = True

        if not contains_unary_bounded:
            new_args.append(arg)
        else:  # If there's a unary and bounded function inside this argument, explore a lower level of the tree
            # Check if the current function is sqrt = Pow and the second argument is 1/2
            is_sqrt, is_div = False, False
            if 'Pow' in str(arg.func):
                if np.abs(float(arg.args[1])) == 1/2:
                    is_sqrt = True
                if arg.args[1] < 0:  # If exponent is negative, is a division
                    is_div = True

            # Check if the function at the current level of the tree is unary and bounded
            if any([f == str(arg.func) for f in list(bounded_ops[0].keys())]) or \
                    any([f == str(arg.func) for f in list(bounded_ops[1].keys())]) or \
                    any([f == str(arg.func) for f in list(bounded_ops[2].keys())]) or \
                    any([f == str(arg.func) for f in bounded_ops[3]]) or is_sqrt or is_div:
                # Now check if there's any other unary and bounded function inside the current function
                # Note that inside this IF statement, it is true that arg is a unary and bounded function, so it has
                # only one argument
                intrig = False
                if any([f == str(arg.func) for f in ['sin', 'cos', 'tan', 'tanh', 'atan']]) : # or '/' in str(arg):
                    intrig = True
                arg_init = copy.deepcopy(arg)
                if any([f in str(arg.args[0]) for f in list(bounded_ops[0].keys())]) or \
                        any([f in str(arg.args[0]) for f in list(bounded_ops[1].keys())]) or \
                        any([f in str(arg.args[0]) for f in list(bounded_ops[2].keys())]) or \
                        any([f in str(arg.args[0]) for f in bounded_ops[3]]) or \
                        '**0.5' in str(arg.args[0]) or '**(-' in str(arg.args[0]) or '/' in str(arg.args[0]):
                    # If that's the case, go to a deeper level of the tree to obtain new function arguments
                    x, arg, avoid_x, contain_singularity = modify_constants_avoidNaNs(arg, x, bounded_ops, npoints,
                                                            Xbounds, variable, extrapolate=extrapolate,
                                                            excl_intervals=excl_intervals, alpha=alpha, in_trig=intrig)
                # Check if the operation function has changed. If yes, the new argument must be a number
                # For example, suppose the original expr is (4x + 7)**.5 and it changes to (4x + 16)**.5, then Sympy will
                # factorize it as 2(x + 4)**.5
                constant = 1
                change_detected = False
                if arg_init.func != arg.func:
                    # if arg.args[0].is_number:
                    change_detected = True
                    constant = arg.args[0]
                    args2 = arg.args[1].args[0]
                else:
                    args2 = arg.args[0]

                # Evaluate the support vector X to find NaNs
                vals = safe_evaluation(arg, x, variable)
                are_nans = any(is_nan(vals))
                if are_nans or ('exp' in str(arg_init.func) and 0 in vals) or \
                        any([f == str(arg_init.func) for f in list(bounded_ops[2].keys())]) or \
                        any([f == str(arg_init.func) for f in ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh']]) or \
                        is_div or is_sqrt or in_trig:
                    # If NaN values were found, evaluate the values of the arguments inside the current function
                    vals_arg = safe_evaluation(args2, x, variable)

                    if any([f == str(arg_init.func) for f in ['sin', 'cos', 'tan']]):
                        arg_range = np.max(vals_arg) - np.min(vals_arg)
                        if arg_range > 150:
                            args2 = args2 / arg_range * 150

                    # If it's a single-bounded unary operation, add an additive constant to avoid NaNs
                    if any([f == str(arg_init.func) for f in list(bounded_ops[0].keys())]) or is_sqrt:
                        if is_sqrt:
                            bound_type, bound = bounded_ops[0].get('sqrt')
                        else:
                            find = \
                                np.where(np.array([f == str(arg_init.func) for f in list(bounded_ops[0].keys())]))[0][0]
                            current_function = list(bounded_ops[0].keys())[find]
                            bound_type, bound = bounded_ops[0].get(current_function)
                        if bound_type == 'min':
                            # If there are more NaNs than actual values, try multiplying vals_arg by -1
                            NaNs = np.sum(is_nan(vals))
                            if NaNs > (len(vals) - NaNs):
                                args2 = - args2
                                vals_arg = - vals_arg
                            # Check if after the previous modification, there's still values in vals_arg that are less than the minimum bound
                            if any(np.array(vals_arg) < bound):
                                # If that's the case, add a horizontal offset
                                minVal = np.min(vals_arg)
                                offset = bound - minVal
                                args2 = args2 + offset + 0.1
                                if offset > 10:
                                    args2 = args2 / offset * 10
                                    vals_arg = safe_evaluation(args2, x, variable)
                                    minVal = np.min(vals_arg)
                                    offset = bound - minVal
                                    args2 = args2 + offset + 0.1
                        else:
                            # Max-bounded operations are simply divided and multiplied
                            if np.any(np.abs(vals_arg) > bound):
                                args2 = args2 / np.max(np.abs(vals_arg)) * bound

                    elif any([f == str(arg_init.func) for f in list(bounded_ops[1].keys())]):
                        # Double-bounded ops: add multiplicative and additive constants to bound the values of vals_arg
                        find = np.where(np.array([f in str(arg_init.func) for f in list(bounded_ops[1].keys())]))[0][0]
                        current_function = list(bounded_ops[1].keys())[find]
                        [min_bound, max_bound] = bounded_ops[1].get(current_function)
                        # Min-max normalization
                        args2 = (args2 - np.min(vals_arg)) / (
                                np.max(vals_arg) - np.min(vals_arg))  # Scaled between 0 and 1
                        args2 = args2 * (max_bound - min_bound) + min_bound  # Scaled between min_bound and max_bound

                    # REDESIGNED: Analytical Singularity handling using gradients
                    elif str(arg_init.func) not in ['sin', 'cos'] or ('Pow' in str(arg.func) and arg.args[1] == -1):
                        # For operations with singularities such as TAN, resample the support vector to avoid singularities
                        # Drop input-output pairs containing NaNs and entries with an absolute value of y above 10000

                        if not in_trig:
                            if is_sqrt or is_div:
                                arg = arg_init.func(*[args2, arg_init.args[-1]])
                            else:
                                arg = arg_init.func(*[args2])
                            new_support, excl_intervals = handle_singularities(expr=arg, variable=variable,
                                                                               n_points=npoints,
                                                                               minb=minb, maxb=maxb,
                                                                               excl_intervals=excl_intervals, alpha=alpha)
                            x = new_support
                            contain_singularity = True
                        elif are_nans:
                            new_support, excl_intervals = handle_nan_singularities(x_grid=x,
                                                                                   minb=minb, maxb=maxb,
                                                                                   excl_intervals=excl_intervals,
                                                                                   y_vals=vals, n_points=npoints)
                            x = new_support

                    # if any(['exp' in str(arg_init.func) or '**3' in str(arg_init.func) or
                    #         '**4' in str(arg_init.func) or '**5' in str(arg_init.func) or '**6' in str(arg_init.func)]):
                    #     arg_function = lambdify(variable, args2)
                    #     vals_arg = np.array(arg_function(x))
                    #     if np.max(vals_arg) > 5:
                    #         args2 = args2 / np.max(np.abs(vals_arg)) * 5

                if is_sqrt or is_div:
                    arg = arg_init.func(*[args2, arg_init.args[-1]])
                else:
                    arg = arg_init.func(*[args2])

                if change_detected:
                    arg = constant * arg
            else:
                result = modify_constants_avoidNaNs(arg, x, bounded_ops, npoints, Xbounds, variable, in_trig=in_trig,
                                                    extrapolate=extrapolate, excl_intervals=excl_intervals, alpha=alpha)
                if result is None:
                    return None
                x, arg, excl_intervals, contain_singularity = result

            new_args.append(arg)

    new_xp = expr.func(*new_args)
    # if contain_singularity:
    #     eval = safe_evaluation(new_xp, x, variable)
    #     x, excl_intervals = handle_singularities(expr=new_xp, variable=variable, n_points=npoints, minb=minb, maxb=maxb,
    #                                                excl_intervals=excl_intervals, alpha=alpha)

    return x, new_xp, excl_intervals, contain_singularity


def handle_singularities(expr, variable, n_points, minb, maxb, excl_intervals=None, alpha=1e4):
    """Globally identifies intervals where |f'(x)/f(x)| >= alpha and isolates them"""
    if excl_intervals is None:
        excl_intervals = []

    # Derive relative gradient expression: | f'(x) / f(x) |
    f_prime = diff(expr, variable)
    try:
        rel_grad = f_prime  # Abs(f_prime / expr)
        rel_grad_func = lambdify(variable, rel_grad, 'numpy')
    except Exception:
        # Fallback if symbolic derivation fails completely
        return np.linspace(minb, maxb, n_points), excl_intervals

    # Evaluate condition globally across a dense initial grid
    grid_size = 20000
    x_grid = np.linspace(minb, maxb, grid_size)

    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            grad_vals = rel_grad_func(x_grid)
            slope_metric = np.abs(grad_vals) / np.sqrt(1 + grad_vals ** 2)
            # Mark point as unsafe if it violates alpha, or data in NaN/Inf
            # unsafe_mask = (grad_vals >= alpha) | np.isnan(grad_vals) | np.isinf(grad_vals)
            unsafe_mask = (slope_metric >= 0.9995) | np.isnan(slope_metric) | np.isinf(grad_vals)
    except Exception:
        unsafe_mask = np.zeros_like(x_grid, dtype=bool)

    # Extract contiguous segments of unsafe indices into exclusion intervals
    in_unsafe = False
    start_x = None

    for i, x_val in enumerate(x_grid):
        if unsafe_mask[i]:
            if not in_unsafe:
                start_x = x_val
                in_unsafe = True
        else:
            if in_unsafe:
                excl_intervals.append((start_x, x_grid[i - 1]))
                in_unsafe = False
    if in_unsafe:
        excl_intervals.append((start_x, maxb))

    # Merge overlapping intervals and find the surviving safe subdomains
    def merge_and_invert(intervals, b_min, b_max):
        if not intervals:
            return [(b_min, b_max)]

        sorted_int = sorted([(max(b_min, s), min(b_max, e)) for s, e in intervals])
        merged = [sorted_int[0]]
        for curr in sorted_int[1:]:
            if curr[0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], curr[1]))
            else:
                merged.append(curr)

        valid = []
        curr_pos = b_min
        for s, e in merged:
            if s > curr_pos + 1e-6:
                valid.append((curr_pos, s))
            curr_pos = max(curr_pos, e)
        if curr_pos < b_max - 1e-6:
            valid.append((curr_pos, b_max))
        return valid

    valid_intervals = merge_and_invert(excl_intervals, minb, maxb)

    # Proportionally sample equidistant points across safe regions
    total_length = sum(end - start for start, end in valid_intervals)
    generated_points = []

    if total_length > 0:
        for start, end in valid_intervals:
            if end - start < 1e-2:
                continue
            points_in_range = max(2, int(np.round(n_points * (end - start) / total_length)))
            if start == -10:
                generated_points.extend(np.linspace(start, end - 0.001, points_in_range))
            elif end == 10:
                generated_points.extend(np.linspace(start + 0.001, end, points_in_range))
            else:
                generated_points.extend(np.linspace(start + 0.001, end - 0.001, points_in_range))


        generated_points = np.sort(np.unique(generated_points))

        # Enforce exact array length constraint
        if len(generated_points) > n_points:
            idx = np.round(np.linspace(0, len(generated_points) - 1, n_points)).astype(int)
            generated_points = generated_points[idx]
        elif len(generated_points) < n_points and len(generated_points) > 0:
            while len(generated_points) < n_points:
                r_range = valid_intervals[np.random.choice(len(valid_intervals))]
                generated_points = np.append(generated_points, np.random.uniform(r_range[0], r_range[1]))
            generated_points = np.sort(generated_points)

    if len(generated_points) == 0:
        generated_points = np.linspace(minb, maxb, n_points)

    return np.array(generated_points), excl_intervals


def handle_nan_singularities(x_grid, minb, maxb, y_vals, n_points, excl_intervals=None):
    """Globally isolates domain spaces where safe_evaluation yields NaN/Inf, adding a 0.01 distance buffer."""
    if excl_intervals is None:
        excl_intervals = []

    buffer = 0.01

    try:
        unsafe_mask = np.isnan(y_vals) | np.isinf(y_vals)
    except Exception:
        unsafe_mask = np.zeros_like(x_grid, dtype=bool)

    # Group contiguous blocks of NaNs into single intervals padded with the 0.01 threshold
    in_unsafe = False
    start_x = None

    for i, x_val in enumerate(x_grid):
        if unsafe_mask[i]:
            if not in_unsafe:
                start_x = x_val
                in_unsafe = True
        else:
            if in_unsafe:
                excl_intervals.append((start_x - buffer, x_grid[i - 1] + buffer))
                in_unsafe = False
    if in_unsafe:
        excl_intervals.append((start_x - buffer, maxb + buffer))

    # Merge overlapping intervals and find the surviving safe subdomains
    def merge_and_invert(intervals, b_min, b_max):
        if not intervals:
            return [(b_min, b_max)]
        sorted_int = sorted([(max(b_min, s), min(b_max, e)) for s, e in intervals])
        merged = [sorted_int[0]]
        for curr in sorted_int[1:]:
            if curr[0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], curr[1]))
            else:
                merged.append(curr)
        valid = []
        curr_pos = b_min
        for s, e in merged:
            if s > curr_pos + 1e-6:
                valid.append((curr_pos, s))
            curr_pos = max(curr_pos, e)
        if curr_pos < b_max - 1e-6:
            valid.append((curr_pos, b_max))
        return valid

    valid_intervals = merge_and_invert(excl_intervals, minb, maxb)

    # Proportionally sample equidistant points across safe regions
    total_length = sum(end - start for start, end in valid_intervals)
    generated_points = []

    if total_length > 0:
        for start, end in valid_intervals:
            if end - start < 1e-2:
                continue
            points_in_range = max(2, int(np.round(n_points * (end - start) / total_length)))
            generated_points.extend(np.linspace(start + 0.001, end - 0.001, points_in_range))

        generated_points = np.sort(np.unique(generated_points))

        # Enforce exact array length constraint
        if len(generated_points) > n_points:
            idx = np.round(np.linspace(0, len(generated_points) - 1, n_points)).astype(int)
            generated_points = generated_points[idx]
        elif 0 < len(generated_points) < n_points:
            while len(generated_points) < n_points:
                r_range = valid_intervals[np.random.choice(len(valid_intervals))]
                generated_points = np.append(generated_points, np.random.uniform(r_range[0], r_range[1]))
            generated_points = np.sort(generated_points)

    if len(generated_points) == 0:
        generated_points = np.linspace(minb, maxb, n_points)

    return np.array(generated_points), excl_intervals