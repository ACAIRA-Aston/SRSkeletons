import collections
from typing import Set
from src.symbolic.normalizer import canonical_sort
from src.utils.constraints import count_nu, max_nd, check_constraints_mult, check_constraints_add, \
    check_power_constraints
from src.symbolic.reducer import simplify, is_less_complex, prune_linear_constants
from src.grammar.engine import find_leftmost_nt, apply_rule, get_subtree, update_subtree, Node


def generate_exhaustive_skeletons(grammar, n_u: int, n_d: int, unary_ops: Set[str]) -> Set[str]:
    """
    Executes a minimal and exhaustive set of unique functional skeletons S.
    """
    S = set()
    Q = collections.deque([Node(grammar.start_symbol)])
    seen_hashes = {hash(grammar.start_symbol)}

    while Q:
        expr = Q.popleft()

        expr_str = grammar.flatten_to_string(expr)
        if grammar.is_fully_resolved(expr_str):
            # Final algebraic reduction and global sorting before adding to S
            s_sorted = canonical_sort(simplify(expr_str))
            s_str = str(s_sorted)
            S.add(s_str)
            print(s_str)
            continue

        nt = find_leftmost_nt(expr, grammar.non_terminals)
        # Expand based on production rules in G
        for production in grammar.get_productions(nt):
            if nt.symbol == 'τ' and (
                    nt.parent.symbol == 'τ' or nt.parent.symbol == 'σ'):  # If it's a second-generation τ, don't add a constant C
                production = ['Ω']
            if nt.symbol == 'τ' and (
                    nt.parent.children[0].symbol == 'exp'):  # If it's a second-generation τ, don't add a constant C
                production = ['Ω']
            path = apply_rule(expr, production, grammar, n_u=n_u)
            if path is None:
                continue

            # Recursive parent look-back to identify the highest-level resolved branch affected by the expansion
            s = get_subtree(path, grammar.non_terminals)

            if s is not None:
                ss = grammar.flatten_to_string(s)
                s_prime = simplify(ss)
                if is_less_complex(s_prime, ss):  # Reject path if s' can be simplified
                    continue

                # Sort the resolved branch to ensure path uniqueness
                s_can = canonical_sort(s_prime)
                path = update_subtree(path, s, s_can, grammar.non_terminals)

            # Ensure path satisfies complexity bounds and hasn't been seen before
            path_str = grammar.flatten_to_string(path)
            path_hash = hash(path_str)

            if path_hash in seen_hashes:
                continue

            if check_power_constraints(path_str):
                continue

            c_nu = count_nu(path, unary_ops)

            if (c_nu <= n_u and path_str.count('P') <= 1 and path_str.count('L') <= 2) and max_nd(path, unary_ops) <= n_d:
                if check_constraints_mult(path, n_u, c_nu) and check_constraints_add(path, n_u, c_nu):
                    Q.append(path)
                    seen_hashes.add(path_hash)
    return S
