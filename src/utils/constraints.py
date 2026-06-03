from typing import Set


def count_nu(node, unary_ops: Set[str]) -> int:
    """ Calculates the total number of unary operators (N_U) present in the derivation tree """
    count = 0
    if node.symbol in unary_ops and not node.children and not node.symbol in {'inv', 'powu'}:
        count = 1
    return count + sum(count_nu(child, unary_ops) for child in node.children)  # Sum up counts from all branches


def max_nd(node, unary_ops: Set[str]) -> int:
    """ Calculates the maximum nesting depth (N_D) of unary operators."""
    if not node.children:
        return 0

    has_unary_child = any((child.symbol in unary_ops and not child.children) for child in node.children)

    if has_unary_child:
        return 1 + max((max_nd(child, unary_ops) for child in node.children), default=0)

    return max((max_nd(child, unary_ops) for child in node.children), default=0)


def check_constraints_mult(path, n_u: int, c_nu: int):
    """ Before checking constraints, find all Φ symbols. If within their corresponding multiplications there exists
    Atoms AND there are no more available unary operators, then prune the tree. """
    # Find all nodes whose symbol is 'Φ' and have no children
    leaf_phi_nodes = []
    def find_leaf_phis(node):
        if node.symbol == 'Φ' and not node.children:
            leaf_phi_nodes.append(node)
        for child in node.children:
            find_leaf_phis(child)
    find_leaf_phis(path)

    for leaf_phi in leaf_phi_nodes:  # For each node found, check conditions
        # Climb up to find the highest contiguous Φ ancestor
        highest_phi = leaf_phi
        while highest_phi.parent and highest_phi.parent.symbol == 'Φ':
            # Check if there's already another Atom in the multipliers
            L_node = next((child for child in highest_phi.children if child.symbol == 'L'), None)
            P_node = next((child for child in highest_phi.children if child.symbol == 'P'), None)
            A_node = next((child for child in highest_phi.children if child.symbol == 'Atom'), None)
            if (L_node or P_node or A_node) and (n_u == c_nu):
                return False
            highest_phi = highest_phi.parent
        L_node = next((child for child in highest_phi.children if child.symbol == 'L'), None)
        P_node = next((child for child in highest_phi.children if child.symbol == 'P'), None)
        A_node = next((child for child in highest_phi.children if child.symbol == 'Atom'), None)
        if (L_node or P_node or A_node) and (n_u == c_nu):
            return False
    return True


def check_constraints_add(path, n_u: int, c_nu: int):
    """ Before checking constraints, find all second-generation τ symbols. In those cases, the only option to avoid
    being absorbed is using another unary operator; however, if we have already reached the maximum number of unary ops
    allowed, we can prune the tree early. """
    # Find all second-generation 'τ' nodes that have no children
    leaf_tau_nodes = []
    def find_leaf_tau(node):
        if node.symbol == 'τ' and node.parent.symbol == 'τ' and not node.children:
            leaf_tau_nodes.append(node)
        for child in node.children:
            find_leaf_tau(child)
    find_leaf_tau(path)
    if n_u == c_nu and len(leaf_tau_nodes) >= 1:
        return False
    return True