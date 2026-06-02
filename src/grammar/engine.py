import copy
import sympy as sp


class Node:
    """Represents a node in the derivation tree."""
    def __init__(self, symbol, parent=None):
        self.symbol = symbol  # Can be a Non-terminal or Terminal
        self.parent = parent
        self.children = []

    def is_terminal(self, non_terminals):
        """Checks if the symbol belongs to the terminal set Σ."""
        return self.symbol in non_terminals


def find_leftmost_nt(node, non_terminals: set):
    """ Performs a depth-first search to find the leftmost node containing a non-terminal symbol. """
    if node.symbol in non_terminals and not node.children:
        return node

    for child in node.children:
        result = find_leftmost_nt(child, non_terminals)
        if result:
            return result
    return None


def apply_rule(expr_tree, production, grammar, n_u: int):
    """
    Applies a production rule by adding children to the non-terminal node.
    Returns a deep copy of the tree to represent the new derivation path.
    """
    non_terminals = grammar.non_terminals
    new_tree = copy.deepcopy(expr_tree)
    target_node = find_leftmost_nt(new_tree, non_terminals)  # Find the corresponding node in the new tree

    for symbol in production:  # assuming a production is a list of symbols, e.g., ['C', '+', 'Ω']
        target_node.children.append(Node(symbol, parent=target_node))

    new_tree.last_modified_node = target_node


    # Prohibit self-nesting of INV and POWER
    if 'inv' in production or '/' in production:
        highest_psi = None
        currI = target_node.parent.parent
        while currI:
            if currI.symbol == 'Ψ':
                highest_psi = currI
                break
            currI = currI.parent
        if highest_psi:
            if 'inv' in [ch.symbol for ch in highest_psi.children] or '/' in [ch.symbol for ch in highest_psi.children]:
                return None
        # Check if there's another inv in the multiplication
        highest_psi = target_node.parent
        if highest_psi.symbol in {'Φ', 'Ψ'}:
            # If we just added a multiplicative Atom, check there's no other Atom in the multiplication
            while highest_psi.parent and highest_psi.parent.symbol in {'Φ', 'Ω'}:
                I_node = next((child for child in highest_psi.children if child.symbol == 'inv'), None)
                if I_node:  # Cases like this `inv(A)*...*inv(B)`
                    return None
                highest_psi = highest_psi.parent

    if 'powu' in production:
        highest_psi = None
        currI = target_node.parent.parent
        while currI:
            if currI.symbol == 'Ψ':
                highest_psi = currI
                break
            currI = currI.parent
        if highest_psi:
            if 'powu' in [ch.symbol for ch in highest_psi.children]:
                return None
        # TODO  # Check if there's another pow in the multiplication


    curr = target_node
    highest_phi, highest_tau, highest_psi = None, None, None

    if target_node.children[0].symbol == 'Atom':
        ## Check multiplicative redundancies
        # Climb up to find the immediate Φ block containing this target_node
        while curr:
            if curr.symbol in {'Φ', 'Ψ'}:
                highest_phi = curr
                break
            if curr.symbol == 'τ':  # Stop immediately if we cross into an addition boundary
                highest_phi = curr
                break
            curr = curr.parent
        # Ride the Φ chain to the top of this specific multiplication scope
        if highest_phi:
            if highest_phi.symbol in {'Φ', 'Ψ'}:
                # If we just added a multiplicative Atom, check there's no other Atom in the multiplication
                ctr = 1
                while highest_phi.parent and highest_phi.parent.symbol in {'Φ', 'Ω'}:
                    # Check if there's already another Atom in the multipliers
                    L_node = next((child for child in highest_phi.children if child.symbol == 'L'), None)
                    P_node = next((child for child in highest_phi.children if child.symbol == 'P'), None)
                    I_node = next((child for child in highest_phi.children if grammar.flatten_to_string(child) in {'inv(P)', 'inv(L)'}), None)
                    if L_node or P_node or I_node:
                        return None
                    highest_phi = highest_phi.parent
                    ctr += 1
                C_node = next((child for child in highest_phi.children if child.symbol == 'C'), None)
                L_node = next((child for child in highest_phi.children if child.symbol == 'L'), None)
                P_node = next((child for child in highest_phi.children if child.symbol == 'P'), None)
                I_node = next((child for child in highest_phi.children if grammar.flatten_to_string(child) in {'inv(P)', 'inv(L)'}), None)
                if C_node:  # Remove 'C*' in cases 'C*Atom'
                    highest_phi.children.remove(C_node)
                    highest_phi.children.remove(next((child for child in highest_phi.children if child.symbol == '*'), None))
                if L_node or P_node or I_node:
                    return None
                # if ctr >= n_u + 1:
                #     return None

        ## Check additive redundancies
        # Climb up to find the immediate τ block containing this target_node
        curr = target_node
        while curr:
            if curr.symbol in {'τ', 'Ω'}:
                highest_tau = curr
                break
            curr = curr.parent
        # Ride the τ chain to the top
        if highest_tau:
            # The second condition implies that the Atom that has been added, while it belongs to a parent Ω, is not
            # multiplied by anything (from 'C*Atom' it was transformed into just 'Atom')
            if highest_tau.symbol == 'τ' or (highest_tau.symbol == 'Ω' and len(highest_tau.children[0].children) == 1):
                # If we just included an additive Atom, check there's no other Atom in the sum
                while highest_tau.parent and highest_tau.parent.symbol in {'τ', 'σ'}:
                    # Check if there's already another Atom in the multipliers
                    L_node = next((child for child in highest_tau.children if child.symbol == 'L'), None)
                    P_node = next((child for child in highest_tau.children if child.symbol == 'P'), None)
                    if L_node or P_node:
                        return None
                    highest_tau = highest_tau.parent
                C_node = next((child for child in highest_tau.children if child.symbol == 'C'), None)
                if C_node:  # Remove 'C+' in cases 'C+Atom'
                    highest_tau.children.remove(C_node)
                    highest_tau.children.remove(next((child for child in highest_tau.children if child.symbol == '+'), None))
                L_node = next((child for child in highest_tau.children if child.symbol == 'L'), None)
                P_node = next((child for child in highest_tau.children if child.symbol == 'P'), None)
                if L_node or P_node:
                    return None

    if target_node.children[0].symbol == 'L' or target_node.children[0].symbol == 'P':
        ## Check inverse redundancies
        # Climb up to find the immediate unary operator
        curr = target_node
        if curr.parent.symbol == 'Ψ' and curr.parent.parent.symbol == 'Φ'  and curr.parent.parent.parent.symbol == 'Ω'\
                and curr.parent.parent.parent.parent.symbol == 'τ' and curr.parent.parent.parent.parent.parent.symbol == 'Ψ':
            highest_psi = curr.parent.parent.parent.parent.parent
            if any([ch.symbol == 'inv' for ch in highest_psi.children]):  # Verifies that we're inside an INV
                if highest_psi.symbol in {'Φ', 'Ψ'}:
                    # If we just added a multiplicative Atom, check there's no other Atom in the multiplication
                    while highest_psi.parent and highest_psi.parent.symbol in {'Φ', 'Ω'}:
                    # Check if there's already another Atom in the multipliers
                        L_node = next((child for child in highest_psi.children if child.symbol == 'L'), None)
                        P_node = next((child for child in highest_psi.children if child.symbol == 'P'), None)
                        if target_node.children[0].symbol == 'L' and L_node: # Cases like this `L*...*inv(L)`
                            return None
                        if target_node.children[0].symbol == 'L' and P_node:  # Cases like this `P*...*inv(L)`
                            return None
                        if target_node.children[0].symbol == 'P' and L_node:  # Cases like this `L*...*inv(P)`
                            return None
                        if target_node.children[0].symbol == 'P' and L_node:  # Cases like this `L*...*inv(P)`
                            return None
                        highest_psi = highest_psi.parent



    return new_tree


def get_subtree(path_tree, non_terminals: set):
    """
    Performs a recursive look-back starting from the modified node. Identifies the highest parent that is fully resolved
    """
    current = path_tree.last_modified_node
    highest_resolved = None

    while current is not None:
        if _is_node_resolved(current, non_terminals):
            highest_resolved = current
            current = current.parent
        else:
            break

    return highest_resolved


def _is_node_resolved(node, non_terminals: set):
    """ Recursively checks if a node and all its descendants consist entirely of terminal symbols. """
    if not node.children:
        # It's a leaf. Is it a placeholder (NT) or a real value (T)?
        return node.symbol not in non_terminals
    return all(_is_node_resolved(child, non_terminals) for child in node.children)


def sympy_to_node_tree(expr, parent=None) -> Node:
    """
    Converts a simplified SymPy expression back into a token-stream
    Node tree so it can be cleanly reinjected into the BFS pipeline.
    """
    # Base Case: It's a leaf node in SymPy (Symbol, Number, or Constant placeholder)
    if not expr.args:
        return Node(str(expr), parent=parent)

    # Create a structural container node named after the operation type
    container = Node(expr.func.__name__, parent=parent)
    children = []

    if isinstance(expr, sp.Add):
        # Interleave arguments with '+' terminal nodes
        for i, arg in enumerate(expr.args):
            children.append(sympy_to_node_tree(arg, parent=container))
            if i < len(expr.args) - 1:
                children.append(Node('+', parent=container))

    elif isinstance(expr, sp.Mul):
        # Interleave arguments with '*' terminal nodes
        for i, arg in enumerate(expr.args):
            children.append(sympy_to_node_tree(arg, parent=container))
            if i < len(expr.args) - 1:
                children.append(Node('*', parent=container))

    else:
        # It's a functional/unary operator, e.g., sin(L + C)
        op_name = expr.func.__name__.lower()
        children = [
            Node(op_name, parent=container),
            Node('(', parent=container),
            sympy_to_node_tree(expr.args[0], parent=container),
            Node(')', parent=container)
        ]

    container.children = children
    return container


def update_subtree(path_tree, s, s_can, non_terminals):
    """ Replaces a resolved branch with its sorted version. """
    # highest = get_subtree(path_tree, non_terminals)  # Find the highest resolved ancestor that needs to be swapped out
    s_can_node = sympy_to_node_tree(s_can)

    # Extract the parent pointer from the old raw branch we are replacing
    parent = s.parent

    if parent:
        # Find exactly where the old node was sitting and swap it out
        idx = parent.children.index(s)
        parent.children[idx] = s_can_node
        s_can_node.parent = parent

        # Check if our new subtree is a terminal Atom (L or P) inside a product
        if parent.symbol == 'Φ' and s.children[0].symbol == 'Atom':
            grandparent = parent.parent
            if grandparent:
                # Find the specific 'C' and '*' sibling instances inside the grandparent block
                c_node = next((child for child in grandparent.children if child.symbol == 'C'), None)
                star_node = next((child for child in grandparent.children if child.symbol == '*'), None)

                # If both are present, wipe them from the grandparent's layout
                if c_node and star_node:
                    grandparent.children.remove(c_node)
                    grandparent.children.remove(star_node)
        return path_tree
    else:
        # If the root of the entire expression tree was what got resolved
        s_new = Node(s.symbol)
        s_new.children.append(s_can_node)
        return s_new
