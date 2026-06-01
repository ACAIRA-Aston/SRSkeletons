from typing import List, Set, Union, Dict
from sympy import Symbol
from src.grammar.engine import Node


class StructuralGrammar:
    """
    Defines the context-free grammar G used to generate univariate skeletons in their recursive canonical form.
    """
    def __init__(self, unary_operators: List[str] = None):
        if unary_operators is None:  # Default fallback list if none provided (for debugging mainly)
            unary_operators = ['sin', 'cos', 'exp', 'log', 'sqrt', 'tan', 'tanh']

        self.start_symbol = 'σ'  # Start Symbol
        self.non_terminals: Set[str] = {'σ', 'τ', 'Ω', 'Φ', 'Ψ', 'Atom'}  # Non-Terminals
        self.terminals: Set[str] = {'+', '*', '(', ')', '/', 'C', 'L', 'P'}  # Terminal Symbols
        self.terminals.update(unary_operators)
        # Production Rules
        self.productions: Dict[str, List[List[str]]] = {
            'σ': [ ['C', '+', 'Ω'],
                   ['C', '+', 'Ω', '+', 'τ']  ],
            'τ': [ ['C', '+', 'Ω'],
                   ['C', '+', 'Ω', '+', 'τ'] ],
            'Ω': [ ['C', '*', 'Φ'] ],
            'Φ': [ ['Ψ'],
                   ['Ψ', '*', 'Φ'] ],
            'Ψ': [ ['Atom'] ] + [[op, '(', 'τ', ')'] for op in unary_operators ],
            # 'Ψ': [ ['Atom'] ] + [[op, '(', 'τ', ')'] if op != 'inv' else ['/', '(', 'τ', ')'] for op in unary_operators ],
            'Atom': [ ['L'], ['P'] ]
        }

    def get_productions(self, nt: Union[str, Node]) -> List[List[str]]:
        """Retrieves the valid expansion choices for a given non-terminal."""
        symbol = nt.symbol if hasattr(nt, 'symbol') else nt
        return self.productions.get(symbol, [])

    def is_fully_resolved(self, expr) -> bool:
        """
        Recursively inspects a derivation tree node to verify if it contains exclusively terminal symbols
        """
        if all(ch not in self.non_terminals for ch in expr) and 'Atom' not in expr:
            pass
        return all(ch not in self.non_terminals for ch in expr) and 'Atom' not in expr

    def flatten_to_string(self, node) -> str:
        """
        Converts a fully expanded derivation tree into a mathematical string representation ready for SymPy parsing.
        """
        if isinstance(node, Symbol):
            return str(node)
        if not node.children:
            return node.symbol
        return "".join(self.flatten_to_string(child) for child in node.children)
