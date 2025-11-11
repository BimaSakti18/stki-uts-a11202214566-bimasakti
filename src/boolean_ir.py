from collections import defaultdict
from typing import List, Dict, Set
import os
from preprocess import preprocess_text

def build_inverted_index(doc_texts: Dict[str,str]) -> Dict[str, Set[str]]:
    inv = defaultdict(set)
    for doc_id, text in doc_texts.items():
        tokens = set(preprocess_text(text))
        for t in tokens:
            inv[t].add(doc_id)
    return dict(inv)

def build_incidence_matrix(inv_index: Dict[str, Set[str]], doc_ids: List[str]):
    # returns dict term
    matrix = {}
    for term, postings in inv_index.items():
        matrix[term] = [1 if d in postings else 0 for d in doc_ids]
    return matrix

# very simple boolean evaluator: supports tokens combined with AND, OR, NOT (no parentheses)
def eval_boolean_query(query: str, inv_index: Dict[str, Set[str]], all_docs: Set[str]) -> Set[str]:
    # normalize query
    q = query.lower().strip()
    # tokenization by spaces; expects format: term (AND|OR|NOT) term ...
    parts = q.split()
    # construct postfix-like left-to-right evaluation: handle NOT unary, AND/OR binary (left assoc.)
    # For simplicity, process sequentially:
    def get_posting(tok):
        if tok == 'not':
            return 'NOT'
        if tok in ('and','or'):
            return tok.upper()
        return inv_index.get(tok, set())
    stack = []
    i = 0
    while i < len(parts):
        p = parts[i]
        val = get_posting(p)
        if val == 'NOT':
            # unary: apply to next term
            i += 1
            t = parts[i]
            postings = inv_index.get(t, set())
            stack.append(all_docs - postings)
        elif val in ('AND','OR'):
            stack.append(val)
        else:
            stack.append(val)
        i += 1

    # reduce stack left-to-right
    res = stack[0] if stack else set()
    j = 1
    while j < len(stack):
        op = stack[j]; right = stack[j+1]
        if op == 'AND':
            res = res & right
        elif op == 'OR':
            res = res | right
        else:
            raise ValueError("Unexpected op")
        j += 2
    return res
