"""Grammar V1: paired Head Initial (HI) / Head Final (HF) CFG.

Single source of truth for the grammar documented in GRAMMAR_V1.md (same
folder). Instead of maintaining two separate nltk CFG strings that could
silently drift apart, ``RULES`` pairs every HI production with its HF
production (the HF right-hand side is a permutation of the HI one). Everything
else is derived from this table:

- ``hi_cfg()`` / ``hf_cfg()``: ``nltk.CFG`` objects, frame-level or lexicalized
- ``cfg_string()``: the underlying nltk grammar strings, if you want to inspect
  or export them
- ``hi_to_hf()``: mirrors an HI parse tree into its HF tree (the hierarchical
  reversal rule), driven by the same rule pairs
- ``enumerate_trees()`` / ``generate_frames()``: exhaustive enumeration of all
  derivations up to a token-length bound, used for frame generation and for
  the ambiguity tests

The derived functions take the tables (rules, terminal categories, lexicon)
as optional arguments defaulting to the V1 tables, so that other grammars in
the same paired-rule format (grammar_number_agreement.py) reuse them.

Frame-level grammar: the terminal categories (N, V_trans, ...) are the
terminals, so derivations yield "syntactic frames" such as ``N V_trans N``
that are later populated with lexical items. Because the lexica of the
terminal categories are disjoint (enforced by tests), any word-level string
maps back to exactly one frame, so ambiguity results proven at the frame
level transfer to the word level.
"""

from functools import lru_cache

from nltk import CFG, Tree

START = "S"

TERMINAL_CATEGORIES = ("N", "V_intrans", "V_trans", "Adv", "C")

# (lhs, hi_rhs, hf_rhs) — hf_rhs must be a permutation of hi_rhs, and no rule
# may repeat a symbol in its RHS (otherwise the HI -> HF mirroring is not
# well defined). Both properties are enforced by the integration tests.
RULES = [
    ("S", ("NP", "VP"), ("NP", "VP")),
    ("NP", ("N",), ("N",)),
    ("NP", ("N", "CP_subj_gap"), ("CP_subj_gap", "N")),
    ("NP", ("N", "CP_obj_gap"), ("CP_obj_gap", "N")),
    ("VP", ("VP_trans",), ("VP_trans",)),
    ("VP", ("VP_intrans",), ("VP_intrans",)),
    ("VP_trans", ("V_trans", "NP"), ("NP", "V_trans")),
    ("VP_trans", ("VP_trans_adv", "NP"), ("NP", "VP_trans_adv")),
    ("VP_trans_adv", ("V_trans", "AdvP"), ("AdvP", "V_trans")),
    ("VP_intrans", ("V_intrans",), ("V_intrans",)),
    ("VP_intrans", ("VP_intrans_adv",), ("VP_intrans_adv",)),
    ("VP_intrans_adv", ("V_intrans", "AdvP"), ("AdvP", "V_intrans")),
    ("AdvP", ("Adv",), ("Adv",)),
    ("CP_subj_gap", ("C", "VP"), ("VP", "C")),
    ("CP_obj_gap", ("C", "S_obj_gap"), ("S_obj_gap", "C")),
    # the embedded subject does not swap with its verb; this is what makes
    # object gaps OOD under the linear reversal rule, and it is also the
    # source of the known HF subj-gap/obj-gap string ambiguity
    ("S_obj_gap", ("NP", "VP_obj_gap"), ("NP", "VP_obj_gap")),
    ("VP_obj_gap", ("V_trans",), ("V_trans",)),
    ("VP_obj_gap", ("VP_trans_adv",), ("VP_trans_adv",)),
]

# 26 words per category (matching the a-z animal nouns); vocabularies must be
# disjoint across categories or word-level parses stop being unique
LEXICON = {
    "N": (
        "ant bird cat dog elephant flamingo giraffe hippo iguana jaguar koala lemur monkey "
        "newt owl penguin quail rabbit shark tiger urchin vulture walrus xerus yak zebra"
    ).split(),
    "V_intrans": (
        "swims dances sings laughs smiles sleeps jumps claps runs walks hops crawls flies "
        "dives spins yawns snores sneezes stretches shivers wanders naps whistles hums stumbles twirls"
    ).split(),
    "V_trans": (
        "chases pursues sees likes meets follows watches greets admires avoids loves hates knows "
        "hugs kisses tickles teases trusts helps fears finds grabs lifts carries pokes nudges"
    ).split(),
    "Adv": (
        "quickly slowly hastily mindfully quietly loudly eagerly calmly gently fiercely gracefully "
        "carefully clumsily happily sadly proudly bravely boldly shyly sweetly wildly softly briskly "
        "lazily keenly warily"
    ).split(),
    "C": ["that"],
}


def cfg_string(order="hi", lexicalized=False, rules=None, terminals=None, lexicon=None):
    """Render the grammar as an nltk-parseable CFG string.

    order: "hi" or "hf".
    lexicalized=False: terminal categories are the terminals (frame grammar).
    lexicalized=True: terminal categories stay nonterminal and expand to words.
    rules / terminals / lexicon: the grammar tables (default: Grammar V1).
    """
    if order not in ("hi", "hf"):
        raise ValueError(f"order must be 'hi' or 'hf', got {order!r}")
    rules = RULES if rules is None else rules
    terminals = TERMINAL_CATEGORIES if terminals is None else terminals
    lexicon = LEXICON if lexicon is None else lexicon

    def render(symbol):
        if not lexicalized and symbol in terminals:
            return f"'{symbol}'"
        return symbol

    lines = []
    for lhs, hi_rhs, hf_rhs in rules:
        rhs = hi_rhs if order == "hi" else hf_rhs
        lines.append(f"{lhs} -> {' '.join(render(s) for s in rhs)}")
    if lexicalized:
        for category in terminals:
            lines.append(f"{category} -> {' | '.join(repr(w) for w in lexicon[category])}")
    return "\n".join(lines)


@lru_cache(maxsize=None)
def hi_cfg(lexicalized=False):
    return CFG.fromstring(cfg_string("hi", lexicalized))


@lru_cache(maxsize=None)
def hf_cfg(lexicalized=False):
    return CFG.fromstring(cfg_string("hf", lexicalized))


@lru_cache(maxsize=None)
def _hf_child_order(rules):
    """(lhs, hi_rhs) -> positions of the HI children in HF order."""
    return {(lhs, hi_rhs): tuple(hi_rhs.index(s) for s in hf_rhs) for lhs, hi_rhs, hf_rhs in rules}


def hi_to_hf(tree, rules=None, terminals=None):
    """Mirror an HI parse tree into its HF tree (the hierarchical reversal rule).

    Works on frame-level trees (leaves are terminal categories) and on
    lexicalized trees (leaves are words under terminal-category preterminals,
    whose expansions are order-preserving).
    """
    child_order = _hf_child_order(tuple(RULES if rules is None else rules))
    terminals = TERMINAL_CATEGORIES if terminals is None else terminals

    def mirror(node):
        if not isinstance(node, Tree):
            return node
        child_symbols = tuple(c.label() if isinstance(c, Tree) else c for c in node)
        order = child_order.get((node.label(), child_symbols))
        if order is not None:
            children = [mirror(node[i]) for i in order]
        elif node.label() in terminals:
            children = list(node)
        else:
            raise ValueError(f"no paired rule for {node.label()} -> {child_symbols}")
        return Tree(node.label(), children)

    return mirror(tree)


def enumerate_trees(order="hi", max_len=13, rules=None, terminals=None, start=START):
    """Exhaustively enumerate every derivation whose frame has <= max_len tokens.

    Returns a list of (frame_tokens, nltk.Tree) pairs, with terminal categories
    as the leaves. The enumeration is exhaustive: recursion is bounded by the
    yield-length budget, with each child reserving the minimum yield lengths of
    its remaining siblings so that recursive cycles strictly shrink the budget.
    """
    rules = RULES if rules is None else rules
    terminals = TERMINAL_CATEGORIES if terminals is None else terminals
    productions = {}
    for lhs, hi_rhs, hf_rhs in rules:
        productions.setdefault(lhs, []).append(hi_rhs if order == "hi" else hf_rhs)

    # fixed-point minimum yield length per symbol
    min_len = {s: 1 for s in terminals}
    min_len.update({lhs: float("inf") for lhs in productions})
    changed = True
    while changed:
        changed = False
        for lhs, rhss in productions.items():
            best = min(sum(min_len[s] for s in rhs) for rhs in rhss)
            if best < min_len[lhs]:
                min_len[lhs] = best
                changed = True

    memo = {}

    def gen(symbol, budget):
        if budget < min_len[symbol]:
            return []
        if symbol in terminals:
            return [((symbol,), symbol)]
        key = (symbol, budget)
        if key not in memo:
            results = []
            for rhs in productions[symbol]:
                partials = [((), ())]
                for i, child in enumerate(rhs):
                    reserved = sum(min_len[s] for s in rhs[i + 1:])
                    partials = [
                        (frame + child_frame, subtrees + (child_tree,))
                        for frame, subtrees in partials
                        for child_frame, child_tree in gen(child, budget - len(frame) - reserved)
                    ]
                results.extend((frame, (symbol, subtrees)) for frame, subtrees in partials)
            memo[key] = results
        return memo[key]

    def to_tree(node):
        if isinstance(node, str):
            return node
        symbol, subtrees = node
        return Tree(symbol, [to_tree(c) for c in subtrees])

    return [(list(frame), to_tree(node)) for frame, node in gen(start, max_len)]


def generate_frames(order="hi", max_len=13, rules=None, terminals=None):
    """All distinct syntactic frames (space-joined category strings) up to max_len tokens."""
    return sorted({" ".join(frame) for frame, _ in enumerate_trees(order, max_len, rules, terminals)})
