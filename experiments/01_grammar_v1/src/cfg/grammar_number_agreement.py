"""Grammar V1 with subject-verb number agreement: paired Head Initial / Head Final CFG.

The rules are exactly those of Grammar V1 (grammar.py) with a number feature
(``sg`` / ``pl``) attached to every symbol that takes part in subject-verb
agreement: nouns, verbs and the phrases that carry a noun's or a verb's
number. Erasing the feature from every symbol gives back the V1 rule set
(``strip_number``; verified by the integration tests), so each V1 tree shape
exists here in 2^(number of nouns) number-decorated variants, and the
all-singular fragment of this grammar is Grammar V1 word for word.

Agreement is written with feature variables (``AGREEING_RULES``): ``{n}`` is
instantiated to the same number at every occurrence within a rule, ``{m}``
independently. This yields three kinds of dependency:

    matrix clause    S -> NP_{n} VP_{n}                     cat swims / cats swim
    subject-gap RC   NP_{n} -> N_{n} CP_subj_gap_{n}         cats that pursue dog swim
                     CP_subj_gap_{n} -> C VP_{n}             (the RC verb agrees with the head noun)
    object-gap RC    S_obj_gap -> NP_{n} VP_obj_gap_{n}      cats that dog hugs tease bird
                     (the RC verb agrees with the embedded subject; the head noun is irrelevant)
    objects are free VP_trans_{n} -> V_trans_{n} NP_{m}      cat chases dogs

In a center-embedded sentence the matrix verb therefore agrees with the matrix
subject across the whole relative clause, whose nouns may carry the other
number ("cats that pursue dog swim": "dog" intervenes between "cats" and
"swim", an agreement attractor), and an object-gap RC adds a second,
independent dependency inside the embedding. ``agreement_dependencies`` and
``n_attractors`` expose both for analysis.

Terminals: N, V_intrans and V_trans are split by number; Adv and C are shared
with V1. The singular and plural forms of one lemma sit at the same index of
LEXICON["N_sg"] / LEXICON["N_pl"] (likewise for verbs), and all categories stay
pairwise disjoint, so a word-level sentence still maps to exactly one frame.

The API mirrors grammar.py (cfg_string, hi_cfg, hf_cfg, hi_to_hf,
enumerate_trees, generate_frames), bound to this grammar's tables.
"""

import re
from functools import lru_cache

from nltk import CFG, Tree

from . import grammar as v1

START = v1.START
NUMBERS = ("sg", "pl")

# Rule templates (lhs, hi_rhs, hf_rhs), the format of grammar.RULES: {n} is
# the agreement number of the clause and takes the same value at every
# occurrence within a rule; {m}, the number of an object NP, is independent.
AGREEING_RULES = [
    ("S", ("NP_{n}", "VP_{n}"), ("NP_{n}", "VP_{n}")),
    ("NP_{n}", ("N_{n}",), ("N_{n}",)),
    ("NP_{n}", ("N_{n}", "CP_subj_gap_{n}"), ("CP_subj_gap_{n}", "N_{n}")),
    ("NP_{n}", ("N_{n}", "CP_obj_gap"), ("CP_obj_gap", "N_{n}")),
    ("VP_{n}", ("VP_trans_{n}",), ("VP_trans_{n}",)),
    ("VP_{n}", ("VP_intrans_{n}",), ("VP_intrans_{n}",)),
    ("VP_trans_{n}", ("V_trans_{n}", "NP_{m}"), ("NP_{m}", "V_trans_{n}")),
    ("VP_trans_{n}", ("VP_trans_adv_{n}", "NP_{m}"), ("NP_{m}", "VP_trans_adv_{n}")),
    ("VP_trans_adv_{n}", ("V_trans_{n}", "AdvP"), ("AdvP", "V_trans_{n}")),
    ("VP_intrans_{n}", ("V_intrans_{n}",), ("V_intrans_{n}",)),
    ("VP_intrans_{n}", ("VP_intrans_adv_{n}",), ("VP_intrans_adv_{n}",)),
    ("VP_intrans_adv_{n}", ("V_intrans_{n}", "AdvP"), ("AdvP", "V_intrans_{n}")),
    ("CP_subj_gap_{n}", ("C", "VP_{n}"), ("VP_{n}", "C")),
    ("S_obj_gap", ("NP_{n}", "VP_obj_gap_{n}"), ("NP_{n}", "VP_obj_gap_{n}")),
    ("VP_obj_gap_{n}", ("V_trans_{n}",), ("V_trans_{n}",)),
    ("VP_obj_gap_{n}", ("VP_trans_adv_{n}",), ("VP_trans_adv_{n}",)),
]
# rules without a number feature
NEUTRAL_RULES = [
    ("AdvP", ("Adv",), ("Adv",)),
    ("CP_obj_gap", ("C", "S_obj_gap"), ("S_obj_gap", "C")),
]


def _instantiate(templates):
    """Expand the feature variables of every template over NUMBERS."""
    rules = []
    for lhs, hi_rhs, hf_rhs in templates:
        object_numbers = NUMBERS if any("{m}" in symbol for symbol in hi_rhs) else (None,)
        for n in NUMBERS:
            for m in object_numbers:
                features = {"n": n, "m": m}
                rules.append(
                    (
                        lhs.format(**features),
                        tuple(symbol.format(**features) for symbol in hi_rhs),
                        tuple(symbol.format(**features) for symbol in hf_rhs),
                    )
                )
    return rules


RULES = _instantiate(AGREEING_RULES) + NEUTRAL_RULES

NUMBERED_CATEGORIES = ("N", "V_intrans", "V_trans")
VERB_CATEGORIES = ("V_intrans", "V_trans")
TERMINAL_CATEGORIES = tuple(f"{c}_{n}" for c in NUMBERED_CATEGORIES for n in NUMBERS) + ("Adv", "C")

# Inflection paradigms, (singular, plural) forms in the order of grammar.LEXICON
# (the singular column is the V1 lexicon, which the tests assert).
NOUNS = [
    ("ant", "ants"), ("bird", "birds"), ("cat", "cats"), ("dog", "dogs"), ("elephant", "elephants"),
    ("flamingo", "flamingos"), ("giraffe", "giraffes"), ("hippo", "hippos"), ("iguana", "iguanas"),
    ("jaguar", "jaguars"), ("koala", "koalas"), ("lemur", "lemurs"), ("monkey", "monkeys"),
    ("newt", "newts"), ("owl", "owls"), ("penguin", "penguins"), ("quail", "quails"),
    ("rabbit", "rabbits"), ("shark", "sharks"), ("tiger", "tigers"), ("urchin", "urchins"),
    ("vulture", "vultures"), ("walrus", "walruses"), ("xerus", "xeruses"), ("yak", "yaks"), ("zebra", "zebras"),
]  # fmt: skip
V_INTRANS = [
    ("swims", "swim"), ("dances", "dance"), ("sings", "sing"), ("laughs", "laugh"), ("smiles", "smile"),
    ("sleeps", "sleep"), ("jumps", "jump"), ("claps", "clap"), ("runs", "run"), ("walks", "walk"),
    ("hops", "hop"), ("crawls", "crawl"), ("flies", "fly"), ("dives", "dive"), ("spins", "spin"),
    ("yawns", "yawn"), ("snores", "snore"), ("sneezes", "sneeze"), ("stretches", "stretch"),
    ("shivers", "shiver"), ("wanders", "wander"), ("naps", "nap"), ("whistles", "whistle"),
    ("hums", "hum"), ("stumbles", "stumble"), ("twirls", "twirl"),
]  # fmt: skip
V_TRANS = [
    ("chases", "chase"), ("pursues", "pursue"), ("sees", "see"), ("likes", "like"), ("meets", "meet"),
    ("follows", "follow"), ("watches", "watch"), ("greets", "greet"), ("admires", "admire"),
    ("avoids", "avoid"), ("loves", "love"), ("hates", "hate"), ("knows", "know"), ("hugs", "hug"),
    ("kisses", "kiss"), ("tickles", "tickle"), ("teases", "tease"), ("trusts", "trust"),
    ("helps", "help"), ("fears", "fear"), ("finds", "find"), ("grabs", "grab"), ("lifts", "lift"),
    ("carries", "carry"), ("pokes", "poke"), ("nudges", "nudge"),
]  # fmt: skip
PARADIGMS = {"N": NOUNS, "V_intrans": V_INTRANS, "V_trans": V_TRANS}

LEXICON = {
    **{
        f"{category}_{number}": [forms[i] for forms in paradigm]
        for category, paradigm in PARADIGMS.items()
        for i, number in enumerate(NUMBERS)
    },
    "Adv": v1.LEXICON["Adv"],
    "C": v1.LEXICON["C"],
}


# ---------------------------------------------------------------------------
# number feature helpers
# ---------------------------------------------------------------------------

_NUMBER_SUFFIX = re.compile(r"_(sg|pl)$")


def number_of(symbol):
    """'sg' or 'pl' for a number-bearing symbol (N_pl, VP_trans_sg, ...), None otherwise."""
    match = _NUMBER_SUFFIX.search(symbol)
    return match.group(1) if match else None


def strip_number(symbol):
    """Erase the number feature: 'CP_subj_gap_pl' -> 'CP_subj_gap' (the V1 symbol)."""
    return _NUMBER_SUFFIX.sub("", symbol)


def strip_number_tree(tree):
    """Erase the number feature from every label and category leaf, mapping a
    frame tree of this grammar onto its V1 frame tree (words and frame indices
    are left alone, so a lexicalized tree only gets V1 labels)."""
    if isinstance(tree, Tree):
        return Tree(strip_number(tree.label()), [strip_number_tree(child) for child in tree])
    return strip_number(tree) if isinstance(tree, str) else tree


def inflect(category, lemma, number):
    """The `number` form of a lemma given by any of its forms, e.g. inflect("N", "cat", "pl") -> "cats"."""
    forms = next(forms for forms in PARADIGMS[category] if lemma in forms)
    return forms[NUMBERS.index(number)]


# ---------------------------------------------------------------------------
# agreement dependencies
# ---------------------------------------------------------------------------

# phrases whose head verb is the verb of the clause they belong to
VERBAL_PHRASES = frozenset({"VP", "VP_trans", "VP_intrans", "VP_trans_adv", "VP_intrans_adv", "VP_obj_gap", "CP_subj_gap"})


def _label(node):
    return node.label() if isinstance(node, Tree) else node


def terminal_labels(tree):
    """Terminal-category labels in linear order, for frame trees (category
    leaves), indexed frame trees and lexicalized trees (category preterminals)."""
    if _label(tree) in TERMINAL_CATEGORIES:
        return [_label(tree)]
    return [label for child in tree for label in terminal_labels(child)]


def agreement_dependencies(tree):
    """Subject-verb dependencies of a tree as (noun, verb, kind) triples: the
    0-based leaf positions, in the tree's own linear order, of a verb and of
    the noun it agrees with; one triple per verb, sorted by verb position.

    kind is "matrix" (S), "subj_gap" (the RC verb agrees with the head noun) or
    "obj_gap" (the RC verb agrees with the embedded subject). Accepts frame
    trees, indexed frame trees and lexicalized trees, in HI or HF order.
    """
    triples = []
    position = [0]

    def visit(node):
        """(head noun position, head verb position) of node, None when absent."""
        label = _label(node)
        base = strip_number(label)
        if label in TERMINAL_CATEGORIES:
            index, position[0] = position[0], position[0] + 1
            return (index if base == "N" else None, index if base in VERB_CATEGORIES else None)
        heads = {strip_number(_label(child)): visit(child) for child in node}
        if base in ("S", "S_obj_gap"):
            noun = heads["NP"][0]
            verb = heads["VP" if base == "S" else "VP_obj_gap"][1]
            triples.append((noun, verb, "matrix" if base == "S" else "obj_gap"))
        elif base == "NP" and "CP_subj_gap" in heads:
            triples.append((heads["N"][0], heads["CP_subj_gap"][1], "subj_gap"))
        head_noun = heads["N"][0] if base == "NP" else None
        head_verb = next((verb for _, verb in heads.values() if verb is not None), None) if base in VERBAL_PHRASES else None
        return (head_noun, head_verb)

    visit(tree)
    return sorted(triples, key=lambda triple: triple[1])


def n_attractors(tree):
    """Agreement attractors of a tree: nouns that lie between a verb and the
    noun it agrees with and carry the other number (Bock & Miller 1991),
    summed over the tree's dependencies. Only center embedding puts a noun
    between a subject and its verb, so attractors single out the frames where
    an "agree with the nearest noun" heuristic fails."""
    labels = terminal_labels(tree)
    count = 0
    for noun, verb, _ in agreement_dependencies(tree):
        low, high = sorted((noun, verb))
        count += sum(
            strip_number(labels[i]) == "N" and number_of(labels[i]) != number_of(labels[noun])
            for i in range(low + 1, high)
        )
    return count


# ---------------------------------------------------------------------------
# grammar API, as in grammar.py
# ---------------------------------------------------------------------------


def cfg_string(order="hi", lexicalized=False):
    return v1.cfg_string(order, lexicalized, rules=RULES, terminals=TERMINAL_CATEGORIES, lexicon=LEXICON)


@lru_cache(maxsize=None)
def hi_cfg(lexicalized=False):
    return CFG.fromstring(cfg_string("hi", lexicalized))


@lru_cache(maxsize=None)
def hf_cfg(lexicalized=False):
    return CFG.fromstring(cfg_string("hf", lexicalized))


def hi_to_hf(tree):
    return v1.hi_to_hf(tree, rules=RULES, terminals=TERMINAL_CATEGORIES)


def enumerate_trees(order="hi", max_len=13, rules=None):
    return v1.enumerate_trees(order, max_len, RULES if rules is None else rules, TERMINAL_CATEGORIES)


def generate_frames(order="hi", max_len=13):
    return v1.generate_frames(order, max_len, RULES, TERMINAL_CATEGORIES)
