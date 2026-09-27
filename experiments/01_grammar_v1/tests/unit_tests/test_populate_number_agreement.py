"""Tests for the number-agreement populated dataset (src/cfg/populate.py --grammar number_agreement).

Closed form of the train token budget. A train row encodes to 2 * n_tokens + 4
non-pad tokens (<bos> src <task> <sep> tgt <eos>), so a (task, category) group
with m rows per frame contributes m * (2 * S + 4 * F) tokens, F the frames in
the category and S the sum of their n_tokens. From cfg.frames_number_agreement
(each V1 frame with k nouns appears 2^k times with the same n_tokens):

    category             F     S
    None/None           12    38
    Subject/SubjectGap  72   456
    Subject/ObjectGap   48   320
    Object/SubjectGap   48   320
    Object/ObjectGap    32   224

At multiplier m with n_val = round(0.1 m), the ce-rb train split is
translate(ambiguous, m - n_val per frame) + copy(ambiguous, m - n_val) +
copy(unambiguous, m), i.e. with the ambiguous categories None/None and
Object/SubjectGap (F = 60, S = 358) and the unambiguous rest (F = 152,
S = 1000):

    tokens(m) = 2 * (m - n_val) * (2*358 + 4*60) + m * (2*1000 + 4*152)
    tokens(18) = 2 * 16 * 956 + 18 * 2608 = 77,536      (Grammar V1 ce-rb: 78,080)
    tokens(19) = 2 * 17 * 956 + 19 * 2608 = 82,056

so the multiplier solved for the V1 budget is 18 (544 tokens under, 0.7%).
The ce / rb variants are boosted to the number grammar's own ce-rb budget.
"""

import re

import pytest
from cfg.grammar_number_agreement import LEXICON, hi_cfg, hi_to_hf, number_of
from cfg.populate import (
    COPY_VARIANTS,
    CSV_COLUMNS,
    GRAMMARS,
    V1,
    build_examples,
    cerb_train_tokens,
    csv_columns,
    dataset_card,
    has_agreement,
    population_table,
    repo_name,
    solve_boost,
    solve_multiplier,
    train_tokens,
)
from nltk.parse.chart import ChartParser

NA = GRAMMARS["number_agreement"]
MULTIPLIER, N_VAL = 18, 2
V1_BUDGET = 78_080

INVENTORY = {
    ("None", "None"): (12, 38),
    ("Subject", "Subject Gap"): (72, 456),
    ("Subject", "Object Gap"): (48, 320),
    ("Object", "Subject Gap"): (48, 320),
    ("Object", "Object Gap"): (32, 224),
}
AMBIGUOUS = {("None", "None"), ("Object", "Subject Gap")}


def group_tokens(category, rows_per_frame):
    frames, n_tokens = INVENTORY[category]
    return rows_per_frame * (2 * n_tokens + 4 * frames)


def closed_form_tokens(copy_attachments, multiplier=MULTIPLIER, n_val=N_VAL):
    total = 0
    for category in INVENTORY:
        ambiguous = category in AMBIGUOUS
        if ambiguous:
            total += group_tokens(category, multiplier - n_val)  # translate
        if category[0] in copy_attachments:
            total += group_tokens(category, multiplier - n_val if ambiguous else multiplier)
    return total


@pytest.fixture(scope="module")
def rows():
    return build_examples(MULTIPLIER, 0, grammar_spec=NA)


def test_closed_form_budget_and_solved_multiplier():
    assert closed_form_tokens(("None", "Subject", "Object")) == 77_536
    assert closed_form_tokens(("None", "Subject", "Object"), 19, 2) == 82_056
    records = NA.records()
    assert cerb_train_tokens(records, 18) == 77_536 and cerb_train_tokens(records, 19) == 82_056
    assert solve_multiplier(V1_BUDGET, grammar_spec=NA) == MULTIPLIER
    assert solve_multiplier(train_tokens(build_examples()), grammar_spec=NA) == MULTIPLIER
    # V1 solves back to its own default multiplier; the closed form matches the built dataset
    assert cerb_train_tokens(V1.records(), 100) == V1_BUDGET == train_tokens(build_examples())
    assert solve_multiplier(V1_BUDGET, grammar_spec=V1) == 100


def test_train_budget_close_to_v1(rows):
    assert train_tokens(rows) == cerb_train_tokens(NA.records(), MULTIPLIER) == 77_536
    assert abs(train_tokens(rows) - V1_BUDGET) / V1_BUDGET < 0.01


def test_required_split_counts(rows):
    train = [r for r in rows if r["Split"] == "train"]
    val = [r for r in rows if r["Split"] == "val"]
    test = [r for r in rows if r["Split"] == "test"]
    assert sum(r["Task"] == "translate" for r in train) == 60 * (MULTIPLIER - N_VAL)  # 960
    assert sum(r["Task"] == "copy" and r["is_ambiguous"] for r in train) == 60 * (MULTIPLIER - N_VAL)
    assert sum(r["Task"] == "copy" and not r["is_ambiguous"] for r in train) == 152 * MULTIPLIER  # 2,736
    assert sum(r["Task"] == "translate" for r in val) == 60 * N_VAL == 120
    assert sum(r["Task"] == "copy" for r in val) == 120
    assert len(test) == 152 * MULTIPLIER and all(r["Task"] == "translate" and not r["is_ambiguous"] for r in test)
    assert len(rows) == 7_632
    assert len({r["HI Frame"] for r in rows}) == 212


def test_columns_and_agreement_metadata(rows):
    assert csv_columns(NA) == CSV_COLUMNS[:4] + ["Matrix Number", "RC Number", "n_attractors"] + CSV_COLUMNS[4:]
    assert csv_columns() == CSV_COLUMNS and repo_name("ce") == "grammar-v1-frames-ce"
    assert repo_name("ce-rb", NA) == "grammar-number-agreement-frames-ce-rb"
    for r in rows:
        assert set(r) == set(csv_columns(NA))
        slots = re.findall(r"\[([A-Za-z_]+)_\d+\]", r["HI Frame"])
        assert r["Matrix Number"] == number_of(slots[0]) in ("sg", "pl")
        assert (r["RC Number"] == "None") == (r["Gap Type"] == "None")
        assert r["n_attractors"] in (0, 1)
        if r["n_attractors"]:
            assert r["Attachment Site"] == "Subject"


def test_words_match_slots_and_sentences_are_grammatical(rows):
    """Every slot is filled from its numbered category, so agreement holds by
    construction; the lexicalized CFG confirms it on the whole validation
    split plus a slice of test and train, with the HF target as the mirror."""
    for r in rows:
        slots = re.findall(r"\[([A-Za-z_]+)_\d+\]", r["HI Frame"])
        words = r["HI Sentence"].split()
        assert len(words) == len(slots) == r["n_tokens"]
        for slot, word in zip(slots, words):
            assert word in LEXICON[slot], (slot, word)
    parser = ChartParser(hi_cfg(lexicalized=True))
    sample = [r for r in rows if r["Split"] == "val"] + [r for r in rows if r["Split"] != "val"][::40]
    assert len(sample) > 300
    for r in sample:
        parses = list(parser.parse(r["HI Sentence"].split()))
        assert len(parses) == 1, r["HI Sentence"]
        mirrored = " ".join(hi_to_hf(parses[0]).leaves())
        assert r["HF Sentence"] == (r["HI Sentence"] if r["Task"] == "copy" else mirrored)


def test_variants_boosted_to_own_cerb_budget(rows):
    for variant in ("ce", "rb"):
        attachments = COPY_VARIANTS[variant]
        boost = solve_boost(attachments, MULTIPLIER, 0, grammar_spec=NA)
        assert boost == pytest.approx(77_536 / closed_form_tokens(attachments))
        variant_rows = build_examples(MULTIPLIER, 0, copy_attachments=attachments, boost=boost, grammar_spec=NA)
        # every kept group has 16 or 18 base rows, so round(size * boost) errs
        # the same way in all of them and the residual is larger than v1's
        # (ce +136, rb -592 tokens) but stays under 1%
        assert abs(train_tokens(variant_rows) - 77_536) / 77_536 < 0.01
        for split in ("val", "test"):
            assert [r for r in variant_rows if r["Split"] == split] == [r for r in rows if r["Split"] == split]
        train_copy_sites = {r["Attachment Site"] for r in variant_rows if r["Split"] == "train" and r["Task"] == "copy"}
        assert train_copy_sites == set(attachments)


def test_v1_default_unchanged_and_deterministic():
    assert build_examples(3, 1) == build_examples(3, 1, grammar_spec=V1)
    a = build_examples(3, 5, grammar_spec=NA)
    assert a == build_examples(3, 5, grammar_spec=NA)
    assert a != build_examples(3, 6, grammar_spec=NA)
    assert a != build_examples(3, 5)


def table_body(lines):
    """(attachment, gap) -> row of a markdown table with those as its first two cells."""
    return {tuple(line.strip("| ").split(" | ")[:2]): line for line in lines[2:-1]}


def test_population_table_counts_every_split(rows):
    """ce-rb: every frame is filled 2 * MULTIPLIER times (copy + translate),
    each a distinct sentence. The RC verb of a subject-gap RC on the subject
    agrees with the matrix subject itself, so those always match; the other
    RC structures agree with an independent noun, so they split half / half."""
    lines = population_table(rows, NA)
    assert lines[0] == (
        "| Attachment Site | Gap Type | Frames | Ambiguous | Unambiguous | Sentences | Sample factor "
        "| RC number = matrix number | RC number differs |"
    )
    body = table_body(lines)
    assert body[("None", "None")] == "| None | None | 12 | 12 | 0 | 432 | 36 | — | — |"
    assert body[("Subject", "Subject Gap")] == "| Subject | Subject Gap | 72 | 0 | 72 | 2,592 | 36 | 2,592 | 0 |"
    assert body[("Subject", "Object Gap")] == "| Subject | Object Gap | 48 | 0 | 48 | 1,728 | 36 | 864 | 864 |"
    assert body[("Object", "Subject Gap")] == "| Object | Subject Gap | 48 | 48 | 0 | 1,728 | 36 | 864 | 864 |"
    assert body[("Object", "Object Gap")] == "| Object | Object Gap | 32 | 0 | 32 | 1,152 | 36 | 576 | 576 |"
    assert lines[-1] == "| **Total** | | **212** | **60** | **152** | **7,632** | | **4,896** | **2,304** |"
    assert len(rows) == 7_632 == len({r["HI Sentence"] for r in rows})


def test_population_table_sample_factor_follows_variant():
    """Dropped train copy groups are never replaced (an ambiguous frame keeps
    only its 2 val copy rows, an unambiguous one none), kept train groups are
    boosted: ce 16 -> 23 / 18 -> 26 rows, rb 16 -> 30 / 18 -> 34 rows."""
    expected = {
        "ce": {
            ("None", "None"): 50,
            ("Subject", "Subject Gap"): 44,
            ("Subject", "Object Gap"): 44,
            ("Object", "Subject Gap"): 27,
            ("Object", "Object Gap"): 18,
        },
        "rb": {
            ("None", "None"): 64,
            ("Subject", "Subject Gap"): 18,
            ("Subject", "Object Gap"): 18,
            ("Object", "Subject Gap"): 64,
            ("Object", "Object Gap"): 52,
        },
    }
    for variant, factors in expected.items():
        attachments = COPY_VARIANTS[variant]
        boost = solve_boost(attachments, MULTIPLIER, 0, grammar_spec=NA)
        variant_rows = build_examples(MULTIPLIER, 0, copy_attachments=attachments, boost=boost, grammar_spec=NA)
        body = table_body(population_table(variant_rows, NA))
        for category, factor in factors.items():
            cells = body[category].strip("| ").split(" | ")
            n_frames, n_sentences, sample = int(cells[2]), int(cells[5].replace(",", "")), int(cells[6])
            assert sample == factor and n_sentences == n_frames * factor, (variant, category)


def test_population_table_without_agreement_columns_for_v1():
    assert has_agreement(NA) and not has_agreement(V1)
    lines = population_table(build_examples(3, 0), V1)
    assert lines[0] == "| Attachment Site | Gap Type | Frames | Ambiguous | Unambiguous | Sentences | Sample factor |"
    assert table_body(lines)[("None", "None")] == "| None | None | 4 | 4 | 0 | 24 | 6 |"
    assert lines[-1] == "| **Total** | | **40** | **12** | **28** | **240** | |"


def test_dataset_card_has_after_population_table(rows):
    card = dataset_card(rows, "ce-rb", COPY_VARIANTS["ce-rb"], 1.0, MULTIPLIER, 0, rows, NA)
    before = card.index("## Frame counts per category (before population)")
    after = card.index("## Sentence counts per category (after population)")
    assert before < after
    assert "\n".join(population_table(rows, NA)) in card[after:]
    assert "`RC number = matrix number` counts sentences" in card[after:]
    v1_rows = build_examples(3, 0)
    v1_card = dataset_card(v1_rows, "ce-rb", COPY_VARIANTS["ce-rb"], 1.0, 3, 0, v1_rows)
    assert "\n".join(population_table(v1_rows)) in v1_card and "RC number" not in v1_card
