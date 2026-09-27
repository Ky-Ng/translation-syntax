"""Tables for results/gpt2-8h2l300Ks4lr-number-agreement-{ce,ce-rb}/experiment_results.md.

Reads artifacts/<run>/eval/step_<STEP>/{validation,test}.csv of the two 2-layer
number-agreement runs, joins every evaluated sentence with its dataset row
(data/grammar-number-agreement-frames-<variant>/{val,test}.csv: Matrix Number,
RC Number, n_attractors, HI Frame) and prints, per run, markdown tables:

  * validation / test accuracy per construction with the dataset README's
    differentiator columns (Frames, Ambiguous, Unambiguous, Sentences, Sample
    factor, RC number = matrix number, RC number differs) plus exact match,
    teacher-forced token accuracy and linear-rule rate;
  * exact match by matrix number, attractor count and object-head number;
  * the trajectory over the evaluated checkpoints (from eval/step_*/summary.csv
    and loss.csv);
  * per construction, the error types, the shape of the wrong outputs (which
    output templates they match), the most frequent orderings with a real
    example each, the missing / foreign words and the teacher-forced confusions.

Usage (from anywhere; light CPU only, no model is loaded):
    python3 experiments/01_grammar_v1/results/analyze_2l_number_agreement.py > out.md
"""
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repository root
ART = ROOT / "experiments/01_grammar_v1/artifacts"
DATA = ROOT / "data"
RUNS = {
    "ce": "gpt2-8h2l300Ks4lr-number-agreement-ce-seed42",
    "ce-rb": "gpt2-8h2l300Ks4lr-number-agreement-ce-rb-seed42",
}
STEP = 300000
SPLIT_FILE = {"validation": "val.csv", "test": "test.csv"}
CATS = [("None", "None"), ("Subject", "Subject Gap"), ("Subject", "Object Gap"), ("Object", "Subject Gap"), ("Object", "Object Gap")]
TERM_RE = re.compile(r"\[(N_(?:sg|pl)|V_intrans_(?:sg|pl)|V_trans_(?:sg|pl)|Adv|C)_(\d+)\]")

# ----------------------------------------------------------------------------- lexicon (sg/pl pairs)
grammar_src = (ROOT / "experiments/01_grammar_v1/src/cfg/grammar_number_agreement.py").read_text()
PAIRS = {}
for block in ("NOUNS", "V_INTRANS", "V_TRANS"):
    body = grammar_src.split(f"{block} = [", 1)[1].split("]  # fmt: skip", 1)[0]
    for sg, pl in re.findall(r'\("(\w+)", "(\w+)"\)', body):
        PAIRS[sg] = pl
        PAIRS[pl] = sg


def other_number(word):
    return PAIRS.get(word)


# ----------------------------------------------------------------------------- data
def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def terminals(frame):
    """[terminal category] in source order from an HI frame."""
    items = sorted(((int(i), c) for c, i in TERM_RE.findall(frame)))
    return [c for _, c in items]


WORD_CAT = {}  # word -> terminal category (e.g. N_pl)
DATASET = {}  # variant -> split -> (task, source) -> row
for variant in RUNS:
    DATASET[variant] = {}
    for split, fname in SPLIT_FILE.items():
        rows = read(DATA / f"grammar-number-agreement-frames-{variant}" / fname)
        index = {}
        for r in rows:
            key = (r["Task"], r["HI Sentence"])
            assert key not in index, key
            index[key] = r
            for w, c in zip(r["HI Sentence"].split(), terminals(r["HI Frame"])):
                assert WORD_CAT.setdefault(w, c) == c, (w, c)
        DATASET[variant][split] = index
    for r in read(DATA / f"grammar-number-agreement-frames-{variant}" / "train.csv"):
        for w, c in zip(r["HI Sentence"].split(), terminals(r["HI Frame"])):
            assert WORD_CAT.setdefault(w, c) == c, (w, c)


def load_eval(run, split):
    rows = read(ART / run / "eval" / f"step_{STEP:06d}" / f"{split}.csv")
    for r in rows:
        r["exact_match"] = r["exact_match"] == "True"
        r["eos_emitted"] = r["eos_emitted"] == "True"
        r["n_target"], r["n_wrong"] = int(r["n_target"]), int(r["n_wrong"])
    return rows


def join(variant, split, rows):
    index = DATASET[variant][split]
    for r in rows:
        d = index[(r["task"], r["source"])]
        r["matrix"] = d["Matrix Number"]
        r["rc"] = d["RC Number"]
        r["n_attractors"] = int(d["n_attractors"])
        r["cats"] = terminals(d["HI Frame"])
        r["agree"] = "none" if r["rc"] in ("None", "") else ("same" if r["rc"] == r["matrix"] else "differs")
        m = re.search(r"\[NP_(sg|pl) \[N_(sg|pl)_\d+\] \[CP_(?:obj_gap|subj_gap_(?:sg|pl))", d["HI Frame"][d["HI Frame"].find("[VP_"):])
        r["obj_head"] = m.group(2) if m else None
    return rows


# ----------------------------------------------------------------------------- helpers
def em(rows):
    return sum(r["exact_match"] for r in rows) / len(rows) if rows else float("nan")


def tok(rows):
    return sum(r["n_target"] - r["n_wrong"] for r in rows) / sum(r["n_target"] for r in rows)


def fmt(x, n=None):
    if x != x:
        return "—"
    return f"{x:.3f}" + (f" (n={n})" if n is not None else "")


def source_positions(source, words):
    index = {}
    for i, w in enumerate(source.split(), start=1):
        index.setdefault(w, i)
    return [str(index.get(w, "?")) for w in words]


def error_type(row):
    """Same labels as training.evaluate.error_type."""
    if row["exact_match"]:
        return "correct"
    if row["rule"] == "linear":
        return "linear"
    out, gold = row["greedy"].split(), row["target"].split()
    flags = []
    if sorted(out) == sorted(gold):
        flags.append("permutation")
    else:
        oc, gc = Counter(out), Counter(gold)
        if any(gc[w] > oc[w] for w in gc):
            flags.append("missing")
        if any(oc[w] > gc[w] for w in oc if w in gc):
            flags.append("repeat")
        if any(w not in gc for w in oc):
            flags.append("foreign")
    if not row["eos_emitted"]:
        flags.append("no-eos")
    return "+".join(flags)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ex(r):
    return f"`{r['source']}` → gold `{r['target']}` → model `{r['greedy']}`"


# ----------------------------------------------------------------------------- per-construction tables
def construction_table(rows, task, with_linear):
    rows = [r for r in rows if r["task"] == task]
    header = ["Attachment Site", "Gap Type", "Frames", "Ambiguous", "Unambiguous", "Sentences", "Sample factor",
              "RC number = matrix number", "RC number differs", "Exact match", "Token acc (teacher forced)"]
    if with_linear:
        header.append("Linear rule")
    body = []
    for site, gap in CATS:
        g = [r for r in rows if (r["attachment_site"], r["gap_type"]) == (site, gap)]
        if not g:
            continue
        frames = {r["hi_frame"] for r in g}
        amb = {r["hi_frame"] for r in g if r["is_ambiguous"] == "True"}
        same = [r for r in g if r["agree"] == "same"]
        diff = [r for r in g if r["agree"] == "differs"]
        line = [site, gap, len(frames), len(amb), len(frames) - len(amb), f"{len(g):,}", f"{len(g) / len(frames):g}",
                fmt(em(same), len(same)) if same else "—", fmt(em(diff), len(diff)) if diff else "—",
                f"{em(g):.3f}", f"{tok(g):.3f}"]
        if with_linear:
            line.append(f"{sum(r['rule'] == 'linear' for r in g) / len(g):.3f}")
        body.append(line)
    g = rows
    frames = {r["hi_frame"] for r in g}
    amb = {r["hi_frame"] for r in g if r["is_ambiguous"] == "True"}
    same = [r for r in g if r["agree"] == "same"]
    diff = [r for r in g if r["agree"] == "differs"]
    line = ["**All**", "", f"**{len(frames)}**", f"**{len(amb)}**", f"**{len(frames) - len(amb)}**", f"**{len(g):,}**", "",
            fmt(em(same), len(same)), fmt(em(diff), len(diff)), f"**{em(g):.3f}**", f"**{tok(g):.3f}**"]
    if with_linear:
        line.append(f"**{sum(r['rule'] == 'linear' for r in g) / len(g):.3f}**")
    body.append(line)
    return table(header, body)


def attractor_table(rows):
    rows = [r for r in rows if r["task"] == "translate"]
    header = ["Attachment Site", "Gap Type", "Matrix sg", "Matrix pl", "0 attractors", "1 attractor", "2 attractors",
              "RC number = object head number", "RC number ≠ object head number"]
    body = []
    for site, gap in CATS:
        g = [r for r in rows if (r["attachment_site"], r["gap_type"]) == (site, gap)]
        if not g:
            continue

        def cell(sub):
            return fmt(em(sub), len(sub)) if sub else "—"

        body.append([site, gap, cell([r for r in g if r["matrix"] == "sg"]), cell([r for r in g if r["matrix"] == "pl"]),
                     cell([r for r in g if r["n_attractors"] == 0]), cell([r for r in g if r["n_attractors"] == 1]),
                     cell([r for r in g if r["n_attractors"] == 2]),
                     cell([r for r in g if r["obj_head"] and r["rc"] == r["obj_head"]]),
                     cell([r for r in g if r["obj_head"] and r["rc"] != r["obj_head"]])])
    return table(header, body)


# ----------------------------------------------------------------------------- trajectory
def trajectory(run):
    steps = {}
    for p in sorted((ART / run / "eval").glob("step_*/summary.csv")):
        s = int(p.parent.name.split("_")[1])
        steps[s] = {(r["split"], r["task"], r["attachment_site"], r["gap_type"]): r for r in read(p)}
    losses = defaultdict(dict)
    for r in read(ART / run / "loss.csv"):
        losses[int(r["step"])][r["split"]] = float(r["loss"])
    return steps, losses


def trajectory_table(steps, losses):
    sample = [2000, 4000, 6000, 10000, 20000, 30000, 50000, 100000, 150000, 200000, 250000, 300000]
    header = ["Step", "Train loss", "Val loss", "Val translate EM", "Val copy EM", "Test EM (all)",
              "Subject / Subject Gap", "Subject / Object Gap", "Object / Object Gap", "Object / Object Gap token acc"]
    body = []
    for s in sample:
        d = steps[s]
        tr = losses[s].get("train", float("nan"))
        body.append([f"{s:,}", f"{tr:.3f}", f"{losses[s]['val']:.3f}",
                     f"{float(d[('validation', 'translate', 'all', 'all')]['exact_match']):.3f}",
                     f"{float(d[('validation', 'copy', 'all', 'all')]['exact_match']):.3f}",
                     f"{float(d[('test', 'translate', 'all', 'all')]['exact_match']):.3f}",
                     f"{float(d[('test', 'translate', 'Subject', 'Subject Gap')]['exact_match']):.3f}",
                     f"{float(d[('test', 'translate', 'Subject', 'Object Gap')]['exact_match']):.3f}",
                     f"{float(d[('test', 'translate', 'Object', 'Object Gap')]['exact_match']):.3f}",
                     f"{float(d[('test', 'translate', 'Object', 'Object Gap')]['token_acc']):.3f}"])
    return table(header, body)


def trajectory_facts(steps):
    facts = []
    ordered = sorted(steps)
    first_val = next((s for s in ordered if float(steps[s][('validation', 'translate', 'all', 'all')]['exact_match']) >= 1.0), None)
    first_val_all = next((s for s in ordered if float(steps[s][('validation', 'all', 'all', 'all')]['exact_match']) >= 1.0), None)
    facts.append(f"first step with validation translate EM = 1.000: {first_val:,}" if first_val else "validation translate EM never reaches 1.000")
    facts.append(f"first step with validation EM = 1.000 on both tasks: {first_val_all:,}" if first_val_all else "validation EM on both tasks never reaches 1.000")
    n_val_1 = sum(float(steps[s][('validation', 'translate', 'all', 'all')]['exact_match']) >= 1.0 for s in ordered)
    facts.append(f"checkpoints with validation translate EM = 1.000: {n_val_1} of {len(ordered)}")
    key = ('test', 'translate', 'Object', 'Object Gap')
    best = max(ordered, key=lambda s: float(steps[s][key]['exact_match']))
    facts.append(f"best Object / Object Gap test EM: {float(steps[best][key]['exact_match']):.3f} at step {best:,}")
    last10 = ordered[-10:]
    vals = [float(steps[s][key]['exact_match']) for s in last10]
    facts.append(f"Object / Object Gap test EM over the last 10 checkpoints ({last10[0]:,}–{last10[-1]:,}): min {min(vals):.3f}, max {max(vals):.3f}")
    for k in (('test', 'translate', 'Subject', 'Subject Gap'), ('test', 'translate', 'Subject', 'Object Gap')):
        mx = max(float(steps[s][k]['exact_match']) for s in ordered)
        n_nonzero = sum(float(steps[s][k]['exact_match']) > 0 for s in ordered)
        facts.append(f"{k[2]} / {k[3]} test EM: max over all checkpoints {mx:.3f}; checkpoints with EM > 0: {n_nonzero}")
        lin = max(float(steps[s][k]['linear_rate']) for s in ordered)
        facts.append(f"{k[2]} / {k[3]} linear-rule rate: max over all checkpoints {lin:.3f}")
    return facts


# ----------------------------------------------------------------------------- error analysis
def analyze_errors(rows, split, task, site, gap, top=8, n_examples=3):
    g = [r for r in rows if (r["task"], r["attachment_site"], r["gap_type"]) == (task, site, gap)]
    wrong = [r for r in g if not r["exact_match"]]
    out = [f"### {split} / {task} / {site} / {gap}: {len(wrong)} wrong of {len(g)} (EM {em(g):.3f})", ""]
    if not wrong:
        return "\n".join(out)
    for r in wrong:
        r["etype"] = error_type(r)
        r["gold_pos"] = source_positions(r["source"], r["target"].split())
        r["out_pos"] = source_positions(r["source"], r["greedy"].split())
        r["ordering"] = f"gold {' '.join(r['gold_pos'])} -> model {' '.join(r['out_pos'])}"
    # error types
    et = Counter(r["etype"] for r in wrong)
    out.append("**Error types** (share of wrong outputs; `permutation` = gold words in a wrong order, `missing` / `repeat` / `foreign` = a gold word absent / a source word emitted too often / a word not in the source):")
    out.append("")
    body = []
    for k, c in et.most_common():
        exs = [r for r in wrong if r["etype"] == k][:n_examples]
        body.append([f"`{k}`", c, f"{c / len(wrong):.3f}", "<br>".join(ex(r) for r in exs)])
    out.append(table(["Error type", "Count", "Share", "Examples (source → gold → model)"], body))
    out.append("")
    # structural template stats
    n = len(wrong)
    subj_first = sum(r["out_pos"][:1] == ["1"] for r in wrong)
    same_len = sum(len(r["greedy"].split()) == len(r["target"].split()) for r in wrong)
    identity = sum(r["greedy"] == r["source"] for r in wrong)
    first_err = Counter()
    for r in wrong:
        items = [m for m in r["mispredictions"].split(";") if m]
        if items:
            first_err[int(items[0].split(":")[0])] += 1
    stats = [f"- output starts with the first source word: {subj_first} / {n} ({subj_first / n:.3f})",
             f"- output has the gold length: {same_len} / {n} ({same_len / n:.3f})",
             f"- output is the source copied verbatim: {identity} / {n} ({identity / n:.3f})",
             f"- `<eos>` emitted: {sum(r['eos_emitted'] for r in wrong)} / {n}",
             f"- first teacher-forced error at target position: " + ", ".join(f"{p}: {c} ({c / n:.3f})" for p, c in sorted(first_err.items()))]
    if task == "translate" and site == "Subject":
        head_moved = starts_1R = ends_V = v_after_subj = 0
        subj_repeated = rc_twice = emb_dropped = emb_replaced = 0
        for r in wrong:
            gp = r["gold_pos"]
            h = gp.index("1")
            R, V = gp[:h], gp[h + 1:]
            op = r["out_pos"]
            head_moved += op == ["1"] + R + V
            starts_1R += op[:1 + len(R)] == ["1"] + R
            ends_V += len(op) > len(V) and op[-len(V):] == V
            v_after_subj += op[1:1 + len(V)] == V
            subj_repeated += op == ["1"] + R + ["1"] + V
            rc_twice += op == ["1"] + R + R[:-1]
            if gap == "Object Gap":
                emb = str(r["source"].split().index("that") + 2)  # the noun after 'that'
                R_no_emb = [p for p in R if p != emb]
                emb_dropped += op == ["1"] + R_no_emb + V
                emb_replaced += op == ["1"] + V[:-1] + R_no_emb + V
        stats += [f"- output = first source word + gold with that word removed (`1 R V`, R = mirrored RC incl. `that`, V = mirrored matrix VP): {head_moved} / {n} ({head_moved / n:.3f})",
                  f"- output starts with `1 R` (first source word, then the mirrored RC exactly as in the gold): {starts_1R} / {n} ({starts_1R / n:.3f})",
                  f"- output ends with `V` (the mirrored matrix VP exactly as in the gold): {ends_V} / {n} ({ends_V / n:.3f})",
                  f"- mirrored matrix VP `V` follows the subject directly (`1 V ...`): {v_after_subj} / {n} ({v_after_subj / n:.3f})",
                  f"- output = `1 R 1 V` (subject emitted first and again at its gold position): {subj_repeated} / {n} ({subj_repeated / n:.3f})",
                  f"- output = `1 R R'` (subject, mirrored RC, then the mirrored RC again without `that`; matrix VP never produced): {rc_twice} / {n} ({rc_twice / n:.3f})"]
        if gap == "Object Gap":
            stats += [f"- output = `1 R\\emb V` (embedded subject of the RC deleted, everything else as in the gold): {emb_dropped} / {n} ({emb_dropped / n:.3f})",
                      f"- output = `1 V\\verb R\\emb V` (embedded subject replaced by the matrix object (+ adverb), then the gold continues): {emb_replaced} / {n} ({emb_replaced / n:.3f})"]
    if task == "translate" and (site, gap) == ("Object", "Object Gap"):
        drop_emb = 0
        for r in wrong:
            words = r["source"].split()
            that = words.index("that") + 1  # 1-based position of 'that'
            emb = str(that + 1)  # embedded subject = the noun after 'that'
            drop_emb += r["out_pos"] == [p for p in r["gold_pos"] if p != emb]
        stats.append(f"- output = gold with the embedded subject (the noun after `that`) deleted: {drop_emb} / {n} ({drop_emb / n:.3f})")
    if task == "copy" and site == "Object":
        ce_rewrite = starts_that = 0
        for r in wrong:
            words = r["source"].split()
            that = words.index("that") + 1
            rc = [str(p) for p in range(that, len(words) + 1)]
            mid = [str(p) for p in range(2, that)]
            ce_rewrite += r["out_pos"] == ["1"] + rc + mid
            starts_that += r["out_pos"][:2] == ["1", str(that)]
        stats += [f"- output = subject + RC + matrix VP (the right-branching input re-attached as a center-embedded sentence): {ce_rewrite} / {n} ({ce_rewrite / n:.3f})",
                  f"- output starts with `subject that`: {starts_that} / {n} ({starts_that / n:.3f})"]
    out.append("**Shape of the wrong outputs:**")
    out.append("")
    out += stats
    out.append("")
    # orderings
    oc = Counter(r["ordering"] for r in wrong)
    out.append(f"**Most frequent output orderings** (words as 1-based source positions, `?` = word not in the source; {len(oc)} distinct orderings in total):")
    out.append("")
    body = []
    for k, c in oc.most_common(top):
        r = next(r for r in wrong if r["ordering"] == k)
        body.append([f"`{k}`", c, f"{c / n:.3f}", f"`{r['etype']}`", ex(r)])
    out.append(table(["Ordering", "Count", "Share", "Error type", "Example"], body))
    out.append("")
    # missing categories
    missing = Counter()
    for r in wrong:
        if "missing" in r["etype"]:
            gc, oc2 = Counter(r["target"].split()), Counter(r["greedy"].split())
            words = r["source"].split()
            for w in gc:
                if gc[w] > oc2[w]:
                    pos = words.index(w) + 1
                    missing[f"{r['cats'][pos - 1]} at source position {pos}"] += 1
    if missing:
        out.append("**Which source word goes missing** (over outputs with a `missing` flag):")
        out.append("")
        out.append(table(["Missing word (frame category, source position)", "Count"], [[k, c] for k, c in missing.most_common(10)]))
        out.append("")
    # foreign words
    foreign = Counter()
    foreign_ex = {}
    for r in wrong:
        if "foreign" in r["etype"]:
            src = set(r["source"].split())
            for w in r["greedy"].split():
                if w not in src:
                    on = other_number(w)
                    if on in src:
                        kind = f"other-number form of a source word ({WORD_CAT.get(w, '?')})"
                    else:
                        kind = f"unrelated {WORD_CAT.get(w, w)}"
                    foreign[kind] += 1
                    foreign_ex.setdefault(kind, r)
    if foreign:
        out.append("**Foreign words** (words in the output that are not in the source):")
        out.append("")
        out.append(table(["Kind", "Count", "Example"], [[k, c, ex(foreign_ex[k])] for k, c in foreign.most_common()]))
        out.append("")
    # teacher-forced confusions: number flips
    conf = Counter()
    for r in wrong:
        for item in filter(None, r["mispredictions"].split(";")):
            gold, pred = item.split(":", 1)[1].split("->")
            if other_number(gold) == pred:
                conf["number flip (same lemma, other number)"] += 1
            elif WORD_CAT.get(gold, "").split("_")[0] == WORD_CAT.get(pred, "?").split("_")[0] and gold != pred:
                conf["same category, other lemma"] += 1
            else:
                conf["other category / special token"] += 1
    tot = sum(conf.values())
    out.append(f"**Teacher-forced mispredictions** ({tot} in this construction): " + "; ".join(f"{k} {c} ({c / tot:.3f})" for k, c in conf.most_common()))
    out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------------- main
def main():
    for variant, run in RUNS.items():
        print(f"\n\n# ===================== {run} =====================\n")
        rows = {s: join(variant, s, load_eval(run, s)) for s in SPLIT_FILE}
        for split in ("validation", "test"):
            for task in ("translate", "copy"):
                if any(r["task"] == task for r in rows[split]):
                    print(f"\n## TABLE {split} {task}\n")
                    print(construction_table(rows[split], task, with_linear=(task == "translate")))
        print("\n## TABLE test by number / attractors\n")
        print(attractor_table(rows["test"]))
        print("\n## TABLE validation by number / attractors\n")
        print(attractor_table(rows["validation"]))
        steps, losses = trajectory(run)
        print(f"\n## TRAJECTORY ({len(steps)} evaluated checkpoints)\n")
        print(trajectory_table(steps, losses))
        print()
        print("\n".join(f"- {f}" for f in trajectory_facts(steps)))
        print("\n## ERRORS\n")
        for split in ("validation", "test"):
            for task in ("translate", "copy"):
                for site, gap in CATS:
                    if any((r["task"], r["attachment_site"], r["gap_type"]) == (task, site, gap) for r in rows[split]):
                        print(analyze_errors(rows[split], split, task, site, gap))
        # the consistent pattern on test, and its training-data cause
        wrong = [r for r in rows["test"] if not r["exact_match"]]
        subj_first = sum(r["greedy"].split()[:1] == r["source"].split()[:1] for r in wrong)
        print(f"\nTEST OVERALL: {len(wrong)} wrong; output starts with first source word in {subj_first} ({subj_first / len(wrong):.3f}); "
              f"first source word == first gold word in {sum(r['target'].split()[0] == r['source'].split()[0] for r in rows['test'])} of {len(rows['test'])} test rows")
        train = read(DATA / f"grammar-number-agreement-frames-{variant}" / "train.csv")
        same = sum(r["HF Sentence"].split()[0] == r["HI Sentence"].split()[0] for r in train)
        print(f"TRAIN: target[0] == source[0] in {same} of {len(train)} rows ({same / len(train):.3f})")
        tr_translate = [r for r in train if r["Task"] == "translate"]
        same_t = sum(r["HF Sentence"].split()[0] == r["HI Sentence"].split()[0] for r in tr_translate)
        print(f"TRAIN translate: target[0] == source[0] in {same_t} of {len(tr_translate)} rows")


if __name__ == "__main__":
    main()
