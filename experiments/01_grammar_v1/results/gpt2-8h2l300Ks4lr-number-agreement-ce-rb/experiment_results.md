# gpt2-8h2l300Ks4lr-number-agreement-ce-rb — experiment results

GPT-2 (2 layers, 8 heads, d_model 128) trained for 300K steps on `kylelovesllms/grammar-number-agreement-frames-ce-rb`: Grammar V1 with subject-verb number agreement, where the copy task contains every structure (simple, center-embedded and right-branching sentences). All numbers are from the final checkpoint (step 300,000) unless a step is given. Every table was computed from the evaluator's per-example CSVs (`eval/step_300000/{validation,test}.csv`, see [`src/training/evaluate.py`](../../src/training/evaluate.py)) by [`../analyze_2l_number_agreement.py`](../analyze_2l_number_agreement.py), which joins each evaluated sentence with the dataset's `Matrix Number`, `RC Number` and `n_attractors` columns.

## Run summary

| | |
| --- | --- |
| Seeds trained | **1** (seed 42). The 2026-09-23 sweep ran one seed per dataset and depth. |
| Checkpoints | **150**, every 2,000 steps from 2,000 to 300,000, each evaluated on validation and test right after saving (`--eval-every` defaults to `--checkpoint-every`) |
| wandb | run [gpt2-8h2l300Ks4lr-number-agreement-ce-rb-seed42](https://wandb.ai/kgng-usc/translation-syntax/runs/n08vdh7y) in project [translation-syntax](https://wandb.ai/kgng-usc/translation-syntax), group `gpt2-8h2l300Ks4lr-number-agreement-ce-rb`. Keys `train/loss`, `val/loss`, `validation/…` and `test/translate/<Site>_<Gap>/{exact_match,token_acc,linear_rate}` are logged at every checkpoint. |
| Dataset | `kylelovesllms/grammar-number-agreement-frames-ce-rb`: train 4,656 rows / 77,536 tokens (960 translate + 960 copy rows of the 60 ambiguous frames, 2,736 copy rows of the 152 unambiguous frames; boost factor 1), validation 240 rows, test 2,736 rows. Local copy: `data/grammar-number-agreement-frames-ce-rb/`. |
| Model | `GPT2LMHeadModel`, 2 layers, 8 heads, d_model 128, vocabulary 189 ids (number-agreement lexicon), loss on every non-pad token including the source, absolute positions shifted by 0–25 pads |
| Optimisation | Adam, lr 1e-4 with linear decay to 0, batch 256, 300,000 steps (about 16K passes over the training set) |
| SLURM job | 6013441 (`number-agreement-ce-rb-2l`), node c06-02, 1 h 37 min, 2026-09-23 |
| Artifacts | [`../../artifacts/gpt2-8h2l300Ks4lr-number-agreement-ce-rb-seed42/`](../../artifacts/gpt2-8h2l300Ks4lr-number-agreement-ce-rb-seed42/): `launch_config.json`, `loss.csv`, `checkpoints/step_XXXXXX/`, `eval/step_XXXXXX/{validation,test,summary,errors_validation,errors_test,confusions_*}.csv` |
| Final losses | train 0.565 (mean of the last 50 steps), validation 3.121 |

The validation loss is the full-sequence LM loss, so it mostly scores the *source* words, which are lexicalizations the model has never seen. It climbs from 1.67 at step 2,000 to 3.12 at 300,000 while validation exact match stays at 1.000: it measures how confidently the model has memorised the training lexicalizations, not translation quality. Exact match and token accuracy are the metrics to read.

## Validation accuracy (step 300,000)

Validation holds 2 held-out lexicalizations per ambiguous frame and task (60 frames × 2 tasks × 2 = 240 rows). Only the two in-distribution constructions exist in this split; the unambiguous constructions are the test set below.

Column meaning (same header as the sentence-count table of the dataset README): `Frames`, `Ambiguous` and `Unambiguous` count the frames of the construction in the split, `Sentences` the rows, `Sample factor` the rows per frame. `RC number = matrix number` and `RC number differs` give the exact match on the sentences of that agreement condition (n in brackets; `—` when the construction has no RC or the condition cannot occur). `Exact match` is greedy decoding == gold, `Token acc (teacher forced)` the next-token accuracy over the target positions including `<eos>` given the gold prefix, `Linear rule` the share of outputs equal to the keep-first-word-reverse-the-rest string.

### Translate

| Attachment Site | Gap Type | Frames | Ambiguous | Unambiguous | Sentences | Sample factor | RC number = matrix number | RC number differs | Exact match | Token acc (teacher forced) | Linear rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| None | None | 12 | 12 | 0 | 24 | 2 | — | — | 1.000 | 1.000 | 0.000 |
| Object | Subject Gap | 48 | 48 | 0 | 96 | 2 | 1.000 (n=48) | 1.000 (n=48) | 1.000 | 1.000 | 0.000 |
| **All** |  | **60** | **60** | **0** | **120** |  | 1.000 (n=48) | 1.000 (n=48) | **1.000** | **1.000** | **0.000** |

### Copy

| Attachment Site | Gap Type | Frames | Ambiguous | Unambiguous | Sentences | Sample factor | RC number = matrix number | RC number differs | Exact match | Token acc (teacher forced) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| None | None | 12 | 12 | 0 | 24 | 2 | — | — | 1.000 | 1.000 |
| Object | Subject Gap | 48 | 48 | 0 | 96 | 2 | 1.000 (n=48) | 1.000 (n=48) | 1.000 | 1.000 |
| **All** |  | **60** | **60** | **0** | **120** |  | 1.000 (n=48) | 1.000 (n=48) | **1.000** | **1.000** |

Validation is solved on both tasks, in both agreement conditions and for singular and plural matrix subjects (12/12 and 12/12 on no-RC sentences, 48/48 and 48/48 on right-branching ones). Validation translate exact match reached 1.000 at step 8,000 (0.200 at step 2,000, 0.917 at 4,000, 0.983 at 6,000) and stayed there at 134 of the 150 checkpoints (every later exception is 0.992, one sentence wrong, the first at step 28,000); validation copy was 1.000 from the first checkpoint on. There are no validation errors to analyse.

## Test accuracy (step 300,000)

The test split holds 18 lexicalizations of every unambiguous frame, translate task only (2,736 rows); none of these sentences appears in training in any task (the unambiguous frames enter training only as copy rows with different lexicalizations). Both accuracy types the evaluator computes are shown: **exact match** (greedy decode from `<bos> src <translate> <sep>` until `<eos>` equals the gold Head-Final sentence) and **teacher-forced token accuracy** (argmax next token given the gold prefix, over the target positions including `<eos>`).

| Attachment Site | Gap Type | Frames | Ambiguous | Unambiguous | Sentences | Sample factor | RC number = matrix number | RC number differs | Exact match | Token acc (teacher forced) | Linear rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Subject | Subject Gap | 72 | 0 | 72 | 1,296 | 18 | 0.000 (n=1296) | — | 0.000 | 0.431 | 0.000 |
| Subject | Object Gap | 48 | 0 | 48 | 864 | 18 | 0.000 (n=432) | 0.000 (n=432) | 0.000 | 0.417 | 0.000 |
| Object | Object Gap | 32 | 0 | 32 | 576 | 18 | 0.708 (n=288) | 0.726 (n=288) | 0.717 | 0.965 | 0.000 |
| **All** |  | **152** | **0** | **152** | **2,736** |  | 0.101 (n=2016) | 0.290 (n=720) | **0.151** | **0.545** | **0.000** |

- **Object-attached object-gap RCs generalise partially** (0.717 exact match, 163 wrong of 576; token accuracy 0.965). This is lower than the `ce` 2-layer model (0.993), although the `ce-rb` model additionally saw these structures in the copy task; with one seed each, the gap between the two runs is not yet a result.
- **Subject-attached RCs never translate** (0 of 2,160), for subject gaps and object gaps alike. The linear rule is never used (0.000 at every one of the 150 checkpoints); every wrong output is `other`. Token accuracy is lower than in the `ce` model (0.43 / 0.42 vs 0.54 / 0.50): given the gold prefix, this model recovers less of the second half of the target.
- **Number agreement does not separate the conditions.** Exact match is the same whether the RC number equals the matrix number or differs (0.708 vs 0.726 on Object / Object Gap, 0.000 vs 0.000 on Subject / Object Gap). The "all" row differs between the two conditions (0.101 vs 0.290) only because Subject / Subject Gap, which can only be in the "same number" condition, dominates the first column.

### Exact match by number, attractors and object-head number

`Matrix sg` / `Matrix pl` split the test rows by the number of the matrix subject, `k attractors` by the dataset's `n_attractors` (nouns of the other number between a verb and the noun it agrees with). The last two columns apply to object-attached RCs: whether the RC verb's number (that of the embedded subject) equals the number of the object head noun it is attached to.

| Attachment Site | Gap Type | Matrix sg | Matrix pl | 0 attractors | 1 attractor | 2 attractors | RC number = object head number | RC number ≠ object head number |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Subject | Subject Gap | 0.000 (n=648) | 0.000 (n=648) | 0.000 (n=864) | 0.000 (n=432) | — | — | — |
| Subject | Object Gap | 0.000 (n=432) | 0.000 (n=432) | 0.000 (n=432) | 0.000 (n=432) | — | — | — |
| Object | Object Gap | 0.712 (n=288) | 0.722 (n=288) | 0.717 (n=576) | — | — | 0.705 (n=288) | 0.729 (n=288) |

No number condition moves Object / Object Gap by more than 0.025; the failures there are not agreement-driven (see Errors).

### Across checkpoints

Selected checkpoints out of 150 (train loss is the running mean of the 50 steps before the checkpoint; exact match unless stated):

| Step | Train loss | Val loss | Val translate EM | Val copy EM | Test EM (all) | Subject / Subject Gap | Subject / Object Gap | Object / Object Gap | Object / Object Gap token acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,000 | 1.651 | 1.668 | 0.200 | 1.000 | 0.001 | 0.000 | 0.000 | 0.007 | 0.806 |
| 4,000 | 1.429 | 1.466 | 0.917 | 1.000 | 0.163 | 0.000 | 0.000 | 0.773 | 0.972 |
| 6,000 | 1.326 | 1.500 | 0.983 | 1.000 | 0.177 | 0.000 | 0.000 | 0.840 | 0.980 |
| 10,000 | 1.102 | 1.705 | 1.000 | 1.000 | 0.170 | 0.000 | 0.000 | 0.809 | 0.976 |
| 20,000 | 0.779 | 2.231 | 1.000 | 1.000 | 0.166 | 0.000 | 0.000 | 0.790 | 0.974 |
| 30,000 | 0.683 | 2.514 | 0.992 | 1.000 | 0.158 | 0.000 | 0.000 | 0.752 | 0.969 |
| 50,000 | 0.628 | 2.732 | 1.000 | 1.000 | 0.181 | 0.000 | 0.000 | 0.858 | 0.982 |
| 100,000 | 0.592 | 2.938 | 1.000 | 1.000 | 0.151 | 0.000 | 0.000 | 0.717 | 0.965 |
| 150,000 | 0.578 | 3.022 | 1.000 | 1.000 | 0.160 | 0.000 | 0.000 | 0.762 | 0.970 |
| 200,000 | 0.571 | 3.074 | 1.000 | 1.000 | 0.156 | 0.000 | 0.000 | 0.741 | 0.967 |
| 250,000 | 0.568 | 3.110 | 1.000 | 1.000 | 0.147 | 0.000 | 0.000 | 0.696 | 0.962 |
| 300,000 | 0.565 | 3.121 | 1.000 | 1.000 | 0.151 | 0.000 | 0.000 | 0.717 | 0.965 |

- The copy task is learned before translation (copy 1.000 at step 2,000, translate 0.200 → 1.000 by step 8,000); the `ce` model had translate at 1.000 already at step 2,000.
- Object / Object Gap test exact match rises with validation (0.007 → 0.773 between steps 2,000 and 4,000), **peaks at 0.938 at step 52,000 and then drifts down** to 0.70–0.86 at the sampled checkpoints, 0.700–0.738 over the last ten checkpoints, ending at 0.717. The generalisation to this construction is not stable under continued training on this dataset, unlike in the `ce` model (0.988–0.993 over its last ten checkpoints).
- Subject / Subject Gap and Subject / Object Gap are 0.000 at every one of the 150 checkpoints, with a linear-rule rate of 0.000 throughout. The failure is not a late collapse; it never works at any point in training.

## Errors

Natural-generation errors are the greedy outputs of the final checkpoint on the 2,323 wrong test sentences (validation has none). Notation: an *ordering* writes the gold and the model output as 1-based positions of the words in the source sentence (`?` = word not in the source); for a subject-attached RC the gold is always `R 1 V` with `R` = the mirrored relative clause ending in `that`, `1` = the head noun (the first source word) and `V` = the mirrored matrix VP. The evaluator's error types are `permutation` (exactly the gold words, wrong order), `missing` (a gold word absent), `repeat` (a source word emitted more often than in the gold), `foreign` (a word that is not in the source), joined with `+` when several apply; `linear` never occurs.

### The consistent pattern: the subject is always emitted first

| | Subject / Subject Gap | Subject / Object Gap | Object / Object Gap |
| --- | --- | --- | --- |
| Wrong outputs | 1,296 of 1,296 | 864 of 864 | 163 of 576 |
| Output starts with the first source word | 1,296 (1.000) | 864 (1.000) | 163 (1.000) |
| First teacher-forced error at target position 1 | 1,278 (0.986) | 847 (0.980) | 0 |
| First teacher-forced error at target position 2 | 18 (0.014) | 17 (0.020) | 162 (0.994) |
| Output has the gold length | 487 (0.376) | 598 (0.692) | 23 (0.141) |
| `<eos>` emitted | all | all | all |
| Distinct orderings | 403 | 222 | 31 |

**Every wrong test output begins with the first source word (2,323 of 2,323), and in 98–99% of the subject-attached sentences that is already the first mistake.** The gold Head-Final sentence for a subject-attached RC starts with the mirrored relative clause (`bird likes that cat dog chases`), but the model starts with the head noun (`cat …`) and never places a mirrored RC in front of it. The 18 + 17 sentences whose first teacher-forced prediction is right are the ones where the first gold word coincides with the head noun (for example `giraffe that giraffe loves …`). This matches the training data exactly: in all 4,656 training rows of `ce-rb` (both tasks) the first target word equals the first source word, because every ambiguous frame has a bare-noun subject and copying is the identity, so fronting a subject's relative clause is never demonstrated. Number agreement adds no pressure against this: the head noun and its verb keep their number wherever the model puts them.

What happens after the first word is *not* one consistent string (403 and 222 distinct orderings on the subject-attached constructions), so the rest of this section lists every error type per construction with real examples. Three secondary regularities recur and are quantified below:

1. **Object / Object Gap failures are one error**: in 110 of the 163 wrong outputs (0.675) the output is exactly the gold with the embedded subject (the noun after `that`) deleted, and the first teacher-forced error is at target position 2, the embedded subject's slot, in 162 of 163.
2. **Subject / Object Gap outputs lack the embedded subject** in 634 of 864 (0.734).
3. **Subject / Subject Gap outputs lose the matrix VP**: only 57 of 1,296 (0.044) end with the mirrored matrix VP (the `ce` model: 0.780); instead the model repeats the mirrored RC (`1 R R'`, 202 = 0.156) or emits foreign transitive verbs (412 of 615 foreign words are transitive verbs).

Number agreement itself is not a failure mode. Among the teacher-forced mispredictions, a prediction that is the other-number form of the gold word accounts for 26 of 5,410 (0.5%) in Subject / Subject Gap, 21 of 3,861 (0.5%) in Subject / Object Gap and 0 of 163 in Object / Object Gap; among the foreign words in greedy outputs only 1 of 615, 3 of 258 and 1 of 20 are the other-number form of a source word. The errors are structural (order, omission, repetition), and since the outputs are recombinations of the source words, the number features travel with the words.

### Test: Object / Object Gap (163 wrong of 576, exact match 0.717)

Error types:

| Error type | Count | Share | Examples (source → gold → model) |
| --- | --- | --- | --- |
| `missing` | 113 | 0.693 | `giraffes trust zebras that penguin likes` → gold `giraffes penguin likes that zebras trust` → model `giraffes likes that zebras trust`<br>`dogs poke walruses that rabbits kiss` → gold `dogs rabbits kiss that walruses poke` → model `dogs kiss that walruses poke`<br>`owl fears quail that iguana tickles carefully` → gold `owl iguana carefully tickles that quail fears` → model `owl carefully tickles that quail fears` |
| `missing+foreign` | 20 | 0.123 | `birds find ants that zebra knows` → gold `birds zebra knows that ants find` → model `birds walrus knows that ants find`<br>`yaks find boldly monkey that tiger hugs` → gold `yaks tiger hugs that monkey boldly find` → model `yaks urchin hugs that monkey boldly find`<br>`penguins admire mindfully zebras that penguin greets` → gold `penguins penguin greets that zebras mindfully admire` → model `penguins penguin tickles that zebras mindfully admire` |
| `repeat` | 18 | 0.110 | `ant pursues shark that flamingo pursues sweetly` → gold `ant flamingo sweetly pursues that shark pursues` → model `ant sweetly flamingo sweetly pursues that shark pursues`<br>`giraffe nudges monkey that sharks kiss shyly` → gold `giraffe sharks shyly kiss that monkey nudges` → model `giraffe shyly sharks shyly kiss that monkey nudges`<br>`rabbit avoids giraffes that newt carries sadly` → gold `rabbit newt sadly carries that giraffes avoids` → model `rabbit sadly newt sadly carries that giraffes avoids` |
| `missing+repeat` | 12 | 0.074 | `giraffe follows happily ants that tiger finds` → gold `giraffe tiger finds that ants happily follows` → model `giraffe ants finds that ants happily follows`<br>`lemur likes mindfully ants that jaguar hugs` → gold `lemur jaguar hugs that ants mindfully likes` → model `lemur ants hugs that ants mindfully likes`<br>`ants avoid wildly vulture that yak hates` → gold `ants yak hates that vulture wildly avoid` → model `ants vulture hates that vulture wildly avoid` |

Shape of the wrong outputs (share of the 163):

- output = gold with the embedded subject deleted, nothing else changed: **110 (0.675)**
- first teacher-forced error at target position 2 (the embedded subject's slot): 162 (0.994); the model predicts the RC verb or adverb there, i.e. it skips the noun after `that`
- output has the gold length: 23 (0.141), all of them `repeat` or `missing+foreign` cases
- the `missing+foreign` cases keep the slot but fill it with an unrelated noun (`birds walrus knows …` for `zebra`); the `repeat` cases put the RC adverb before the embedded subject and again after it; the `missing+repeat` cases put the object head noun into the embedded-subject slot (`giraffe ants finds that ants …`)

The most frequent orderings (top 8 of 31; the embedded subject is source position 5, or 6 when the matrix VP has an adverb):

| Ordering | Count | Share | Error type | Example |
| --- | --- | --- | --- | --- |
| `gold 1 5 7 6 4 3 2 -> model 1 7 6 4 3 2` | 67 | 0.411 | `missing` | `owl fears quail that iguana tickles carefully` → gold `owl iguana carefully tickles that quail fears` → model `owl carefully tickles that quail fears` |
| `gold 1 6 8 7 5 4 3 2 -> model 1 8 7 5 4 3 2` | 35 | 0.215 | `missing` | `ant chases softly bird that tiger avoids proudly` → gold `ant tiger proudly avoids that bird softly chases` → model `ant proudly avoids that bird softly chases` |
| `gold 1 6 8 7 5 4 3 2 -> model 1 8 6 8 7 5 4 3 2` | 11 | 0.067 | `repeat` | `newt chases wildly penguin that xerus grabs gently` → gold `newt xerus gently grabs that penguin wildly chases` → model `newt gently xerus gently grabs that penguin wildly chases` |
| `gold 1 6 8 7 5 4 3 2 -> model 1 ? 8 7 5 4 3 2` | 7 | 0.043 | `missing+foreign` | `newt greets loudly walrus that shark grabs warily` → gold `newt shark warily grabs that walrus loudly greets` → model `newt monkeys warily grabs that walrus loudly greets` |
| `gold 1 5 7 6 4 3 2 -> model 1 7 5 7 6 4 3 2` | 5 | 0.031 | `repeat` | `giraffe nudges monkey that sharks kiss shyly` → gold `giraffe sharks shyly kiss that monkey nudges` → model `giraffe shyly sharks shyly kiss that monkey nudges` |
| `gold 1 6 8 7 5 4 3 2 -> model 1 4 8 7 5 4 3 2` | 4 | 0.025 | `missing+repeat` | `owl fears lazily elephant that monkey lifts slowly` → gold `owl monkey slowly lifts that elephant lazily fears` → model `owl elephant slowly lifts that elephant lazily fears` |
| `gold 1 6 7 5 4 3 2 -> model 1 4 7 5 4 3 2` | 3 | 0.018 | `missing+repeat` | `giraffe follows happily ants that tiger finds` → gold `giraffe tiger finds that ants happily follows` → model `giraffe ants finds that ants happily follows` |
| `gold 1 5 7 6 4 3 2 -> model 1 ? 7 6 4 3 2` | 3 | 0.018 | `missing+foreign` | `giraffe hugs rabbit that sharks like carefully` → gold `giraffe sharks carefully like that rabbit hugs` → model `giraffe cat carefully like that rabbit hugs` |

Every one of the top orderings has an adverb somewhere in the sentence, and the failures concentrate on the longer frames: exact match is 0.979 on the 6-word frames, which have no adverb (141 of 144), 0.688 on 7-word frames (198 of 288) and 0.514 on 8-word frames (74 of 144); with an adverb anywhere in the sentence it is 0.630 (n=432) against 0.979 without (n=144). The exceptions on the short frames are the `missing` examples above (`giraffes likes that zebras trust`).

Which source word goes missing (over the 145 outputs with a `missing` flag):

| Missing word (frame category, source position) | Count |
| --- | --- |
| N_sg at source position 5 | 40 |
| N_pl at source position 5 | 37 |
| N_sg at source position 6 | 33 |
| N_pl at source position 6 | 24 |
| V_trans_sg at source position 7 | 4 |
| N_sg at source position 3 | 2 |
| N_sg at source position 1 | 2 |
| V_trans_sg at source position 6 | 2 |
| V_trans_sg at source position 2 | 2 |
| N_pl at source position 3 | 1 |

Positions 5 and 6 are the embedded subject (134 of the 150 missing words; the table shows the top 10); singular and plural embedded subjects go missing alike (73 vs 61).

Foreign words (20 tokens in 20 outputs):

| Kind | Count | Example |
| --- | --- | --- |
| unrelated N_sg | 10 | `birds find ants that zebra knows` → gold `birds zebra knows that ants find` → model `birds walrus knows that ants find` |
| unrelated N_pl | 4 | `owl follows iguanas that jaguars watch gently` → gold `owl jaguars gently watch that iguanas follows` → model `owl flamingos gently watch that iguanas follows` |
| unrelated V_trans_sg | 3 | `penguins admire mindfully zebras that penguin greets` → gold `penguins penguin greets that zebras mindfully admire` → model `penguins penguin tickles that zebras mindfully admire` |
| unrelated Adv | 1 | `walrus helps warily bird that giraffes chase calmly` → gold `walrus giraffes calmly chase that bird warily helps` → model `walrus gracefully chase that bird warily helps` |
| other-number form of a source word (N_sg) | 1 | `yak pokes wildly xeruses that yak loves bravely` → gold `yak yak bravely loves that xeruses wildly pokes` → model `yak xerus bravely loves that xeruses wildly pokes` |
| unrelated V_trans_pl | 1 | `zebras kiss gracefully elephant that penguins avoid gently` → gold `zebras penguins gently avoid that elephant gracefully kiss` → model `zebras gently follow that elephant gracefully kiss` |

When the embedded subject is replaced rather than dropped, the replacement noun has the number the RC verb requires in only 6 of the 15 cases (`birds walrus knows` does; `newt monkeys warily grabs` and `giraffe cat carefully like` do not): the model fills the slot with some noun without tracking the number the RC verb will need.

Teacher-forced mispredictions (163 in this construction): other category or special token 140 (0.859); same category, other lemma 23 (0.141); number flips 0.

### Test: Subject / Subject Gap (1,296 wrong of 1,296, exact match 0.000)

Error types:

| Error type | Count | Share | Examples (source → gold → model) |
| --- | --- | --- | --- |
| `missing+repeat` | 580 | 0.448 | `xerus that flies swims` → gold `flies that xerus swims` → model `xerus swims that xerus swims`<br>`shark that dives flies` → gold `dives that shark flies` → model `shark flies that shark flies`<br>`flamingo that snores smiles` → gold `snores that flamingo smiles` → model `flamingo smiles that flamingo smiles` |
| `missing+repeat+foreign` | 376 | 0.290 | `vulture that smiles jumps` → gold `smiles that vulture jumps` → model `vulture jumps that vulture pursues`<br>`jaguar that wanders hops` → gold `wanders that jaguar hops` → model `jaguar hops that jaguar greets`<br>`newt that sneezes sneezes` → gold `sneezes that newt sneezes` → model `newt sneezes that newt carries` |
| `missing+foreign` | 123 | 0.095 | `penguins that stretch hastily run` → gold `hastily stretch that penguins run` → model `penguins hastily stretch that giraffes know`<br>`elephant that sleeps spins shyly` → gold `sleeps that elephant shyly spins` → model `elephant shyly spins that dog pursue`<br>`owl that crawls finds penguin` → gold `crawls that owl penguin finds` → model `owl penguin crawls that vulture sees` |
| `permutation` | 111 | 0.086 | `lemurs that sleep happily spin` → gold `happily sleep that lemurs spin` → model `lemurs happily sleep that spin`<br>`dog that shivers grabs elephants` → gold `shivers that dog elephants grabs` → model `dog elephants shivers that grabs`<br>`cat that loves owl flies` → gold `owl loves that cat flies` → model `cat flies that owl loves` |
| `repeat` | 76 | 0.059 | `owl that smiles crawls` → gold `smiles that owl crawls` → model `owl crawls that owl smiles`<br>`xerus that swims wanders` → gold `swims that xerus wanders` → model `xerus wanders that xerus swims`<br>`giraffes that sneeze dive` → gold `sneeze that giraffes dive` → model `giraffes dive that giraffes sneeze` |
| `missing` | 13 | 0.010 | `iguana that swims warily fears flamingo` → gold `warily swims that iguana flamingo fears` → model `iguana flamingo warily swims`<br>`urchin that hops boldly carries newt` → gold `boldly hops that urchin newt carries` → model `urchin newt boldly hops`<br>`newt that swims happily trusts cat` → gold `happily swims that newt cat trusts` → model `newt cat happily trusts` |
| `repeat+foreign` | 11 | 0.008 | `zebra that stumbles dances` → gold `stumbles that zebra dances` → model `zebra dances that zebra gently stumbles`<br>`quails that hop jump` → gold `hop that quails jump` → model `quails jump that quails hastily hop`<br>`urchins that wander whistle` → gold `wander that urchins whistle` → model `urchins whistle that urchins gently wander` |
| `foreign` | 6 | 0.005 | `yaks that stretch grab flamingo` → gold `stretch that yaks flamingo grab` → model `yaks flamingo stretch that cat grab`<br>`hippos that carry tigers fly proudly` → gold `tigers carry that hippos proudly fly` → model `hippos proudly fly that tigers loudly carry`<br>`shark that finds zebra helps lazily urchin` → gold `zebra finds that shark urchin lazily helps` → model `shark urchin lazily finds that zebra shyly helps` |

Shape of the wrong outputs (share of the 1,296):

- output = `1 R V` (only the head noun fronted, everything else as in the gold): 4 (0.003); in the `ce` model this was the largest class (0.330)
- output starts with `1 R` (head noun, then the mirrored RC exactly as in the gold): 363 (0.280)
- output ends with `V` (the mirrored matrix VP exactly as in the gold): 57 (0.044)
- output = `1 R R'` (head noun, mirrored RC, then the mirrored RC again without `that`; the matrix VP is never produced): **202 (0.156)**
- mirrored matrix VP directly after the head noun (`1 V …`): 269 (0.208); for the shortest frames this yields the right-branching template `N V that N V` (`xerus swims that xerus swims`)
- output = `1 R 1 V`: 1 (0.001)

The most frequent orderings (top 8 of 403):

| Ordering | Count | Share | Error type | Example |
| --- | --- | --- | --- | --- |
| `gold 5 4 3 2 1 7 6 -> model 1 5 4 3 2 5 4 3` | 80 | 0.062 | `missing+repeat` | `walrus that trusts warily elephant sleeps quickly` → gold `elephant warily trusts that walrus quickly sleeps` → model `walrus elephant warily trusts that elephant warily trusts` |
| `gold 5 4 3 2 1 6 -> model 1 5 4 3 2 5 4 3` | 54 | 0.042 | `missing+repeat` | `walrus that likes boldly rabbit twirls` → gold `rabbit boldly likes that walrus twirls` → model `walrus rabbit boldly likes that rabbit boldly likes` |
| `gold 4 3 2 1 6 5 -> model 1 6 3 2 4 5` | 50 | 0.039 | `permutation` | `dog that fears quail greets hippo` → gold `quail fears that dog hippo greets` → model `dog hippo fears that quail greets` |
| `gold 4 3 2 1 5 -> model 1 4 3 2 4 3` | 43 | 0.033 | `missing+repeat` | `cat that watches iguana spins` → gold `iguana watches that cat spins` → model `cat iguana watches that iguana watches` |
| `gold 4 3 2 1 6 5 -> model 1 6 3 2 4 3` | 31 | 0.024 | `missing+repeat` | `urchin that avoids iguana spins slowly` → gold `iguana avoids that urchin slowly spins` → model `urchin slowly avoids that iguana avoids` |
| `gold 3 2 1 5 4 -> model 1 5 3 2 1 ?` | 29 | 0.022 | `missing+repeat+foreign` | `tiger that flies sings warily` → gold `flies that tiger warily sings` → model `tiger warily flies that tiger pursues` |
| `gold 4 3 2 1 7 6 5 -> model 1 7 6 3 2 4 6 3` | 29 | 0.022 | `missing+repeat` | `giraffe that greets elephant knows warily urchin` → gold `elephant greets that giraffe urchin warily knows` → model `giraffe urchin warily greets that elephant warily greets` |
| `gold 4 3 2 1 6 5 -> model 1 6 5 2 4 5` | 28 | 0.022 | `missing+repeat` | `urchin that pokes newt fears vulture` → gold `newt pokes that urchin vulture fears` → model `urchin vulture fears that newt fears` |

Reading: after `head noun + mirrored RC` the model has to produce the matrix VP but instead produces the RC again (`… that elephant warily trusts`, rows 1, 2, 4), or it swaps the RC object and the matrix object (row 3: `dog hippo fears that quail greets`, the last noun of the sentence is placed after the subject as in the trained right-branching translations, where the noun after the subject is the RC's object). Foreign transitive verbs appear where the matrix verb of the shortest frames should go (`tiger warily flies that tiger pursues`).

Which source word goes missing (over the 1,092 outputs with a `missing` flag; the matrix verb, the RC adverb and, on short frames, the RC verb):

| Missing word (frame category, source position) | Count |
| --- | --- |
| V_trans_sg at source position 6 | 107 |
| V_trans_pl at source position 6 | 106 |
| Adv at source position 4 | 104 |
| V_trans_sg at source position 5 | 92 |
| V_trans_pl at source position 5 | 85 |
| Adv at source position 7 | 85 |
| V_intrans_pl at source position 3 | 78 |
| V_intrans_sg at source position 3 | 74 |
| V_intrans_pl at source position 6 | 70 |
| V_intrans_sg at source position 5 | 68 |

Foreign words (615 tokens in 516 outputs; two thirds are transitive verbs put where an intransitive matrix verb belongs):

| Kind | Count | Example |
| --- | --- | --- |
| unrelated V_trans_sg | 214 | `vulture that smiles jumps` → gold `smiles that vulture jumps` → model `vulture jumps that vulture pursues` |
| unrelated V_trans_pl | 198 | `vultures that hum shiver` → gold `hum that vultures shiver` → model `vultures shiver that vultures know` |
| unrelated Adv | 138 | `zebra that stumbles dances` → gold `stumbles that zebra dances` → model `zebra dances that zebra gently stumbles` |
| unrelated N_sg | 23 | `giraffe that shivers calmly twirls` → gold `calmly shivers that giraffe twirls` → model `giraffe calmly twirls that bird calmly helps` |
| unrelated N_pl | 18 | `penguins that stretch hastily run` → gold `hastily stretch that penguins run` → model `penguins hastily stretch that giraffes know` |
| unrelated V_intrans_pl | 14 | `zebras that twirl quickly dive` → gold `quickly twirl that zebras dive` → model `zebras quickly dive that zebras quickly stretch` |
| unrelated V_intrans_sg | 9 | `bird that twirls sings wildly` → gold `twirls that bird wildly sings` → model `bird wildly twirls that bird warily flies` |
| other-number form of a source word (N_pl) | 1 | `flamingos that stumble admire carefully koala` → gold `stumble that flamingos koala carefully admire` → model `flamingos koala carefully stumble that koalas flamingos trust` |

401 of the 435 invented verbs carry the number of the matrix subject (`vulture … pursues`, `vultures … know`), although they are the wrong lemma and usually the wrong transitivity; only 143 of the 262 that directly follow a noun agree with that noun, so the number is copied from the subject, not from the nearest noun.

Teacher-forced mispredictions (5,410 in this construction): same category, other lemma 2,826 (0.522); other category or special token 2,558 (0.473); number flip 26 (0.005).

### Test: Subject / Object Gap (864 wrong of 864, exact match 0.000)

Error types:

| Error type | Count | Share | Examples (source → gold → model) |
| --- | --- | --- | --- |
| `missing+repeat` | 561 | 0.649 | `yak that koala loves laughs` → gold `koala loves that yak laughs` → model `yak laughs that yak loves`<br>`penguin that elephant avoids shivers` → gold `elephant avoids that penguin shivers` → model `penguin shivers that penguin avoids`<br>`rabbit that walrus chases flies` → gold `walrus chases that rabbit flies` → model `rabbit flies that rabbit chases` |
| `missing+repeat+foreign` | 189 | 0.219 | `flamingo that yaks hate dives` → gold `yaks hate that flamingo dives` → model `flamingo spin that flamingo hate`<br>`quail that monkeys find bravely smiles` → gold `monkeys bravely find that quail smiles` → model `quail bravely stretches that monkeys bravely hugs`<br>`dogs that vulture tickles mindfully snore` → gold `vulture mindfully tickles that dogs snore` → model `dogs mindfully snore that dogs mindfully meet` |
| `permutation` | 64 | 0.074 | `zebra that giraffe hugs hops` → gold `giraffe hugs that zebra hops` → model `zebra hops that giraffe hugs`<br>`xerus that tiger pursues swims` → gold `tiger pursues that xerus swims` → model `xerus swims that tiger pursues`<br>`koala that elephant hugs stumbles` → gold `elephant hugs that koala stumbles` → model `koala stumbles that elephant hugs` |
| `missing+foreign` | 39 | 0.045 | `walrus that monkey follows hums` → gold `monkey follows that walrus hums` → model `walrus hums that shark follows`<br>`iguana that birds grab shivers` → gold `birds grab that iguana shivers` → model `iguana shivers that shark grab`<br>`dog that ants see naps` → gold `ants see that dog naps` → model `dog hums that ants see` |
| `repeat` | 8 | 0.009 | `ant that tiger hates happily loves cat` → gold `tiger happily hates that ant cat loves` → model `ant cat happily loves that tiger happily hates`<br>`zebra that giraffes meet happily loves lemur` → gold `giraffes happily meet that zebra lemur loves` → model `zebra lemur happily loves that giraffes happily meet`<br>`ant that monkeys pursue briskly helps rabbits` → gold `monkeys briskly pursue that ant rabbits helps` → model `ant rabbits briskly pursue that monkeys briskly helps` |
| `missing` | 3 | 0.003 | `dog that lemur helps yawns happily` → gold `lemur helps that dog happily yawns` → model `dog happily helps`<br>`giraffe that giraffe loves slowly jumps wildly` → gold `giraffe slowly loves that giraffe wildly jumps` → model `giraffe wildly loves that giraffe slowly`<br>`penguin that xeruses grab slowly watches keenly zebra` → gold `xeruses slowly grab that penguin zebra keenly watches` → model `penguin watches that xeruses slowly grab` |

Shape of the wrong outputs (share of the 864):

- the embedded subject (the noun right after `that`, source position 3) is absent from the output: **634 (0.734)**
- mirrored matrix VP directly after the head noun (`1 V …`): 196 (0.227); on the shortest frames this is the right-branching template `N V that N V` with the head noun repeated in place of the embedded subject (`yak laughs that yak loves`)
- output = `1 R\emb V` (embedded subject deleted, otherwise gold): 0; output = `1 V\verb R\emb V` (embedded subject replaced by the matrix object): 9 (0.010). The `ce` model's two main shapes (0.116 and 0.286) hardly occur here.
- output = `1 R V`: 0; output starts with `1 R`: 23 (0.027); output ends with `V`: 15 (0.017); output = `1 R R'`: 2 (0.002)

The most frequent orderings (top 8 of 222):

| Ordering | Count | Share | Error type | Example |
| --- | --- | --- | --- | --- |
| `gold 3 4 2 1 6 5 -> model 1 6 4 2 1 4` | 63 | 0.073 | `missing+repeat` | `yak that flamingo sees shivers slowly` → gold `flamingo sees that yak slowly shivers` → model `yak slowly sees that yak sees` |
| `gold 3 5 4 2 1 7 6 -> model 1 7 5 4 2 1 5 4` | 52 | 0.060 | `missing+repeat` | `flamingo that walrus greets sadly fears jaguar` → gold `walrus sadly greets that flamingo jaguar fears` → model `flamingo jaguar sadly greets that flamingo sadly greets` |
| `gold 3 5 4 2 1 6 -> model 1 6 2 1 5 4` | 46 | 0.053 | `missing+repeat` | `penguin that ant carries sweetly flies` → gold `ant sweetly carries that penguin flies` → model `penguin flies that penguin sweetly carries` |
| `gold 3 4 2 1 5 -> model 1 5 2 1 4` | 45 | 0.052 | `missing+repeat` | `yak that koala loves laughs` → gold `koala loves that yak laughs` → model `yak laughs that yak loves` |
| `gold 3 4 2 1 6 5 -> model 1 6 4 2 3 4` | 26 | 0.030 | `missing+repeat` | `tiger that cat pokes sleeps briskly` → gold `cat pokes that tiger briskly sleeps` → model `tiger briskly pokes that cat pokes` |
| `gold 3 4 2 1 5 -> model 1 5 2 3 4` | 21 | 0.024 | `permutation` | `zebra that giraffe hugs hops` → gold `giraffe hugs that zebra hops` → model `zebra hops that giraffe hugs` |
| `gold 3 4 2 1 7 6 5 -> model 1 7 6 4 2 1 6 4` | 21 | 0.024 | `missing+repeat` | `penguin that yak carries kisses gracefully monkey` → gold `yak carries that penguin monkey gracefully kisses` → model `penguin monkey gracefully carries that penguin gracefully carries` |
| `gold 3 5 4 2 1 7 6 -> model 1 7 5 4 2 3 5 4` | 20 | 0.023 | `missing+repeat` | `xerus that zebra watches happily hates iguana` → gold `zebra happily watches that xerus iguana hates` → model `xerus iguana happily watches that zebra happily watches` |

Reading: the head noun takes the place of the embedded subject on both sides of `that` (`yak … sees that yak sees`, `flamingo … greets that flamingo sadly greets`), i.e. the model treats the object-gap RC as if it were a subject-gap RC whose verb belongs to the head noun, and the matrix VP material is attached to the head noun directly after it. When the embedded subject does appear, it appears after `that` in the position a right-branching translation would give it (`tiger briskly pokes that cat pokes`, `xerus … that zebra happily watches`).

Which source word goes missing (over the 792 outputs with a `missing` flag):

| Missing word (frame category, source position) | Count |
| --- | --- |
| N_sg at source position 3 | 333 |
| N_pl at source position 3 | 301 |
| V_trans_pl at source position 6 | 125 |
| V_trans_sg at source position 6 | 122 |
| V_trans_sg at source position 5 | 115 |
| V_trans_pl at source position 5 | 87 |
| Adv at source position 7 | 76 |
| V_intrans_sg at source position 5 | 35 |
| V_intrans_pl at source position 5 | 33 |
| V_intrans_sg at source position 6 | 27 |

Foreign words (258 tokens in 228 outputs):

| Kind | Count | Example |
| --- | --- | --- |
| unrelated V_trans_sg | 73 | `quail that monkeys find bravely smiles` → gold `monkeys bravely find that quail smiles` → model `quail bravely stretches that monkeys bravely hugs` |
| unrelated V_trans_pl | 62 | `dogs that vulture tickles mindfully snore` → gold `vulture mindfully tickles that dogs snore` → model `dogs mindfully snore that dogs mindfully meet` |
| unrelated Adv | 51 | `tigers that zebras grab boldly snore` → gold `zebras boldly grab that tigers snore` → model `tigers happily snore that tigers boldly grab` |
| unrelated N_pl | 31 | `tigers that xerus knows boldly laugh` → gold `xerus boldly knows that tigers laugh` → model `tigers laugh that sharks boldly knows` |
| unrelated N_sg | 20 | `walrus that monkey follows hums` → gold `monkey follows that walrus hums` → model `walrus hums that shark follows` |
| unrelated V_intrans_pl | 10 | `flamingo that yaks hate dives` → gold `yaks hate that flamingo dives` → model `flamingo spin that flamingo hate` |
| unrelated V_intrans_sg | 8 | `dog that ants see naps` → gold `ants see that dog naps` → model `dog hums that ants see` |
| other-number form of a source word (V_trans_pl) | 2 | `dog that hippos like follows bravely ants` → gold `hippos like that dog ants bravely follows` → model `dog ants bravely like that ants bravely follow` |
| other-number form of a source word (N_sg) | 1 | `penguins that shark avoids tickle carefully rabbit` → gold `shark avoids that penguins rabbit carefully tickle` → model `penguins rabbit carefully avoids that penguin carefully avoids` |

The three number-variant cases are the only outputs in this construction where a verb or noun changed number, and two of them are the model re-agreeing an RC verb with the noun it wrongly put in front of it (`ants bravely follow`, `penguin carefully avoids`); `flamingo spin that flamingo hate` and `quail … monkeys bravely hugs` break agreement outright (a plural verb after a singular head noun, a singular verb after a plural one); over all 155 invented verbs of this construction, 117 carry the matrix subject's number.

Teacher-forced mispredictions (3,861 in this construction): same category, other lemma 2,239 (0.580); other category or special token 1,601 (0.415); number flip 21 (0.005).
