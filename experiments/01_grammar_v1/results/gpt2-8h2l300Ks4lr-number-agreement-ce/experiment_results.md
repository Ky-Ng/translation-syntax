# gpt2-8h2l300Ks4lr-number-agreement-ce — experiment results

GPT-2 (2 layers, 8 heads, d_model 128) trained for 300K steps on `kylelovesllms/grammar-number-agreement-frames-ce`: Grammar V1 with subject-verb number agreement, where the copy task contains only simple (no RC) and center-embedded (subject-attached RC) sentences. All numbers are from the final checkpoint (step 300,000) unless a step is given. Every table was computed from the evaluator's per-example CSVs (`eval/step_300000/{validation,test}.csv`, see [`src/training/evaluate.py`](../../src/training/evaluate.py)) by [`../analyze_2l_number_agreement.py`](../analyze_2l_number_agreement.py), which joins each evaluated sentence with the dataset's `Matrix Number`, `RC Number` and `n_attractors` columns.

## Run summary

| | |
| --- | --- |
| Seeds trained | **1** (seed 42). The 2026-09-23 sweep ran one seed per dataset and depth. |
| Checkpoints | **150**, every 2,000 steps from 2,000 to 300,000, each evaluated on validation and test right after saving (`--eval-every` defaults to `--checkpoint-every`) |
| wandb | run [gpt2-8h2l300Ks4lr-number-agreement-ce-seed42](https://wandb.ai/kgng-usc/translation-syntax/runs/fmbmjmh2) in project [translation-syntax](https://wandb.ai/kgng-usc/translation-syntax), group `gpt2-8h2l300Ks4lr-number-agreement-ce`. Keys `train/loss`, `val/loss`, `validation/…` and `test/translate/<Site>_<Gap>/{exact_match,token_acc,linear_rate}` are logged at every checkpoint. |
| Dataset | `kylelovesllms/grammar-number-agreement-frames-ce`: train 4,776 rows / 77,672 tokens (1,380 translate + 276 copy rows of the 60 ambiguous frames, 3,120 copy rows of the 120 center-embedded frames; boost factor 1.4397), validation 240 rows, test 2,736 rows. Local copy: `data/grammar-number-agreement-frames-ce/`. |
| Model | `GPT2LMHeadModel`, 2 layers, 8 heads, d_model 128, vocabulary 189 ids (number-agreement lexicon), loss on every non-pad token including the source, absolute positions shifted by 0–25 pads |
| Optimisation | Adam, lr 1e-4 with linear decay to 0, batch 256, 300,000 steps (about 16K passes over the training set) |
| SLURM job | 6013438 (`number-agreement-ce-2l`), node c06-01, 1 h 36 min, 2026-09-23 |
| Artifacts | [`../../artifacts/gpt2-8h2l300Ks4lr-number-agreement-ce-seed42/`](../../artifacts/gpt2-8h2l300Ks4lr-number-agreement-ce-seed42/): `launch_config.json`, `loss.csv`, `checkpoints/step_XXXXXX/`, `eval/step_XXXXXX/{validation,test,summary,errors_validation,errors_test,confusions_*}.csv` |
| Final losses | train 0.574 (mean of the last 50 steps), validation 5.590 |

The validation loss is the full-sequence LM loss, so it mostly scores the *source* words, which are lexicalizations the model has never seen. It climbs from 2.37 at step 2,000 to 5.59 at 300,000 while validation exact match stays at 1.000: it measures how confidently the model has memorised the training lexicalizations, not translation quality. Exact match and token accuracy are the metrics to read.

## Validation accuracy (step 300,000)

Validation holds 2 held-out lexicalizations per ambiguous frame and task (60 frames × 2 tasks × 2 = 240 rows). Only the two in-distribution constructions exist in this split; the unambiguous constructions are the test set below.

Column meaning (same header as the sentence-count table of the dataset README): `Frames`, `Ambiguous` and `Unambiguous` count the frames of the construction in the split, `Sentences` the rows, `Sample factor` the rows per frame. `RC number = matrix number` and `RC number differs` give the exact match on the sentences of that agreement condition (n in brackets; `—` when the construction has no RC or the condition cannot occur). `Exact match` is greedy decoding == gold, `Token acc (teacher forced)` the next-token accuracy over the target positions including `<eos>` given the gold prefix, `Linear rule` the share of outputs equal to the keep-first-word-reverse-the-rest string.

### Translate

| Attachment Site | Gap Type | Frames | Ambiguous | Unambiguous | Sentences | Sample factor | RC number = matrix number | RC number differs | Exact match | Token acc (teacher forced) | Linear rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| None | None | 12 | 12 | 0 | 24 | 2 | — | — | 1.000 | 1.000 | 0.000 |
| Object | Subject Gap | 48 | 48 | 0 | 96 | 2 | 1.000 (n=48) | 1.000 (n=48) | 1.000 | 1.000 | 0.000 |
| **All** |  | **60** | **60** | **0** | **120** |  | 1.000 (n=48) | 1.000 (n=48) | **1.000** | **1.000** | **0.000** |

Both in-distribution constructions are translated perfectly, in both agreement conditions and for singular and plural matrix subjects (12/12 and 12/12 on no-RC sentences, 48/48 and 48/48 on right-branching ones). Validation translate exact match was already 1.000 at the first checkpoint (step 2,000) and stayed there at 140 of the 150 checkpoints (lowest value 0.983 at step 20,000).

### Copy

| Attachment Site | Gap Type | Frames | Ambiguous | Unambiguous | Sentences | Sample factor | RC number = matrix number | RC number differs | Exact match | Token acc (teacher forced) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| None | None | 12 | 12 | 0 | 24 | 2 | — | — | 1.000 | 1.000 |
| Object | Subject Gap | 48 | 48 | 0 | 96 | 2 | 0.000 (n=48) | 0.000 (n=48) | 0.000 | 0.408 |
| **All** |  | **60** | **60** | **0** | **120** |  | 0.000 (n=48) | 0.000 (n=48) | **0.200** | **0.478** |

Copying right-branching (object-attached RC) validation sentences fails completely. This is expected: the `ce` dataset removes exactly those copy rows from training, while the validation split is shared with `ce-rb` and still contains them. The model does not copy them, it rewrites them into center-embedded sentences (see [Errors](#errors)). Validation copy exact match was 0.200 (= the 24 no-RC rows) at every one of the 150 checkpoints.

## Test accuracy (step 300,000)

The test split holds 18 lexicalizations of every unambiguous frame, translate task only (2,736 rows); none of these sentences appears in training in any task. Both accuracy types the evaluator computes are shown: **exact match** (greedy decode from `<bos> src <translate> <sep>` until `<eos>` equals the gold Head-Final sentence) and **teacher-forced token accuracy** (argmax next token given the gold prefix, over the target positions including `<eos>`).

| Attachment Site | Gap Type | Frames | Ambiguous | Unambiguous | Sentences | Sample factor | RC number = matrix number | RC number differs | Exact match | Token acc (teacher forced) | Linear rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Subject | Subject Gap | 72 | 0 | 72 | 1,296 | 18 | 0.000 (n=1296) | — | 0.000 | 0.538 | 0.000 |
| Subject | Object Gap | 48 | 0 | 48 | 864 | 18 | 0.000 (n=432) | 0.000 (n=432) | 0.000 | 0.503 | 0.000 |
| Object | Object Gap | 32 | 0 | 32 | 576 | 18 | 0.993 (n=288) | 0.993 (n=288) | 0.993 | 0.999 | 0.000 |
| **All** |  | **152** | **0** | **152** | **2,736** |  | 0.142 (n=2016) | 0.397 (n=720) | **0.209** | **0.629** | **0.000** |

- **Object-attached object-gap RCs generalise** (0.993 exact match, 4 wrong of 576), although no object-gap RC on an object ever appears in `ce` training in either task. The hierarchical rule inside the VP is applied to a structure the model has only seen center-embedded, and only in the copy task.
- **Subject-attached RCs never translate** (0 of 2,160), for subject gaps and object gaps alike. The linear rule is never used either (0.000 at every one of the 150 checkpoints); every wrong output is `other`. Token accuracy of about 0.5 means the model recovers the second half of the target once given the gold prefix, but the first target position is wrong in 98–99% of these sentences (see Errors).
- **Number agreement does not separate the conditions.** Exact match is identical whether the RC number equals the matrix number or differs (0.993 vs 0.993 on Object / Object Gap, 0.000 vs 0.000 on Subject / Object Gap), and attractors (only center-embedded sentences have them) do not create a gradient because the center-embedded constructions are at zero anyway. The "all" row differs between the two conditions (0.142 vs 0.397) only because the Subject / Subject Gap construction, which can only be in the "same number" condition, dominates the first column.

### Exact match by number, attractors and object-head number

`Matrix sg` / `Matrix pl` split the test rows by the number of the matrix subject, `k attractors` by the dataset's `n_attractors` (nouns of the other number between a verb and the noun it agrees with). The last two columns apply to object-attached RCs: whether the RC verb's number (that of the embedded subject) equals the number of the object head noun it is attached to.

| Attachment Site | Gap Type | Matrix sg | Matrix pl | 0 attractors | 1 attractor | 2 attractors | RC number = object head number | RC number ≠ object head number |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Subject | Subject Gap | 0.000 (n=648) | 0.000 (n=648) | 0.000 (n=864) | 0.000 (n=432) | — | — | — |
| Subject | Object Gap | 0.000 (n=432) | 0.000 (n=432) | 0.000 (n=432) | 0.000 (n=432) | — | — | — |
| Object | Object Gap | 0.990 (n=288) | 0.997 (n=288) | 0.993 (n=576) | — | — | 1.000 (n=288) | 0.986 (n=288) |

The only non-trivial split is Object / Object Gap: all 4 failures are sentences where the embedded subject's number differs from the object head's number (0.986 vs 1.000). With 4 errors this is suggestive at most.

### Across checkpoints

Selected checkpoints out of 150 (train loss is the running mean of the 50 steps before the checkpoint; exact match unless stated):

| Step | Train loss | Val loss | Val translate EM | Val copy EM | Test EM (all) | Subject / Subject Gap | Subject / Object Gap | Object / Object Gap | Object / Object Gap token acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,000 | 1.601 | 2.371 | 1.000 | 0.200 | 0.207 | 0.000 | 0.000 | 0.983 | 0.998 |
| 4,000 | 1.395 | 2.504 | 1.000 | 0.200 | 0.204 | 0.000 | 0.000 | 0.969 | 0.996 |
| 6,000 | 1.305 | 2.663 | 1.000 | 0.200 | 0.189 | 0.000 | 0.000 | 0.898 | 0.986 |
| 10,000 | 1.092 | 3.042 | 1.000 | 0.200 | 0.201 | 0.000 | 0.001 | 0.953 | 0.994 |
| 20,000 | 0.775 | 3.968 | 0.983 | 0.200 | 0.207 | 0.000 | 0.000 | 0.983 | 0.998 |
| 30,000 | 0.685 | 4.390 | 1.000 | 0.200 | 0.208 | 0.000 | 0.001 | 0.988 | 0.998 |
| 50,000 | 0.634 | 4.773 | 1.000 | 0.200 | 0.210 | 0.000 | 0.000 | 0.997 | 1.000 |
| 100,000 | 0.600 | 5.040 | 1.000 | 0.200 | 0.208 | 0.000 | 0.000 | 0.990 | 0.999 |
| 150,000 | 0.586 | 5.327 | 1.000 | 0.200 | 0.209 | 0.000 | 0.000 | 0.993 | 0.999 |
| 200,000 | 0.579 | 5.424 | 1.000 | 0.200 | 0.209 | 0.000 | 0.000 | 0.993 | 0.999 |
| 250,000 | 0.577 | 5.542 | 1.000 | 0.200 | 0.209 | 0.000 | 0.000 | 0.993 | 0.999 |
| 300,000 | 0.574 | 5.590 | 1.000 | 0.200 | 0.209 | 0.000 | 0.000 | 0.993 | 0.999 |

- Validation translate exact match is 1.000 from the first checkpoint on (140 of 150 checkpoints).
- Object / Object Gap test exact match is 0.983 at step 2,000, dips to 0.898 at step 6,000, is at or above 0.98 from step 20,000 on, peaks at 1.000 (step 138,000) and is 0.988–0.993 over the last ten checkpoints.
- Subject / Subject Gap reaches at most 0.001 (one checkpoint above zero); Subject / Object Gap at most 0.002 (15 checkpoints above zero). The linear-rule rate is 0.000 at every checkpoint for both. The failure on subject-attached RCs is therefore not a late collapse; it never works at any point in training.

## Errors

Natural-generation errors are the greedy outputs of the final checkpoint on the 2,164 wrong test sentences and the 96 wrong validation copies. Notation: an *ordering* writes the gold and the model output as 1-based positions of the words in the source sentence (`?` = word not in the source); for a subject-attached RC the gold is always `R 1 V` with `R` = the mirrored relative clause ending in `that`, `1` = the head noun (the first source word) and `V` = the mirrored matrix VP. The evaluator's error types are `permutation` (exactly the gold words, wrong order), `missing` (a gold word absent), `repeat` (a source word emitted more often than in the gold), `foreign` (a word that is not in the source), joined with `+` when several apply; `linear` never occurs.

### The consistent pattern: the subject is always emitted first

| | Subject / Subject Gap | Subject / Object Gap | Object / Object Gap | Validation copy, Object / Subject Gap |
| --- | --- | --- | --- | --- |
| Wrong outputs | 1,296 of 1,296 | 864 of 864 | 4 of 576 | 96 of 96 |
| Output starts with the first source word | 1,296 (1.000) | 864 (1.000) | 4 (1.000) | 96 (1.000) |
| First teacher-forced error at target position 1 | 1,278 (0.986) | 847 (0.980) | 0 | 0 |
| First teacher-forced error at target position 2 | 18 (0.014) | 13 (0.015) | 2 | 96 (1.000) |
| Output has the gold length | 635 (0.490) | 408 (0.472) | 0 | 75 (0.781) |
| `<eos>` emitted | all | all | all | all |
| Distinct orderings | 173 | 206 | 3 | 25 |

**Every wrong test output begins with the first source word (2,164 of 2,164), and in 98–99% of the subject-attached sentences that is already the first mistake.** The gold Head-Final sentence for a subject-attached RC starts with the mirrored relative clause (`bird likes that cat dog chases`), but the model starts with the head noun (`cat …`) and never places a mirrored RC in front of it. The 18 + 13 sentences whose first teacher-forced prediction is right are the ones where the first gold word coincides with the head noun (for example `giraffe that giraffe loves …`). This matches the training data exactly: in all 4,776 training rows of `ce` (both tasks) the first target word equals the first source word, because every ambiguous frame has a bare-noun subject and copying is the identity, so fronting a subject's relative clause is never demonstrated. Number agreement adds no pressure against this: the head noun and its verb keep their number wherever the model puts them.

What happens after the first word is *not* one consistent string (173 and 206 distinct orderings), so the rest of this section lists every error type per construction with real examples. Two secondary regularities do recur and are quantified below: in Subject / Subject Gap sentences the model keeps all the right constituents and only fronts the subject (`1 R V`, 33%), and in Subject / Object Gap sentences the embedded subject of the RC is dropped or replaced (75.6% of the failures lack it).

Number agreement itself is not a failure mode. Among the teacher-forced mispredictions, a prediction that is the other-number form of the gold word (`swims` → `swim`, `cat` → `cats`) accounts for 32 of 4,395 (0.7%) in Subject / Subject Gap and 26 of 3,294 (0.8%) in Subject / Object Gap, and among the foreign words in greedy outputs only 2 of 297 and 3 of 172 are the other-number form of a source word. The errors are structural (order, omission, repetition), and since the outputs are recombinations of the source words, the number features travel with the words.

### Test: Subject / Subject Gap (1,296 wrong of 1,296, exact match 0.000)

Error types:

| Error type | Count | Share | Examples (source → gold → model) |
| --- | --- | --- | --- |
| `permutation` | 439 | 0.339 | `owl that whistles gracefully wanders` → gold `gracefully whistles that owl wanders` → model `owl gracefully whistles that wanders`<br>`lemur that smiles gracefully swims` → gold `gracefully smiles that lemur swims` → model `lemur gracefully smiles that swims`<br>`dogs that smile briskly fly` → gold `briskly smile that dogs fly` → model `dogs briskly smile that fly` |
| `repeat` | 348 | 0.269 | `newt that sneezes sneezes` → gold `sneezes that newt sneezes` → model `newt sneezes that newt sneezes`<br>`zebra that hums twirls` → gold `hums that zebra twirls` → model `zebra twirls that zebra hums`<br>`owl that shivers shivers` → gold `shivers that owl shivers` → model `owl shivers that owl shivers` |
| `missing+repeat` | 238 | 0.184 | `vulture that smiles jumps` → gold `smiles that vulture jumps` → model `vulture jumps that vulture jumps`<br>`xerus that flies swims` → gold `flies that xerus swims` → model `xerus swims that xerus swims`<br>`flamingo that snores smiles` → gold `snores that flamingo smiles` → model `flamingo smiles that flamingo smiles` |
| `missing+repeat+foreign` | 137 | 0.106 | `owl that smiles crawls` → gold `smiles that owl crawls` → model `owl crawls that owl wanders`<br>`jaguar that wanders hops` → gold `wanders that jaguar hops` → model `jaguar hops that jaguar knows`<br>`zebra that stumbles dances` → gold `stumbles that zebra dances` → model `zebra dances that zebra avoids` |
| `missing` | 45 | 0.035 | `rabbit that sleeps carefully twirls` → gold `carefully sleeps that rabbit twirls` → model `rabbit carefully sleeps that`<br>`walrus that twirls gently snores` → gold `gently twirls that walrus snores` → model `walrus gently twirls that`<br>`owl that stumbles sadly sneezes` → gold `sadly stumbles that owl sneezes` → model `owl sadly stumbles that` |
| `foreign` | 43 | 0.033 | `yak that laughs happily shivers` → gold `happily laughs that yak shivers` → model `yak happily laughs that giraffe shivers`<br>`ants that stretch quickly nap` → gold `quickly stretch that ants nap` → model `ants quickly stretch that newts nap`<br>`penguins that stretch hastily run` → gold `hastily stretch that penguins run` → model `penguins hastily stretch that giraffe run` |
| `missing+foreign` | 43 | 0.033 | `ants that sneeze swim quietly` → gold `sneeze that ants quietly swim` → model `ants quietly sneeze that urchin follow`<br>`yak that sings mindfully sneezes sweetly` → gold `mindfully sings that yak sweetly sneezes` → model `yak mindfully sings that flamingos sneezes`<br>`owl that sleeps sadly runs briskly` → gold `sadly sleeps that owl briskly runs` → model `owl briskly sleeps that tigers chases` |
| `repeat+foreign` | 3 | 0.002 | `yak that sleeps sneezes boldly` → gold `sleeps that yak boldly sneezes` → model `yak boldly sleeps that yak quietly sneezes`<br>`tigers that whistle follow boldly owl` → gold `whistle that tigers owl boldly follow` → model `tigers koala boldly whistle that owl boldly follow`<br>`walrus that hugs yak wanders lazily` → gold `yak hugs that walrus lazily wanders` → model `walrus yak lazily hugs that urchin lazily wanders` |

Shape of the wrong outputs (share of the 1,296):

- output = `1 R V`, i.e. the gold with the head noun moved to the front and nothing else changed: **428 (0.330)**. Both mirrorings (inside the RC and inside the matrix VP) are right; only the position of the head noun relative to its RC is wrong.
- output starts with `1 R` (head noun, then the mirrored RC exactly as in the gold): 729 (0.562)
- output ends with `V` (the mirrored matrix VP exactly as in the gold): 1,011 (0.780)
- output = `1 R 1 V` (head noun fronted *and* kept at its gold position): 128 (0.099)
- mirrored matrix VP directly after the head noun (`1 V …`): 48 (0.037)

The most frequent orderings (top 8 of 173):

| Ordering | Count | Share | Error type | Example |
| --- | --- | --- | --- | --- |
| `gold 4 3 2 1 6 5 -> model 1 4 3 2 6 5` | 193 | 0.149 | `permutation` | `monkey that laughs mindfully walks hastily` → gold `mindfully laughs that monkey hastily walks` → model `monkey mindfully laughs that hastily walks` |
| `gold 5 4 3 2 1 7 6 -> model 1 5 4 3 2 7 6` | 120 | 0.093 | `permutation` | `lemur that nudges softly shark jumps proudly` → gold `shark softly nudges that lemur proudly jumps` → model `lemur shark softly nudges that proudly jumps` |
| `gold 5 4 3 2 1 8 7 6 -> model 1 5 7 3 2 8 7 6` | 103 | 0.079 | `missing+repeat` | `iguana that knows mindfully penguin nudges sadly tiger` → gold `penguin mindfully knows that iguana tiger sadly nudges` → model `iguana penguin sadly knows that tiger sadly nudges` |
| `gold 4 3 2 1 7 6 5 -> model 1 4 6 3 2 7 6 5` | 102 | 0.079 | `repeat` | `flamingo that sees owl helps sadly vulture` → gold `owl sees that flamingo vulture sadly helps` → model `flamingo owl sadly sees that vulture sadly helps` |
| `gold 4 3 2 1 7 6 5 -> model 1 4 3 2 7 6 5` | 62 | 0.048 | `permutation` | `monkey that sneezes clumsily loves gracefully vulture` → gold `clumsily sneezes that monkey vulture gracefully loves` → model `monkey clumsily sneezes that vulture gracefully loves` |
| `gold 4 3 2 1 5 -> model 1 4 3 2 1 5` | 54 | 0.042 | `repeat` | `bird that smiles quietly yawns` → gold `quietly smiles that bird yawns` → model `bird quietly smiles that bird yawns` |
| `gold 5 4 3 2 1 6 -> model 1 5 4 3 2 1 6` | 47 | 0.036 | `repeat` | `walrus that likes boldly rabbit twirls` → gold `rabbit boldly likes that walrus twirls` → model `walrus rabbit boldly likes that walrus twirls` |
| `gold 3 2 1 5 4 -> model 1 5 3 2 5 ?` | 40 | 0.031 | `missing+repeat+foreign` | `owl that crawls finds penguin` → gold `crawls that owl penguin finds` → model `owl penguin crawls that penguin pokes` |

Three sub-patterns are visible in the table: (a) the `1 R V` fronting on frames whose RC and VP both contain an adverb or object (rows 1, 2, 5); (b) the matrix adverb leaks into the RC when both clauses have a transitive verb (`sadly` in rows 3 and 4: the model puts the matrix adverb before the RC verb *and* before the matrix verb); (c) on the shortest frames (`N that V V`, `N that V Adv V`) the output takes the shape of an in-distribution right-branching translation `N V that N V` (subject, matrix verb, `that`, subject again, RC verb or an invented one), for example `zebra that hums twirls` → `zebra twirls that zebra hums`, `owl that smiles crawls` → `owl crawls that owl wanders`.

Which source word goes missing (over the 463 outputs with a `missing` flag; the RC-final adverb and the verbs of short frames):

| Missing word (frame category, source position) | Count |
| --- | --- |
| Adv at source position 4 | 162 |
| V_intrans_sg at source position 5 | 37 |
| V_trans_pl at source position 4 | 37 |
| Adv at source position 6 | 36 |
| V_intrans_pl at source position 5 | 35 |
| V_trans_sg at source position 4 | 30 |
| V_intrans_pl at source position 3 | 26 |
| V_intrans_sg at source position 3 | 21 |
| V_trans_pl at source position 3 | 18 |
| V_intrans_pl at source position 4 | 16 |

Foreign words (words in the output that are not in the source; 244 tokens in 226 outputs). They are almost always a verb or noun of the right category but a different lemma, i.e. the model fills a slot with a plausible word rather than the source word; only 2 are a number variant of a source word:

| Kind | Count | Example |
| --- | --- | --- |
| unrelated V_trans_pl | 73 | `koalas that hum yawn` → gold `hum that koalas yawn` → model `koalas yawn that koalas see` |
| unrelated V_trans_sg | 53 | `jaguar that wanders hops` → gold `wanders that jaguar hops` → model `jaguar hops that jaguar knows` |
| unrelated N_pl | 47 | `ants that stretch quickly nap` → gold `quickly stretch that ants nap` → model `ants quickly stretch that newts nap` |
| unrelated N_sg | 32 | `yak that laughs happily shivers` → gold `happily laughs that yak shivers` → model `yak happily laughs that giraffe shivers` |
| unrelated Adv | 21 | `yak that sleeps sneezes boldly` → gold `sleeps that yak boldly sneezes` → model `yak boldly sleeps that yak quietly sneezes` |
| unrelated V_intrans_sg | 10 | `owl that smiles crawls` → gold `smiles that owl crawls` → model `owl crawls that owl wanders` |
| unrelated V_intrans_pl | 6 | `tigers that clap gently dance` → gold `gently clap that tigers dance` → model `tigers gently clap that tigers nap` |
| other-number form of a source word (V_trans_sg) | 1 | `xeruses that snore poke jaguar` → gold `snore that xeruses jaguar poke` → model `xeruses jaguar snore that jaguar pokes` |
| other-number form of a source word (N_pl) | 1 | `quail that pursues shark crawls` → gold `shark pursues that quail crawls` → model `quail shark pursues that sharks crawls` |

Teacher-forced mispredictions (4,395 in this construction): other category or special token 2,350 (0.535); same category, other lemma 2,013 (0.458); number flip (same lemma, other number) 32 (0.007). In the examples above the foreign transitive verbs fill an intransitive slot (`koalas yawn that koalas see`): when the model invents a verb it does not respect the frame's transitivity, but 137 of the 143 invented verbs carry the number of the matrix subject (`koalas … see`, `jaguar … knows`); only 72 of the 113 that directly follow a noun agree with that noun, so the number comes from the subject, not from the nearest noun.

### Test: Subject / Object Gap (864 wrong of 864, exact match 0.000)

Error types:

| Error type | Count | Share | Examples (source → gold → model) |
| --- | --- | --- | --- |
| `missing+repeat` | 520 | 0.602 | `rabbit that walrus chases flies` → gold `walrus chases that rabbit flies` → model `rabbit chases that walrus chases`<br>`owl that yak chases dives` → gold `yak chases that owl dives` → model `owl yak chases that yak chases`<br>`quail that owl grabs runs` → gold `owl grabs that quail runs` → model `quail runs that quail grabs` |
| `missing` | 121 | 0.140 | `flamingo that zebra pokes gracefully wanders` → gold `zebra gracefully pokes that flamingo wanders` → model `flamingo gracefully pokes that wanders`<br>`owl that monkey fears quietly flies` → gold `monkey quietly fears that owl flies` → model `owl quietly fears that flies`<br>`hippo that birds pursue softly whistles` → gold `birds softly pursue that hippo whistles` → model `hippo softly pursue that` |
| `missing+repeat+foreign` | 75 | 0.087 | `elephant that koalas lift twirls` → gold `koalas lift that elephant twirls` → model `elephant clumsily lift that elephant twirls`<br>`flamingos that vulture nudges sing` → gold `vulture nudges that flamingos sing` → model `flamingos vulture lifts that vulture nudges`<br>`zebras that xerus kisses jump` → gold `xerus kisses that zebras jump` → model `zebras wildly kisses that xerus kisses` |
| `missing+foreign` | 74 | 0.086 | `yak that koala loves laughs` → gold `koala loves that yak laughs` → model `yak hops that koala loves`<br>`walrus that monkey follows hums` → gold `monkey follows that walrus hums` → model `walrus sleeps that monkey follows`<br>`koala that elephant hugs stumbles` → gold `elephant hugs that koala stumbles` → model `koala stumbles that lemur hugs` |
| `permutation` | 63 | 0.073 | `penguin that elephant avoids shivers` → gold `elephant avoids that penguin shivers` → model `penguin shivers that elephant avoids`<br>`zebra that giraffe hugs hops` → gold `giraffe hugs that zebra hops` → model `zebra hops that giraffe hugs`<br>`xerus that tiger pursues swims` → gold `tiger pursues that xerus swims` → model `xerus swims that tiger pursues` |
| `repeat` | 8 | 0.009 | `elephants that giraffes see smile` → gold `giraffes see that elephants smile` → model `elephants giraffes see that giraffes smile`<br>`quail that newts trust jumps sadly` → gold `newts trust that quail sadly jumps` → model `quail sadly newts trust that quail jumps`<br>`dogs that ant carries hop proudly` → gold `ant carries that dogs proudly hop` → model `dogs proudly carries that ant proudly hop` |
| `foreign` | 3 | 0.003 | `flamingo that yaks hate dives` → gold `yaks hate that flamingo dives` → model `flamingo sweetly dives that yaks hate`<br>`hippos that jaguars watch briskly hop` → gold `jaguars briskly watch that hippos hop` → model `hippos briskly watch that jaguars softly hop`<br>`zebra that sharks love crawls bravely` → gold `sharks love that zebra bravely crawls` → model `zebra bravely love that sharks keenly crawls` |

Shape of the wrong outputs (share of the 864):

- the embedded subject (the noun right after `that`, source position 3) is absent from the output: **653 (0.756)**, the dominant pattern of this construction
- output = `1 R\emb V` (gold with the head noun fronted and the embedded subject deleted, nothing else changed): 100 (0.116)
- output = `1 V\verb R\emb V` (head noun, then the matrix object and adverb, then the RC without its embedded subject, then the full mirrored matrix VP): 247 (0.286). The model fills the embedded-subject slot with material from the matrix VP, which is then produced again at the end.
- output = `1 R V` (only the head noun fronted): 12 (0.014); output starts with `1 R`: 49 (0.057); output ends with `V`: 505 (0.584)
- mirrored matrix VP directly after the head noun (`1 V …`): 85 (0.098)

The most frequent orderings (top 8 of 206):

| Ordering | Count | Share | Error type | Example |
| --- | --- | --- | --- | --- |
| `gold 3 4 2 1 7 6 5 -> model 1 7 6 4 2 7 6 5` | 90 | 0.104 | `missing+repeat` | `monkey that hippo sees lifts fiercely walrus` → gold `hippo sees that monkey walrus fiercely lifts` → model `monkey walrus fiercely sees that walrus fiercely lifts` |
| `gold 3 4 2 1 6 5 -> model 1 6 4 2 6 5` | 85 | 0.098 | `missing+repeat` | `dog that lemur helps yawns happily` → gold `lemur helps that dog happily yawns` → model `dog happily helps that happily yawns` |
| `gold 3 5 4 2 1 7 6 -> model 1 7 5 4 2 7 6` | 55 | 0.064 | `missing+repeat` | `shark that walrus carries sweetly pokes tiger` → gold `walrus sweetly carries that shark tiger pokes` → model `shark tiger sweetly carries that tiger pokes` |
| `gold 3 5 4 2 1 7 6 -> model 1 5 4 2 7 6` | 48 | 0.056 | `missing` | `quail that shark admires eagerly helps cat` → gold `shark eagerly admires that quail cat helps` → model `quail eagerly admires that cat helps` |
| `gold 3 5 4 2 1 8 7 6 -> model 1 7 4 2 8 7 6` | 41 | 0.047 | `missing+repeat` | `shark that quail tickles bravely knows mindfully urchin` → gold `quail bravely tickles that shark urchin mindfully knows` → model `shark mindfully tickles that urchin mindfully knows` |
| `gold 3 5 4 2 1 8 7 6 -> model 1 5 4 2 8 7 6` | 38 | 0.044 | `missing` | `elephant that dog finds boldly meets mindfully yak` → gold `dog boldly finds that elephant yak mindfully meets` → model `elephant boldly finds that yak mindfully meets` |
| `gold 3 4 2 1 6 5 -> model 1 6 4 2 6 4` | 35 | 0.041 | `missing+repeat` | `tiger that walrus carries chases shark` → gold `walrus carries that tiger shark chases` → model `tiger shark carries that shark carries` |
| `gold 3 4 2 1 5 -> model 1 5 2 3 4` | 26 | 0.030 | `permutation` | `penguin that elephant avoids shivers` → gold `elephant avoids that penguin shivers` → model `penguin shivers that elephant avoids` |

Reading: the model treats the object-gap RC on the subject as if its verb belonged to the matrix subject (`monkey … sees that …`), which leaves no slot for the embedded subject; the matrix object (`walrus`) is then used both as the "object" of the RC verb and in the matrix VP. For the shortest frames (`N that N V V`) the output is again the right-branching template `N V that N V` (`penguin shivers that elephant avoids`).

Which source word goes missing (over the 790 outputs with a `missing` flag):

| Missing word (frame category, source position) | Count |
| --- | --- |
| N_pl at source position 3 | 337 |
| N_sg at source position 3 | 316 |
| Adv at source position 5 | 145 |
| V_intrans_pl at source position 5 | 45 |
| V_intrans_sg at source position 6 | 40 |
| V_intrans_sg at source position 5 | 37 |
| V_trans_sg at source position 5 | 34 |
| V_trans_sg at source position 4 | 31 |
| V_intrans_pl at source position 6 | 28 |
| V_trans_pl at source position 4 | 27 |

Foreign words (175 tokens in 152 outputs):

| Kind | Count | Example |
| --- | --- | --- |
| unrelated Adv | 34 | `flamingo that yaks hate dives` → gold `yaks hate that flamingo dives` → model `flamingo sweetly dives that yaks hate` |
| unrelated N_sg | 32 | `koala that elephant hugs stumbles` → gold `elephant hugs that koala stumbles` → model `koala stumbles that lemur hugs` |
| unrelated V_trans_sg | 25 | `flamingos that vulture nudges sing` → gold `vulture nudges that flamingos sing` → model `flamingos vulture lifts that vulture nudges` |
| unrelated N_pl | 24 | `lemur that hippos lift bravely sneezes` → gold `hippos bravely lift that lemur sneezes` → model `lemur sneezes that jaguars lift` |
| unrelated V_trans_pl | 21 | `walruses that giraffe chases yawn` → gold `giraffe chases that walruses yawn` → model `walruses softly see that giraffe softly see` |
| unrelated V_intrans_sg | 20 | `yak that koala loves laughs` → gold `koala loves that yak laughs` → model `yak hops that koala loves` |
| unrelated V_intrans_pl | 16 | `walrus that hippos tease stretches` → gold `hippos tease that walrus stretches` → model `walrus sneeze that hippo tease` |
| other-number form of a source word (N_pl) | 2 | `bird that vultures greet gently kisses walrus` → gold `vultures gently greet that bird walrus kisses` → model `bird birds gently greet that walrus kisses` |
| other-number form of a source word (N_sg) | 1 | `walrus that hippos tease stretches` → gold `hippos tease that walrus stretches` → model `walrus sneeze that hippo tease` |

Teacher-forced mispredictions (3,294 in this construction): same category, other lemma 2,124 (0.645); other category or special token 1,144 (0.347); number flip 26 (0.008). The last example above is one of the few genuine agreement errors: `walrus sneeze that hippo tease` puts a plural verb on a singular subject and a singular noun with a plural verb.

### Test: Object / Object Gap (4 wrong of 576, exact match 0.993)

| Error type | Count | Share | Examples (source → gold → model) |
| --- | --- | --- | --- |
| `missing` | 2 | 0.500 | `yak pursues dog that cats watch proudly` → gold `yak cats proudly watch that dog pursues` → model `yak proudly watch that dog pursues`<br>`ant helps giraffes that owl follows mindfully` → gold `ant owl mindfully follows that giraffes helps` → model `ant mindfully follows that giraffes helps` |
| `repeat` | 1 | 0.250 | `newt grabs gently newts that shark teases` → gold `newt shark teases that newts gently grabs` → model `newt shark gently teases that newts gently grabs` |
| `foreign` | 1 | 0.250 | `elephants fear quails that shark carries loudly` → gold `elephants shark loudly carries that quails fear` → model `elephants shark loudly carries that quails shyly fear` |

Orderings: `gold 1 5 7 6 4 3 2 -> model 1 7 6 4 3 2` (2, the embedded subject `cats` / `owl` is dropped, the same omission that dominates Subject / Object Gap), `gold 1 6 7 5 4 3 2 -> model 1 6 3 7 5 4 3 2` (1, the matrix adverb `gently` is repeated inside the RC) and `gold 1 5 7 6 4 3 2 -> model 1 5 7 6 4 3 ? 2` (1, an adverb `shyly` is inserted before the matrix verb). All 4 sentences have an embedded subject whose number differs from the object head's number; first teacher-forced errors are at target positions 2, 2, 3 and 7.

### Validation copy: Object / Subject Gap (96 wrong of 96, exact match 0.000)

The `ce` model never saw a right-branching sentence in the copy task. Given one, it produces the center-embedded sentence with the same words in every case:

| Error type | Count | Share | Examples (source → gold → model) |
| --- | --- | --- | --- |
| `permutation` | 56 | 0.583 | `monkey likes bird that yawns loudly` → gold `monkey likes bird that yawns loudly` → model `monkey that yawns loudly likes bird`<br>`giraffe pokes cats that hum slowly` → gold `giraffe pokes cats that hum slowly` → model `giraffe that hum slowly pokes cats`<br>`tigers watch jaguar that claps wildly` → gold `tigers watch jaguar that claps wildly` → model `tigers that claps wildly watch jaguar` |
| `missing+repeat` | 31 | 0.323 | `elephant fears urchin that shivers` → gold `elephant fears urchin that shivers` → model `elephant that shivers shivers`<br>`ant grabs shark that twirls` → gold `ant grabs shark that twirls` → model `ant that twirls twirls`<br>`flamingo lifts elephants that hum` → gold `flamingo lifts elephants that hum` → model `flamingo that hum hum` |
| `missing+foreign` | 5 | 0.052 | `elephant likes lazily birds that jump` → gold `elephant likes lazily birds that jump` → model `elephant that jump kisses lazily birds`<br>`elephants meet keenly dogs that hop` → gold `elephants meet keenly dogs that hop` → model `elephants that hop slowly fly keenly`<br>`newt finds gently quail that hops quietly` → gold `newt finds gently quail that hops quietly` → model `newt that hops quietly hug gently quail` |
| `missing` | 3 | 0.031 | `zebra fears loudly ant that dives` → gold `zebra fears loudly ant that dives` → model `zebra that dives`<br>`elephants grab softly xerus that naps` → gold `elephants grab softly xerus that naps` → model `elephants that naps`<br>`urchins greet keenly yaks that smile` → gold `urchins greet keenly yaks that smile` → model `urchins that smile` |
| `missing+repeat+foreign` | 1 | 0.010 | `yak chases fiercely jaguars that stumble` → gold `yak chases fiercely jaguars that stumble` → model `yak that stumble keenly stumble` |

- output starts with `subject that`: 96 of 96 (1.000); first teacher-forced error at target position 2 in all 96, where the model predicts `that` instead of the matrix verb (96 of 96 teacher-forced predictions at that position are `that`)
- output = subject + RC + matrix VP, i.e. the input re-attached as a center-embedded sentence with no other change: 51 of 96 (0.531); output has the gold length: 75 of 96
- the remaining cases drop the matrix verb or object after the RC (`elephant that shivers shivers`: the RC verb is repeated where the matrix verb should be) or replace them with an unrelated verb

Most frequent orderings (top 8 of 25):

| Ordering | Count | Share | Error type | Example |
| --- | --- | --- | --- | --- |
| `gold 1 2 3 4 5 6 -> model 1 4 5 6 2 3` | 18 | 0.188 | `permutation` | `monkey likes bird that yawns loudly` → gold `monkey likes bird that yawns loudly` → model `monkey that yawns loudly likes bird` |
| `gold 1 2 3 4 5 6 7 -> model 1 4 5 6 7 2 3` | 15 | 0.156 | `permutation` | `newt pokes lemur that chases carefully ant` → gold `newt pokes lemur that chases carefully ant` → model `newt that chases carefully ant pokes lemur` |
| `gold 1 2 3 4 5 6 7 8 -> model 1 5 6 7 8 2 3 4` | 14 | 0.146 | `permutation` | `iguana tickles gently rabbit that hates softly urchin` → gold `iguana tickles gently rabbit that hates softly urchin` → model `iguana that hates softly urchin tickles gently rabbit` |
| `gold 1 2 3 4 5 6 7 -> model 1 5 6 7 2 3 7` | 11 | 0.115 | `missing+repeat` | `ant lifts mindfully urchin that meets cat` → gold `ant lifts mindfully urchin that meets cat` → model `ant that meets cat lifts mindfully cat` |
| `gold 1 2 3 4 5 -> model 1 4 5 5` | 8 | 0.083 | `missing+repeat` | `elephant fears urchin that shivers` → gold `elephant fears urchin that shivers` → model `elephant that shivers shivers` |
| `gold 1 2 3 4 5 6 7 -> model 1 5 6 7 2 3 4` | 4 | 0.042 | `permutation` | `cat avoids fiercely koala that dances slowly` → gold `cat avoids fiercely koala that dances slowly` → model `cat that dances slowly avoids fiercely koala` |
| `gold 1 2 3 4 5 6 -> model 1 5 6` | 3 | 0.031 | `missing` | `zebra fears loudly ant that dives` → gold `zebra fears loudly ant that dives` → model `zebra that dives` |
| `gold 1 2 3 4 5 3 -> model 1 4 5 3 2 3` | 3 | 0.031 | `permutation` | `jaguar chases shark that grabs shark` → gold `jaguar chases shark that grabs shark` → model `jaguar that grabs shark chases shark` |

Note that the rewritten sentences violate agreement whenever the object and subject numbers differ (`giraffe that hum slowly pokes cats`, `tigers that claps wildly watch jaguar`): in the copy task the model reproduces the words it was given and does not repair the RC verb to agree with its new head noun. Teacher-forced mispredictions (436): other category or special token 340 (0.780); same category, other lemma 92 (0.211); number flip 4 (0.009).

Missing words in these copies (over the 40 outputs with a `missing` flag): the object noun (N at position 4: 23; at position 3: 10), the matrix verb (V_trans at position 2: 26) and the adverb (position 3: 7).
