# Grammar V1
- All rules are unary or binary (no ternary rules).
- Parse trees and translations are always built from the Head Initial (HI) CFG, which is unambiguous
  (mechanically verified: all 908 derivations up to length 13 yield distinct strings).
- The Head Final (HF) CFG has one known string ambiguity: subject-gap and object-gap relative
  clauses linearize identically (e.g. "dog chases that cat swims") because the embedded subject
  does not swap with its verb. This never affects generation, since generation starts from HI parses.
- AdvP is its own phrase. It adjoins to the verb, forming VP_trans_adv / VP_intrans_adv
  (following the previous grammar's VP_dp_adv pattern): HI is V AdvP, HF is AdvP V.
  Adjoining AdvP to the full VP instead would make the HI grammar ambiguous
  (adverb attachment: "sees dog that swims quickly").
- Adverbs are depth 1 (no recursion): at most one AdvP per verb.
- Relative clauses recurse to arbitrary depth (capped at dataset generation time).
  Each noun takes at most one relative clause (no stacking), which removes
  high- vs. low-attachment ambiguity.
- S_obj_gap / VP_obj_gap binarize the object-gap relative clause; S_obj_gap does not
  reverse between HI and HF (the subject stays preverbal), which is what makes
  object gaps unambiguous (OOD) under the linear reversal rule.
- All terminal categories have exactly 26 words with disjoint vocabularies
  (disjointness is required for HI unambiguity).

## Production Rules

| Head Initial Rule               | Head Final Rule                 |
| ------------------------------- | ------------------------------- |
| S -> NP VP                      | S -> NP VP                      |
| NP -> N                         | NP -> N                         |
| NP -> N CP_subj_gap             | NP -> CP_subj_gap N             |
| NP -> N CP_obj_gap              | NP -> CP_obj_gap N              |
| VP -> VP_trans                  | VP -> VP_trans                  |
| VP -> VP_intrans                | VP -> VP_intrans                |
| VP_trans -> V_trans NP          | VP_trans -> NP V_trans          |
| VP_trans -> VP_trans_adv NP     | VP_trans -> NP VP_trans_adv     |
| VP_trans_adv -> V_trans AdvP    | VP_trans_adv -> AdvP V_trans    |
| VP_intrans -> V_intrans         | VP_intrans -> V_intrans         |
| VP_intrans -> VP_intrans_adv    | VP_intrans -> VP_intrans_adv    |
| VP_intrans_adv -> V_intrans AdvP| VP_intrans_adv -> AdvP V_intrans|
| AdvP -> Adv                     | AdvP -> Adv                     |
| CP_subj_gap -> C VP             | CP_subj_gap -> VP C             |
| CP_obj_gap -> C S_obj_gap       | CP_obj_gap -> S_obj_gap C       |
| S_obj_gap -> NP VP_obj_gap      | S_obj_gap -> NP VP_obj_gap      |
| VP_obj_gap -> V_trans           | VP_obj_gap -> V_trans           |
| VP_obj_gap -> VP_trans_adv      | VP_obj_gap -> VP_trans_adv      |

## Terminal Categories (26 words each)

| Category  | Terminals |
| --------- | --------- |
| N         | ant, bird, cat, dog, elephant, flamingo, giraffe, hippo, iguana, jaguar, koala, lemur, monkey, newt, owl, penguin, quail, rabbit, shark, tiger, urchin, vulture, walrus, xerus, yak, zebra |
| V_intrans | swims, dances, sings, laughs, smiles, sleeps, jumps, claps, runs, walks, hops, crawls, flies, dives, spins, yawns, snores, sneezes, stretches, shivers, wanders, naps, whistles, hums, stumbles, twirls |
| V_trans   | chases, pursues, sees, likes, meets, follows, watches, greets, admires, avoids, loves, hates, knows, hugs, kisses, tickles, teases, trusts, helps, fears, finds, grabs, lifts, carries, pokes, nudges |
| Adv       | quickly, slowly, hastily, mindfully, quietly, loudly, eagerly, calmly, gently, fiercely, gracefully, carefully, clumsily, happily, sadly, proudly, bravely, boldly, shyly, sweetly, wildly, softly, briskly, lazily, keenly, warily |
| C         | that |