# Pipeline of the project

Main objective: obtain a dataset similar to the one in https://huggingface.co/datasets/HiTZ/meta4xnli but in catalan. To obtain this, we have to achieve 2 subtasks:
- Obtain some instances (data points) already labelled
- Fine tune a model using the previous data points to label the rest of the dataset.


## Step 0: Load and tokenize catalan xnli dataset

## Step 1: Align also the meta4xnli dataset (the splits of xnli seem aligned - check because some sentences are missing as ds is smaller)

splits | ES/EN | CAT | comparison

esxnli_hyp | - | not found | -
esxnli_prem | - | not found | - 
xnli_dev_hyp | 2490 | 2490 | matches perfectly
xnli_dev_prem | 830 | 847 | 13 instances drop in the catalan dataset  

xnli_test_hyp |  5010 | 5008 | 2 instances droped in the english and spanish datasets
xnli_test_prem |  1670  | 1759 | 88 instances drop in the catalan dataset


{137-138, 191-192, 204-205, 230-231, 335-337, 383-384, 414-415, 501-502, 504-505, 521-522, 532-533, 536-537, 541-542, 548-549, 642-643, 649-650}

{75-76, 79-80, 81-82, 84-85, 88-89, 95-96, 98-99, 104-105, 106-107, 108-109, 112-113, 117-118, 125-126, 138-139, 140-141, 142-143, 152-153, 154-155, 157-158, 160-161, 162-163, 164-165, 171-172, 174-175, 177-178, 181-182, 184-185, 188-189, 191-192, 194-195, 202-203, 311-312, 314-315, 318-319, 323-324, 325-326, 328-329, 349-350, 372-373, 384-385, 386-387, 400-401, 419-420, 482-483, 488-489, 492-493, 494-495, 511-512, 520-521, 533-534, 546-547, 554-555, 558-559, 562-563, 616-617, 749-750, 758-759, 760-761, 764-765, 966-967, 969-970, 985-986, 988-989, 993-994, 1113-1114, 1145-1146, 1148-1149, 1152-1153, 1155-1156, 1167-1168, 1169-1170, 1181-1182, 1183-1184, 1187-1188, 1201-1202, 1204-1205, 1207-1208, 1209-1210, 1216-1217, 1219-1220, 1222-1223, 1558-1559, 1568-1569, 1577-1579, 1580-1581, 1601-1602, 1605-1606, 1637-1638}


## Step 2: Align the token and propagate the labels

align tokens (es-cat and en-cat), propagate the labels for the aligned tokens. Two possible labels one per language.

## Step 3: Align the spanish tokens and the english tokens

-Sentences have id
-2 aligned tokens (es-en) can have different labels, if so -> the token is set to uncertain in the aligned token in catalan (in both projections)
-There will be tokens that are not aligned in English-Spanish, thus they do not give us any info and that is all.
- This is only used to mark as uncertain some label that has been projected, but if there is no consensum (not alignment in ES-EN), then we skip that token

## Step 4: There will be label projections that do not match -> uncertain cases

- with this we unlabelled the cases that show missmatch to ensure robutness in the ones that did work



## Step 5: Automatic Gap-Filling

Fine-tune Multilingual Model on Projected Data, for example: "projecte-aina/roberta-base-ca-v2" and then, once fine tuned, predict the labels in the unlabeled sentences.

```
from transformers import AutoModelForTokenClassification, Trainer

# Use your projected CA data as training set
model = AutoModelForTokenClassification.from_pretrained(
    "projecte-aina/roberta-base-ca-v2",  # Or xlm-roberta-large
    num_labels=2  # 0: literal, 1: metaphor
)

# Fine-tune on projected Catalan labels
trainer = Trainer(
    model=model,
    train_dataset=projected_ca_dataset,
    # ... training args
)
trainer.train()
``` 


## Example of how the final dataset must look like:

{
  "sentence_id": "xnli_ca_dev_001",
  "text": "El món és en gran part fruit de la ciència.",
  "tokens": ["El", "món", "és", "en", "gran", "part", "fruit", "de", "la", "ciència", "."],
  "labels": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
  "pos_tags": ["DET", "NOUN", "VERB", "ADP", "ADJ", "NOUN", "NOUN", "ADP", "DET", "NOUN", "PUNCT"],
}

























----


## Step 1: Project EN -> CAT

- Take the English split, detokenize it, translate it to catalan, tokenize the catalan translation (points or commas goes with the previous token, as it is how is done in the meta4xnli dataset), align tokens (en-cat), propagate the labels for the aligned tokens. 

## Step 2: Project ES -> CAT

- Take the Spanish split, detokenize it, traslate it to catalan, tokenize the catalan translation (points or commas goes with the previous token, as it is how is done in the meta4xnli dataset), align tokens (es-cat), propagate the labels for the aligned tokens.


If we would like to do the translation to catalan instead of using aina dataset:


## Step 4: Choose the best catalan translation based on the projections (the objective will be to maintain the translation that keeps more metaphors - more 1 in the labels of the tokens)

-We have to define a metric for that, here is a draft:

score =
  3.0 * coverage_met_cons +
  2.0 * coverage_met +
  1.0 * coverage_total +
  1.5 * mean_conf -
  2.5 * conflict_rate -
  3.0 * unaligned_met_rate

### `coverage_total`

**What it measures:** the fraction of source tokens (EN/ES) that “find a counterpart” in the Catalan translation.

**How `score_projection` computes it:**

* A source token `i` is considered *covered* if `src2tgt[i]` has at least one alignment link.
* In other words, the aligner found at least one CA token `j` aligned to that source token.

**Interpretation:**

* High (`≈1.0`) → the CA translation closely follows the structure of the source sentence (and the aligner agrees).
* Low → many source tokens are not clearly reflected in CA (paraphrasing, omissions, odd tokenization, etc.).

---

### `coverage_met`

**What it measures:** the same as `coverage_total`, but **restricted to source tokens with label = 1** (metaphorical / non-literal according to Meta4XNLI).

**How `score_projection` computes it:**

* It collects indices `src_met = [i for i, l in enumerate(src_labels) if l == 1]`.
* It then counts how many of those tokens have at least one alignment in `src2tgt[i]`.

**Interpretation (more important than `coverage_total`):**

* High → the metaphorical parts of the source sentence survive in an alignable way in CA.
* Low → the CA translation likely “lost” or paraphrased exactly the content you wanted to label.

---

### `coverage_met_cons`

**What it measures:** metaphor coverage **restricted to “consensus” source tokens**, if you are doing triangulation by source concept.

**How `score_projection` computes it:**

* It is only defined if you pass `consensus_src_indices` (a `set` of source indices considered reliable).
* It takes the metaphorical tokens that are also in `consensus_src_indices` and computes the fraction that align.
* If `consensus_src_indices is None` → `coverage_met_cons = 0.0`.

**Interpretation:**

* High → you are not only preserving metaphors, but preserving **cross-lingually stable metaphors**.
* Low → what you preserve may be exactly what shifts between EN and ES (possible metaphor shifts).

---

### `mean_conf`

**What it measures:** the average “quality” of alignment links, using the `conf` values.

**How `score_projection` computes it:**

* For each aligned source token, it takes **only the highest-confidence link** (`best`) for that token.
* It then averages these `best_confs`.

**Interpretation:**

* High → the aligner is fairly confident about the source↔CA mappings.
* Low → there is substantial ambiguity or many weak alignments (risky for projection).

**Practical note:** if your aligner does not provide `conf`, you can use proxies (e.g. 1.0 if bidirectional, 0.7 if unidirectional), but then `mean_conf` is a more heuristic signal.

---

### `conflict_rate`

**What it measures:** the fraction of CA tokens that receive **contradictory labels** from the source.

**How `score_projection` computes it:**

* For each CA token `j`, it gathers all labels of the source tokens aligned to `j` (`tgt2src_labels[j]`).
* If the same CA token receives both `0` and `1`, it is counted as a conflict.
* `conflict_rate = conflict_targets / len(ca_tokens)`.

**Interpretation:**

* High → the projection will be noisy, because a CA token inherits incompatible signals.
* This often happens with:

  * many-to-one alignments (two source tokens collapse into one CA token)
  * multiword expressions
  * alignment errors

---

### `unaligned_met_rate`

**What it measures:** the proportion of metaphorical source tokens (label = 1) that **do not align to anything** in CA.

**How `score_projection` computes it:**

* `unaligned_met = [i for i in src_met if len(src2tgt[i]) == 0]`
* `unaligned_met_rate = len(unaligned_met) / max(1, n_met)`

**Interpretation:**

* High → metaphors from the source are being lost (bad translation for your goal, or bad alignment).
* This is one of the strongest signals to **discard** a CA translation as a basis for projection.


**Recommended usage:**

* Compute `score_projection` for `CA_from_EN` and `CA_from_ES`
* Keep the version with the higher `details["score"]`
* Optionally store `details` for auditing and debugging

```
def score_projection(src_tokens, src_labels, ca_tokens, align_links,
                     consensus_src_indices=None):
    """
    align_links: list of dicts { "i": int, "j": int, "conf": float }
    consensus_src_indices: set[int] or None
    """

    n_src = len(src_tokens)
    src_met = [i for i,l in enumerate(src_labels) if l == 1]
    n_met = len(src_met)

    # Map source token -> list of aligned target tokens with confidence
    src2tgt = {i: [] for i in range(n_src)}
    tgt2src_labels = {j: [] for j in range(len(ca_tokens))}

    for link in align_links:
        i, j, conf = link["i"], link["j"], link.get("conf", 1.0)
        src2tgt[i].append((j, conf))
        tgt2src_labels[j].append(src_labels[i])

    # Coverage
    aligned_src = {i for i in range(n_src) if len(src2tgt[i]) > 0}
    coverage_total = len(aligned_src) / max(1, n_src)

    aligned_met = {i for i in src_met if len(src2tgt[i]) > 0}
    coverage_met = len(aligned_met) / max(1, n_met)

    unaligned_met = [i for i in src_met if len(src2tgt[i]) == 0]
    unaligned_met_rate = len(unaligned_met) / max(1, n_met)

    # Consensus metaphor coverage (optional)
    if consensus_src_indices is None:
        coverage_met_cons = 0.0
    else:
        met_cons = [i for i in src_met if i in consensus_src_indices]
        n_met_cons = len(met_cons)
        aligned_met_cons = [i for i in met_cons if len(src2tgt[i]) > 0]
        coverage_met_cons = len(aligned_met_cons) / max(1, n_met_cons)

    # Mean confidence: take best link per source token
    best_confs = []
    for i in aligned_src:
        best_confs.append(max(conf for _,conf in src2tgt[i]))
    mean_conf = sum(best_confs) / max(1, len(best_confs))

    # Conflict rate: target tokens that receive both 0 and 1 from source
    conflict_targets = 0
    for j, lbls in tgt2src_labels.items():
        if 0 in lbls and 1 in lbls:
            conflict_targets += 1
    conflict_rate = conflict_targets / max(1, len(ca_tokens))

    score = (
        3.0 * coverage_met_cons +
        2.0 * coverage_met +
        1.0 * coverage_total +
        1.5 * mean_conf -
        2.5 * conflict_rate -
        3.0 * unaligned_met_rate
    )

    details = {
        "coverage_total": coverage_total,
        "coverage_met": coverage_met,
        "coverage_met_cons": coverage_met_cons,
        "mean_conf": mean_conf,
        "conflict_rate": conflict_rate,
        "unaligned_met_rate": unaligned_met_rate,
        "score": score,
    }
    return score, details
```
