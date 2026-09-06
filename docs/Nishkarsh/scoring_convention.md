# Scoring convention: F1 is graded with pos_label=0

## The finding

Every model through v3.1's first pass was tuned and threshold-searched to
maximize F1 with **churn (`exit_status=1`) as the positive class** — the
natural reading of "predict who churns." Local out-of-fold F1 for our best
models topped out around 0.65-0.66 under that convention.

Actual leaderboard scores (Manya's and later our own submission of the exact
same model/predictions) came back around 0.91-0.92 — a gap far too large to
be explained by better modeling.

Root cause, confirmed directly with a competition organizer: **the grading
script computes F1 with `pos_label=0`** (non-churn / "stays" as positive),
not `pos_label=1`. This was accidental on the organizers' side, not a
documented choice — `docs/rules.md` just says "Metric: F1 score" with no
positive-class specified.

## Why this produces such a large score gap

F1 depends heavily on which class is "positive" when the classes are
imbalanced (79% stay / 21% churn here). The same trained model, same
predictions, same threshold:

| convention | F1 |
|---|---|
| pos_label=1 (churn) | ~0.66 |
| pos_label=0 (non-churn) | ~0.92 |

This isn't two views of the same number — it's the same predictions scored
against two different definitions of "positive," and the majority class is
inherently much easier to get high precision/recall on.

## What this changes

- **All threshold searches must target `pos_label=0`.** This is a real
  optimization target, not a formality — the optimal decision threshold is
  very different (~0.82-0.83 for pos_label=0 vs ~0.62-0.63 for pos_label=1),
  since it controls how conservatively the model predicts churn.
- **Hyperparameter search should also target `pos_label=0`**, not just the
  final threshold — the best-scoring model config isn't guaranteed to be the
  same under both conventions. v3.1 was fully rebuilt with this as the
  search objective throughout.
- **Class-weighting (`sample_weight='balanced'`) turned out not to matter
  much either way** — checked empirically: 0.9158 vs 0.9159 achievable F1
  with vs without balanced weighting, under pos_label=0. The threshold search
  already absorbs most of that effect. Kept balanced weighting for the minor
  stability it gives the ranking, not because it changes the ceiling.
- **v1.1 through v2.2's reported F1 numbers are not comparable to v3.1's.**
  They were all computed under the wrong (`pos_label=1`) convention. They're
  still useful as relative comparisons *among themselves*, just not against
  anything computed after this fix.

## What this does *not* change

This does not retroactively excuse or explain away the `customer_id`/
`last_name` leakage finding (see the NaN-handling and feature-engineering
docs, and the conversation history) — that leak is still real, still present
in the data, and still against `docs/rules.md`'s prohibition on "exploiting
metadata/leakage to reconstruct held-out labels." It just turned out **not**
to be the primary explanation for the high leaderboard scores observed —
this scoring-convention bug alone accounts for nearly all of it, including
very likely Manya's original 0.918, even though her pipeline never used
`customer_id` and only used `last_name` (which, separately, remains something
to not use going forward).
