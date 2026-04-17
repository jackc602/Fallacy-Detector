## TF-IDF - all 13 classes
- 39.65% accuracy (random baseline ~7.7% so still meaningful)
- equivocation 0 F1 again, consistent across both models
- intentional and faulty generalization act as catch-alls due to class size (99 and 84 dev samples)
- false dilemma precision 0.909 - highest in the whole report, "either/or" framing has very distinctive vocabulary
- fallacy of extension and fallacy of logic nearly useless (F1 0.167 and 0.146)
- ad hominem and ad populum do ok (F1 0.552, 0.477) - rhetorical classes have more consistent vocabulary than expected

## TF-IDF - informal fallacies only (7 non-SMT classes)
- 48.87% accuracy (random baseline ~14.3%)
- lower than the SMT subset (57.87%) confirming rhetorical fallacies are harder to classify from text alone
- intentional is the catch-all again (99 samples), swallows most uncertain predictions
- fallacy of relevance and fallacy of extension have near-perfect precision but basically 0 recall - model almost never predicts them
- ad hominem (0.567) and ad populum (0.629) are the strongest, consistent with having more distinctive vocabulary
- 100 epochs barely improved over 50 (0.4887 vs 0.4802), same overfitting pattern as SMT run

## TF-IDF - logical fallacies only (6 smt solvable classes)
- 57.87% accuracy, only about 5% behind DAN on the same task
- most of the signal in these 6 classes comes from individual word identity, not semantic relationships - embeddings buying very little
- model peaked around epoch 69, extra training just overfits (train loss 0.31, dev loss 1.16 at epoch 200, same best checkpoint, 100 epochs and 200 epochs showed no difference, 100 better than 50)
- same faulty generalization dominance and equivocation failure as DAN
- high precision / low recall pattern across most classes - model is conservative, rarely commits but usually right when it does
