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


## DAN - all 13 classes
- 40.35% accuracy, basically identical to TF-IDF on all 13 (39.65%)
- equivocation and fallacy of relevance both 0 F1, model can't learn these
- circular reasoning the clearest signal (0.690)
- training very noisy throughout, dev loss barely trending down
- 13 classes likely too many for small data set

## DAN - logical fallacies only (6 smt solvable classes)
- 62.5% accuracy
- model defaults to faulty generalization when unclear (largest class, 84 dev samples)
- equivocation is a total miss (0 F1) likely due to lack of training examples (9 dev samples)
- circular reasoning is the strongest class (F1 0.789), probably has distinctive phrasing
- false dilemma has high precision (0.867) but low recall meaning model rarely predicts it but is usually right when it does
- fallacy of logic is weak (F1 0.364), likely absorbed into faulty generalization

## DAN - informal fallacies only (7 non-SMT classes)
- 48.31% accuracy, essentially tied with TF-IDF on the same task (48.87%), embeddings not helping here either
- training curve much noisier than TF-IDF, dev acc bouncing around the whole run suggesting the model is struggling to find stable signal
- fallacy of relevance is a total miss (0 F1), all 44 samples swallowed by intentional and other classes
- ad populum strongest (F1 0.698), better than TF-IDF's 0.629 - one place DAN edges out
- fallacy of extension nearly useless (0.163), same story as TF-IDF


