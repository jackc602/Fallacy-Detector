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


## DAN + TF-IDF - all 13 classes
- 35.26% accuracy, actually worse than either single-stream baseline (TF-IDF 39.65%, DAN 40.35%) - naive concatenation is net negative here
- severe overfitting: train loss 0.24, dev loss 4.19 by epoch 100, dev acc plateaus around epoch 30 and oscillates 0.33-0.35 after
- TF-IDF vocab 4333, plus the 128d projection + 300d avg embedding feeding into the MLP is likely too many parameters for ~2000 training examples split across 13 classes
- equivocation F1 0.214 (nonzero for the first time across any all-13 model) and fallacy of logic F1 0.102 both improve vs baselines, but faulty generalization (0.400) and intentional (0.318) lose some ground - the catch-all effect gets redistributed, not eliminated
- ad hominem (0.514) and ad populum (0.516) hold up, same rhetorical-vocab story as the baselines
- main takeaway: adding TF-IDF features did not help the DAN extract better signal, and the extra capacity probably hurt generalization

## DAN + TF-IDF - logical fallacies only (6 smt solvable classes)
- 52.78% accuracy, below both TF-IDF (57.87%) and DAN (62.50%) on the same subset
- TF-IDF vocab 2189, model hits 0.52 dev by epoch 9 then plateaus while train loss keeps dropping (0.04 by epoch 100, dev loss 3.64) - same overfit signature as all-13 run
- circular reasoning still the strongest signal (F1 0.684), faulty generalization close behind (0.616)
- fallacy of logic F1 0.409 - notably better than DAN's 0.364, the one place fusion seems to help
- equivocation 0 F1 again, consistent with every other model on this subset - no amount of feature engineering rescues the 9-sample class
- false causality flipped to high-precision / low-recall (0.643 / 0.419), model got more conservative on that class than DAN did

## DAN + TF-IDF - informal fallacies only (7 non-SMT classes)
- 40.68% accuracy, ~8 points below both TF-IDF (48.87%) and DAN (48.31%) - biggest gap of the three hybrid runs
- TF-IDF vocab 3142, same overfitting curve: train loss 0.13 by epoch 100, dev loss ~4.0
- fallacy of relevance F1 0.151 (nonzero, DAN was 0) - again the hybrid spreads predictions instead of letting a catch-all swallow a class
- intentional F1 0.434, way below its TF-IDF / DAN dominance - dropped from catch-all behavior, but the redistributed mass went into weaker classes and dragged accuracy down
- ad populum (0.612) and ad hominem (0.525) survive but trail DAN's ad populum (0.698) - the TF-IDF projection seems to blur what DAN does well on rhetorical classes
- hybrid's pattern across all three runs: flattens the class distribution slightly (fewer 0 F1s, weaker catch-alls) at the cost of overall accuracy - the two streams appear to interfere rather than complement, at least with this architecture and data size


## Argument Structure Features (14 features, no word identity)
- 28.6% accuracy with handcrafted features only (premise/conclusion density, sentiment, text stats, etc.)
- equivocation F1 0.233 which is the first none zero sxore we got
- top features: pronoun usage, question marks, causal words, premise/conclusion ratio

## TF-IDF + Argument Features
- 39.12% accuracy and got a lot better class distribution
- equivocation F1 0.261 and ad populum F1 0.604
- structural features complement lexical ones, especially for pattern-defined fallacies


# BERT models show that we are missing semantic understanding
## DistilBERT
- 50.18% accuracy
- even BERT can't learn equivocation with 58 train / 9 dev samples
- got a 0 f1

## RoBERTa 
- 51.23% untuned, 52.28% tuned with class weighting 
- equivocation F1 0.500 (precision 1.000), macro F1 0.530
