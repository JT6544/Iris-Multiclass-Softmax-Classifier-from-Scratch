# Iris Multiclass Softmax Classifier — Gate 3 Improvement Record

This document records the changes made when rebuilding `Multi- Class Classification - Iris Date Set.py` into the repository distributed as `Iris-Multiclass-Softmax-Classifier-from-Scratch-Gate3.zip`. The Python file beside this document is the untouched original source used as the starting point.

## Project objective

The original script demonstrated a from-scratch multiclass linear classifier on the Iris dataset. The rebuild preserves that educational objective while turning the demonstration into a deterministic, leakage-resistant experiment with a documented data source, defensible model selection, automated verification, and reproducible outputs.

## Summary of improvements

| Area | Original implementation | Rebuilt repository | Why the change matters | Impact |
|---|---|---|---|---|
| Data acquisition | Downloaded data at run time through `ucimlrepo` | Bundles the corrected UCI `bezdekIris.data`, attribution, and SHA-256 manifest | Removes a network dependency and makes the exact input auditable | The experiment can be reproduced offline from a verified 150-row dataset |
| Data splitting | Normalised the full dataset before splitting | Creates a stratified development/test split and fits preprocessing only on training data | Prevents holdout information from influencing training | Reported test performance measures genuinely unseen data |
| Duplicate control | Ordinary sample-level split | Groups exact duplicate feature rows before splitting and cross-validation | Prevents an identical observation appearing on both sides of an evaluation boundary | The single duplicated record pair remains in one group |
| Model selection | Trained four configurations and selected the best one on the test set | Selects among the same four candidates with grouped five-fold cross-validation on the development set | The test set must not be used to choose hyperparameters | The 30-sample test set is evaluated once after selection |
| Numerical stability | Direct exponentiation and logarithms | Uses shifted softmax and stable log-sum-exp calculations | Avoids overflow, underflow, and `log(0)` failures | Loss and probability calculations remain finite for large logits |
| Regularisation | Cost and gradient used inconsistent L2 scaling | Uses one explicit non-bias L2 definition in both loss and gradient | An optimiser must differentiate the objective it reports | Training now follows the stated regularised objective |
| Training control | Fixed iterations with no validation-based stopping | Adds validation monitoring, early stopping in each fold, and deterministic final refitting | Reduces unnecessary fitting and makes the stopping rule auditable | Final training length is derived from selected-fold behaviour |
| Evaluation | Accuracy only | Adds confusion matrix, per-class precision/recall/F1, macro and weighted averages, and cross-validation statistics | Accuracy alone can conceal class-specific errors | Performance is reported at class and aggregate levels |
| Reproducibility | Notebook-style execution and transient plots | Adds a CLI, seeds, saved model parameters, JSON/CSV results, and versioned figures | A repository should regenerate and retain its evidence | A clean run produces the same split, selection process, metrics, and artefacts |
| Quality assurance | No automated checks | Adds tests and continuous integration | Mathematical and data-pipeline regressions should be caught automatically | Loader, grouping, gradients, softmax, splits, and determinism are verified |

## Mathematical corrections

### Stable softmax

For input vector $x_i$ and class $k$, the linear score is

$$
z_{ik} = x_i^\mathsf{T}w_k.
$$

The class probability is evaluated after subtracting the largest score in the row:

$$
p_{ik}
= \frac{\exp(z_{ik}-m_i)}
       {\sum_{j=1}^{K}\exp(z_{ij}-m_i)},
\qquad
m_i=\max_j z_{ij}.
$$

Subtracting $m_i$ does not change the probabilities, but it prevents very large logits from overflowing during exponentiation.

### Cross-entropy with consistent L2 regularisation

With one-hot target matrix $Y$, $N$ observations, and a weight matrix $W$, the rebuild minimises

$$
\mathcal{L}(W)
= -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}
Y_{ik}\log p_{ik}
+ \frac{\lambda}{2}\lVert W_{\text{non-bias}}\rVert_F^2.
$$

The corresponding gradient is

$$
\nabla_W\mathcal{L}
= \frac{1}{N}X^\mathsf{T}(P-Y)
+ \lambda W_{\text{non-bias}},
$$

with zero regularisation applied to the intercept row. The original script's penalty scaling and gradient were not an exact derivative pair; the rebuild makes them consistent.

### Leakage-resistant standardisation

For a training partition with feature mean $\mu_j$ and standard deviation $\sigma_j$, each feature is transformed as

$$
x'_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}.
$$

Crucially, $\mu_j$ and $\sigma_j$ are computed from the relevant training fold only, then applied unchanged to that fold's validation data. The final scaler is fitted on the complete development set and applied to the untouched test set.

### Model-selection metric

For class $k$,

$$
F_{1,k}=\frac{2\,\mathrm{precision}_k\,\mathrm{recall}_k}
{\mathrm{precision}_k+\mathrm{recall}_k},
\qquad
F_{1,\mathrm{macro}}=\frac{1}{K}\sum_{k=1}^{K}F_{1,k}.
$$

Candidates are ranked first by mean cross-validation macro F1, then by mean accuracy, then by lower validation loss. This treats the three Iris classes equally and states the tie-breaking rule explicitly.

## Rebuilt experimental workflow

1. Load and checksum the local UCI data file.
2. Validate its 150 observations, four numeric features, three labels, and 50 observations per class.
3. Assign exact feature duplicates to groups; the dataset contains 149 distinct groups.
4. Reserve a stratified 20% holdout set: 30 observations, with 10 from each class.
5. Use the remaining 120 observations for grouped five-fold cross-validation.
6. Fit preprocessing independently inside every fold.
7. Compare the four original learning-rate/regularisation configurations using validation data only.
8. Select the winning configuration by the documented ranking rule.
9. Refit on all development data for a training length derived from the selected cross-validation runs.
10. Evaluate the holdout once and save the model, metrics, tables, and figures.

## Measured impact

The selected configuration used standardisation, a learning rate of $0.1$, L2 strength $\lambda=0.001$, and a maximum of 5,000 update steps. Its mean grouped cross-validation macro F1 was approximately $0.95817$.

On the untouched 30-observation holdout, the final model achieved:

| Metric | Value |
|---|---:|
| Accuracy | 1.0000 |
| Macro precision | 1.0000 |
| Macro recall | 1.0000 |
| Macro F1 | 1.0000 |
| Weighted F1 | 1.0000 |

All 30 test observations were correctly classified. This is evidence that the final fitted model performed well on this particular deterministic holdout; it is not a claim of universal 100% generalisation. The grouped cross-validation distribution is the broader estimate and the holdout is small.

## Repository and publication improvements

The rebuilt repository adds:

- `iris_softmax_classifier.py` as an importable implementation and command-line entry point;
- the corrected UCI data file, its checksum, and `DATASET_ATTRIBUTION.md`;
- a deterministic data-preparation helper;
- unit tests for the mathematical core and data pipeline;
- a GitHub Actions workflow;
- machine-readable cross-validation and final-metric files;
- saved training, comparison, feature, and confusion-matrix figures;
- dependency and ignore files suitable for a clean repository.

## Gate 3 documentation improvements

Gate 3 retains the verified Gate 2 implementation, data, tests, continuous integration, results, and figures, and adds a publication-quality `README.md`. The guide documents the dataset provenance, duplicate grouping, leakage-resistant cross-validation, stable softmax equations, regularised objective, commands, saved artefacts, test process, and limitations. It reports both the perfect 30-observation holdout result and the broader cross-validation estimate so that the small holdout is not presented as universal 100% generalisation.

The README makes the repository usable without reverse-engineering the source and gives GitHub a correctly rendered mathematical and experimental narrative. Gate 3 is a documentation-only refinement: it does not change the model, split, data, test results, or generated metrics.

## Files represented by this folder

- `Multi- Class Classification - Iris Date Set.py` — untouched original source.
- `IMPROVEMENTS.md` — this improvement and impact record.

The original source is included for comparison and provenance. Run the rebuilt repository from its own archive; this folder is documentation support, not a replacement distribution.
