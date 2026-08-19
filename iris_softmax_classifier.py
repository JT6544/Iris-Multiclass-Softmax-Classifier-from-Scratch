"""Multiclass softmax classification of the corrected UCI Iris dataset.

The model, gradient, grouped splitting, cross-validation, and metrics are
implemented with NumPy for educational transparency.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "data" / "raw" / "bezdekIris.data"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs"

FEATURE_NAMES = (
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
)
CLASS_NAMES = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    group_ids: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    class_names: tuple[str, ...] = CLASS_NAMES


@dataclass(frozen=True)
class CandidateConfig:
    normalize: bool
    learning_rate: float
    l2_strength: float

    @property
    def name(self) -> str:
        scale = "standardized" if self.normalize else "unscaled"
        return f"{scale}_lr-{self.learning_rate:g}_l2-{self.l2_strength:g}"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    test_fraction: float = 0.20
    folds: int = 5
    max_iterations: int = 5_000
    patience: int = 300
    min_delta: float = 1e-7
    l2_strength: float = 0.001

    def validate(self) -> None:
        if not 0 < self.test_fraction < 0.5:
            raise ValueError("Test fraction must be between 0 and 0.5.")
        if self.folds < 2:
            raise ValueError("At least two cross-validation folds are required.")
        if self.max_iterations <= 0 or self.patience <= 0:
            raise ValueError("Maximum iterations and patience must be positive.")
        if self.min_delta < 0 or self.l2_strength < 0:
            raise ValueError("Minimum improvement and L2 strength cannot be negative.")


@dataclass
class StandardScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "StandardScaler":
        features = _as_feature_matrix(features)
        self.mean_ = features.mean(axis=0)
        scale = features.std(axis=0)
        self.scale_ = np.where(scale == 0, 1.0, scale)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("The scaler must be fitted before transform is called.")
        features = _as_feature_matrix(features)
        return (features - self.mean_) / self.scale_

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)


class MulticlassSoftmaxRegression:
    """Full-batch softmax regression with stable loss and analytical gradient."""

    def __init__(
        self,
        learning_rate: float,
        max_iterations: int,
        l2_strength: float,
        patience: int = 300,
        min_delta: float = 1e-7,
        num_classes: int = 3,
    ) -> None:
        if learning_rate <= 0 or max_iterations <= 0:
            raise ValueError("Learning rate and maximum iterations must be positive.")
        if l2_strength < 0 or patience <= 0 or num_classes < 2:
            raise ValueError("Invalid regularisation, patience, or class count.")
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.l2_strength = l2_strength
        self.patience = patience
        self.min_delta = min_delta
        self.num_classes = num_classes
        self.weights_: np.ndarray | None = None
        self.best_iteration_: int | None = None
        self.history_: dict[str, list[float]] = {
            "iteration": [],
            "training_loss": [],
            "validation_loss": [],
            "training_accuracy": [],
            "validation_accuracy": [],
        }

    def fit(
        self,
        training_features: np.ndarray,
        training_labels: np.ndarray,
        validation_features: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
    ) -> "MulticlassSoftmaxRegression":
        training_features = _as_feature_matrix(training_features)
        training_labels = _as_class_labels(training_labels, self.num_classes)
        if training_features.shape[0] != training_labels.size:
            raise ValueError("Training features and labels have different lengths.")

        has_validation = validation_features is not None or validation_labels is not None
        if has_validation:
            if validation_features is None or validation_labels is None:
                raise ValueError("Validation features and labels must be provided together.")
            validation_features = _as_feature_matrix(validation_features)
            validation_labels = _as_class_labels(validation_labels, self.num_classes)
            if validation_features.shape[0] != validation_labels.size:
                raise ValueError("Validation features and labels have different lengths.")
            if validation_features.shape[1] != training_features.shape[1]:
                raise ValueError("Training and validation feature counts differ.")

        training_design = _add_bias_column(training_features)
        validation_design = (
            _add_bias_column(validation_features) if validation_features is not None else None
        )
        weights = np.zeros((training_design.shape[1], self.num_classes), dtype=float)
        best_weights = weights.copy()
        best_validation_loss = np.inf
        iterations_without_improvement = 0

        for iteration in range(1, self.max_iterations + 1):
            probabilities = stable_softmax(training_design @ weights)
            indicators = np.eye(self.num_classes)[training_labels]
            gradient = training_design.T @ (probabilities - indicators)
            gradient /= training_labels.size
            gradient[1:, :] += self.l2_strength * weights[1:, :]
            weights -= self.learning_rate * gradient

            training_loss = softmax_loss(
                training_design, training_labels, weights, self.l2_strength
            )
            training_predictions = np.argmax(training_design @ weights, axis=1)
            training_accuracy = float(np.mean(training_predictions == training_labels))

            if validation_design is not None and validation_labels is not None:
                validation_loss = softmax_loss(
                    validation_design, validation_labels, weights, self.l2_strength
                )
                validation_predictions = np.argmax(validation_design @ weights, axis=1)
                validation_accuracy = float(
                    np.mean(validation_predictions == validation_labels)
                )
            else:
                validation_loss = np.nan
                validation_accuracy = np.nan

            self.history_["iteration"].append(float(iteration))
            self.history_["training_loss"].append(float(training_loss))
            self.history_["validation_loss"].append(float(validation_loss))
            self.history_["training_accuracy"].append(training_accuracy)
            self.history_["validation_accuracy"].append(validation_accuracy)

            if validation_design is None:
                best_weights = weights.copy()
                self.best_iteration_ = iteration
                continue

            if validation_loss < best_validation_loss - self.min_delta:
                best_validation_loss = validation_loss
                best_weights = weights.copy()
                self.best_iteration_ = iteration
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
                if iterations_without_improvement >= self.patience:
                    break

        self.weights_ = best_weights
        if self.best_iteration_ is None:
            raise RuntimeError("Training did not produce a valid model.")
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("The model must be fitted before prediction.")
        design = _add_bias_column(_as_feature_matrix(features))
        return stable_softmax(design @ self.weights_)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(features), axis=1)


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> Dataset:
    """Load and validate the corrected Bezdek variant of the UCI Iris data."""

    path = Path(path)
    features: list[list[float]] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != 5:
                raise ValueError(f"Expected 5 columns at line {line_number}; found {len(row)}.")
            measurements = [float(value) for value in row[:4]]
            species = row[4]
            if species not in CLASS_TO_INDEX:
                raise ValueError(f"Unexpected class {species!r} at line {line_number}.")
            features.append(measurements)
            labels.append(CLASS_TO_INDEX[species])

    feature_array = np.asarray(features, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    group_ids = identical_record_groups(feature_array, label_array)
    dataset = Dataset(feature_array, label_array, group_ids)
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: Dataset) -> None:
    features = _as_feature_matrix(dataset.features)
    labels = _as_class_labels(dataset.labels, len(CLASS_NAMES))
    groups = np.asarray(dataset.group_ids).reshape(-1)
    if features.shape != (150, 4):
        raise ValueError(f"Expected a 150 x 4 feature matrix; found {features.shape}.")
    if labels.size != features.shape[0] or groups.size != features.shape[0]:
        raise ValueError("Features, labels, and group IDs must have equal row counts.")
    if not np.isfinite(features).all():
        raise ValueError("The feature matrix contains missing or non-finite values.")
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    if not np.array_equal(class_counts, np.asarray([50, 50, 50])):
        raise ValueError(f"Expected 50 observations per species; found {class_counts.tolist()}.")
    if np.any(features <= 0):
        raise ValueError("All flower measurements must be positive.")


def identical_record_groups(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Assign one group ID to every identical feature-and-class record."""

    features = _as_feature_matrix(features)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if features.shape[0] != labels.size:
        raise ValueError("Features and labels must have equal lengths.")
    mapping: dict[tuple[float | int, ...], int] = {}
    group_ids = np.empty(labels.size, dtype=int)
    for index, (row, label) in enumerate(zip(features, labels)):
        key = (*row.tolist(), int(label))
        if key not in mapping:
            mapping[key] = len(mapping)
        group_ids[index] = mapping[key]
    return group_ids


def stratified_group_partition(
    group_ids: Sequence[int],
    labels: Sequence[int],
    fractions: Sequence[float],
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Greedily balance class counts and sizes while keeping groups intact."""

    groups = np.asarray(group_ids).reshape(-1)
    labels = _as_class_labels(np.asarray(labels), len(CLASS_NAMES))
    fractions_array = np.asarray(fractions, dtype=float)
    if groups.size != labels.size:
        raise ValueError("Group IDs and labels must have equal lengths.")
    if fractions_array.ndim != 1 or fractions_array.size < 2:
        raise ValueError("At least two partition fractions are required.")
    if np.any(fractions_array <= 0) or not np.isclose(fractions_array.sum(), 1.0):
        raise ValueError("Partition fractions must be positive and sum to 1.")

    rng = np.random.default_rng(seed)
    group_records = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        counts = np.bincount(labels[indices], minlength=len(CLASS_NAMES)).astype(float)
        group_records.append((group, indices, counts, rng.random()))
    group_records.sort(key=lambda item: (-item[1].size, item[3]))

    total_class_counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(float)
    target_class_counts = fractions_array[:, None] * total_class_counts[None, :]
    target_sizes = fractions_array * labels.size
    assigned_class_counts = np.zeros_like(target_class_counts)
    assigned_sizes = np.zeros(fractions_array.size, dtype=float)
    partition_indices: list[list[int]] = [[] for _ in fractions_array]

    for _group, indices, class_counts, _tie_breaker in group_records:
        candidate_costs = []
        for partition in range(fractions_array.size):
            next_counts = assigned_class_counts.copy()
            next_sizes = assigned_sizes.copy()
            next_counts[partition] += class_counts
            next_sizes[partition] += indices.size
            class_error = np.sum(
                ((next_counts - target_class_counts) / np.maximum(target_class_counts, 1.0))
                ** 2
            )
            size_error = np.sum(
                ((next_sizes - target_sizes) / np.maximum(target_sizes, 1.0)) ** 2
            )
            overflow = np.sum(
                np.maximum(next_counts - target_class_counts, 0.0)
                / np.maximum(target_class_counts, 1.0)
            )
            candidate_costs.append(class_error + 0.25 * size_error + 0.5 * overflow)
        chosen = int(np.argmin(candidate_costs))
        partition_indices[chosen].extend(indices.tolist())
        assigned_class_counts[chosen] += class_counts
        assigned_sizes[chosen] += indices.size

    output = tuple(np.sort(np.asarray(indices, dtype=int)) for indices in partition_indices)
    if any(indices.size == 0 for indices in output):
        raise RuntimeError("Partitioning produced an empty subset.")
    return output


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("Softmax scores must form a two-dimensional matrix.")
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def softmax_loss(
    design_matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    l2_strength: float,
) -> float:
    labels = _as_class_labels(labels, weights.shape[1])
    scores = design_matrix @ weights
    maxima = np.max(scores, axis=1)
    log_sum_exp = maxima + np.log(np.exp(scores - maxima[:, None]).sum(axis=1))
    data_loss = np.mean(log_sum_exp - scores[np.arange(labels.size), labels])
    penalty = 0.5 * l2_strength * float(np.sum(weights[1:, :] ** 2))
    return float(data_loss + penalty)


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, object]:
    labels = _as_class_labels(labels, len(class_names))
    predictions = _as_class_labels(predictions, len(class_names))
    if labels.size != predictions.size:
        raise ValueError("Labels and predictions must have equal lengths.")

    matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)
    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    def safe_divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    for index, class_name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted = int(matrix[:, index].sum())
        precision = safe_divide(true_positive, predicted)
        recall = safe_divide(true_positive, support)
        f1_score = safe_divide(2 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        per_class[class_name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
        }

    accuracy = float(np.mean(labels == predictions))
    macro_recall = float(np.mean(recalls))
    return {
        "observations": int(labels.size),
        "confusion_matrix": matrix.tolist(),
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": macro_recall,
        "macro_f1": float(np.mean(f1_scores)),
        "balanced_accuracy": macro_recall,
        "per_class": per_class,
    }


def build_candidates(l2_strength: float) -> tuple[CandidateConfig, ...]:
    return tuple(
        CandidateConfig(normalize, learning_rate, l2_strength)
        for normalize in (False, True)
        for learning_rate in (0.001, 0.1)
    )


def cross_validate_candidates(
    dataset: Dataset,
    development_indices: np.ndarray,
    config: ExperimentConfig,
    candidates: Sequence[CandidateConfig],
) -> tuple[CandidateConfig, list[dict[str, object]], dict[str, dict[str, float]]]:
    """Select a configuration without examining the held-out test subset."""

    local_partitions = stratified_group_partition(
        dataset.group_ids[development_indices],
        dataset.labels[development_indices],
        fractions=np.repeat(1.0 / config.folds, config.folds),
        seed=config.seed + 1,
    )
    validation_folds = tuple(development_indices[local] for local in local_partitions)
    records: list[dict[str, object]] = []

    for candidate in candidates:
        for fold_number, validation_indices in enumerate(validation_folds, start=1):
            training_indices = np.setdiff1d(
                development_indices, validation_indices, assume_unique=True
            )
            training_features = dataset.features[training_indices]
            validation_features = dataset.features[validation_indices]

            if candidate.normalize:
                scaler = StandardScaler()
                training_features = scaler.fit_transform(training_features)
                validation_features = scaler.transform(validation_features)

            model = MulticlassSoftmaxRegression(
                learning_rate=candidate.learning_rate,
                max_iterations=config.max_iterations,
                l2_strength=candidate.l2_strength,
                patience=config.patience,
                min_delta=config.min_delta,
            ).fit(
                training_features,
                dataset.labels[training_indices],
                validation_features,
                dataset.labels[validation_indices],
            )
            predictions = model.predict(validation_features)
            metrics = classification_metrics(dataset.labels[validation_indices], predictions)
            best_history_index = int(model.best_iteration_) - 1
            records.append(
                {
                    "candidate": candidate.name,
                    "normalize": candidate.normalize,
                    "learning_rate": candidate.learning_rate,
                    "l2_strength": candidate.l2_strength,
                    "fold": fold_number,
                    "training_observations": int(training_indices.size),
                    "validation_observations": int(validation_indices.size),
                    "best_iteration": int(model.best_iteration_),
                    "validation_loss": float(
                        model.history_["validation_loss"][best_history_index]
                    ),
                    "validation_accuracy": metrics["accuracy"],
                    "validation_macro_f1": metrics["macro_f1"],
                    "validation_balanced_accuracy": metrics["balanced_accuracy"],
                }
            )

    summaries: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        candidate_records = [row for row in records if row["candidate"] == candidate.name]
        macro_f1 = np.asarray([row["validation_macro_f1"] for row in candidate_records])
        accuracy = np.asarray([row["validation_accuracy"] for row in candidate_records])
        loss = np.asarray([row["validation_loss"] for row in candidate_records])
        iterations = np.asarray([row["best_iteration"] for row in candidate_records])
        summaries[candidate.name] = {
            "mean_macro_f1": float(macro_f1.mean()),
            "std_macro_f1": float(macro_f1.std()),
            "mean_accuracy": float(accuracy.mean()),
            "mean_validation_loss": float(loss.mean()),
            "median_best_iteration": float(np.median(iterations)),
        }

    selected = max(
        candidates,
        key=lambda candidate: (
            summaries[candidate.name]["mean_macro_f1"],
            summaries[candidate.name]["mean_accuracy"],
            -summaries[candidate.name]["mean_validation_loss"],
        ),
    )
    return selected, records, summaries


def run_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    config: ExperimentConfig,
    create_plots: bool = True,
) -> dict[str, object]:
    config.validate()
    dataset = load_dataset(data_path)
    development_indices, test_indices = stratified_group_partition(
        dataset.group_ids,
        dataset.labels,
        fractions=(1.0 - config.test_fraction, config.test_fraction),
        seed=config.seed,
    )
    candidates = build_candidates(config.l2_strength)
    selected, cv_records, cv_summaries = cross_validate_candidates(
        dataset, development_indices, config, candidates
    )
    final_iterations = max(
        1, int(round(cv_summaries[selected.name]["median_best_iteration"]))
    )

    development_features = dataset.features[development_indices]
    test_features = dataset.features[test_indices]
    if selected.normalize:
        scaler = StandardScaler()
        development_features = scaler.fit_transform(development_features)
        test_features = scaler.transform(test_features)
    else:
        scaler = StandardScaler(
            mean_=np.zeros(dataset.features.shape[1]),
            scale_=np.ones(dataset.features.shape[1]),
        )

    final_model = MulticlassSoftmaxRegression(
        learning_rate=selected.learning_rate,
        max_iterations=final_iterations,
        l2_strength=selected.l2_strength,
        patience=config.patience,
        min_delta=config.min_delta,
    ).fit(development_features, dataset.labels[development_indices])

    development_predictions = final_model.predict(development_features)
    test_predictions = final_model.predict(test_features)
    development_metrics = classification_metrics(
        dataset.labels[development_indices], development_predictions
    )
    test_metrics = classification_metrics(dataset.labels[test_indices], test_predictions)

    results: dict[str, object] = {
        "dataset": {
            "source": "UCI Iris, corrected Bezdek data file",
            "observations": int(dataset.labels.size),
            "features": list(FEATURE_NAMES),
            "units": "centimetres",
            "class_counts": {
                class_name: int(np.sum(dataset.labels == index))
                for index, class_name in enumerate(CLASS_NAMES)
            },
            "unique_record_groups": int(np.unique(dataset.group_ids).size),
        },
        "config": asdict(config),
        "split_sizes": {
            "development": int(development_indices.size),
            "test": int(test_indices.size),
        },
        "selected_candidate": asdict(selected) | {"name": selected.name},
        "final_iterations": final_iterations,
        "cross_validation_summary": cv_summaries,
        "metrics": {
            "development": development_metrics,
            "test": test_metrics,
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metrics.json", results)
    _write_records_csv(output_dir / "cross_validation_results.csv", cv_records)
    _write_history(output_dir / "training_history.csv", final_model.history_)
    np.savez(
        output_dir / "model.npz",
        weights=final_model.weights_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        feature_names=np.asarray(FEATURE_NAMES),
        class_names=np.asarray(CLASS_NAMES),
        normalize=selected.normalize,
        learning_rate=selected.learning_rate,
        l2_strength=selected.l2_strength,
    )

    if create_plots:
        _plot_cross_validation(cv_summaries, selected.name, output_dir)
        _plot_training_history(final_model.history_, output_dir)
        _plot_confusion_matrix(test_metrics["confusion_matrix"], output_dir)
        _plot_feature_distributions(dataset, output_dir)

    return results


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _write_records_csv(path: Path, records: Sequence[dict[str, object]]) -> None:
    if not records:
        raise ValueError("Cannot write an empty cross-validation result table.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_history(path: Path, history: dict[str, list[float]]) -> None:
    columns = tuple(history)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*(history[column] for column in columns)))


def _plot_cross_validation(
    summaries: dict[str, dict[str, float]], selected_name: str, output_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    names = list(summaries)
    means = [summaries[name]["mean_macro_f1"] for name in names]
    errors = [summaries[name]["std_macro_f1"] for name in names]
    colours = ["#B91C1C" if name == selected_name else "#2563EB" for name in names]
    labels = [name.replace("_", "\n") for name in names]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(np.arange(len(names)), means, yerr=errors, color=colours, capsize=5)
    axis.set_xticks(np.arange(len(names)), labels=labels)
    axis.set_ylim(max(0.0, min(means) - 0.08), 1.01)
    axis.set(ylabel="Mean validation macro F1", title="Five-fold configuration comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "cross_validation_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_training_history(history: dict[str, list[float]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(history["iteration"], history["training_loss"], color="#2563EB")
    axes[0].set(xlabel="Iteration", ylabel="Loss", title="Final development-set loss")
    axes[1].plot(history["iteration"], history["training_accuracy"], color="#059669")
    axes[1].set(
        xlabel="Iteration", ylabel="Accuracy", title="Final development-set accuracy"
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Selected softmax model training history")
    figure.tight_layout()
    figure.savefig(output_dir / "training_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_confusion_matrix(matrix: object, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    values = np.asarray(matrix, dtype=int)
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(values, cmap="Blues")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, int(values[row, column]), ha="center", va="center")
    short_names = [name.replace("Iris-", "") for name in CLASS_NAMES]
    axis.set_xticks(np.arange(3), labels=short_names, rotation=20, ha="right")
    axis.set_yticks(np.arange(3), labels=short_names)
    axis.set(xlabel="Predicted species", ylabel="Actual species", title="Held-out test confusion matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_dir / "test_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_feature_distributions(
    dataset: Dataset,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    colours = ("#2563EB", "#D97706", "#059669")
    for feature_index, axis in enumerate(axes.ravel()):
        for class_index, class_name in enumerate(CLASS_NAMES):
            values = dataset.features[dataset.labels == class_index, feature_index]
            axis.hist(
                values,
                bins=10,
                alpha=0.35,
                color=colours[class_index],
                label=class_name.replace("Iris-", ""),
            )
        axis.set(
            xlabel=FEATURE_NAMES[feature_index].replace("_", " "),
            ylabel="Observations",
        )
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    figure.suptitle("Corrected Iris feature distributions (all 150 observations)")
    figure.tight_layout()
    figure.savefig(output_dir / "feature_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _as_feature_matrix(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("Features must form a two-dimensional matrix.")
    return features


def _as_class_labels(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError(f"Labels must be integers from 0 to {num_classes - 1}.")
    return labels


def _add_bias_column(features: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(features.shape[0]), features))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a NumPy softmax classifier on the corrected UCI Iris data."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=5_000)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--l2-strength", type=float, default=0.001)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(arguments: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(arguments)
    config = ExperimentConfig(
        seed=args.seed,
        test_fraction=args.test_fraction,
        folds=args.folds,
        max_iterations=args.max_iterations,
        patience=args.patience,
        l2_strength=args.l2_strength,
    )
    results = run_experiment(
        args.data, args.output_dir, config, create_plots=not args.no_plots
    )
    selected = results["selected_candidate"]
    test_metrics = results["metrics"]["test"]  # type: ignore[index]
    print("Configuration selection used development-set cross-validation only.")
    print(f"Selected candidate: {selected['name']}")  # type: ignore[index]
    print(f"Final iterations: {results['final_iterations']}")
    print(f"Held-out test observations: {test_metrics['observations']}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro precision: {test_metrics['macro_precision']:.4f}")
    print(f"Test macro recall: {test_metrics['macro_recall']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Results written to: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
