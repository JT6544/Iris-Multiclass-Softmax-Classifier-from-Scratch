from __future__ import annotations

import unittest

import numpy as np

from iris_softmax_classifier import (
    MulticlassSoftmaxRegression,
    classification_metrics,
    softmax_loss,
    stable_softmax,
)


class ModelTests(unittest.TestCase):
    def test_stable_softmax_handles_extreme_scores(self) -> None:
        scores = np.asarray([[1_000.0, 999.0, -1_000.0], [-1_000.0, -1_000.0, -1_000.0]])
        probabilities = stable_softmax(scores)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_analytical_gradient_matches_finite_difference_with_l2(self) -> None:
        rng = np.random.default_rng(3)
        features = rng.normal(size=(8, 3))
        design = np.column_stack((np.ones(features.shape[0]), features))
        labels = np.asarray([0, 1, 2, 0, 1, 2, 1, 0])
        weights = rng.normal(scale=0.2, size=(4, 3))
        l2_strength = 0.07

        probabilities = stable_softmax(design @ weights)
        indicators = np.eye(3)[labels]
        analytical = design.T @ (probabilities - indicators) / labels.size
        analytical[1:, :] += l2_strength * weights[1:, :]

        numerical = np.zeros_like(weights)
        epsilon = 1e-6
        for row in range(weights.shape[0]):
            for column in range(weights.shape[1]):
                plus = weights.copy()
                minus = weights.copy()
                plus[row, column] += epsilon
                minus[row, column] -= epsilon
                numerical[row, column] = (
                    softmax_loss(design, labels, plus, l2_strength)
                    - softmax_loss(design, labels, minus, l2_strength)
                ) / (2 * epsilon)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-7)

    def test_model_learns_three_separable_classes(self) -> None:
        features = np.asarray(
            [
                [-3.0, -3.0],
                [-2.5, -2.0],
                [3.0, -3.0],
                [2.5, -2.0],
                [0.0, 3.0],
                [0.5, 2.5],
            ]
        )
        labels = np.asarray([0, 0, 1, 1, 2, 2])
        model = MulticlassSoftmaxRegression(
            learning_rate=0.2,
            max_iterations=800,
            l2_strength=0.0,
            patience=100,
        ).fit(features, labels)
        np.testing.assert_array_equal(model.predict(features), labels)
        self.assertLess(model.history_["training_loss"][-1], model.history_["training_loss"][0])

    def test_multiclass_metrics(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 2, 2])
        predictions = np.asarray([0, 1, 1, 1, 2, 0])
        metrics = classification_metrics(labels, predictions)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertAlmostEqual(metrics["per_class"]["Iris-versicolor"]["recall"], 1.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], (0.5 + 1.0 + 0.5) / 3)


if __name__ == "__main__":
    unittest.main()
