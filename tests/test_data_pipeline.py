from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from iris_softmax_classifier import (
    CLASS_NAMES,
    DEFAULT_DATA_PATH,
    FEATURE_NAMES,
    StandardScaler,
    identical_record_groups,
    load_dataset,
    stratified_group_partition,
)
from scripts.prepare_data import write_prepared_csv


class DataPipelineTests(unittest.TestCase):
    def test_corrected_dataset_structure_and_values(self) -> None:
        dataset = load_dataset(DEFAULT_DATA_PATH)
        self.assertEqual(dataset.features.shape, (150, 4))
        self.assertEqual(tuple(dataset.feature_names), FEATURE_NAMES)
        self.assertEqual(tuple(dataset.class_names), CLASS_NAMES)
        self.assertTrue(np.isfinite(dataset.features).all())
        np.testing.assert_array_equal(np.bincount(dataset.labels), [50, 50, 50])

        # Corrected values documented by the UCI dataset record.
        np.testing.assert_allclose(dataset.features[34], [4.9, 3.1, 1.5, 0.2])
        np.testing.assert_allclose(dataset.features[37], [4.9, 3.6, 1.4, 0.1])

    def test_identical_records_share_one_group(self) -> None:
        dataset = load_dataset(DEFAULT_DATA_PATH)
        self.assertEqual(np.unique(dataset.group_ids).size, 149)
        np.testing.assert_allclose(dataset.features[101], dataset.features[142])
        self.assertEqual(dataset.labels[101], dataset.labels[142])
        self.assertEqual(dataset.group_ids[101], dataset.group_ids[142])

        groups = identical_record_groups(dataset.features, dataset.labels)
        np.testing.assert_array_equal(groups, dataset.group_ids)

    def test_holdout_partition_is_deterministic_balanced_and_grouped(self) -> None:
        dataset = load_dataset(DEFAULT_DATA_PATH)
        first = stratified_group_partition(
            dataset.group_ids, dataset.labels, (0.8, 0.2), seed=42
        )
        second = stratified_group_partition(
            dataset.group_ids, dataset.labels, (0.8, 0.2), seed=42
        )
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second)))
        self.assertEqual(sum(part.size for part in first), 150)
        self.assertEqual(len(set(np.concatenate(first).tolist())), 150)

        group_sets = [set(dataset.group_ids[indices]) for indices in first]
        self.assertTrue(group_sets[0].isdisjoint(group_sets[1]))
        for indices in first:
            class_counts = np.bincount(dataset.labels[indices], minlength=3)
            self.assertLessEqual(int(class_counts.max() - class_counts.min()), 1)

    def test_five_grouped_folds_cover_development_once(self) -> None:
        dataset = load_dataset(DEFAULT_DATA_PATH)
        development, _test = stratified_group_partition(
            dataset.group_ids, dataset.labels, (0.8, 0.2), seed=42
        )
        local_folds = stratified_group_partition(
            dataset.group_ids[development],
            dataset.labels[development],
            np.repeat(0.2, 5),
            seed=43,
        )
        folds = [development[local] for local in local_folds]
        self.assertEqual(len(set(np.concatenate(folds).tolist())), development.size)
        for left in range(5):
            for right in range(left + 1, 5):
                self.assertTrue(
                    set(dataset.group_ids[folds[left]]).isdisjoint(
                        set(dataset.group_ids[folds[right]])
                    )
                )

    def test_standardisation_uses_fitted_values(self) -> None:
        features = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        transformed = scaler.fit_transform(features)
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-12)
        np.testing.assert_allclose(scaler.transform([[3.0, 4.0]]), [[0.0, 0.0]])

    def test_prepared_csv_is_row_oriented_and_non_destructive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "iris.csv"
            write_prepared_csv(DEFAULT_DATA_PATH, destination)
            with destination.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], [*FEATURE_NAMES, "species"])
            self.assertEqual(len(rows), 151)
            self.assertEqual(rows[1][-1], "Iris-setosa")
            with self.assertRaises(FileExistsError):
                write_prepared_csv(DEFAULT_DATA_PATH, destination)


if __name__ == "__main__":
    unittest.main()
