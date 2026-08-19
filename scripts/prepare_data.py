"""Create a named, row-oriented CSV from the bundled corrected UCI data."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from iris_softmax_classifier import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_DATA_PATH,
    FEATURE_NAMES,
    load_dataset,
)


def write_prepared_csv(source: Path, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {destination}; pass --force to replace it.")
    dataset = load_dataset(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((*FEATURE_NAMES, "species"))
        for features, label in zip(dataset.features, dataset.labels):
            writer.writerow((*features.tolist(), CLASS_NAMES[int(label)]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--destination",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "processed" / "iris_corrected.csv",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    write_prepared_csv(args.source, args.destination, force=args.force)
    print(f"Prepared dataset written to {args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
