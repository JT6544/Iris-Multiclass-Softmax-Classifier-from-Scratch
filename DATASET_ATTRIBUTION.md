# Dataset attribution

This project includes the corrected **Iris** dataset from the UCI Machine
Learning Repository.

- Creator: R. A. Fisher
- Dataset record: https://archive.ics.uci.edu/dataset/53/iris
- DOI: https://doi.org/10.24432/C56C76
- Citation: Fisher, R. (1936). *Iris* [Dataset]. UCI Machine Learning Repository.
- Dataset licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Licence text: https://creativecommons.org/licenses/by/4.0/

## Included file

`data/raw/bezdekIris.data` is the corrected UCI data file downloaded from:

https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data

Its SHA-256 checksum is recorded in `data/raw/SHA256SUMS`.

The file is retained unchanged. The UCI dataset record notes corrections to
samples 35 and 38; the Bezdek file contains those corrected measurements.

## Project processing

The four measurements are parsed as centimetres and the species names are
mapped to integer model labels:

- `Iris-setosa` becomes `0`
- `Iris-versicolor` becomes `1`
- `Iris-virginica` becomes `2`

No measurement is modified. Records with identical measurements and species
are assigned one shared grouping identifier so they cannot be divided across
the development/test split or cross-validation folds.

Run `python scripts/prepare_data.py` to create an optional named, row-oriented
CSV derivative. The raw source file is never overwritten.
