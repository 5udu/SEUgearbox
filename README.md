# SEUgearbox

Reproducible gearbox fault diagnosis experiments based on DWT feature extraction, random-forest feature ranking, and an HHO-optimized BP neural network.

## Reproducible workflow

The repository is now organized around a repeatable research pipeline instead of a notebook-only workflow:

1. Prepare aligned raw signal tables from the eight source CSV files.
2. Segment each signal into sliding windows.
3. Extract wavelet, frequency, and SVD-based features.
4. Rank feature importance with a random forest.
5. Run the HHO-BP experiment and save metrics as artifacts.
6. Optionally generate GADF images and serialized image datasets.

All generated outputs are intended to live under `artifacts/`, so the source code and derived data stay separate.

## Project layout

- `seugearbox/`: reusable preprocessing, feature extraction, and experiment modules.
- `scripts/prepare_research_data.py`: creates `data_1.csv`, `data_2.csv`, features, and feature-importance artifacts.
- `scripts/run_experiment.py`: runs the HHO-BP experiment and writes metrics and confusion matrices.
- `scripts/generate_gaf_dataset.py`: reproduces the notebook's GADF image generation flow.
- `BP测试.py`: compatibility entry point that delegates to the new experiment runner.
- `hho.py`: standalone HHO optimizer implementation.
- `tests/`: lightweight regression tests for optimizer, preprocessing, and feature generation.
- `数据处理部分，完整版.ipynb`: original notebook kept for traceability.

## Installation

```bash
python -m pip install -r requirements.txt
```

Or install as a package:

```bash
python -m pip install -e .[dev]
```

## Data assumptions

The raw-signal preparation step expects these files in one directory:

- `Health_20_0.csv`
- `Chipped_20_0.csv`
- `Miss_20_0.csv`
- `Root_20_0.csv`
- `Surface_20_0.csv`
- `ball_20_0.csv`
- `inner_20_0.csv`
- `outer_20_0.csv`

The default slicing logic reproduces the notebook:

- row range: `1015:202063`
- signal column index: `2`
- window size: `2048`
- step size: `1000`

## End-to-end usage

Prepare tabular research artifacts:

```bash
python scripts/prepare_research_data.py --input-dir path/to/raw_csv_dir --output-dir artifacts/data
```

Run the main experiment:

```bash
python scripts/run_experiment.py --data artifacts/data/features_reconstructed_simplified_df.csv --output-dir artifacts/experiments/default
```

Generate GADF images and an image dataset:

```bash
python scripts/generate_gaf_dataset.py --segmented-data artifacts/data/data_2.csv --image-dir artifacts/gaf_images --output-path artifacts/gaf_dataset.npz
```

Use the legacy entry point if you want the old command name:

```bash
python BP测试.py --data artifacts/data/features_reconstructed_simplified_df.csv --output-dir artifacts/experiments/default
```

## Saved outputs

The experiment runner writes:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.csv`

The data-preparation runner writes:

- `data_1.csv`
- `data_2.csv`
- `features_reconstructed_simplified_df.csv`
- `feature_importance.csv`

## Verification

Run the test suite with:

```bash
pytest -q
```

## Notes

- The original notebook has some text-encoding issues, so the scripted workflow is the preferred path for reproducibility.
- The experiment runner still supports the notebook's mojibake feature names when working with already exported CSV files.
