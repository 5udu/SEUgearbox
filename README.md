# SEUgearbox

Gearbox fault diagnosis experiments based on wavelet features, random-forest feature selection, and an HHO-optimized BP neural network.

## What was improved

- Reworked the HHO implementation with input validation, deterministic seeding, and clearer progress reporting.
- Turned the training script into a reusable CLI instead of a one-off notebook export.
- Added robust handling for both normal Chinese column names and the mojibake column names already present in the notebook export.
- Documented dependencies and added a small optimizer regression test.

## Repository layout

- `BP测试.py`: training entry point for the HHO-BP classifier.
- `hho.py`: Harris Hawks Optimization implementation.
- `数据处理部分，完整版.ipynb`: original data processing notebook exported from the author workflow.
- `requirements.txt`: Python dependencies for the training script and tests.
- `tests/test_hho.py`: minimal sanity test for the optimizer.

## Expected dataset

The training script expects a CSV named `features_reconstructed_simplified_df.csv` by default. It should contain:

- 9 selected feature columns used by the classifier
- 1 label column named either `故障类别` or the notebook's existing mojibake equivalent

If your notebook export still contains mojibake column names, the script will still try to work with it.

## Quick start

```bash
python -m pip install -r requirements.txt
python BP测试.py --data features_reconstructed_simplified_df.csv --iters 50 --hawks 20 --plot
```

## Notes

- The raw dataset is not stored in this repository.
- The notebook appears to have been saved with encoding issues in several text cells. The Python entry point now avoids depending on those broken labels as much as possible.
- If you want a full restructuring of the notebook pipeline into standalone modules, that can be done in a follow-up pass.
