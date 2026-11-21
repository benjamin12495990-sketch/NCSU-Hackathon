# pipeline/ — alternative training pipeline (neural network)

An alternative to the live app's model (`train/train.py`, a `GradientBoostingRegressor`). Same
task, same data, same leak-free evaluation — a different model family, so the two can be compared.

- **Expected strength:** a small ReLU MLP (`[9, 64, 32, 1]`), trained with mean-squared error in
  pure NumPy (hand-written forward/backward + Adam, learning-rate decay).
- **Cautious lower estimate:** the expected model minus a **split-conformal offset**, calibrated on
  a held-out set of mixes so a target share of lab results clear it. This is the calibrated
  alternative to GBR's quantile-loss model. (A direct pinball-loss MLP was tried first; it would not
  calibrate reliably with this optimizer, so conformal calibration replaced it — see the commit
  history / `train_nn.py` docstring.)

## Run

```bash
python3 pipeline/train_nn.py
```

Only dependency is NumPy (`pipeline/requirements.txt`). No scikit-learn, no PyTorch. Trains in a
few seconds. Outputs go to `pipeline/out/<task>/` — **never** to `public/model/**, so this pipeline
does not disturb the app's own exported models.

Outputs:
- `metrics.json` — test metrics (same shape as the app's).
- `nn_model.json` — standardization + weight matrices + the cautious offset. Pure matmuls, so it
  can run in the browser as an alternative inference path to the tree walk.
- `test_points.json` — predicted vs measured on the held-out mixes.

## Results (concrete, held-out mixes)

| Model | Expected MAE | RMSE | R² | Cautious coverage (target 0.90) |
| --- | --- | --- | --- | --- |
| GBR — live app (`train/train.py`) | 4.46 | 6.60 | 0.839 | 0.838 |
| **NN — this pipeline** | 5.68 | 7.81 | 0.823 | **0.927** |

Reading: the gradient-boosted trees are a little more accurate on this small tabular dataset (the
usual outcome), while the neural net's conformal lower bound is better calibrated to its target. A
useful second opinion, not a replacement — the same honesty rules apply (predictions labelled as
predictions; the tool narrows the experiment list).

## Reusing it on another domain

`train_nn.py` is driven by the `CONFIG` block at the top: dataset path, feature columns, derived
ratios, target, and the group columns that define a leak-free split. Point those at any tabular
dataset from `domains/` to get the same expected + calibrated-cautious pair. See `../docs/SCOPE.md`.
