# pipeline/ — alternative training pipeline (neural network)

An alternative to the live app's model (`train/train.py`, a `GradientBoostingRegressor`). Same
task, same data and the same rule for what counts as one mix, with a different model family. **The
live page does not use it.** The two scripts draw their test mixes separately and this one sets
aside a calibration set, so compare the numbers loosely.

- **Expected strength:** a small ReLU MLP (`[9, 64, 32, 1]`), trained with mean-squared error in
  pure NumPy (hand-written forward/backward + Adam, learning-rate decay).
- **Cautious lower estimate:** the expected model minus a **split-conformal offset**, calibrated on
  a held-out set of mixes so a target share of lab results clear it. This is an
  alternative to GBR's quantile-loss model. (A direct pinball-loss MLP was tried first; it would
  not calibrate reliably with this optimizer, so conformal calibration replaced it. That attempt is not in this
  repository; it is only noted in the message of commit e55cd76.)

## Run

```bash
python3 pipeline/train_nn.py
```

Only dependency is NumPy (`pipeline/requirements.txt`). No scikit-learn, no PyTorch. Trains in a
few seconds. Outputs go to `pipeline/out/<task>/`, **never** to `public/model/`, so this pipeline
does not disturb the app's own exported models.

Outputs:
- `metrics.json` — test metrics (same shape as the app's).
- `nn_model.json` — standardization + weight matrices + the cautious offset. Pure matmuls, so it
  can run in the browser as an alternative inference path to the tree walk.
- `test_points.json` — predicted vs measured on the held-out mixes.

## Results (concrete, held-out mixes)

| Model | Expected MAE | RMSE | R² | Cautious coverage (target 0.90) |
| --- | --- | --- | --- | --- |
| GBR, live app (`train/train.py`) | 3.29 | 4.37 | 0.912 | 0.877 |
| NN, this pipeline | 5.29 | 7.88 | 0.745 | 0.809 |

Both rows drop the 25 repeated rows and treat recipes within 5 kg on every ingredient as one mix,
so no test mix has a near-copy in training. The neural net was tested on 61 held-out mixes
(194 lab results) and the trees on 62 (219).

Reading: on this small tabular dataset the gradient-boosted trees are clearly more accurate, and
the neural net's conformal lower bound also misses its 90% target (81% here). An earlier
version of this table, made with the older split that kept the 25 repeated rows and let
near-duplicate mixes into both training and test, showed the neural net better calibrated (0.927).
With the stricter split it is not. The neural net is a second opinion that did not win, kept for
comparison.

## Reusing it on another domain

`train_nn.py` is driven by the `CONFIG` block at the top: dataset path, feature columns, derived
ratios, target, and the group columns that define a leak-free split. In principle it could be
pointed at another all-numeric table. That has not been done: it has only been run on the concrete
data, and the folders in `domains/` hold plans, not datasets. See `../docs/SCOPE.md`.
