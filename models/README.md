# Saved models

`python -m mu2e_tranad` scores data with a saved model by default and does not train. The
default is the file below. To use another one, pass `--model PATH` or set
`MU2E_TRANAD_MODEL`. To train a new model, add `--retrain`.

| File | Source | Trained | Rows / runs | Inputs |
|---|---|---|---|---|
| `crv_tranad_006-001.pkl` (default) | `crvreco,kpp_crv,file,006-001` (sid 118) | 2026-11-18 | 4231 subruns, runs 120065–125895 | TranAD: 12 (8 coincidence-cluster inputs with validity flags, 2 sector digi centroids, `eventHasError`, `h2_channels_nzero`); window 4; threshold 0.2014 |

`crv_tranad_006-001.pkl` was trained with the default options (`--tranad-features response`,
`--train-mode stratified`, `--threshold-mode heldout`, `--threshold-percentile 99.5`,
`--seed 0`) on 1268 training rows, with the threshold set on 423 held-out rows. It is the
same file dqm-assistant uses (sha256 `ac2f95a0…7715`). The `source` recorded inside it is
the path of the CSV it was trained from, a temporary file that no longer exists.

Its PCA part keeps all 9 components, so PCA's reconstruction error is rounding noise. PCA is
skipped when scoring with this model, and the online evaluator refuses `detectors: [pca]`.

## Replacing the default

1. Train: `python -m mu2e_tranad --dqm-source <source> --retrain --save-model models/<name>.pkl`
2. Check the scores of a few known runs with `--model models/<name>.pkl`.
3. Point `DEFAULT_MODEL` in `training/bundle.py` at the new file and add a row above.
   In a DQM driver config, also bump `config:`, because it is part of alarm identity.
