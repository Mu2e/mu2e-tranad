# mu2e-tranad — CRV DQM anomaly detection

`mu2e-tranad` flags anomalous run/subruns in Mu2e Cosmic Ray Veto (CRV) Data Quality Monitoring metrics. It takes the per-subrun CRV metrics from the DQM database as a time series, then runs three detectors over them: a Z-score check, a PCA reconstruction check and a TranAD transformer. **TranAD is the primary detector.** The other two act as cross-checks.

**By default nothing is trained.** The data is scored with a saved model: the CRV model trained on `crvreco,kpp_crv,file,006-001`, kept in [`models/`](models/README.md). Use `--model PATH` (or `MU2E_TRANAD_MODEL`) for another saved model. Use `--retrain` to train the detectors on the data instead, which also saves the new model.

---

## Quick start

The TranAD model is a copy of the one in the `anldq` package from the ANL [CrossExperimentalAIDQM](https://github.com/anl-hep-ai-ml/CrossExperimentalAIDQM) repository, kept in `training/anl_tranad/`. You do not need a CrossExperimentalAIDQM checkout. Clone this repository **as `mu2e_tranad/`** (underscore: the repository is `mu2e-tranad`, but the directory is also the Python package, and a package name cannot contain a hyphen).

On a Mu2e gpvm (CPU is enough):

```bash
git clone https://github.com/sophiemiddleton/mu2e-tranad.git mu2e_tranad

source mu2e_tranad/setup_env.sh      # must print "mu2e_tranad env ready (TranAD available)"

# Fetch a DQM source and score it with the saved 006-001 model (needs the FNAL network, see Environment)
python -m mu2e_tranad --dqm-source crvreco,kpp_crv,file,006-001 --output results/crvreco_006-001

# ...or score a CSV you already have, for example the filtered CSV saved by the run above
python -m mu2e_tranad --csv results/crvreco_006-001/dqm_crvreco_kpp_crv_file_006-001_filtered.csv \
    --output results/rerun_006-001

# Train a new model on the data instead (slow: 50 TranAD epochs), and save it
python -m mu2e_tranad --dqm-source crvreco,kpp_crv,file,006-001 --output results/train_006-001 \
    --retrain --save-model mu2e_tranad/models/crv_tranad_006-001_v2.pkl
```

The saved model only fits data with the metrics it was trained on. Data from another source that lacks some of them stops with an error naming the missing metrics: train a model for that source with `--retrain`.

In CrossExperimentalAIDQM, this repository is the git submodule `mu2e_tranad/`. To get it there, use `git clone --recurse-submodules`, or run `git submodule update --init mu2e_tranad` in an existing clone.

Run these commands from the directory that contains `mu2e_tranad/`. Open `results/<dir>/anomaly_report_*_anomalies_only.csv` and `results/<dir>/diagnostics/diagnostic_report.txt` first.

> **If TranAD cannot be imported, it is silently disabled.** When `torch` cannot be imported, the pipeline prints `⚠️ PyTorch not available` and still runs. Every TranAD score and prediction is then 0, so the primary `anomaly` column contains no anomalies. The usual causes are skipping `setup_env.sh` or cloning this repository under a name other than `mu2e_tranad`.

---

## Environment

The only dependency for TranAD is `torch`. `python -m mu2e_tranad` needs the directory that contains `mu2e_tranad/` on `PYTHONPATH` (or as the current directory).

**Recommended on a Mu2e gpvm: `setup_env.sh`.** This script uses the Mu2e cvmfs `ana` environment (Python 3.12, torch 2.5.1, pandas, scikit-learn, matplotlib). On first use it installs `seaborn`, which the correlation heatmap in `analysis/visualization.py` needs, into `mu2e_tranad/.ana_extras/` (git-ignored).

The script also puts the parent directory and the extras on `PYTHONPATH`, and sets `MPLBACKEND=Agg` and `PYTHONWARNINGS=ignore`. It warns if the directory is not named `mu2e_tranad`.

**Alternative: your own Python environment.** Install the dependencies yourself, ideally in a venv on top of the cvmfs `ana` env (`python -m venv --system-site-packages`):

```bash
pip install -e "./mu2e_tranad[tranad]"   # pandas, numpy, scikit-learn, matplotlib, seaborn, torch
```

This also installs a `mu2e-tranad` console command.

**DQM MCP server.** `--dqm-source` and `export_from_dqm` fetch data from the DQM MCP server at `http://mu2eaigpvm01.fnal.gov:8001/mcp`, which is only reachable from the FNAL network. Outside FNAL, use `--csv` with a CSV exported on site. To use a different server, set `DQM_MCP_URL`. The client only uses the Python standard library, so no extra packages are required.

---

## How it works

```
 DQM MCP server ──► fetch matrix ──► drop static columns ──► ingest ──► preprocess
                    (dqm_mcp.py)     (filter_from_dqm.py)              │
                                                                       ├─► Z-score ┐
                                                         TranAD prep ──┼─► TranAD  ├─► vote ─► reports, diagnostics, plots
                                                      (tranad_prep.py) └─► PCA     ┘
```

The code is split into four subpackages, in the order the data flows through them:

| Directory | Stage | Phases |
|-----------|-------|--------|
| `reading/` | Fetch, filter and load the DQM metrics | 0, 1 |
| `training/` | Preprocess, prepare the TranAD inputs, fit the detectors, save the model bundle | 2, 2b, 3, 3b |
| `analysis/` | Diagnostics and plots of the results | 5, 6 |
| `evaluator/` | Score new DQM points online with a saved bundle (not part of a batch run) | — |

`run_pipeline.py` at the top level runs the batch phases in order and writes the reports (Phase 4).

### Phase 0 — Fetch (only with `--dqm-source`)

`dqm_mcp.fetch_matrix()` turns the source string `process,stream,aggregation,version` (for example `crvreco,kpp_crv,file,005-001`) into a DQM source ID, `sid`. It lists every metric in the value group (default `CRV`) and downloads the full timeline of each metric, using 8 threads in parallel. The results are combined into a single wide table:

```
run, subrun, timestamp, CRV_<subgroup>_<name>, ...
118, 1,      2026-02-13 11:07:29-06:00, 1.0, ...
```

Each row is one run/subrun interval, keyed by its start run and start subrun. Columns that are constant, or that are 1.0 in every row, are dropped. Both the raw CSV and the filtered CSV are saved to the output directory as `dqm_<source>.csv` and `dqm_<source>_filtered.csv`. `--after` and `--before` limit the fetch to a time window given in ISO UTC.

### Phase 1 — Ingestion (`ingestion.py`)

This phase loads the CSV and parses `timestamp`, which may include a timezone, into a UTC `datetime` column. It also runs quality checks such as counting missing values. Every column except `run`, `subrun`, `timestamp` and `datetime` is treated as a feature.

### Phase 2 — Preprocessing (`preprocessing.py`)

Only "active" features are kept. A feature is active if it contains at least one non-zero value, is not 1.0 in every row, and takes more than one distinct value. The active features are then standardised to mean 0 and standard deviation 1, and this matrix is the input to Z-score and PCA.

IQR outlier removal is available, but it is **off by default** because it would remove the anomalies you want to find. Enable it with `--apply-iqr` only to clean training data.

### Phase 2b — TranAD input preparation (`tranad_prep.py`)

Recalibrations are expected events, but TranAD would flag every one of them as an anomaly if it saw the raw CRV metrics. This phase prevents that:

1. **Split the features.** *Condition* metrics (`crvCalibConstants*`, `crvPedestals*`) describe calibration inputs. *Response* metrics are everything else.
2. **Find calibration epochs.** An epoch is a block of consecutive rows in which every condition `Centroid` value is identical. Blocks shorter than 20 rows are labelled epoch `-1` (transient). One example is a glitch where the constants read 0. Blocks that share the same values, such as the stretches before and after a glitch, are given the same epoch. The epochs are written to `calibration_epochs.csv`, and each row's epoch is stored in a `calib_epoch` column.
3. **Build the TranAD inputs**, controlled by `--tranad-features`:
   - `response` (default): response metrics only. Features with fewer than 6 distinct values are dropped. "No fit" sentinel values (0 for `crvPEsMPV_*Centroid`, −1 for `crvCoincidencesClusters_*`) are replaced by the median, and each such feature gets a 0/1 `<feature>__valid` column.
   - `changes`: everything in `response`, plus each condition metric expressed as `(value − epoch median) / |epoch median|`. TranAD then sees glitches in the constants but not legitimate recalibrations.
   - `all`: the same scaled matrix that Z-score and PCA use.
4. **Choose the training and threshold rows**, controlled by `--train-mode` and `--seed`. Training uses 30% of rows and threshold validation uses a separate 10%:
   - `stratified` (default): 30% and 10% are sampled from *every* calibration epoch, and transients are excluded. This way, every calibration state counts as normal.
   - `random`: rows are sampled uniformly from the whole dataset.
   - `chronological`: the first 30% of rows are used for training and the next 10% for validation. This matches the original prototype of this pipeline.

   The TranAD inputs are then standardised using the training rows only.

### Phase 3 — Detection (`models.py`, `tranad_model.py`)

**Default (scoring):** `models.score_with_bundle` loads the saved model and applies its TranAD weights, input scaling, sentinel fills and threshold, and its drift reference, all frozen at training time. Phase 2b's input preparation and row selection are skipped. Z-score has no saved model and is computed on the data, as below. PCA from a model that keeps every component is skipped, because its reconstruction error is rounding noise (the 006-001 model's PCA is one of these). Scoring the data a model was trained on reproduces that training run's TranAD flags and scores exactly.

**With `--retrain`:** the detectors are trained as described in the table below, and the model is saved.

| Detector | What it does | Output columns | Flags a row when |
|----------|--------------|----------------|------------------|
| **TranAD** (primary) | Uses the ANL TranAD (`training/anl_tranad/`) with window 4, `d_model` 32, 4 heads, `d_ff` 64 and 2 layers. It trains for 50 epochs with Adam (lr 1e-3) on the training rows, and each row is reconstructed from the 4 rows before it. The score is the mean squared reconstruction error. | `tranad_anomaly`, `tranad_reconstruction_error` | The score is above the `--threshold-percentile` (default **99.5**) of the scores on the held-out rows (`--threshold-mode heldout`) or on the training rows (`train`). |
| Z-score | Computes the absolute z-score of each feature, using the mean and standard deviation of the whole dataset. | `z_score_anomaly`, `z_score_max` | Any feature has \|z\| > 3. |
| PCA | Fits PCA on the whole dataset and computes the squared reconstruction error. | `pca_anomaly`, `pca_reconstruction_error` | The error is above its own 95th percentile, so 5% of rows are flagged by construction. |
| Ensemble | Adds the three 0/1 votes. | `ensemble_voting_score` (0–3) | The vote is at least `--voting-threshold` (default 2). The ensemble is only used for diagnostics and comparison, and it does **not** set `anomaly`. |

The first 4 rows (the window length) always have a TranAD score of 0.

### Phase 3b — Training drift (`tranad_prep.py`)

TranAD's scaling and threshold are fixed when it is trained. When a metric drifts slowly away from the training data, TranAD eventually flags it as a run of anomalies, and those flags look the same as a fault. The drift check reports this case directly. For each row and each input, it takes the median of the last `--drift-window` rows (default 50) and compares it with that input's median on the training rows, in units of the training standard deviation. A row is flagged when any input has moved more than `--drift-sigma` (default 3.0). The median ignores single anomalous rows, so only sustained shifts count.

Drift is measured on the TranAD inputs, standardised the same way as for TranAD. If TranAD did not run, it uses the inputs that Z-score and PCA use. The `__valid` flags are not measured; drift in a sentinel feature shows up in the feature itself. Rows with fewer than half a window of rows before them are not scored.

This is not the same as the `drift` check of the DQM `trends` evaluator. That check compares a metric with its own recent past. This one compares it with the data the model was trained on, so it answers the question "is it time to retrain?".

### Phase 4 — Reports

`anomaly_report_<YYYYMMDD_HHMMSS>.csv` contains every row with all of its original features. It adds `calib_epoch`, `anomaly` (equal to `tranad_anomaly`), the prediction and score columns listed above, and `ensemble_voting_score`. It also adds the drift columns: `training_drift` (0/1), `drift_max_shift` (the largest shift, in training standard deviations), `drift_worst_feature` and `drift_n_features` (the number of inputs beyond `--drift-sigma`). `..._anomalies_only.csv` contains only the rows where `anomaly == 1`.

### Phase 5 — Diagnostics (`diagnostics.py`, saved to `diagnostics/`)

- `feature_statistics.csv` compares the mean, standard deviation, minimum and maximum of each feature between normal and anomalous rows.
- `extreme_features.csv` lists the features with extreme z-scores in anomalous rows.
- `diagnostic_report.txt` is a text summary. It covers how often the detectors agree, a breakdown of the votes, when anomalies occurred, and the features that best separate anomalous rows from normal ones. The same text is printed to the console.

See [analysis/DIAGNOSTICS.md](analysis/DIAGNOSTICS.md) for how to read these files, and how to use `AnomalyDiagnostics` on its own.

### Phase 6 — Plots (`visualization.py`, saved to `plots/`, skip with `--no-plots`)

| File | Content |
|------|---------|
| `tranad_training_curve.png` | TranAD loss for each epoch |
| `timeseries_overview.png` | Four-panel overview |
| `{z-score,pca,tranad,ensemble}_anomaly_scores_{full,zoomed}.png` | Each detector's score over time, with TranAD anomalies highlighted |
| `detector_comparison.png` | The three detectors' scores side by side |
| `anomaly_distribution.png`, `anomaly_timeline.png` | Number of anomalies, and when they occurred |
| `feature_distributions.png`, `correlation_heatmap.png` | The top 6 features, and the correlations between features |
| `timeseries/NNN_<feature>.png` | One time series for each active feature, with anomalies marked |

The threshold line on the TranAD plot is the threshold the model actually used. The lines on the Z-score and PCA plots are the 95th percentile of their scores.

---

## Command-line reference

```
python -m mu2e_tranad (--csv PATH | --dqm-source P,S,A,V) [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--csv PATH` | — | Analyse an existing filtered CSV |
| `--dqm-source P,S,A,V` | — | Fetch this source from the DQM MCP server first |
| `--dqm-group` | `CRV` | DQM value group to fetch |
| `--after`, `--before` | — | ISO UTC time window used with `--dqm-source` |
| `--output DIR` | `./results` | Output directory, created if it does not exist |
| `--model PATH` | `models/crv_tranad_006-001.pkl` | Saved model to score with (`MU2E_TRANAD_MODEL` overrides the default). Not used with `--retrain` |
| `--retrain` | off | Train the detectors on this data instead of loading a saved model. The options marked *(training)* need it |
| `--tranad-features` *(training)* | `response` | `response`, `changes` or `all` (see Phase 2b) |
| `--train-mode` *(training)* | `stratified` | `stratified`, `random` or `chronological` (see Phase 2b) |
| `--threshold-mode` *(training)* | `heldout` | Set the TranAD threshold on `heldout` or `train` rows |
| `--threshold-percentile` *(training)* | `99.5` | Percentile of TranAD error used as the threshold. Lower values flag more rows. |
| `--voting-threshold` | `2.0` | Number of detectors that must agree for the ensemble |
| `--seed` *(training)* | `0` | Random seed for selecting TranAD rows |
| `--drift-window` | `50` | Number of rows in the training-drift median (see Phase 3b). Also saved as the evaluator's default. |
| `--drift-sigma` | `3.0` | Training standard deviations an input's median can move before it counts as drift. Also saved as the evaluator's default. |
| `--apply-iqr` / `--iqr K` | off / `1.5` | Remove IQR outliers before detection. Use only to clean training data. |
| `--no-plots` | — | Skip Phase 6, which is the slowest step for wide datasets |
| `--save-model PATH` *(training)* | `<output>/tranad_model_<timestamp>.pkl` | Where to save the trained models for the online evaluator (see Online evaluation). Every run saves them; with `--scale-by period`, `--drop-empty`, `--train-clean` or `--tranad-features changes` an explicit path is refused and the default run saves `tranad_weights_<timestamp>.pkl` (weights only, not loadable online) instead |
| `--no-save-model` | off | With `--retrain`: don't save the trained model |

### Exporting and filtering on their own

```bash
# Raw matrix only (no detection)
python -m mu2e_tranad.reading.export_from_dqm --source crvreco,kpp_crv,file,005-001 --output raw.csv \
    [--group CRV] [--after 2026-03-01T00:00:00] [--before ...] [--url ...]

# Drop static columns (reads dqm_crv_anomaly_dataset_004-000.csv in the current directory,
# writes ..._filtered.csv; the file names are set at the top of the script)
python mu2e_tranad/reading/filter_from_dqm.py
```

---

## Python API

```python
from mu2e_tranad.run_pipeline import main, fetch_from_dqm

csv = fetch_from_dqm("crvreco,kpp_crv,file,006-001", "results/006-001")
main(csv, output_dir="results/006-001")                    # score with the default model
main(csv, output_dir="results/006-001", model="other.pkl")  # ...or another saved model

main(csv, output_dir="results/train", retrain=True,         # train instead
     tranad_features="response", train_mode="stratified",
     threshold_mode="heldout", threshold_percentile=99.5,
     save_model="models/new.pkl")
```

You can also run the phases one at a time:

```python
from mu2e_tranad.reading import ingestion
from mu2e_tranad.training import preprocessing, models, tranad_prep
from mu2e_tranad.reading.dqm_mcp import DqmMcpClient, fetch_matrix

df = fetch_matrix("crvreco,kpp_crv,file,005-001")            # wide DataFrame
df, feats = ingestion.ingest_data("data.csv")
df, active, X, scaler = preprocessing.preprocess_pipeline(df, feats)
cond, resp = tranad_prep.split_condition_features(active)
epochs, summary = tranad_prep.detect_calibration_epochs(df.reset_index(drop=True), cond)
ensemble, results = models.run_ensemble_detection(X, df, active)   # TranAD on X, first 30% rows

client = DqmMcpClient()                 # lower-level MCP access
client.list_sources(process="crvreco")
client.list_values(groupx="CRV")
```

---

## Online evaluation (Mu2e DQM driver)

The trained models can run inside the Mu2e DQM evaluation framework
([Mu2e/DQM](https://github.com/Mu2e/DQM), `python/dqm/evaluate`), next to the `limits` and
`trends` evaluators. The DQM driver hands an evaluator one new timeline point at a time,
together with the points before it, and commits what it reports as alarms. Training stays
here, offline:

```bash
source mu2e_tranad/setup_env.sh      # also puts ./DQM/python (a Mu2e/DQM clone) on PYTHONPATH

# 1. Only for a new model: train and save it (TranAD threshold, scaling and weights are
#    frozen here). Without pars.model the driver uses models/crv_tranad_006-001.pkl.
python -m mu2e_tranad --dqm-source crvreco,kpp_crv,file,006-001 --output results/006-001 \
    --no-plots --retrain --save-model models/crv_ml_v1.pkl

# 2. Configure a driver (copy mu2e_tranad/configs/crv_ml.example.yaml) and check it
python -m mu2e_tranad.evaluator.dqm_driver crv_ml.yaml --check

# 3. Run it, from cron; report mode (commit: false) writes nothing to the database
python -m mu2e_tranad.evaluator.dqm_driver crv_ml.yaml
```

`dqm_driver` is the DQM `dqm-driver` command with the `crv_ml` algorithm added to its
registry, so the code does not need to live in the DQM repository. The evaluator
(`evaluator/ml_evaluator.py`) reports these subjects:

| Subject | Raised when |
|---------|-------------|
| `tranad_outlier` | TranAD's reconstruction error for the point, from it and the 4 points before it, is above the trained threshold |
| `pca_outlier` | The PCA reconstruction error is above the trained threshold. Refused until PCA keeps fewer components (see Known limitations). |
| `training_drift` | Over the point and the `drift_window − 1` points before it, the median of some input has moved more than `drift_sigma` training standard deviations from its training median (see Phase 3b). Needs a bundle saved with this version. |

Each finding names the `top_causes` worst-reconstructed metrics as its causes, and its note
holds the score, the threshold, those metrics' values and errors, and the training date.
Replaying the 004-000 training data through the driver reproduces the batch
`tranad_reconstruction_error` and flags to float32 precision. Scoring takes about 5 ms per
point on a gpvm CPU; most of the time goes to importing torch and to the database reads.

- **Start with a floor.** Set `min_time` (or `min_run`) to the go-live date. Without one, the
  first invocations examine the whole back history and open an alarm episode for every old
  anomaly. To look at the history, run in report mode first.
- **Retrain when `training_drift` stays open.** An episode that lasts means the detector has
  moved away from the data the model learned. Its `tranad_outlier` findings then reflect that
  move rather than faults. Retrain on recent data (`--dqm-source ... --after <date>`) and bump
  `config`. Before going live, check the `training_drift` column of a batch run on the
  history to choose `drift_sigma`.
- **Retrain when the metrics change.** The bundle stores the feature names it was trained on.
  A point that lacks one of them, or has one flagged by its producer, is skipped and the skip is
  logged. New metrics are ignored. After retraining, change `config` (for example
  `crv-tranad-v2`), so the new model's alarms do not extend the old model's episodes.
- **The bundle is a pickle of numpy arrays.** Read it with the numpy major version it was
  written with (the `ana` environment).
- `--tranad-features changes` cannot be saved yet, because calibration-epoch references
  need the whole dataset.

---

## Files

| File | Role |
|------|------|
| `__main__.py`, `run_pipeline.py` | Command-line interface and the code that runs each phase in order; Phase 4 reports |
| **`reading/`** | |
| `reading/dqm_mcp.py` | JSON-RPC client for the DQM MCP server, and `fetch_matrix()` (Phase 0) |
| `reading/export_from_dqm.py` | Command-line tool to export a DQM source to a CSV |
| `reading/filter_from_dqm.py` | `filter_static_columns()` and a standalone script for filtering |
| `reading/ingestion.py` | Phase 1 |
| **`training/`** | |
| `training/preprocessing.py` | Phase 2 |
| `training/tranad_prep.py` | Phase 2b: condition/response split, calibration epochs and row selection; Phase 3b: training drift |
| `training/models.py` | Phase 3: Z-score and PCA detectors, and the ensemble |
| `training/tranad_model.py` | Wrapper around the TranAD model |
| `training/anl_tranad/` | Copy of the TranAD model from `anldq` in CrossExperimentalAIDQM; the source commit is `ANLDQ_COMMIT` in its `__init__.py` |
| `training/bundle.py` | Saves and loads the trained models for the online evaluator |
| **`evaluator/`** | |
| `evaluator/ml_evaluator.py` | `MLEvaluator`, the `crv_ml` evaluator for the Mu2e DQM driver (also importable as `mu2e_tranad.evaluator.MLEvaluator`) |
| `evaluator/dqm_driver.py` | `dqm-driver` with `crv_ml` registered |
| **`analysis/`** | |
| `analysis/diagnostics.py` | Phase 5 |
| `analysis/visualization.py` | Phase 6 |
| [analysis/DIAGNOSTICS.md](analysis/DIAGNOSTICS.md) | Guide to the Phase 5 diagnostics |
| **Top level** | |
| `configs/crv_ml.example.yaml` | Example driver config |
| `configs/periods_005-001.csv` | Example `--periods` file |
| `setup_env.sh` | Environment for a Mu2e gpvm (see Environment) |
| `setup.py` | Optional `pip install -e` support |

The pipeline started as a notebook prototype (referred to as "Genesis" in some code comments) that used the Mu2e `DqmTool` exporter. This repository replaces it.

---

## Known limitations

- **TranAD and the `anomaly` column depend on the environment.** If TranAD cannot be imported, nothing is flagged and the pipeline does not stop with an error. Check the console for the `⚠️` import warnings.
- **The TranAD copy does not follow upstream automatically.** `training/anl_tranad/` is a copy of the `anldq` TranAD from the commit in `ANLDQ_COMMIT`. Fixes made to `anldq` later must be copied in by hand (see the docstring in `training/anl_tranad/__init__.py`).
- **Z-score and PCA are fitted on the data they score.** Their thresholds are relative to the dataset being analysed, so the 5% PCA rate is fixed by construction. Use them as cross-checks, not as absolute tests.
- **PCA keeps every component** (`n_components=None`, and `variance_threshold` is not used), so it reconstructs the data exactly. In the `005-001` run, the reconstruction errors were about 1e-29, which is floating-point noise. The 5% of rows that PCA flags are therefore effectively random until a component count is set.
- **TranAD trains sequentially, one window at a time, with no batching.** This is fine for thousands of rows but becomes slow on much larger datasets.
- **The order of rows matters.** TranAD windows are built from consecutive rows, sorted by (run, subrun) when fetched from the MCP server. A CSV you supply yourself should be sorted the same way.
- **`reading/filter_from_dqm.py` has hard-coded file names** when run as a script.
