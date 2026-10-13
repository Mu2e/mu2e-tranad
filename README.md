# dqm_pipeline — CRV DQM anomaly detection

`dqm_pipeline` flags anomalous run/subruns in Mu2e Cosmic Ray Veto (CRV) Data Quality Monitoring metrics. It takes the per-subrun CRV metrics from the DQM database as a time series, then runs three detectors over them: a Z-score check, a PCA reconstruction check and a TranAD transformer. **TranAD is the primary detector.** The other two act as cross-checks.

---

## Quick start

On a Mu2e gpvm (CPU is enough):

```bash
cd /exp/mu2e/app/users/<you>/CrossExperimentalAIDQM
source envs/setup_ana_env.sh          # must print "dqm_pipeline env ready (TranAD available)"

# Fetch a DQM source directly and analyse it
python -m dqm_pipeline --dqm-source crvreco,kpp_crv,file,005-001 --output results/crvreco_005-001

# ...or analyse a CSV you already have
python -m dqm_pipeline --csv dqm_pipeline/dqm_crv_anomaly_dataset_004-000_filtered.csv --output results/004-000
```

Open `results/<dir>/anomaly_report_*_anomalies_only.csv` and `results/<dir>/diagnostics/diagnostic_report.txt` first.

> **If you skip `setup_ana_env.sh`, TranAD is silently disabled.** When `torch` or `anldq` cannot be imported, the pipeline prints `⚠️ ANL package not available` (or `PyTorch not available`) and still runs. Every TranAD score and prediction is then 0, so the primary `anomaly` column contains no anomalies.

---

## Environment

The pipeline must be run from inside this repository. `tranad_model.py` imports the TranAD model from `anldq`, which lives in `../src`.

**Recommended: `envs/setup_ana_env.sh`.** This script uses the Mu2e cvmfs `ana` environment (Python 3.12, torch 2.5.1, pandas, scikit-learn, matplotlib). On first use it adds the missing packages to `envs/ana_extras/`, which is git-ignored:

| Package | Why it is needed |
|---------|------------------|
| `seaborn` | Correlation heatmap in `visualization.py` |
| `tensorboard` | Imported by `anldq/trainer/utils/logging.py` |
| `spt3g` (no-op stub) | Imported by the SPT-3G loader in `anldq`, which this pipeline never uses |

The script also puts the repository and the extras on `PYTHONPATH`, and sets `MPLBACKEND=Agg` and `PYTHONWARNINGS=ignore`.

**Alternative: pip.** From the repository root:

```bash
pip install -e ".[tranad]"    # pandas, numpy, scikit-learn, matplotlib, seaborn, torch
```

This also installs a `dqm_pipeline` console command. You still need `anldq` and its dependencies to be importable for TranAD to run.

**DQM MCP server.** `--dqm-source` and `export_from_dqm` fetch data from the DQM MCP server at `http://mu2eaigpvm01.fnal.gov:8001/mcp`. To use a different server, set `DQM_MCP_URL`. The client only uses the Python standard library, so no extra packages are required.

---

## How it works

```
 DQM MCP server ──► fetch matrix ──► drop static columns ──► ingest ──► preprocess
                    (dqm_mcp.py)     (filter_from_dqm.py)              │
                                                                       ├─► Z-score ┐
                                                         TranAD prep ──┼─► TranAD  ├─► vote ─► reports, diagnostics, plots
                                                      (tranad_prep.py) └─► PCA     ┘
```

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
   - `chronological`: the first 30% of rows are used for training and the next 10% for validation. This matches the original Genesis notebook.

   The TranAD inputs are then standardised using the training rows only.

### Phase 3 — Detection (`models.py`, `tranad_model.py`)

| Detector | What it does | Output columns | Flags a row when |
|----------|--------------|----------------|------------------|
| **TranAD** (primary) | Uses the `anldq` TranAD with window 4, `d_model` 32, 4 heads, `d_ff` 64 and 2 layers. It trains for 50 epochs with Adam (lr 1e-3) on the training rows, and each row is reconstructed from the 4 rows before it. The score is the mean squared reconstruction error. | `tranad_anomaly`, `tranad_reconstruction_error` | The score is above the `--threshold-percentile` (default **99.5**) of the scores on the held-out rows (`--threshold-mode heldout`) or on the training rows (`train`). |
| Z-score | Computes the absolute z-score of each feature, using the mean and standard deviation of the whole dataset. | `z_score_anomaly`, `z_score_max` | Any feature has \|z\| > 3. |
| PCA | Fits PCA on the whole dataset and computes the squared reconstruction error. | `pca_anomaly`, `pca_reconstruction_error` | The error is above its own 95th percentile, so 5% of rows are flagged by construction. |
| Ensemble | Adds the three 0/1 votes. | `ensemble_voting_score` (0–3) | The vote is at least `--voting-threshold` (default 2). The ensemble is only used for diagnostics and comparison, and it does **not** set `anomaly`. |

The first 4 rows (the window length) always have a TranAD score of 0.

### Phase 4 — Reports

`anomaly_report_<YYYYMMDD_HHMMSS>.csv` contains every row with all of its original features. It adds `calib_epoch`, `anomaly` (equal to `tranad_anomaly`), the prediction and score columns listed above, and `ensemble_voting_score`. `..._anomalies_only.csv` contains only the rows where `anomaly == 1`.

### Phase 5 — Diagnostics (`diagnostics.py`, saved to `diagnostics/`)

- `feature_statistics.csv` compares the mean, standard deviation, minimum and maximum of each feature between normal and anomalous rows.
- `extreme_features.csv` lists the features with extreme z-scores in anomalous rows.
- `diagnostic_report.txt` is a text summary. It covers how often the detectors agree, a breakdown of the votes, when anomalies occurred, and the features that best separate anomalous rows from normal ones. The same text is printed to the console.

See [`../docs/DIAGNOSTICS.md`](../docs/DIAGNOSTICS.md) to use `AnomalyDiagnostics` on its own.

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
python -m dqm_pipeline (--csv PATH | --dqm-source P,S,A,V) [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--csv PATH` | — | Analyse an existing filtered CSV |
| `--dqm-source P,S,A,V` | — | Fetch this source from the DQM MCP server first |
| `--dqm-group` | `CRV` | DQM value group to fetch |
| `--after`, `--before` | — | ISO UTC time window used with `--dqm-source` |
| `--output DIR` | `./results` | Output directory, created if it does not exist |
| `--tranad-features` | `response` | `response`, `changes` or `all` (see Phase 2b) |
| `--train-mode` | `stratified` | `stratified`, `random` or `chronological` (see Phase 2b) |
| `--threshold-mode` | `heldout` | Set the TranAD threshold on `heldout` or `train` rows |
| `--threshold-percentile` | `99.5` | Percentile of TranAD error used as the threshold. Lower values flag more rows. |
| `--voting-threshold` | `2.0` | Number of detectors that must agree for the ensemble |
| `--seed` | `0` | Random seed for selecting TranAD rows |
| `--apply-iqr` / `--iqr K` | off / `1.5` | Remove IQR outliers before detection. Use only to clean training data. |
| `--no-plots` | — | Skip Phase 6, which is the slowest step for wide datasets |

### Exporting and filtering on their own

```bash
# Raw matrix only (no detection)
python -m dqm_pipeline.export_from_dqm --source crvreco,kpp_crv,file,005-001 --output raw.csv \
    [--group CRV] [--after 2026-03-01T00:00:00] [--before ...] [--url ...]

# Drop static columns (reads dqm_crv_anomaly_dataset_004-000.csv in the current directory,
# writes ..._filtered.csv; the file names are set at the top of the script)
python dqm_pipeline/filter_from_dqm.py
```

---

## Python API

```python
from dqm_pipeline.run_pipeline import main, fetch_from_dqm

csv = fetch_from_dqm("crvreco,kpp_crv,file,005-001", "results/005-001")
main(csv, output_dir="results/005-001",
     tranad_features="response", train_mode="stratified",
     threshold_mode="heldout", threshold_percentile=99.5,
     generate_plots=True)
```

You can also run the phases one at a time:

```python
from dqm_pipeline import ingestion, preprocessing, models, tranad_prep
from dqm_pipeline.dqm_mcp import DqmMcpClient, fetch_matrix

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

## Files

| File | Role |
|------|------|
| `__main__.py`, `run_pipeline.py` | Command-line interface and the code that runs each phase in order |
| `dqm_mcp.py` | JSON-RPC client for the DQM MCP server, and `fetch_matrix()` |
| `export_from_dqm.py` | Command-line tool to export a DQM source to a CSV |
| `filter_from_dqm.py` | `filter_static_columns()` and a standalone script for filtering |
| `ingestion.py` | Phase 1 |
| `preprocessing.py` | Phase 2 |
| `tranad_prep.py` | Phase 2b: condition/response split, calibration epochs and row selection |
| `models.py` | Phase 3: Z-score and PCA detectors, and the ensemble |
| `tranad_model.py` | Wrapper around the `anldq` TranAD model |
| `diagnostics.py` | Phase 5 |
| `visualization.py` | Phase 6 |
| `dqm_crv_anomaly_dataset_004-000_filtered.csv` | Sample input: `crvreco 004-000`, about 1,100 rows |

The original prototype is in `../Genesis/`. It contains a `DqmTool`-based exporter, `filter.py` and `GenesisExample_v0.ipynb`.

---

## Known limitations

- **TranAD and the `anomaly` column depend on the environment.** If TranAD cannot be imported, nothing is flagged and the pipeline does not stop with an error. Check the console for the `⚠️` import warnings.
- **Z-score and PCA are fitted on the data they score.** Their thresholds are relative to the dataset being analysed, so the 5% PCA rate is fixed by construction. Use them as cross-checks, not as absolute tests.
- **PCA keeps every component** (`n_components=None`, and `variance_threshold` is not used), so it reconstructs the data exactly. In the `005-001` run, the reconstruction errors were about 1e-29, which is floating-point noise. The 5% of rows that PCA flags are therefore effectively random until a component count is set.
- **TranAD trains sequentially, one window at a time, with no batching.** This is fine for thousands of rows but becomes slow on much larger datasets.
- **The order of rows matters.** TranAD windows are built from consecutive rows, sorted by (run, subrun) when fetched from the MCP server. A CSV you supply yourself should be sorted the same way.
- **`filter_from_dqm.py` has hard-coded file names** when run as a script.
