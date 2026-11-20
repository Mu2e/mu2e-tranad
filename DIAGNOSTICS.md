# Anomaly diagnostics (`diagnostics.py`)

`AnomalyDiagnostics` helps explain *why* rows were flagged. It looks at:

1. **Feature differences**: which features differ most between normal and anomalous rows
2. **Detector agreement**: how often Z-score, PCA and TranAD flag the same rows
3. **Voting breakdown**: how many detectors voted for each row
4. **Temporal patterns**: on which dates and at which hours anomalies occur
5. **Extreme features**: which features have |z| above a threshold in anomalous rows

## In the pipeline

Diagnostics run automatically as Phase 5 of `python -m mu2e_tranad` (see [README.md](README.md)). The results are written to `<output>/diagnostics/`:

| File | Content |
|------|---------|
| `diagnostic_report.txt` | Text summary of everything below. It is also printed to the console. |
| `feature_statistics.csv` | Mean and standard deviation of each feature for normal and anomalous rows, sorted by `mean_diff_pct` |
| `extreme_features.csv` | How many anomalous rows have \|z\| > 3 for each feature, and the largest \|z\| |

**Which rows count as anomalous.** The pipeline sets the `anomaly` column to the TranAD prediction. The feature-difference, extreme-feature and temporal sections use that column. The "Detected Anomalies" count at the top of `diagnostic_report.txt` is different: it counts rows where the ensemble vote reached `--voting-threshold`. The two numbers usually differ.

## Example output

These numbers come from the `crvreco 004-000` dataset (1,114 rows) with default options:

```
DETECTOR AGREEMENT
all_agree_anomaly: 1        # flagged by all three
all_agree_normal: 1079
only_z_score: 1
only_pca: 11
only_tranad: 3
z_score_pca: 18             # flagged by Z-score and PCA, not TranAD
z_score_tranad: 1
pca_tranad: 0

VOTING BREAKDOWN
zero_votes: 1079
one_vote: 15
two_votes: 19
three_votes: 1
split_decisions: 34         # one or two votes
pct_split: 3.05

TOP 5 FEATURES BY DIFFERENCE (Normal vs Anomaly)
CRV_crvDigisPerChannelAndEvent_CRVsector0_LeftEdge | Diff:     0.0113 (1586.77%)
CRV_crvCoincidencesClusters_CCR1 | Diff:     0.3359 (974.05%)
...
```

## Reading the results

- **Start with TranAD's rows** (`anomaly == 1`, or `anomaly_report_*_anomalies_only.csv`), since TranAD is the primary detector. Then check `feature_statistics.csv` to see which features drive them.
- **`mean_diff_pct` is relative to the normal mean.** Features whose normal mean is close to 0 can therefore show very large percentages from small absolute changes. Look at `mean_diff` as well.
- **Agreement with PCA means little.** PCA flags 5% of rows by construction, and these rows are effectively random (see *Known limitations* in the README). A 2/3 or 3/3 vote is therefore not by itself a sign of high confidence.
- **Z-score and PCA see more features than TranAD.** They use every active feature, including the calibration-constant and pedestal metrics. With the default `--tranad-features response`, TranAD does not see those metrics (see Phase 2b in the README). Rows flagged only by Z-score and PCA (`z_score_pca` above) can therefore come from a recalibration.
- **Temporal clusters** (several anomalies on the same date) are worth checking against the run log.

## Standalone use

```python
import pandas as pd
from mu2e_tranad.diagnostics import AnomalyDiagnostics

df = pd.read_csv("results/004-000/anomaly_report_<timestamp>.csv", parse_dates=["datetime"])
meta = {"run", "subrun", "timestamp", "datetime", "calib_epoch", "anomaly",
        "ensemble_voting_score"}
features = [c for c in df.columns
            if c not in meta and not c.startswith(("z_score_", "pca_", "tranad_"))]

diag = AnomalyDiagnostics(df, features)            # uses df["anomaly"] (TranAD)

z, pca, tranad = (df[c].to_numpy() for c in ("z_score_anomaly", "pca_anomaly", "tranad_anomaly"))
ensemble = (df["ensemble_voting_score"] >= 2).astype(int).to_numpy()

diag.feature_statistics()                          # DataFrame, sorted by mean_diff_pct
diag.get_top_features(n=5)                         # list of feature names
diag.outlier_feature_analysis(z_threshold=3.0)     # DataFrame
diag.temporal_patterns()                           # dict
diag.detector_agreement(z, pca, tranad)            # dict of counts
diag.voting_breakdown(z, pca, tranad, ensemble)    # dict of counts
print(diag.anomaly_summary_report(z, pca, tranad, ensemble))
diag.export_diagnostics(z, pca, tranad, ensemble, "my_diagnostics/")
```

Every method that takes an `anomaly_col` argument defaults to `"anomaly"`. To analyse another detector's rows, set that column first, for example `df["anomaly"] = df["z_score_anomaly"]`.

`get_anomaly_records(ensemble_pred, confidence=...)` returns the rows where `ensemble_pred == 1`. Its `confidence` argument is currently ignored.
