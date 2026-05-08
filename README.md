# infonet

Information-theoretic network analysis for univariate and multivariate time series.

## Layout

- `experiments/`: simulation study scripts
- `results/`: simulation study results
- `experiments/replication/`: continuous-time analysis replication
- `src/infonet/core/`: shared abstractions
- `src/infonet/jidt/`: JVM/JIDT runtime helpers
- `src/infonet/univariate/`: 1d pipeline (AR1, NAR; AIS estimators)
- `src/infonet/multivariate/`: nd pipeline (VAR1, NVAR; OLS, TE, IDTxl)
 
## Requirements

- missing

## Install

```bash
pip install -e .
```

## JIDT classpath

JIDT is loaded from the path in the `INFONET_JIDT_JAR` environment variable, defaulting to `/home/r-env/jidt/infodynamics.jar`. Override before importing any estimator:

```bash
export INFONET_JIDT_JAR=/path/to/infodynamics.jar
```

## Univariate Workflow

```sh
# Run sweeps (each writes its own combined.parquet)
python experiments/univariate/run_sweep_parameter.py
python experiments/univariate/run_sweep_n.py
python experiments/univariate/run_sweep_bins.py
python experiments/univariate/run_sweep_joint.py
python experiments/univariate/run_sweep_stationarity.py

# Generate plots from whichever combined parquets exist
python experiments/univariate/plots/make_plots.py
```

## Multivariate Workflow

```sh
# Run sweep (not recommended because it takes long)
python experiments/multivariate/run_sweep.py

# Combine checkpoints (automatica at end of sweep, only for intermediate)
python experiments/multivariate/run_combine.py

# Make plots (run if `results/multivariate/combined.parquet` exists)
python experiments/multivariate/make_accuracy_plots.py
python experiments/multivariate/make_rank_plots.py
```