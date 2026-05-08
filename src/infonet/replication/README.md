# Replication study

This directory holds a separate replication study of continuous-time analysis that's distinct from the univariate/multivariate pipelines in `src/infonet/`. 

It may import a few pieces (e.g. specific generators) from the package, but its logic and goals are independent.

## Inputs

The plots expect two sets of CSVs under `data/`:

- `data/Results_OpenMx/`: Batra's replicated OpenMx estimates.
- `data/batra_results_replication.csv`: Corrected replication.

## Outputs

Run from the repo root:
```sh
python experiments/replication/run_plots_batra.py
python experiments/replication/run_plots_theory.py
```
Figures are written to `plots/replication/`.