# InfraMIND Proteus - A predictive model of dengue cases

InfraMIND Proteus is the first of the InfraMIND (Infrastructure for Modeling Infectious Diseases) family of models. Proteus was specifically developed for the 3rd edition of the [Infodengue-Mosqlimate Dengue Challenge](https://sprint.mosqlimate.org/) (IMDC).

GitHub repository: [github.com/InfraMIND-models/3rd_imdc_ifgw_inframind-proteus](https://github.com/InfraMIND-models/3rd_imdc_ifgw_inframind-proteus).

## Team and contributors

- Paulo Cesar Ventura (IFGW-UNICAMP, Brazil)
- Alberto Aleta (BIFI-Unizar, Spain)
- Marco Fernandez (BIFI-Unizar, Spain)

# Quick start

Requirements:
- python
- [Astral UV](https://docs.astral.sh/uv/)
- git

Create a fork and clone it locally:
```bash
git clone https://github.com/<your_username>/3rd_imdc_ifgw_inframind-proteus.git
cd 3rd_imdc_ifgw_inframind-proteus
```

Setup the development environment (assumes you have uv).
Follow the instructions as prompted.
```
python3 setup_env.py
```

| Obs: To manually configure a uv project with development dependencies, run `uv sync --extra dev --group dev`.'

Activate the environment (do that every new session):
```
source .venv/bin/activate

```

Alternatively, you can run without activating the environment with `uv run [commands]`.
For example, running  `uv run jupyter notebook`
will start a jupyter server in the configured environment.

## Data storing and versioning

This project uses [DVC](https://dvc.org/) to manage large data files, 
which are stored in an internal server. For more information, please check 
the [internal server documentation](docs/internal_server.md).

To reproduce results without access to the server, you have to 
paste the IMDC-provided data files in the `data/data_imdc_2026`, 
keeping the same file names. Other files managed by the server
can be recreated using code in this repository.

# Required information


> [!NOTE]
> The model has two major components: Outbreak Features and Outbreak Dynamics. 
> Each component contains more detailed information in its own README file:
> - **Outbreak Features component** (statistical macro-feature predictions per UF/year): [`src/inframind_proteus/outbreak_features/README.md`](https://github.com/InfraMIND-models/3rd_imdc_ifgw_inframind-proteus/tree/main/src/inframind_proteus/outbreak_features/README.md)
> - **Outbreak Dynamics component** (mechanistic weekly trajectory predictions) [`src/inframind_proteus/outbreak_dynamics/README.md`](https://github.com/InfraMIND-models/3rd_imdc_ifgw_inframind-proteus/tree/main/src/inframind_proteus/outbreak_dynamics/README.md)

## 1. Team and Contributors

- Paulo Cesar Ventura (IFGW-UNICAMP, Brazil)
- Alberto Aleta (BIFI-Unizar, Spain), 
- Marco Fernandez (BIFI-Unizar, Spain).

## 2. Repository Structure

General repository structure:

```
3rd_imdc_ifgw_inframind-proteus/
├── README.md                         # Project overview and quick start
├── pyproject.toml                    # Dependencies and CLI entry points
├── setup_env.py                      # Environment bootstrap helper
├── LICENSE
├── configs/                          # Default YAML configs for calibration/projection workflows
├── data/
│   ├── data_imdc_2026/               # IMDC-provided raw inputs
│   ├── demographic/                  # Processed demographic tables (UF-level)
│   └── disease/                      # Processed dengue time series (UF weekly)
├── docs/                             # Project documentation and notes
├── img/                              # Static images/figures
├── outputs/                          # Intermediate and final pipeline outputs
├── predictions/                      # Outbreak-features exported prediction files
├── src/
│   ├── inframind_proteus/
│   │   ├── outbreak_features/        # Statistical macro-feature model component
│   │   ├── outbreak_dynamics/        # Mechanistic renewal model component
│   │   └── empirical_data/           # Data-access helpers and contracts
│   ├── scripts/                      # Pipeline scripts (calibrate/process/project)
│   └── prototypes/                   # Experimental notebooks/scripts
└── tests/                            # Tests and validation notebooks
```

Component-specific structure described in detail in each component's readme file.


## 3. Libraries and Dependencies

The following python libraries are core dependencies: 
catboost,
dvc-ssh,
epiweeks, 
matplotlib,
mosqlient,
numba,
numpy, 
pandas,
pathos, 
scikit-learn, 
scipy, 
statsmodels, 
torch

Full dependency list in `pyproject.toml`. 


## 4. Data and Variables

**Outbreak Features.** This component uses IMDC datasets  including dengue weekly notifications, DataSUS population, ERA5 observed climate, Copernicus seasonal climate forecast, ocean teleconnections, chikungunya cases (cross-disease signal), and the regional-health mapping table. The model is built on incidence/rate targets (cases per 100k), converted to counts at export time using UF population, and the feature matrix is assembled in `src/inframind_proteus/outbreak_features/season_features.py` and `src/inframind_proteus/outbreak_features/features.py` with epidemiological history, immunity proxies, static demographic/environmental variables, climate summaries, and forecast features.

**Outbreak Dynamics.** This component uses processed UF-level dengue case time series and UF population metadata derived from IMDC files (`src/scripts/prepare_dengue_time_series.py` and `src/make_uf_table.py`). 

## 5. Model Training

**Outbreak Features.** Training is rolling-origin by projection year: for each target year `Y`, models are fit on completed seasons before `Y` and issue predictions from EW25 covariates. Size targets (`case_attack_rate`, `peak_amplitude`) are trained with `CatBoostAnomModel` (climatology-anomaly strategy with quantile models), while timing (`peak_week`) is generated with `SarimaxModel`; orchestration is implemented in `src/inframind_proteus/outbreak_features/export.py` and executed with `uv run run-outbreak-features --n-samples 500`.

**Outbreak Dynamics.** Training/calibration is a three-stage Bayesian workflow per `(UF, year)` in `src/scripts/calibrate_3rd_imdc/`: broad Sobol exploration (stage 1), reduced two-parameter posterior/KDE refinement (stage 2), and coverage-based overdispersion adjustment (stage 3). Projection generation is then split into posterior processing with outbreak-feature updates (`src/scripts/process_data_for_projections.py`) and forward simulations (`src/scripts/project_3rd_imdc.py`), run via `uv run calibrate-3rd-imdc`, `uv run process-data-for-projections`, and `uv run project-3rd-imdc`.

## 6. Data Usage Restriction

**Outbreak Features.** The issue point is fixed at EW25: only variables available up to `t0 = EW25` are used for each target season, and all fitted transforms (including climatology/scaling and early-stopping splits) are learned from training seasons only. SARIMAX is fit on weeks up to EW25 and then forecast over the required EW41 to EW40 horizon.

**Outbreak Dynamics.** Calibration windows are defined to avoid leakage and the projection pipeline only uses calibration years strictly earlier than the projection year (`year < projection_year`) when building priors. Calibration scoring period runs until EW25 to keep it standard for all years.

## 7. Predictive Uncertainty

**Outbreak Features.** Uncertainty for size targets is sampled from a predictive quantile ladder (piecewise-linear inverse CDF with tail extrapolation), while timing uncertainty is obtained from stochastic SARIMAX trajectories using the sampled peak week (argmax) as output. The component exports stochastic samples per `(UF, year, feature)`, and count outputs are rounded to non-negative integers after sampling.

**Outbreak Dynamics.** Uncertainty is represented through posterior parameter distributions and stochastic observation noise: projected trajectories are generated from posterior parameter samples and one negative-binomial stochastic case trajectory is sampled per parameter set. Predictive intervals are computed empirically as trajectory quantiles at each week, including median and required interval levels (50/80/90/95).

## 8. References

The InfraMIND-Proteus model has no associated publications yet.

