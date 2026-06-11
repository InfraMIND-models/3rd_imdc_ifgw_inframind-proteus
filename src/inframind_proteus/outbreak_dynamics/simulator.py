"""Core renewal equation simulator.

Orchestrates one or more vectorised runs of the renewal model by combining:

- A reproduction number model (:class:`~.rt_models.BaseRT` subclass)
- A generation time model (:class:`~.generation_time.BaseGT` subclass)
- A negative-binomial observation (notification) model

Modes
-----
``"calibration"``
    Trajectories are scored against empirical data via WIS.
``"projection"``
    Forward-only run; no data comparison.

Configuration is held in the :class:`SimulationConfig` dataclass (and its
children :class:`TemporalConfig` and :class:`LocationConfig`).
Output is returned as a :class:`SimulationOutput` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numba as nb
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from .generation_time import BaseGT, ConstantGammaGT
from .initial_infections import (
    InitialInfectionsConfig,
    build_initial_infec_df,
    parse_initial_infections_config,
)
from .rt_models import BaseRT, LogisticRT, get_rt_model
from .sampling import SamplingConfig, parse_calibration_sampling_config
from .scoring import nbinom_ppf_cf, wis_score_vectorized, rmse_vectorized, nb_loglikelihood_vectorized
from .utils import parse_timestamp


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TemporalConfig:
    """Temporal settings for a simulation run.

    Attributes
    ----------
    zero_date:
        Reference date; t = 0 in float-based time stamping.
    sim_start:
        Date at which the simulation begins (may differ from ``zero_date``).
    step_dt:
        Duration of each simulation step in days (default: 7 = weekly).
    calibration_start:
        Start of the calibration window (calibration mode only).
    calibration_end:
        End of the calibration window (calibration mode only).
    """

    zero_date: pd.Timestamp
    sim_start: pd.Timestamp
    step_dt: int = 7
    calibration_start: pd.Timestamp | None = None
    calibration_end: pd.Timestamp | None = None


@dataclass
class LocationConfig:
    """Location identification for a simulation run.

    Attributes
    ----------
    location_id_variable:
        Name of the ID variable as used in mosqlimate data.
        Examples: ``"uf"``, ``"geocode"``, ``"regional_geocode"``,
        ``"macrorregional_geocode"``, ``"uf_code"``.
    location_id:
        Value of the location identifier.
    """

    location_id_variable: str
    location_id: str | int


@dataclass
class SimulationConfig:
    """Top-level configuration for the renewal simulator.

    Attributes
    ----------
    mode:
        ``"calibration"`` — compare trajectories against empirical data;
        ``"projection"`` — run forward with no data comparison.
    num_simulations:
        Number of parallel simulation trajectories.
    num_time_steps:
        Number of simulation time steps (excluding the warm-up window).
    gt_max:
        Maximum generation time in days (determines warm-up window size).
    temporal:
        Temporal settings (:class:`TemporalConfig`).
    location:
        Location settings (:class:`LocationConfig`).
    notif_nb_overdispersion:
        Overdispersion parameter for the negative-binomial observation model.
        Can be overridden per-simulation via ``params_df``.
    notif_scaling_factor:
        Scaling factor from abstract infections to expected reported cases.
        Can be overridden per-simulation via ``params_df``.
    case_beam_quantiles:
        Quantiles used to build the deterministic case prediction beam.
    sampling:
        Optional sampling settings parsed from ``sampling`` plus fixed
        parameter dictionaries used to build calibration ``params_df``.
    initial_infections:
        Initial infection seeding configuration.
    rng_seed:
        Global RNG seed for the observation model sampling.
    reference_population_size:
        Denominator for normalizing incidences per population size. Default
        is 100k (1E5), only change this if there is a clear reason.
    population_size:
        Population size for the simulated location. Required if using relative
        scaling scheme for the observation model. Defaults to the reference
        population.
    """

    mode: Literal["calibration", "projection"] = "projection"
    num_simulations: int = 1000
    num_time_steps: int = 50
    gt_max: int = 49  # days
    temporal: TemporalConfig = field(
        default_factory=lambda: TemporalConfig(
            zero_date=pd.Timestamp("2023-10-01"),
            sim_start=pd.Timestamp("2023-10-01"),
        )
    )
    location: LocationConfig = field(
        default_factory=lambda: LocationConfig(
            location_id_variable="uf",
            location_id="DF",
        )
    )
    notif_nb_overdispersion: float = 10.0
    notif_scaling_factor: float = 1.0
    case_beam_quantiles: list[float] = field(
        default_factory=lambda: [0.025, 0.25, 0.5, 0.75, 0.975]
    )
    sampling: SamplingConfig | None = None
    initial_infections: InitialInfectionsConfig = field(
        default_factory=InitialInfectionsConfig
    )
    rng_seed: int = 0

    reference_population_size: int = int(1E5)  # For normalizing incidence per population
    population_size: int = int(1E5)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimulationScoring:
    """
    Attributes
    ----------
    wis_array:
        Per-simulation WIS scores over the calibration window.  ``None`` in
        projection mode.  Shape ``(num_simulations, n_cal)`` where ``n_cal``
        is the number of observation timestamps that fall within
        ``[calibration_start, calibration_end]``.
    summary:
        Summary scores for all simulations, with one scalar for each score
        and for each simulation.
        Data frame shape is ``(num_simulations, num_scores)``,
        where ``num_scores`` is the number of calculated scores.
    """
    wis_array: np.ndarray
    summary: pd.DataFrame


@dataclass
class SimulationOutput:
    """Outputs from a simulation run.

    Attributes
    ----------
    infec_df:
        Abstract infection counts.
        Index = ``i_simulation``, columns = time step values in days.
    cases_df:
        Sampled reported case counts (observation model draws).
        Same shape as ``infec_df``.
    case_beam_df:
        Deterministic quantile case beam.
        MultiIndex ``(quantile, i_simulation)``, same time-step columns.
    scoring:
        Scoring results for calibration mode; ``None`` in projection mode.
    config:
        The :class:`SimulationConfig` used to produce these outputs.
    """

    infec_df: pd.DataFrame
    cases_df: pd.DataFrame
    case_beam_df: pd.DataFrame
    scoring: SimulationScoring | None
    config: SimulationConfig


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def _validate_population_size(population_size):
    if population_size is None:
        raise ValueError(
            "population_size must be provided when using "
            "notif_relative_scale"
        )
    if population_size < 0:
        raise ValueError(f"population_size must be non-negative. Got {population_size}.")


class RenewalSimulator:
    """Vectorised renewal equation simulator.

    Parameters
    ----------
    rt_model:
        An instance of a :class:`~.rt_models.BaseRT` subclass.
    gt_model:
        An instance of a :class:`~.generation_time.BaseGT` subclass.
    config:
        A :class:`SimulationConfig` instance.
    """

    def __init__(
        self,
        rt_model: BaseRT,
        gt_model: BaseGT,
        config: SimulationConfig,
    ) -> None:
        self.rt_model = rt_model
        self.gt_model = gt_model
        self.config = config

        self._step_dt = config.temporal.step_dt
        self._gt_max_steps = int(np.ceil(config.gt_max / self._step_dt))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        params_df: pd.DataFrame,
        initial_infec_df: pd.DataFrame,
        observations_sr: pd.Series | None = None,
    ) -> SimulationOutput:
        """Run the simulator.

        Parameters
        ----------
        params_df:
            Parameter table.  One row per simulation.  Must contain all
            columns required by ``rt_model``, plus
            ``notif_nb_overdispersion`` and ``notif_scaling_factor`` when
            those should vary across simulations. May also contain the optional
            ``notif_relative_scale`` column for relative scaling.
        initial_infec_df:
            Seed infection values for the warm-up window.
            Shape ``(num_simulations, gt_max_steps)``.
        observations_sr:
            Observed case counts indexed by time step.  Required in
            calibration mode; ignored in projection mode.

        Returns
        -------
        SimulationOutput
        """
        cfg = self.config
        num_sim = cfg.num_simulations
        num_steps = cfg.num_time_steps
        gt_steps = self._gt_max_steps
        step_dt = self._step_dt

        # ------------------------------------------------------------------
        # 1. Validate inputs
        # ------------------------------------------------------------------
        if params_df.shape[0] != num_sim:
            raise ValueError(
                f"params_df has {params_df.shape[0]} rows; expected {num_sim}"
            )
        if initial_infec_df.shape != (num_sim, gt_steps):
            raise ValueError(
                f"initial_infec_df must have shape ({num_sim}, {gt_steps}); "
                f"got {initial_infec_df.shape}"
            )
        if cfg.mode == "calibration" and observations_sr is None:
            raise ValueError("observations_sr is required in calibration mode")
        if cfg.mode == "calibration" and (
            cfg.temporal.calibration_start is None
            or cfg.temporal.calibration_end is None
        ):
            raise ValueError(
                "TemporalConfig.calibration_start and calibration_end must both be "
                "set when mode='calibration'"
            )

        _params = params_df.copy()

        # Fill config-default observation params when not in params_df
        # NOTE: Disabled since we want to enforce presence of these parameters.
        # if "notif_nb_overdispersion" not in _params.columns:
        #     _params["notif_nb_overdispersion"] = cfg.notif_nb_overdispersion
        # if "notif_scaling_factor" not in _params.columns:
        #     _params["notif_scaling_factor"] = cfg.notif_scaling_factor

        self.rt_model.validate_params(_params)

        # ------------------------------------------------------------------
        # 2. GT PMF — shape (num_steps, gt_steps), reversed-lag convention
        # ------------------------------------------------------------------
        gt_pmf = self.gt_model.get_pmf(
            gt_max_steps=gt_steps,
            num_time_steps=num_steps,
            step_dt=step_dt,
        )

        # ------------------------------------------------------------------
        # 3. R(t) — shape (num_sim, gt_steps + num_steps)
        #
        # The RT time grid is in days from zero_date.  The warm-up window
        # starts at (sim_start − gt_steps × step_dt) days from zero_date,
        # so R(t) parameters such as rt_logist_start are expressed as days
        # from zero_date independently of the warm-up size.
        # ------------------------------------------------------------------
        sim_start_day = float((cfg.temporal.sim_start - cfg.temporal.zero_date).days)
        t_start = sim_start_day - gt_steps * step_dt

        rt_vec = self.rt_model.generate(
            params_df=_params,
            num_time_steps=gt_steps + num_steps,
            step_dt=step_dt,
            t_start=t_start,
        )

        # ------------------------------------------------------------------
        # 4. Assemble infection array (warm-up pre-filled, rest zero)
        # ------------------------------------------------------------------
        infec_vec = np.concatenate(
            [
                initial_infec_df.to_numpy(dtype=float),
                np.zeros((num_sim, num_steps), dtype=float),
            ],
            axis=1,
        )

        # ------------------------------------------------------------------
        # 5. Core renewal loop
        # ------------------------------------------------------------------
        infec_vec = self._run_renewal_loop(
            infec_vec, rt_vec, gt_pmf, gt_steps, num_steps
        )

        # ------------------------------------------------------------------
        # 6. Observation model (crop warm-up first)
        # ------------------------------------------------------------------
        rng = np.random.default_rng(cfg.rng_seed)
        infec_sim = infec_vec[:, gt_steps:]  # (num_sim, num_steps)

        cases_vec, case_beam_df = self._apply_observation_model(
            infec_sim, _params, rng,
            population_size=cfg.population_size,
            reference_population_size=cfg.reference_population_size,
        )

        # ------------------------------------------------------------------
        # 7. Assign timestamp columns
        # ------------------------------------------------------------------
        sim_timestamps = pd.date_range(
            start=cfg.temporal.sim_start,
            periods=num_steps,
            freq=pd.tseries.offsets.Day(step_dt),
        )

        infec_df = pd.DataFrame(infec_sim, columns=sim_timestamps)
        infec_df.index.name = "i_simulation"
        infec_df.columns.name = "t"

        cases_df = pd.DataFrame(cases_vec, columns=sim_timestamps)
        cases_df.index.name = "i_simulation"
        cases_df.columns.name = "t"

        case_beam_df.columns = sim_timestamps
        case_beam_df.columns.name = "t"

        # ------------------------------------------------------------------
        # 8. Scoring (calibration mode only)
        # ------------------------------------------------------------------
        wis_array = None
        if cfg.mode == "calibration":
            observations_sr: pd.Series
            scoring = self.score_simulations(
                cfg, case_beam_df, observations_sr, params_df
            )

        else:
            scoring = None

        return SimulationOutput(
            infec_df=infec_df,
            cases_df=cases_df,
            case_beam_df=case_beam_df,
            scoring=scoring,
            config=cfg,
        )

    def build_initial_infec_df(self) -> pd.DataFrame:
        """Build the warm-up infection matrix using the config settings."""
        return build_initial_infec_df(
            num_simulations=self.config.num_simulations,
            gt_max_steps=self._gt_max_steps,
            step_dt=self._step_dt,
            initial_config=self.config.initial_infections,
        )

    @staticmethod
    def _run_renewal_loop(
        infec_vec: np.ndarray,
        rt_vec: np.ndarray,
        gt_pmf: np.ndarray,
        gt_max_steps: int,
        num_time_steps: int,
    ) -> np.ndarray:
        """Core renewal equation time loop (numba-compatible structure).

        Advances ``infec_vec`` in-place through ``num_time_steps`` steps.

        Parameters
        ----------
        infec_vec:
            Full infection array of shape
            ``(num_simulations, gt_max_steps + num_time_steps)``.
            The first ``gt_max_steps`` columns are pre-filled (warm-up).
        rt_vec:
            R(t) array of the same shape as ``infec_vec``.
        gt_pmf:
            Generation time PMF of shape ``(num_time_steps, gt_max_steps)``.
            Axis 1 follows the *reversed-lag* convention: index 0 is the
            largest lag (oldest), index ``-1`` is lag 1 (most recent step).
            This ordering aligns directly with the look-back window slices
            so no further reversal is needed inside the loop.
        gt_max_steps:
            Size of the warm-up / look-back window.
        num_time_steps:
            Number of steps to advance.

        Returns
        -------
        np.ndarray
            Updated ``infec_vec`` (modified in-place and returned).

        Notes
        -----
        This method intentionally avoids pandas objects and Python-level
        data structures so that a future ``@numba.njit`` decoration requires
        only minimal changes.

        Renewal equation at simulation step ``i`` (0-based)::

            I(t_i) = Σ_s  R(t_{i-s}) · I(t_{i-s}) · w_i(s)

        where the sum runs over ``s = 1..gt_max_steps`` (the look-back
        window), and ``w_i`` is the GT PMF row for step ``i``.
        """
        # Delegate to external function that can be numba-compiled
        return _run_renewal_loop_numba(
            infec_vec=infec_vec,
            rt_vec=rt_vec,
            gt_pmf=gt_pmf,
            gt_max_steps=gt_max_steps,
            num_time_steps=num_time_steps,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_observation_model(
        self,
        infec_vec: np.ndarray,
        params_df: pd.DataFrame,
        rng: np.random.Generator,
        population_size: int | None = None,
        reference_population_size: int = SimulationConfig.reference_population_size
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Apply the negative-binomial observation (notification) model.

        Parameters
        ----------
        infec_vec:
            Abstract infection counts.
            Shape ``(num_simulations, num_time_steps)`` (warm-up excluded).
        params_df:
            Parameter table with ``notif_nb_overdispersion`` and
            ``notif_scaling_factor`` columns. Optionally may contain
            ``notif_relative_scale`` for population-relative scaling, in which
            case ``notif_scaling_factor`` is overridden by the internal
            algorithm.
        rng:
            NumPy random generator.

        Returns
        -------
        cases_vec : np.ndarray
            Sampled case counts.  Shape ``(num_simulations, num_time_steps)``.
            dtype int64.
        case_beam_df : pd.DataFrame
            Deterministic case beam.
            MultiIndex ``(quantile, i_simulation)``, integer columns
            ``0..num_time_steps-1`` (renamed to timestamps by the caller).
        """
        required_cols = ["notif_nb_overdispersion", "notif_scaling_factor"]
        missing_cols = [c for c in required_cols if c not in params_df.columns]
        if missing_cols:
            raise ValueError(
                "_apply_observation_model requires params_df columns "
                f"{required_cols}; missing: {missing_cols}"
            )


        overdisp = params_df["notif_nb_overdispersion"].to_numpy()[:, np.newaxis]
        # Get alternative scaling scheme
        if "notif_relative_scale" in params_df.columns:
            # Population-relative scaling factor
            _validate_population_size(population_size)
            scale_f = (
                    params_df["notif_relative_scale"].to_numpy()[:, np.newaxis]
                    * population_size
                    / reference_population_size
            )
        else:
            # Directly provided scaling factor
            scale_f = params_df["notif_scaling_factor"].to_numpy()[:, np.newaxis]

        # Expected reported cases; clip to avoid negative expectations
        expectancy = np.clip(infec_vec * scale_f, 0.0, None)

        # NB success probability: p = n / (n + μ)
        # When expectancy = 0 → p = 1 → NB always draws 0 (correct)
        p = overdisp / (overdisp + expectancy)

        # Stochastic sample (integer counts)
        cases_vec = rng.negative_binomial(n=overdisp, p=p)

        # Deterministic quantile beam via Cornish-Fisher approximation
        beam_frames = [
            pd.DataFrame(
                nbinom_ppf_cf(q=q, n=overdisp, p=p, continuity=False)
            )
            for q in self.config.case_beam_quantiles
        ]
        case_beam_df = pd.concat(
            beam_frames,
            keys=self.config.case_beam_quantiles,
            names=["quantile", "i_simulation"],
        )

        return cases_vec, case_beam_df

    @staticmethod
    def score_simulations(
            cfg: SimulationConfig,
            case_beam_df: DataFrame,
            observations_sr: Series,
            params_df: pd.DataFrame,
    ) -> SimulationScoring:
        """Score simulated trajectories against observations via WIS.

        Computes per-simulation score metrics over the
        declared calibration window using the deterministic case beam
        quantiles produced by the observation model.

        Parameters
        ----------
        case_beam_df:
            Deterministic case prediction beam with MultiIndex
            ``(quantile, i_simulation)`` and timestamp columns.
        cfg:
            Simulation configuration. Uses
            ``cfg.temporal.calibration_start`` and
            ``cfg.temporal.calibration_end`` to define the scoring window.
        observations_sr:
            Observed case counts indexed by timestamp.

        Returns
        -------
        SimulationScoring
            Scoring container with ``wis_array`` of shape
            ``(num_simulations, n_cal)``, where ``n_cal`` is the number of
            observation timestamps inside the calibration window.

        Raises
        ------
        ValueError
            If no observations fall within the calibration window.
        ValueError
            If any calibration timestamp in ``observations_sr`` is absent
            from the simulation timestamps in ``case_beam_df``.
        """
        cal_start = cfg.temporal.calibration_start
        cal_end = cfg.temporal.calibration_end

        # Slice observations to the declared calibration window
        obs_cal = observations_sr.loc[
            (observations_sr.index >= cal_start)
            & (observations_sr.index <= cal_end)
            ]
        if obs_cal.empty:
            raise ValueError(
                f"No observation data within the calibration window "
                f"[{cal_start.date()}, {cal_end.date()}]"
            )

        # Every calibration timestamp must align exactly with a simulation step
        missing = obs_cal.index.difference(case_beam_df.columns)
        if not missing.empty:
            raise ValueError(
                f"{len(missing)} calibration timestamp(s) are absent from the "
                f"simulation period. Missing: {missing[:5].tolist()}"
            )

        simulations_df = case_beam_df[obs_cal.index]
        simulations_median_df = simulations_df.xs(0.5, level="quantile")
        summary_df = pd.DataFrame(
            {}, index=simulations_df.index.get_level_values("i_simulation").unique()
        )
        # ^ Shape: summary_df[i_simulation, score_name] = summary score scalar value
        # Expects that `simulations_df` is sorted by `i_simulation`.
        # Warning: Resulting WIS values may be randomized if this is not satisfied.

        # Weighted Interval Scores (WIS)
        wis_array = wis_score_vectorized(
            simulations_df=simulations_df,
            observations_sr=obs_cal,
        )
        summary_df["wis"] = wis_array.sum(axis=1)

        # Root Mean Squared Error - individual components
        rmse_array = rmse_vectorized(
            simulations_df=simulations_median_df,
            observations_sr=observations_sr,
        )
        summary_df["rmse"] = rmse_array

        # # Negative-binomial loglikelihood (TEMPORARILY DISABLED)
        # summary_df["nb_loglikelihood"] = nb_loglikelihood_vectorized(
        #     simulations_df=simulations_median_df,
        #     observations_sr=observations_sr,
        #     overdisp=params_df["notif_nb_overdispersion"].to_numpy(),
        # )

        scoring = SimulationScoring(
            wis_array=wis_array,
            summary=summary_df,
        )

        return scoring

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config_dict(
        cls,
        config_dict: dict,
    ) -> "RenewalSimulator":
        """Construct a simulator from a raw config dictionary (e.g. from YAML).

        Parameters
        ----------
        config_dict:
            Parsed YAML dictionary.

        Returns
        -------
        RenewalSimulator
        """
        defaults = SimulationConfig()

        sim_cfg = config_dict.get("simulation", {}) or {}
        temporal_cfg = config_dict.get("temporal", {}) or {}
        location_cfg = config_dict.get("location", {}) or {}
        obs_params = (config_dict.get("observation_model", {}) or {}).get("params", {}) or {}
        scoring_cfg = config_dict.get("scoring", {}) or {}
        rt_cfg = config_dict.get("reproduction_number", {}) or {}
        gt_cfg = config_dict.get("generation_time", {}) or {}
        sampling_cfg = parse_calibration_sampling_config(config_dict)
        initial_infections_cfg = parse_initial_infections_config(config_dict)

        rt_model_name = str(rt_cfg.get("model", "logistic")).strip().lower()
        rt_model = get_rt_model(rt_model_name)

        gt_model_name = str(gt_cfg.get("model", "constant_gamma")).strip().lower()
        gt_params = gt_cfg.get("params", {}) or {}
        if gt_model_name == "constant_gamma":
            gt_model = ConstantGammaGT(
                shape=float(gt_params.get("shape", 10.0)),
                scale=float(gt_params.get("scale", 1.8)),
            )
        else:
            raise ValueError(
                "Unsupported generation_time.model: "
                f"{gt_cfg.get('model')!r}. Supported models: ['constant_gamma']"
            )

        mode = sim_cfg.get("mode", defaults.mode)
        if mode not in ("calibration", "projection"):
            raise ValueError(
                f"simulation.mode must be 'calibration' or 'projection'; got {mode!r}"
            )

        def _to_timestamp(
            value: str | int | float | pd.Timestamp,
            zero_date: pd.Timestamp | None = None,
        ) -> pd.Timestamp:
            if isinstance(value, pd.Timestamp):
                return value
            return parse_timestamp(value, zero_date=zero_date)

        zero_date = _to_timestamp(
            temporal_cfg.get("zero_date", defaults.temporal.zero_date)
        )
        sim_start = _to_timestamp(
            temporal_cfg.get("sim_start", defaults.temporal.sim_start),
            zero_date=zero_date,
        )

        calibration_start_raw = temporal_cfg.get("calibration_start")
        calibration_end_raw = temporal_cfg.get("calibration_end")
        calibration_start = (
            _to_timestamp(calibration_start_raw, zero_date=zero_date)
            if calibration_start_raw is not None
            else defaults.temporal.calibration_start
        )
        calibration_end = (
            _to_timestamp(calibration_end_raw, zero_date=zero_date)
            if calibration_end_raw is not None
            else defaults.temporal.calibration_end
        )

        case_beam_quantiles = [
            float(q)
            for q in scoring_cfg.get(
                "case_beam_quantiles",
                sim_cfg.get("case_beam_quantiles", defaults.case_beam_quantiles),
            )
        ]

        config = SimulationConfig(
            mode=mode,
            num_simulations=int(sim_cfg.get("num_simulations", defaults.num_simulations)),
            num_time_steps=int(sim_cfg.get("num_time_steps", defaults.num_time_steps)),
            gt_max=int(sim_cfg.get("gt_max", defaults.gt_max)),
            temporal=TemporalConfig(
                zero_date=zero_date,
                sim_start=sim_start,
                step_dt=int(temporal_cfg.get("step_dt", defaults.temporal.step_dt)),
                calibration_start=calibration_start,
                calibration_end=calibration_end,
            ),
            location=LocationConfig(
                location_id_variable=location_cfg.get(
                    "location_id_variable", defaults.location.location_id_variable
                ),
                location_id=location_cfg.get("location_id", defaults.location.location_id),
            ),
            notif_nb_overdispersion=float(
                obs_params.get(
                    "notif_nb_overdispersion",
                    sim_cfg.get(
                        "notif_nb_overdispersion",
                        defaults.notif_nb_overdispersion,
                    ),
                )
            ),
            notif_scaling_factor=float(
                obs_params.get(
                    "notif_scaling_factor",
                    sim_cfg.get(
                        "notif_scaling_factor",
                        defaults.notif_scaling_factor,
                    ),
                )
            ),
            case_beam_quantiles=case_beam_quantiles,
            sampling=sampling_cfg,
            initial_infections=initial_infections_cfg,
            rng_seed=int(sim_cfg.get("rng_seed", defaults.rng_seed)),
        )

        return cls(rt_model=rt_model, gt_model=gt_model, config=config)


_nb_readonly_arr = nb.types.Array(nb.types.float64, 2, 'A', readonly=True)
@nb.njit(
    nb.float64[:,:](
        nb.float64[:,:],
        nb.float64[:,:],
        _nb_readonly_arr,
        nb.int64,
        nb.int64,
    ),
)
def _run_renewal_loop_numba(
    infec_vec: np.ndarray,
    rt_vec: np.ndarray,
    gt_pmf: np.ndarray,
    gt_max_steps: int,
    num_time_steps: int,
) -> np.ndarray:
    """Core renewal equation time loop (numba-compatible structure).

    Advances ``infec_vec`` in-place through ``num_time_steps`` steps.

    Parameters
    ----------
    infec_vec:
        Full infection array of shape
        ``(num_simulations, gt_max_steps + num_time_steps)``.
        The first ``gt_max_steps`` columns are pre-filled (warm-up).
    rt_vec:
        R(t) array of the same shape as ``infec_vec``.
    gt_pmf:
        Generation time PMF of shape ``(num_time_steps, gt_max_steps)``.
        Axis 1 follows the *reversed-lag* convention: index 0 is the
        largest lag (oldest), index ``-1`` is lag 1 (most recent step).
        This ordering aligns directly with the look-back window slices
        so no further reversal is needed inside the loop.
    gt_max_steps:
        Size of the warm-up / look-back window.
    num_time_steps:
        Number of steps to advance.

    Returns
    -------
    np.ndarray
        Updated ``infec_vec`` (modified in-place and returned).

    Notes
    -----
    This method intentionally avoids pandas objects and Python-level
    data structures so that a future ``@numba.njit`` decoration requires
    only minimal changes.

    Renewal equation at simulation step ``i`` (0-based)::

        I(t_i) = Σ_s  R(t_{i-s}) · I(t_{i-s}) · w_i(s)

    where the sum runs over ``s = 1..gt_max_steps`` (the look-back
    window), and ``w_i`` is the GT PMF row for step ``i``.
    """
    for i_sim_step in range(num_time_steps):
        i_full = gt_max_steps + i_sim_step
        # Look-back window: columns [i_sim_step, i_full)
        # Shape of each slice: (num_simulations, gt_max_steps)
        # gt_pmf[i_sim_step] broadcasts as (gt_max_steps,)
        infec_vec[:, i_full] = np.sum(
            rt_vec[:, i_sim_step:i_full]
            * infec_vec[:, i_sim_step:i_full]
            * gt_pmf[i_sim_step],
            axis=1,
        )
    return infec_vec