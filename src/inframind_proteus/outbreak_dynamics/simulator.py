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

import numpy as np
import pandas as pd

from .generation_time import BaseGT
from .rt_models import BaseRT


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
    rng_seed:
        Global RNG seed for the observation model sampling.
    """

    mode: Literal["calibration", "projection"] = "projection"
    num_simulations: int = 1000
    num_time_steps: int = 50
    gt_max: int = 49  # days
    temporal: TemporalConfig = field(
        default_factory=lambda: TemporalConfig(
            zero_date=pd.Timestamp("2023-10-02"),
            sim_start=pd.Timestamp("2023-10-02"),
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
    rng_seed: int = 0


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

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
    wis_array:
        Per-simulation WIS scores over time.  ``None`` in projection mode.
        Shape ``(num_simulations, num_time_steps)``.
    config:
        The :class:`SimulationConfig` used to produce these outputs.
    """

    infec_df: pd.DataFrame
    cases_df: pd.DataFrame
    case_beam_df: pd.DataFrame
    wis_array: np.ndarray | None
    config: SimulationConfig


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

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
            columns required by ``rt_model`` and ``gt_model``, plus
            ``notif_nb_overdispersion`` and ``notif_scaling_factor`` when
            those should vary across simulations.
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
        # TODO: implement
        #   1. Validate shapes and required columns
        #   2. Generate GT PMF via self.gt_model.get_pmf(...)
        #   3. Generate R(t) array via self.rt_model.generate(...)
        #   4. Assemble full infec_vec (warm-up + zero-filled future)
        #   5. Run core renewal loop via _run_renewal_loop(...)
        #   6. Apply observation model via _apply_observation_model(...)
        #   7. If calibration mode, score with wis_score_vectorized(...)
        #   8. Pack results into SimulationOutput and return
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_renewal_loop(
        infec_vec: np.ndarray,
        rt_vec: np.ndarray,
        gt_pmf_reverse: np.ndarray,
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
        gt_pmf_reverse:
            Reversed GT PMF of shape ``(num_simulations, gt_max_steps)``.
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
        """
        # TODO: implement — port the time loop from
        #   proto_renewal_model.ProtoDynModel.run_multiple
        raise NotImplementedError

    def _apply_observation_model(
        self,
        infec_vec: np.ndarray,
        params_df: pd.DataFrame,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Apply the negative-binomial observation (notification) model.

        Parameters
        ----------
        infec_vec:
            Abstract infection counts.
            Shape ``(num_simulations, num_time_steps)`` (warm-up excluded).
        params_df:
            Parameter table with ``notif_nb_overdispersion`` and
            ``notif_scaling_factor`` columns.
        rng:
            NumPy random generator.

        Returns
        -------
        cases_vec : np.ndarray
            Sampled case counts.  Shape ``(num_simulations, num_time_steps)``.
        case_beam_df : pd.DataFrame
            Deterministic case beam.
            MultiIndex ``(quantile, i_simulation)``, columns = time steps.
        """
        # TODO: implement
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config_dict(
        cls,
        config_dict: dict,
        rt_model: BaseRT,
        gt_model: BaseGT,
    ) -> "RenewalSimulator":
        """Construct a simulator from a raw config dictionary (e.g. from YAML).

        Parameters
        ----------
        config_dict:
            Parsed YAML dictionary.
        rt_model:
            R(t) model instance.
        gt_model:
            GT model instance.

        Returns
        -------
        RenewalSimulator
        """
        # TODO: implement — build SimulationConfig (and children) from dict,
        #       then call __init__
        raise NotImplementedError
