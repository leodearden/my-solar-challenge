"""Battery dispatch strategy framework.

This module provides an abstract framework for battery dispatch strategies,
allowing different algorithms to decide when and how to charge/discharge
batteries based on generation, demand, tariffs, and other factors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class DispatchDecision:
    """Decision from a dispatch strategy for a single timestep.

    Attributes:
        charge_kw: Requested battery charging power in kW (non-negative)
        discharge_kw: Requested battery discharge power in kW (non-negative)
    """

    charge_kw: float
    discharge_kw: float

    def __post_init__(self) -> None:
        """Validate dispatch decision parameters."""
        if self.charge_kw < 0:
            raise ValueError(
                f"Charge power must be non-negative, got {self.charge_kw} kW"
            )
        if self.discharge_kw < 0:
            raise ValueError(
                f"Discharge power must be non-negative, got {self.discharge_kw} kW"
            )
        if self.charge_kw > 0 and self.discharge_kw > 0:
            raise ValueError(
                "Cannot charge and discharge simultaneously: "
                f"charge={self.charge_kw} kW, discharge={self.discharge_kw} kW"
            )


class DispatchStrategy(ABC):
    """Abstract base class for battery dispatch strategies.

    A dispatch strategy determines when and how to charge/discharge a battery
    based on current conditions (generation, demand, SOC, time, tariffs, etc.).

    Subclasses must implement decide_action() to return a DispatchDecision
    for each timestep.
    """

    @abstractmethod
    def decide_action(
        self,
        timestamp: datetime,
        generation_kw: float,
        demand_kw: float,
        battery_soc_kwh: float,
        battery_capacity_kwh: float,
        timestep_minutes: float = 1.0,
    ) -> DispatchDecision:
        """Decide battery charge/discharge action for current timestep.

        Args:
            timestamp: Current simulation timestamp
            generation_kw: PV generation power in kW
            demand_kw: Demand/consumption power in kW
            battery_soc_kwh: Current battery state of charge in kWh
            battery_capacity_kwh: Total battery capacity in kWh
            timestep_minutes: Duration of timestep in minutes

        Returns:
            DispatchDecision specifying charge_kw or discharge_kw

        Raises:
            ValueError: If inputs are invalid (negative values, etc.)
        """
        pass
