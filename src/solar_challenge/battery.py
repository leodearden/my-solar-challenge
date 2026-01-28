"""Battery storage configuration and modelling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryConfig:
    """Configuration for a battery storage system.

    Attributes:
        capacity_kwh: Total energy capacity in kilowatt-hours
        max_charge_kw: Maximum charging power in kilowatts
        max_discharge_kw: Maximum discharging power in kilowatts
        name: Optional identifier for the battery
    """

    capacity_kwh: float
    max_charge_kw: float = 2.5
    max_discharge_kw: float = 2.5
    name: str = ""

    def __post_init__(self) -> None:
        """Validate battery configuration parameters."""
        if self.capacity_kwh <= 0:
            raise ValueError(f"Capacity must be positive, got {self.capacity_kwh} kWh")
        if self.max_charge_kw <= 0:
            raise ValueError(
                f"Max charge power must be positive, got {self.max_charge_kw} kW"
            )
        if self.max_discharge_kw <= 0:
            raise ValueError(
                f"Max discharge power must be positive, got {self.max_discharge_kw} kW"
            )

    @classmethod
    def default_5kwh(cls) -> "BatteryConfig":
        """Create a typical UK domestic 5 kWh battery.

        Returns:
            BatteryConfig with 5 kWh, 2.5 kW charge/discharge
        """
        return cls(
            capacity_kwh=5.0,
            max_charge_kw=2.5,
            max_discharge_kw=2.5,
            name="5 kWh domestic battery"
        )
