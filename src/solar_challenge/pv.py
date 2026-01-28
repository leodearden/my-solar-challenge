"""PV system configuration and modelling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PVConfig:
    """Configuration for a photovoltaic system.

    Attributes:
        capacity_kw: Rated DC capacity in kilowatts
        azimuth: Panel orientation in degrees (0=North, 90=East, 180=South, 270=West)
        tilt: Panel tilt angle from horizontal in degrees (0=flat, 90=vertical)
        name: Optional identifier for the system
    """

    capacity_kw: float
    azimuth: float = 180.0  # South-facing default (UK optimal)
    tilt: float = 35.0  # Optimal for UK latitude
    name: str = ""

    def __post_init__(self) -> None:
        """Validate PV configuration parameters."""
        if self.capacity_kw <= 0:
            raise ValueError(f"Capacity must be positive, got {self.capacity_kw} kW")
        if not 0 <= self.azimuth <= 360:
            raise ValueError(f"Azimuth must be 0-360 degrees, got {self.azimuth}")
        if not 0 <= self.tilt <= 90:
            raise ValueError(f"Tilt must be 0-90 degrees, got {self.tilt}")

    @classmethod
    def default_4kw(cls) -> "PVConfig":
        """Create a typical UK domestic 4 kW system.

        Returns:
            PVConfig with 4 kW, south-facing, 35° tilt
        """
        return cls(
            capacity_kw=4.0,
            azimuth=180.0,
            tilt=35.0,
            name="4 kW domestic system"
        )
