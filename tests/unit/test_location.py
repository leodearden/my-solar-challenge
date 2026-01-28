"""Tests for Location class."""

import pytest
from solar_challenge.location import Location


class TestLocationBasics:
    """Test basic Location functionality."""

    def test_create_location_with_all_params(self):
        """Location can be created with all parameters."""
        loc = Location(
            latitude=51.5,
            longitude=-0.1,
            timezone="Europe/London",
            altitude=20.0,
            name="London"
        )
        assert loc.latitude == 51.5
        assert loc.longitude == -0.1
        assert loc.timezone == "Europe/London"
        assert loc.altitude == 20.0
        assert loc.name == "London"

    def test_create_location_with_defaults(self):
        """Location uses sensible defaults."""
        loc = Location(latitude=51.0, longitude=-2.0)
        assert loc.timezone == "Europe/London"
        assert loc.altitude == 0.0
        assert loc.name == ""

    def test_location_is_frozen(self):
        """Location is immutable (frozen dataclass)."""
        loc = Location(latitude=51.0, longitude=-2.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            loc.latitude = 52.0  # type: ignore


class TestBristolLocation:
    """Test Bristol default location."""

    def test_bristol_factory(self):
        """Bristol factory creates correct location."""
        loc = Location.bristol()
        assert loc.latitude == 51.45
        assert loc.longitude == -2.58
        assert loc.timezone == "Europe/London"
        assert loc.altitude == 11.0
        assert loc.name == "Bristol, UK"

    def test_bristol_coordinates(self):
        """Bristol coordinates are correct (51.45°N, 2.58°W)."""
        loc = Location.bristol()
        # Latitude is positive (North)
        assert loc.latitude > 0
        # Longitude is negative (West)
        assert loc.longitude < 0


class TestLocationHashability:
    """Test that Location can be used as dict key."""

    def test_location_is_hashable(self):
        """Location instances are hashable."""
        loc = Location.bristol()
        assert hash(loc) is not None

    def test_location_as_dict_key(self):
        """Location can be used as dictionary key."""
        loc1 = Location.bristol()
        loc2 = Location(latitude=52.0, longitude=-1.0)

        data = {
            loc1: "Bristol data",
            loc2: "Other data"
        }

        assert data[loc1] == "Bristol data"
        assert data[loc2] == "Other data"

    def test_equal_locations_same_hash(self):
        """Equal locations have same hash."""
        loc1 = Location.bristol()
        loc2 = Location.bristol()

        assert loc1 == loc2
        assert hash(loc1) == hash(loc2)


class TestLocationValidation:
    """Test parameter validation."""

    def test_invalid_latitude_too_high(self):
        """Latitude > 90 raises error."""
        with pytest.raises(ValueError, match="Latitude"):
            Location(latitude=91.0, longitude=0.0)

    def test_invalid_latitude_too_low(self):
        """Latitude < -90 raises error."""
        with pytest.raises(ValueError, match="Latitude"):
            Location(latitude=-91.0, longitude=0.0)

    def test_invalid_longitude_too_high(self):
        """Longitude > 180 raises error."""
        with pytest.raises(ValueError, match="Longitude"):
            Location(latitude=0.0, longitude=181.0)

    def test_invalid_longitude_too_low(self):
        """Longitude < -180 raises error."""
        with pytest.raises(ValueError, match="Longitude"):
            Location(latitude=0.0, longitude=-181.0)

    def test_boundary_values_valid(self):
        """Boundary values are valid."""
        # Should not raise
        Location(latitude=90.0, longitude=180.0)
        Location(latitude=-90.0, longitude=-180.0)
