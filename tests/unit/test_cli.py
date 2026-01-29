"""Tests for the CLI module."""

import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from solar_challenge.cli.main import app

runner = CliRunner()


class TestMainCLI:
    """Tests for main CLI commands."""

    def test_help(self) -> None:
        """Test --help shows usage."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Solar Challenge" in result.stdout
        assert "home" in result.stdout
        assert "fleet" in result.stdout
        assert "validate" in result.stdout
        assert "config" in result.stdout

    def test_version(self) -> None:
        """Test --version shows version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "solar-challenge version" in result.stdout

    def test_no_args_shows_help(self) -> None:
        """Test running with no args shows help (exit code 0 or 2 depending on Typer version)."""
        result = runner.invoke(app, [])
        # Typer's no_args_is_help can return 0 or 2 depending on version
        assert result.exit_code in (0, 2)
        assert "Usage" in result.stdout or "Solar Challenge" in result.stdout


class TestHomeCLI:
    """Tests for home subcommands."""

    def test_home_help(self) -> None:
        """Test home --help."""
        result = runner.invoke(app, ["home", "--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "quick" in result.stdout

    def test_home_run_help(self) -> None:
        """Test home run --help."""
        result = runner.invoke(app, ["home", "run", "--help"])
        assert result.exit_code == 0
        assert "--start" in result.stdout
        assert "--end" in result.stdout
        assert "--output" in result.stdout
        assert "--pv-kw" in result.stdout
        assert "--battery-kwh" in result.stdout

    def test_home_quick_help(self) -> None:
        """Test home quick --help."""
        result = runner.invoke(app, ["home", "quick", "--help"])
        assert result.exit_code == 0
        assert "PV_KW" in result.stdout
        assert "--days" in result.stdout


class TestFleetCLI:
    """Tests for fleet subcommands."""

    def test_fleet_help(self) -> None:
        """Test fleet --help."""
        result = runner.invoke(app, ["fleet", "--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "bristol-phase1" in result.stdout

    def test_fleet_run_help(self) -> None:
        """Test fleet run --help."""
        result = runner.invoke(app, ["fleet", "run", "--help"])
        assert result.exit_code == 0
        assert "CONFIG" in result.stdout
        assert "--start" in result.stdout
        assert "--output" in result.stdout

    def test_fleet_bristol_phase1_help(self) -> None:
        """Test fleet bristol-phase1 --help."""
        result = runner.invoke(app, ["fleet", "bristol-phase1", "--help"])
        assert result.exit_code == 0
        assert "--start" in result.stdout
        assert "--days" in result.stdout
        assert "100 homes" in result.stdout


class TestValidateCLI:
    """Tests for validate subcommands."""

    def test_validate_help(self) -> None:
        """Test validate --help."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "results" in result.stdout
        assert "config" in result.stdout

    def test_validate_results_help(self) -> None:
        """Test validate results --help."""
        result = runner.invoke(app, ["validate", "results", "--help"])
        assert result.exit_code == 0
        assert "CSV" in result.stdout
        assert "--pv-kw" in result.stdout

    def test_validate_config_help(self) -> None:
        """Test validate config --help."""
        result = runner.invoke(app, ["validate", "config", "--help"])
        assert result.exit_code == 0


class TestConfigCLI:
    """Tests for config subcommands."""

    def test_config_help(self) -> None:
        """Test config --help."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "show" in result.stdout
        assert "template" in result.stdout
        assert "locations" in result.stdout

    def test_config_template_help(self) -> None:
        """Test config template --help."""
        result = runner.invoke(app, ["config", "template", "--help"])
        assert result.exit_code == 0
        assert "home" in result.stdout.lower()
        assert "fleet" in result.stdout.lower()
        assert "scenario" in result.stdout.lower()

    def test_config_template_home(self) -> None:
        """Test config template home outputs YAML."""
        result = runner.invoke(app, ["config", "template", "home"])
        assert result.exit_code == 0
        # Check for key elements in the template
        assert "location:" in result.stdout or "latitude" in result.stdout

    def test_config_template_fleet(self) -> None:
        """Test config template fleet outputs YAML."""
        result = runner.invoke(app, ["config", "template", "fleet"])
        assert result.exit_code == 0
        assert "homes:" in result.stdout or "homes" in result.stdout

    def test_config_template_scenario(self) -> None:
        """Test config template scenario outputs YAML."""
        result = runner.invoke(app, ["config", "template", "scenario"])
        assert result.exit_code == 0
        assert "period:" in result.stdout or "period" in result.stdout

    def test_config_template_to_file(self) -> None:
        """Test config template writes to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test-config.yaml"
            result = runner.invoke(
                app, ["config", "template", "home", "-o", str(output_path)]
            )
            assert result.exit_code == 0
            assert output_path.exists()
            content = output_path.read_text()
            assert "location" in content or "pv" in content

    def test_config_template_invalid_type(self) -> None:
        """Test config template with invalid type."""
        result = runner.invoke(app, ["config", "template", "invalid"])
        assert result.exit_code == 1
        assert "Unknown template type" in result.stdout

    def test_config_locations(self) -> None:
        """Test config locations shows Bristol."""
        result = runner.invoke(app, ["config", "locations"])
        assert result.exit_code == 0
        assert "bristol" in result.stdout.lower()
        assert "51.45" in result.stdout
        assert "Europe/London" in result.stdout

    def test_config_show_valid_yaml(self) -> None:
        """Test config show with valid YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.yaml"
            config_path.write_text(
                """
home:
  pv:
    capacity_kw: 4.0
  load:
    annual_consumption_kwh: 3400.0
"""
            )
            result = runner.invoke(app, ["config", "show", str(config_path)])
            assert result.exit_code == 0
            assert "4.0" in result.stdout or "capacity_kw" in result.stdout

    def test_config_show_nonexistent_file(self) -> None:
        """Test config show with nonexistent file."""
        result = runner.invoke(app, ["config", "show", "/nonexistent/file.yaml"])
        assert result.exit_code != 0


class TestValidateConfig:
    """Tests for validate config command."""

    def test_validate_config_valid(self) -> None:
        """Test validate config with valid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "valid.yaml"
            config_path.write_text(
                """
home:
  pv:
    capacity_kw: 4.0
    tilt: 35.0
    azimuth: 180.0
  battery:
    capacity_kwh: 5.0
  load:
    annual_consumption_kwh: 3400.0
    household_occupants: 3
"""
            )
            result = runner.invoke(app, ["validate", "config", str(config_path)])
            assert result.exit_code == 0

    def test_validate_config_invalid_pv(self) -> None:
        """Test validate config with invalid PV capacity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.yaml"
            config_path.write_text(
                """
home:
  pv:
    capacity_kw: -4.0
"""
            )
            result = runner.invoke(app, ["validate", "config", str(config_path)])
            assert result.exit_code == 1
            assert "must be positive" in result.stdout

    def test_validate_config_invalid_tilt(self) -> None:
        """Test validate config with invalid tilt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.yaml"
            config_path.write_text(
                """
home:
  pv:
    capacity_kw: 4.0
    tilt: 100.0
"""
            )
            result = runner.invoke(app, ["validate", "config", str(config_path)])
            assert result.exit_code == 1
            assert "0-90" in result.stdout

    def test_validate_config_warning_high_consumption(self) -> None:
        """Test validate config warns about high consumption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "warning.yaml"
            config_path.write_text(
                """
home:
  pv:
    capacity_kw: 4.0
  load:
    annual_consumption_kwh: 50000.0
"""
            )
            result = runner.invoke(app, ["validate", "config", str(config_path)])
            # Should pass but with warning
            assert result.exit_code == 0
            assert "WARNING" in result.stdout or "seems high" in result.stdout


class TestErrorHandling:
    """Tests for CLI error handling."""

    def test_missing_config_file(self) -> None:
        """Test error when config file doesn't exist."""
        result = runner.invoke(app, ["home", "run", "/nonexistent/config.yaml"])
        # Typer handles file existence check
        assert result.exit_code != 0

    def test_invalid_yaml_syntax(self) -> None:
        """Test error with invalid YAML syntax."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.yaml"
            config_path.write_text("invalid: yaml: syntax: [")
            result = runner.invoke(app, ["config", "show", str(config_path)])
            assert result.exit_code != 0


class TestCLIOutputFormats:
    """Tests for CLI output generation."""

    def test_template_generates_valid_yaml(self) -> None:
        """Test that generated templates are valid YAML."""
        import yaml

        for template_type in ["home", "fleet", "scenario"]:
            result = runner.invoke(app, ["config", "template", template_type])
            assert result.exit_code == 0

            # Extract YAML content (may have ANSI codes from Rich)
            # The actual content is in the output
            # For CLI output, we need to write to file to get clean YAML
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / f"{template_type}.yaml"
                result = runner.invoke(
                    app, ["config", "template", template_type, "-o", str(output_path)]
                )
                assert result.exit_code == 0

                content = output_path.read_text()
                # Should be valid YAML
                parsed = yaml.safe_load(content)
                assert parsed is not None
                assert isinstance(parsed, dict)


class TestLocationParsing:
    """Tests for location parsing in CLI."""

    def test_parse_bristol_preset(self) -> None:
        """Test parsing 'bristol' preset."""
        from solar_challenge.cli.utils import parse_location

        loc = parse_location("bristol")
        assert loc.latitude == 51.45
        assert loc.longitude == -2.58

    def test_parse_bristol_case_insensitive(self) -> None:
        """Test parsing 'BRISTOL' is case-insensitive."""
        from solar_challenge.cli.utils import parse_location

        loc = parse_location("BRISTOL")
        assert loc.latitude == 51.45

    def test_parse_lat_lon(self) -> None:
        """Test parsing lat,lon format."""
        from solar_challenge.cli.utils import parse_location

        loc = parse_location("51.50,-0.12")
        assert loc.latitude == 51.50
        assert loc.longitude == -0.12

    def test_parse_lat_lon_altitude(self) -> None:
        """Test parsing lat,lon,altitude format."""
        from solar_challenge.cli.utils import parse_location

        loc = parse_location("51.50,-0.12,25")
        assert loc.latitude == 51.50
        assert loc.longitude == -0.12
        assert loc.altitude == 25.0

    def test_parse_invalid_location(self) -> None:
        """Test parsing invalid location raises error."""
        from solar_challenge.cli.utils import parse_location

        with pytest.raises(ValueError, match="Invalid location"):
            parse_location("invalid")

    def test_parse_invalid_coordinates(self) -> None:
        """Test parsing invalid coordinates raises error."""
        from solar_challenge.cli.utils import parse_location

        with pytest.raises(ValueError, match="Invalid coordinates"):
            parse_location("abc,def")
