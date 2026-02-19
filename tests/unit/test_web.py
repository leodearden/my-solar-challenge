"""Tests for the Flask web dashboard module."""

import pytest
from flask import Flask
from flask.testing import FlaskClient

from solar_challenge.web.app import create_app


@pytest.fixture
def app() -> Flask:
    """Create a test Flask application."""
    test_app = create_app(
        test_config={
            "TESTING": True,
            "SECRET_KEY": "test-secret-key",
            "WTF_CSRF_ENABLED": False,
        }
    )
    return test_app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Create a Flask test client."""
    return app.test_client()


class TestIndexRoute:
    """Tests for the GET / route."""

    def test_get_index_returns_200(self, client: FlaskClient) -> None:
        """Test GET / returns HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_get_index_returns_html(self, client: FlaskClient) -> None:
        """Test GET / returns HTML content."""
        response = client.get("/")
        assert b"html" in response.data.lower() or b"<" in response.data

    def test_get_index_content_type_html(self, client: FlaskClient) -> None:
        """Test GET / returns text/html content type."""
        response = client.get("/")
        assert "text/html" in response.content_type


class TestSimulateRoute:
    """Tests for the POST /simulate route."""

    def test_simulate_valid_params_returns_200(self, client: FlaskClient) -> None:
        """Test POST /simulate with valid params returns 200."""
        response = client.post(
            "/simulate",
            data={
                "pv_kw": "4.0",
                "battery_kwh": "0",
                "occupants": "3",
                "location": "bristol",
                "days": "1",
            },
        )
        assert response.status_code == 200

    def test_simulate_valid_params_returns_html(self, client: FlaskClient) -> None:
        """Test POST /simulate with valid params returns HTML content."""
        response = client.post(
            "/simulate",
            data={
                "pv_kw": "4.0",
                "battery_kwh": "0",
                "occupants": "3",
                "location": "bristol",
                "days": "1",
            },
        )
        assert response.status_code == 200
        assert b"<" in response.data
        assert "text/html" in response.content_type

    def test_simulate_with_battery_returns_200(self, client: FlaskClient) -> None:
        """Test POST /simulate with battery capacity returns 200."""
        response = client.post(
            "/simulate",
            data={
                "pv_kw": "5.0",
                "battery_kwh": "10.0",
                "occupants": "2",
                "location": "london",
                "days": "1",
            },
        )
        assert response.status_code == 200

    def test_simulate_pv_too_small_returns_400(self, client: FlaskClient) -> None:
        """Test POST /simulate with pv_kw below minimum returns 400."""
        response = client.post(
            "/simulate",
            data={
                "pv_kw": "0.1",
                "battery_kwh": "0",
                "occupants": "3",
                "location": "bristol",
                "days": "1",
            },
        )
        assert response.status_code == 400

    def test_simulate_pv_too_large_returns_400(self, client: FlaskClient) -> None:
        """Test POST /simulate with pv_kw above maximum returns 400."""
        response = client.post(
            "/simulate",
            data={
                "pv_kw": "99.9",
                "battery_kwh": "0",
                "occupants": "3",
                "location": "bristol",
                "days": "1",
            },
        )
        assert response.status_code == 400

    def test_simulate_negative_battery_returns_400(self, client: FlaskClient) -> None:
        """Test POST /simulate with negative battery capacity returns 400."""
        response = client.post(
            "/simulate",
            data={
                "pv_kw": "4.0",
                "battery_kwh": "-5.0",
                "occupants": "3",
                "location": "bristol",
                "days": "1",
            },
        )
        assert response.status_code == 400

    def test_simulate_invalid_pv_returns_html_error(self, client: FlaskClient) -> None:
        """Test POST /simulate with invalid params returns HTML error body."""
        response = client.post(
            "/simulate",
            data={
                "pv_kw": "0.1",
                "battery_kwh": "0",
                "occupants": "3",
                "location": "bristol",
                "days": "1",
            },
        )
        assert response.status_code == 400
        assert b"<" in response.data
        assert "text/html" in response.content_type

    def test_simulate_json_valid_params_returns_200(self, client: FlaskClient) -> None:
        """Test POST /simulate with JSON body and valid params returns 200."""
        response = client.post(
            "/simulate",
            json={
                "pv_kw": 4.0,
                "battery_kwh": 0,
                "occupants": 3,
                "location": "bristol",
                "days": 1,
            },
        )
        assert response.status_code == 200

    def test_simulate_json_invalid_pv_returns_400(self, client: FlaskClient) -> None:
        """Test POST /simulate with JSON body and invalid pv_kw returns 400."""
        response = client.post(
            "/simulate",
            json={
                "pv_kw": 0.1,
                "battery_kwh": 0,
                "occupants": 3,
                "location": "bristol",
                "days": 1,
            },
        )
        assert response.status_code == 400


class TestDownloadRoutes:
    """Tests for download routes content-type headers."""

    def _run_simulation(self, client: FlaskClient) -> None:
        """Helper: run a simulation so session and cache are populated."""
        response = client.post(
            "/simulate",
            data={
                "pv_kw": "4.0",
                "battery_kwh": "0",
                "occupants": "3",
                "location": "bristol",
                "days": "1",
            },
        )
        assert response.status_code == 200, (
            f"Simulation setup failed with status {response.status_code}"
        )

    def test_download_csv_no_session_redirects(self, client: FlaskClient) -> None:
        """Test GET /download/csv with no session redirects (no results cached)."""
        response = client.get("/download/csv")
        # Should redirect to index when no result_key in session
        assert response.status_code in (302, 301, 303)

    def test_download_report_no_session_redirects(self, client: FlaskClient) -> None:
        """Test GET /download/report with no session redirects (no results cached)."""
        response = client.get("/download/report")
        # Should redirect to index when no result_key in session
        assert response.status_code in (302, 301, 303)

    def test_download_csv_correct_content_type(self, client: FlaskClient) -> None:
        """Test GET /download/csv returns text/csv content-type after simulation."""
        self._run_simulation(client)
        response = client.get("/download/csv")
        assert response.status_code == 200
        assert "text/csv" in response.content_type

    def test_download_csv_has_attachment_header(self, client: FlaskClient) -> None:
        """Test GET /download/csv returns Content-Disposition attachment header."""
        self._run_simulation(client)
        response = client.get("/download/csv")
        assert response.status_code == 200
        disposition = response.headers.get("Content-Disposition", "")
        assert "attachment" in disposition
        assert ".csv" in disposition

    def test_download_report_correct_content_type(self, client: FlaskClient) -> None:
        """Test GET /download/report returns text/html content-type after simulation."""
        self._run_simulation(client)
        response = client.get("/download/report")
        assert response.status_code == 200
        assert "text/html" in response.content_type

    def test_download_report_has_attachment_header(self, client: FlaskClient) -> None:
        """Test GET /download/report returns Content-Disposition attachment header."""
        self._run_simulation(client)
        response = client.get("/download/report")
        assert response.status_code == 200
        disposition = response.headers.get("Content-Disposition", "")
        assert "attachment" in disposition
        assert ".html" in disposition

    def test_download_csv_contains_data(self, client: FlaskClient) -> None:
        """Test GET /download/csv response body is non-empty CSV."""
        self._run_simulation(client)
        response = client.get("/download/csv")
        assert response.status_code == 200
        assert len(response.data) > 0

    def test_download_report_contains_html(self, client: FlaskClient) -> None:
        """Test GET /download/report response body contains valid HTML."""
        self._run_simulation(client)
        response = client.get("/download/report")
        assert response.status_code == 200
        assert b"<!DOCTYPE html>" in response.data or b"<html" in response.data
