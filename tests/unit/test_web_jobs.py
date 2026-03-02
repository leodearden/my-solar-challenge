"""Tests for the background simulation API endpoints and job lifecycle."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from flask.testing import FlaskClient

from solar_challenge.web.app import create_app


@pytest.fixture
def app(tmp_path) -> Flask:
    """Create a test Flask application with temporary database."""
    db_path = str(tmp_path / "test.db")
    data_dir = str(tmp_path / "data")
    test_app = create_app(
        test_config={
            "TESTING": True,
            "SECRET_KEY": "test-secret-key",
            "WTF_CSRF_ENABLED": False,
            "DATABASE": db_path,
            "DATA_DIR": data_dir,
        }
    )
    return test_app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Create a Flask test client."""
    return app.test_client()


class TestSimulateHomeEndpoint:
    """Tests for POST /api/simulate/home."""

    def test_submit_home_job_returns_201(self, client: FlaskClient) -> None:
        """Test POST /api/simulate/home returns 201 with job_id and run_id."""
        response = client.post(
            "/api/simulate/home",
            json={
                "pv_kw": 4.0,
                "battery_kwh": 0,
                "occupants": 3,
                "location": "bristol",
                "days": 1,
                "name": "Test Simulation",
            },
        )
        assert response.status_code == 201
        data = response.get_json()
        assert "job_id" in data
        assert "run_id" in data
        assert len(data["job_id"]) > 0
        assert len(data["run_id"]) > 0

    def test_submit_home_job_invalid_pv_returns_400(self, client: FlaskClient) -> None:
        """Test POST /api/simulate/home with invalid PV returns 400."""
        response = client.post(
            "/api/simulate/home",
            json={
                "pv_kw": 0.1,
                "battery_kwh": 0,
                "occupants": 3,
                "location": "bristol",
                "days": 1,
            },
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_submit_home_job_negative_battery_returns_400(self, client: FlaskClient) -> None:
        """Test POST /api/simulate/home with negative battery returns 400."""
        response = client.post(
            "/api/simulate/home",
            json={
                "pv_kw": 4.0,
                "battery_kwh": -5.0,
                "occupants": 3,
                "location": "bristol",
                "days": 1,
            },
        )
        assert response.status_code == 400

    def test_submit_home_job_no_json_returns_400(self, client: FlaskClient) -> None:
        """Test POST /api/simulate/home with no JSON body returns 400."""
        response = client.post(
            "/api/simulate/home",
            data="not json",
            content_type="text/plain",
        )
        assert response.status_code == 400


class TestGetJobStatusEndpoint:
    """Tests for GET /api/jobs/<id>."""

    def test_get_job_status_returns_200(self, client: FlaskClient) -> None:
        """Test GET /api/jobs/<id> returns job status after submission."""
        # Submit a job first
        submit_resp = client.post(
            "/api/simulate/home",
            json={
                "pv_kw": 4.0,
                "battery_kwh": 0,
                "occupants": 3,
                "location": "bristol",
                "days": 1,
            },
        )
        assert submit_resp.status_code == 201
        job_id = submit_resp.get_json()["job_id"]

        # Check status
        status_resp = client.get(f"/api/jobs/{job_id}")
        assert status_resp.status_code == 200
        data = status_resp.get_json()
        assert data["job_id"] == job_id
        assert "status" in data
        assert "progress_pct" in data
        assert "current_step" in data

    def test_get_unknown_job_returns_404(self, client: FlaskClient) -> None:
        """Test GET /api/jobs/<unknown_id> returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id-12345")
        assert response.status_code == 404
        data = response.get_json()
        assert "error" in data


class TestJobProgressEndpoint:
    """Tests for GET /api/jobs/<id>/progress (SSE)."""

    def test_progress_returns_event_stream(self, client: FlaskClient) -> None:
        """Test GET /api/jobs/<id>/progress returns text/event-stream content type."""
        # Submit a job first
        submit_resp = client.post(
            "/api/simulate/home",
            json={
                "pv_kw": 4.0,
                "battery_kwh": 0,
                "occupants": 3,
                "location": "bristol",
                "days": 1,
            },
        )
        assert submit_resp.status_code == 201
        job_id = submit_resp.get_json()["job_id"]

        # Request progress stream
        response = client.get(f"/api/jobs/{job_id}/progress")
        assert response.status_code == 200
        assert "text/event-stream" in response.content_type

    def test_progress_unknown_job_sends_error(self, client: FlaskClient) -> None:
        """Test GET /api/jobs/<unknown>/progress sends error event."""
        response = client.get("/api/jobs/nonexistent-id/progress")
        assert response.status_code == 200
        assert "text/event-stream" in response.content_type
        # The stream should contain an error event
        data = response.get_data(as_text=True)
        assert "error" in data


class TestJobResultsEndpoint:
    """Tests for GET /api/jobs/<id>/results."""

    def test_results_returns_409_while_running(self, client: FlaskClient) -> None:
        """Test GET /api/jobs/<id>/results returns 409 while job is running."""
        # Submit a job
        submit_resp = client.post(
            "/api/simulate/home",
            json={
                "pv_kw": 4.0,
                "battery_kwh": 0,
                "occupants": 3,
                "location": "bristol",
                "days": 1,
            },
        )
        assert submit_resp.status_code == 201
        job_id = submit_resp.get_json()["job_id"]

        # Immediately check results - should be 409 (not complete yet)
        # or possibly 200 if it completed very fast
        results_resp = client.get(f"/api/jobs/{job_id}/results")
        # It should either be 409 (still running) or 200 (completed very fast)
        assert results_resp.status_code in (200, 409)

    def test_results_returns_404_unknown_job(self, client: FlaskClient) -> None:
        """Test GET /api/jobs/<unknown_id>/results returns 404."""
        response = client.get("/api/jobs/nonexistent-id/results")
        assert response.status_code == 404


class TestJobCompletion:
    """Test that a job completes successfully end-to-end."""

    def test_home_job_completes(self, client: FlaskClient) -> None:
        """Test that a submitted home job eventually completes with results."""
        # Submit a minimal 1-day simulation
        submit_resp = client.post(
            "/api/simulate/home",
            json={
                "pv_kw": 4.0,
                "battery_kwh": 0,
                "occupants": 3,
                "location": "bristol",
                "days": 1,
                "name": "E2E Test",
            },
        )
        assert submit_resp.status_code == 201
        job_id = submit_resp.get_json()["job_id"]
        run_id = submit_resp.get_json()["run_id"]

        # Poll until completed (with timeout)
        deadline = time.monotonic() + 120  # 120 second timeout
        status = "queued"
        while time.monotonic() < deadline:
            status_resp = client.get(f"/api/jobs/{job_id}")
            assert status_resp.status_code == 200
            status_data = status_resp.get_json()
            status = status_data["status"]
            if status in ("completed", "failed"):
                break
            time.sleep(1)

        assert status == "completed", f"Job did not complete in time, last status: {status}"

        # Now fetch results
        results_resp = client.get(f"/api/jobs/{job_id}/results")
        assert results_resp.status_code == 200
        results_data = results_resp.get_json()
        assert results_data["run_id"] == run_id
        assert "summary" in results_data
        assert results_data["summary"].get("total_generation_kwh") is not None

    def test_home_job_with_battery_completes(self, client: FlaskClient) -> None:
        """Test that a home job with battery completes successfully."""
        submit_resp = client.post(
            "/api/simulate/home",
            json={
                "pv_kw": 4.0,
                "battery_kwh": 5.0,
                "occupants": 2,
                "location": "bristol",
                "days": 1,
                "name": "Battery Test",
            },
        )
        assert submit_resp.status_code == 201
        job_id = submit_resp.get_json()["job_id"]

        # Poll until completed
        deadline = time.monotonic() + 120
        status = "queued"
        while time.monotonic() < deadline:
            status_resp = client.get(f"/api/jobs/{job_id}")
            status_data = status_resp.get_json()
            status = status_data["status"]
            if status in ("completed", "failed"):
                break
            time.sleep(1)

        assert status == "completed", f"Job did not complete, last status: {status}"


class TestSimulateFleetEndpoint:
    """Tests for POST /api/simulate/fleet."""

    def test_submit_fleet_job_returns_201(self, client: FlaskClient) -> None:
        """Test POST /api/simulate/fleet returns 201 with job_id and run_id."""
        response = client.post(
            "/api/simulate/fleet",
            json={
                "name": "Test Fleet",
                "homes": [
                    {
                        "pv_kw": 4.0,
                        "battery_kwh": 0,
                        "occupants": 3,
                        "location": "bristol",
                        "days": 1,
                    },
                    {
                        "pv_kw": 3.0,
                        "battery_kwh": 5.0,
                        "occupants": 2,
                        "location": "london",
                        "days": 1,
                    },
                ],
            },
        )
        assert response.status_code == 201
        data = response.get_json()
        assert "job_id" in data
        assert "run_id" in data

    def test_submit_fleet_job_empty_homes_returns_400(self, client: FlaskClient) -> None:
        """Test POST /api/simulate/fleet with empty homes returns 400."""
        response = client.post(
            "/api/simulate/fleet",
            json={
                "name": "Empty Fleet",
                "homes": [],
            },
        )
        assert response.status_code == 400


class TestJobManagerIntegration:
    """Tests for JobManager class directly."""

    def test_job_manager_exists_on_app(self, app: Flask) -> None:
        """Test that JobManager is registered as an app extension."""
        assert "job_manager" in app.extensions
        from solar_challenge.web.jobs import JobManager
        assert isinstance(app.extensions["job_manager"], JobManager)

    def test_get_job_status_returns_none_for_unknown(self, app: Flask) -> None:
        """Test that get_job_status returns None for unknown job IDs."""
        jm = app.extensions["job_manager"]
        assert jm.get_job_status("nonexistent") is None
