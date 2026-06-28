"""
Tests for JSON API routes — these should all return JSON, not crash.
"""
from unittest.mock import patch


def test_scores_api_returns_json(client):
    """GET /api/scores should return JSON (even if empty list)."""
    with patch("web.get_today_schedule", return_value=[]):
        response = client.get("/api/scores")
        assert response.status_code == 200
        assert response.is_json
        data = response.get_json()
        assert isinstance(data, list)


def test_index_loads(client):
    """GET / should return 200 — main page must not crash."""
    with patch("web.get_today_schedule", return_value=[]):
        with patch("web.get_requests_remaining", return_value=100):
            response = client.get("/")
            assert response.status_code == 200
