"""
Shared test setup. pytest loads this file automatically before any test runs.

We create a test version of the Flask app that:
- Uses a fake SECRET_KEY so the app starts without needing real env vars
- Has TESTING=True so Flask gives clearer error messages during tests
- Has WTF_CSRF_ENABLED=False so we don't need CSRF tokens in test requests
"""
import os

import pytest

# Set env vars BEFORE importing web.py so the app doesn't raise on missing keys
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "fake-key")


@pytest.fixture()
def app():
    """Return a configured Flask test app."""
    from web import app as flask_app

    flask_app.config.update(
        TESTING=True,
        WTF_CSRF_ENABLED=False,
        SECRET_KEY="test-secret-key-not-for-production",
    )
    yield flask_app


@pytest.fixture()
def client(app):
    """Return a test HTTP client — use this in tests to make fake requests."""
    return app.test_client()
