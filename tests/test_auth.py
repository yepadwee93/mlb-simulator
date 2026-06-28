"""
Tests for login, register, and logout routes.

These are "smoke tests" — they don't check every detail, just that the
critical paths don't crash and return sensible HTTP status codes.
"""


def test_login_page_loads(client):
    """GET /login should return 200 OK (page loads without crashing)."""
    response = client.get("/login")
    assert response.status_code == 200
    assert b"Sign" in response.data  # "Sign In" text is on the page


def test_register_page_loads(client):
    """GET /register should return 200 OK."""
    response = client.get("/register")
    assert response.status_code == 200


def test_login_wrong_password(client):
    """POSTing a bad username/password should not crash — should show error or redirect."""
    response = client.post(
        "/login",
        data={"username": "nobody", "password": "wrongpassword"},
        follow_redirects=True,
    )
    # Either stays on login page (200) or redirects — must not be a 500 server error
    assert response.status_code != 500


def test_logout_redirects(client):
    """GET /logout should redirect (even if not logged in)."""
    response = client.get("/logout")
    assert response.status_code in (302, 200)


def test_protected_route_redirects_when_not_logged_in(client):
    """Routes that require login should redirect to /login instead of crashing."""
    response = client.get("/my-picks")
    # Should redirect to login, not crash
    assert response.status_code in (302, 401)
