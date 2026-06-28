"""
auth.py
-------
User account management using Supabase (PostgreSQL).
Passwords are hashed with werkzeug.security (pbkdf2:sha256) — we never
store plaintext passwords, and Supabase never sees them unmasked.
"""

import re

from data.db import supa


def create_user(username, password):
    """
    Create a new user. Returns (True, user_id) on success,
    or (False, error_message) on failure.
    """
    from werkzeug.security import generate_password_hash

    username = username.strip()

    # --- Validation ---
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one capital letter."
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number."

    pw_hash = generate_password_hash(password)

    try:
        res = supa().table("users").insert({
            "username": username,
            "password": pw_hash,
        }).execute()
        if res.data:
            return True, res.data[0]["id"]
        return False, "Could not create account. Try again."
    except Exception as e:
        err = str(e).lower()
        if "unique" in err or "duplicate" in err or "23505" in err:
            return False, "Username already taken. Choose another."
        return False, f"Error creating account: {e}"


def check_password(username, password):
    """
    Verify login credentials. Returns user dict on success, None on failure.
    Uses case-insensitive username match.
    """
    from werkzeug.security import check_password_hash

    # ilike = case-insensitive LIKE in Supabase/PostgreSQL
    res = supa().table("users").select("*").ilike("username", username.strip()).execute()
    if not res.data:
        return None
    user = res.data[0]
    if check_password_hash(user["password"], password):
        return user
    return None


def get_user_by_id(user_id):
    """Return user dict for Flask-Login's user_loader. Returns None if not found."""
    res = supa().table("users").select("*").eq("id", int(user_id)).execute()
    return res.data[0] if res.data else None


def user_data_dir(user_id):
    """No-op kept for backwards compatibility. Data lives in Supabase now."""
    return None
