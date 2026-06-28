"""
db.py
-----
Supabase client singleton. All data modules import `supa` from here.

Usage:
    from data.db import supa
    res = supa().table("bets").select("*").eq("user_id", uid).execute()
"""

import os

_client = None


def supa():
    """Return the shared Supabase client, creating it on first call."""
    global _client
    if _client is None:
        from supabase import create_client

        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set in your .env file.\n"
                "Get them from your Supabase project → Settings → API."
            )
        _client = create_client(url, key)
    return _client
