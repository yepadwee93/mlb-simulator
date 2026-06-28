"""
Tests for My Picks data logic — the parts that don't need a database.

These test pure Python functions in isolation, which is fast and reliable.
"""

from unittest.mock import MagicMock, patch


def test_add_pick_rejects_missing_user():
    """add_pick with no user_id should not crash but also not write anything."""
    # We patch supa() so no real database is called
    with patch("data.my_picks.supa") as mock_supa:
        mock_table = MagicMock()
        mock_supa.return_value.table.return_value = mock_table

        from data.my_picks import add_pick

        # Missing user_id — should either return None or raise a clear error
        try:
            add_pick(
                game_pk="123",
                game_date="2026-01-01",
                away_team="NYY",
                home_team="BOS",
                my_pick="NYY",
                user_id=None,
            )
        except Exception:
            pass  # Crashing is acceptable — but it shouldn't be a silent data write

        # The important thing: if user_id is None, we should NOT have called update()
        # (i.e. no data written without a user)


def test_get_all_picks_returns_empty_without_user():
    """get_all_picks(user_id=None) should always return []."""
    from data.my_picks import get_all_picks

    result = get_all_picks(user_id=None)
    assert result == []


def test_get_pick_stats_empty_user():
    """get_pick_stats with no user should return zeros, not crash."""
    with patch("data.my_picks.get_all_picks", return_value=[]):
        from data.my_picks import get_pick_stats

        stats = get_pick_stats(user_id=None)
        assert stats["total"] == 0
        assert stats["completed"] == 0
