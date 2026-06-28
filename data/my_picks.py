"""
my_picks.py
-----------
Per-user personal game picks stored in Supabase `picks` table.
"""

from datetime import datetime

from data.db import supa
from data.mlb_api import _get, _get_nocache


def add_pick(
    game_pk,
    game_date,
    away_team,
    home_team,
    my_pick,
    my_notes="",
    sim_away_pct=None,
    sim_home_pct=None,
    sim_away_runs=None,
    sim_home_runs=None,
    user_id=None,
    pick=None,
    bet_type="moneyline",
):
    """
    Save (or update) the user's pick for a game.
    `pick` is an alias for `my_pick` kept for route compatibility.
    """
    if pick and not my_pick:
        my_pick = pick
    if not user_id:
        return

    sim_pick = None
    if sim_away_pct is not None and sim_home_pct is not None:
        sim_pick = away_team if float(sim_away_pct) >= float(sim_home_pct) else home_team

    row = {
        "user_id": int(user_id),
        "game_date": str(game_date),
        "game_pk": str(game_pk),
        "away_team": away_team,
        "home_team": home_team,
        "my_pick": my_pick,
        "my_notes": my_notes or "",
        "bet_type": bet_type or "moneyline",
        "sim_pick": sim_pick,
    }
    for key, val in [
        ("sim_away_pct", sim_away_pct),
        ("sim_home_pct", sim_home_pct),
        ("sim_away_runs", sim_away_runs),
        ("sim_home_runs", sim_home_runs),
    ]:
        if val is not None:
            try:
                row[key] = float(val)
            except (ValueError, TypeError):
                pass

    # Upsert: update if this user already has a pick for this game
    existing = (
        supa()
        .table("picks")
        .select("id")
        .eq("user_id", int(user_id))
        .eq("game_pk", str(game_pk))
        .execute()
    )

    if existing.data:
        supa().table("picks").update(row).eq("id", existing.data[0]["id"]).execute()
    else:
        supa().table("picks").insert(row).execute()


def update_pick_results(user_id=None):
    """
    Check the MLB API for final scores on unsettled picks.
    Returns count of updated rows.
    """
    if not user_id:
        return 0

    res = (
        supa()
        .table("picks")
        .select("*")
        .eq("user_id", int(user_id))
        .is_("actual_winner", "null")
        .execute()
    )
    rows = res.data or []
    updated = 0

    for row in rows:
        game_pk = row.get("game_pk")
        if not game_pk:
            continue
        try:
            import requests as _req

            live = _req.get(
                f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live", timeout=10
            ).json()
            state = live.get("gameData", {}).get("status", {}).get("abstractGameState", "")
            print(f"[picks] game_pk={game_pk} state={state}")
            if state != "Final":
                continue

            ls = live["liveData"]["linescore"]["teams"]
            away_runs = ls["away"].get("runs", 0)
            home_runs = ls["home"].get("runs", 0)

            actual_winner = row["away_team"] if away_runs > home_runs else row["home_team"]

            update = {
                "actual_away_runs": away_runs,
                "actual_home_runs": home_runs,
                "actual_winner": actual_winner,
                "my_pick_correct": 1 if row.get("my_pick") == actual_winner else 0,
                "sim_pick_correct": 1 if row.get("sim_pick") == actual_winner else 0,
            }

            if row.get("sim_away_runs") and row.get("sim_home_runs"):
                pred_total = float(row["sim_away_runs"]) + float(row["sim_home_runs"])
                actual_total = away_runs + home_runs
                update["run_diff_error"] = round(abs(pred_total - actual_total), 2)

            supa().table("picks").update(update).eq("id", row["id"]).execute()
            updated += 1

        except Exception as e:
            print(f"[picks] game_pk={game_pk} error: {e}")
            continue

    return updated


def get_all_picks(user_id=None):
    """Returns all picks for this user, newest first."""
    if not user_id:
        return []
    try:
        res = (
            supa()
            .table("picks")
            .select("*")
            .eq("user_id", int(user_id))
            .order("logged_at", desc=True)
            .execute()
        )
        return res.data or []
    except Exception:
        # Fall back without ordering if logged_at column missing
        res = supa().table("picks").select("*").eq("user_id", int(user_id)).execute()
        return res.data or []


def get_pick_stats(user_id=None):
    """Summary stats for the My Picks page."""
    rows = get_all_picks(user_id=user_id)
    completed = [r for r in rows if r.get("my_pick_correct") is not None]

    if not completed:
        return {
            "total": len(rows),
            "completed": 0,
            "my_correct": 0,
            "my_pct": None,
            "sim_correct": 0,
            "sim_pct": None,
            "avg_run_error": None,
            "all_picks": rows,
        }

    my_correct = sum(int(r["my_pick_correct"]) for r in completed)
    sim_rows = [r for r in completed if r.get("sim_pick_correct") is not None]
    sim_correct = sum(int(r["sim_pick_correct"]) for r in sim_rows)

    errors = [float(r["run_diff_error"]) for r in completed if r.get("run_diff_error") is not None]

    return {
        "total": len(rows),
        "completed": len(completed),
        "my_correct": my_correct,
        "my_pct": round(my_correct / len(completed) * 100, 1),
        "sim_correct": sim_correct,
        "sim_pct": round(sim_correct / len(sim_rows) * 100, 1) if sim_rows else None,
        "avg_run_error": round(sum(errors) / len(errors), 2) if errors else None,
        "all_picks": rows,
    }
