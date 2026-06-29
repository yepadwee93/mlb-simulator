"""
tracker.py
----------
Logs every simulation prediction and compares it to real results.
Storage: Supabase `predictions` table (shared across all users).
"""

from datetime import datetime

from data.db import supa
from data.mlb_api import _get, _get_nocache


def log_prediction(
    game_pk,
    game_date,
    away_team,
    home_team,
    away_win_pct,
    home_win_pct,
    away_avg_runs,
    home_avg_runs,
    n_sims,
    source="bulk",
):
    """
    Upsert one simulation result. If this game_pk already has a row,
    update the prediction values but preserve any real results already filled in.
    """
    predicted_winner = away_team if away_win_pct >= home_win_pct else home_team

    # Check if already logged
    existing = (
        supa()
        .table("predictions")
        .select("id, actual_winner")
        .eq("game_pk", str(game_pk))
        .execute()
    )

    row = {
        "game_date": str(game_date),
        "game_pk": str(game_pk),
        "away_team": away_team,
        "home_team": home_team,
        "away_win_pct": round(float(away_win_pct), 2),
        "home_win_pct": round(float(home_win_pct), 2),
        "away_avg_runs": round(float(away_avg_runs), 2),
        "home_avg_runs": round(float(home_avg_runs), 2),
        "predicted_winner": predicted_winner,
        "n_sims": int(n_sims),
        "source": source,
    }

    if existing.data:
        # Update prediction fields but don't wipe real results
        supa().table("predictions").update(row).eq("game_pk", str(game_pk)).execute()
    else:
        supa().table("predictions").insert(row).execute()


def update_results():
    """
    For every prediction without a real result, check the MLB API.
    Returns count of newly settled rows.
    """
    res = supa().table("predictions").select("*").is_("actual_winner", "null").execute()
    rows = res.data or []
    updated = 0

    for row in rows:
        game_pk = row.get("game_pk")
        if not game_pk:
            continue
        try:
            live = _get_nocache(f"/game/{game_pk}/feed/live")
            state = live.get("gameData", {}).get("status", {}).get("abstractGameState", "")
            if state != "Final":
                continue

            ls = live["liveData"]["linescore"]["teams"]
            away_runs = ls["away"].get("runs", 0)
            home_runs = ls["home"].get("runs", 0)

            actual_winner = row["away_team"] if away_runs > home_runs else row["home_team"]
            correct = 1 if actual_winner == row["predicted_winner"] else 0

            pred_total = float(row["away_avg_runs"] or 0) + float(row["home_avg_runs"] or 0)
            actual_total = away_runs + home_runs
            run_diff_err = round(abs(pred_total - actual_total), 2)

            supa().table("predictions").update(
                {
                    "actual_away_runs": away_runs,
                    "actual_home_runs": home_runs,
                    "actual_winner": actual_winner,
                    "correct_pick": correct,
                    "run_diff_error": run_diff_err,
                }
            ).eq("game_pk", str(game_pk)).execute()
            updated += 1

        except Exception:
            continue

    return updated


def log_odds(
    game_pk,
    game_date,
    away_team,
    home_team,
    away_ml,
    home_ml,
    away_implied_pct,
    home_implied_pct,
    over_under=None,
    model_away_pct=None,
    model_home_pct=None,
    model_away_runs=None,
    model_home_runs=None,
):
    """Save a snapshot of Vegas odds for this game to odds_history."""
    try:
        existing = (
            supa()
            .table("odds_history")
            .select("id, away_ml_open")
            .eq("game_pk", str(game_pk))
            .eq("game_date", str(game_date))
            .execute()
        )
        row = {
            "game_pk": str(game_pk),
            "game_date": str(game_date),
            "away_team": away_team,
            "home_team": home_team,
            "away_ml": int(away_ml) if away_ml else None,
            "home_ml": int(home_ml) if home_ml else None,
            "away_implied_pct": round(float(away_implied_pct), 2) if away_implied_pct else None,
            "home_implied_pct": round(float(home_implied_pct), 2) if home_implied_pct else None,
            "over_under": float(over_under) if over_under else None,
        }
        if model_away_pct is not None:
            row["model_away_pct"] = round(float(model_away_pct), 2)
        if model_home_pct is not None:
            row["model_home_pct"] = round(float(model_home_pct), 2)
        if model_away_runs is not None:
            row["model_away_runs"] = round(float(model_away_runs), 2)
        if model_home_runs is not None:
            row["model_home_runs"] = round(float(model_home_runs), 2)
        if existing.data:
            has_open = existing.data[0].get("away_ml_open") is not None
            if not has_open and away_ml:
                row["away_ml_open"] = int(away_ml)
                row["home_ml_open"] = int(home_ml) if home_ml else None
            supa().table("odds_history").update(row).eq("game_pk", str(game_pk)).eq(
                "game_date", str(game_date)
            ).execute()
        else:
            if away_ml:
                row["away_ml_open"] = int(away_ml)
            if home_ml:
                row["home_ml_open"] = int(home_ml)
            supa().table("odds_history").insert(row).execute()
    except Exception:
        pass


def get_odds_history(limit=200, date_from=None, date_to=None):
    """Returns recent odds history rows, newest first."""
    q = supa().table("odds_history").select("*")
    if date_from:
        q = q.gte("game_date", str(date_from))
    if date_to:
        q = q.lte("game_date", str(date_to))
    res = q.order("game_date", desc=True).order("logged_at", desc=True).limit(limit).execute()
    return res.data or []


def settle_odds_history():
    """Fill in actual results for unsettled odds_history rows using predictions table."""
    try:
        unsettled = (
            supa()
            .table("odds_history")
            .select("game_pk, game_date")
            .is_("actual_winner", "null")
            .limit(100)
            .execute()
        )
        if not unsettled.data:
            return 0

        game_pks = [r["game_pk"] for r in unsettled.data]
        settled = (
            supa()
            .table("predictions")
            .select("game_pk, actual_winner, actual_away_runs, actual_home_runs")
            .in_("game_pk", game_pks)
            .not_.is_("actual_winner", "null")
            .execute()
        )
        count = 0
        for pred in settled.data or []:
            gpk = pred["game_pk"]
            winner = pred["actual_winner"]
            supa().table("odds_history").update(
                {
                    "actual_winner": winner,
                    "actual_away_runs": pred.get("actual_away_runs"),
                    "actual_home_runs": pred.get("actual_home_runs"),
                }
            ).eq("game_pk", gpk).execute()
            count += 1
        return count
    except Exception:
        return 0


def save_game_note(user_id, game_pk, game_date, away_team, home_team, note):
    existing = (
        supa()
        .table("game_notes")
        .select("id")
        .eq("user_id", str(user_id))
        .eq("game_pk", str(game_pk))
        .execute()
    )
    row = {
        "user_id": str(user_id),
        "game_pk": str(game_pk),
        "game_date": str(game_date),
        "away_team": away_team,
        "home_team": home_team,
        "note": note.strip(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    if existing.data:
        supa().table("game_notes").update(row).eq("user_id", str(user_id)).eq(
            "game_pk", str(game_pk)
        ).execute()
    else:
        supa().table("game_notes").insert(row).execute()


def delete_game_note(user_id, game_pk):
    supa().table("game_notes").delete().eq("user_id", str(user_id)).eq(
        "game_pk", str(game_pk)
    ).execute()


def get_game_notes(user_id):
    """Returns all notes for a user keyed by game_pk."""
    res = supa().table("game_notes").select("*").eq("user_id", str(user_id)).execute()
    return {r["game_pk"]: r for r in (res.data or [])}


def get_all_predictions():
    """Returns all predictions, newest first."""
    res = supa().table("predictions").select("*").order("logged_at", desc=True).execute()
    return res.data or []


def get_single_game_predictions():
    """Returns only manually-run (single-game) predictions, newest first."""
    res = (
        supa()
        .table("predictions")
        .select("*")
        .eq("source", "single")
        .order("logged_at", desc=True)
        .execute()
    )
    return res.data or []


def get_accuracy_stats():
    """
    Accuracy metrics for the /accuracy page.
    Only counts manually-run (single-game) simulations.
    Returns dict with total_predictions, results_available, correct_picks,
    accuracy_pct, avg_run_diff_error, by_confidence, recent, all_single.
    """
    all_rows = get_single_game_predictions()
    completed = [r for r in all_rows if r.get("correct_pick") is not None]

    if not completed:
        return {
            "total_predictions": len(all_rows),
            "results_available": 0,
            "correct_picks": 0,
            "accuracy_pct": None,
            "avg_run_diff_error": None,
            "by_confidence": [],
            "recent": [],
            "all_single": all_rows,
        }

    correct = sum(int(r["correct_pick"]) for r in completed)
    accuracy_pct = round(correct / len(completed) * 100, 1)

    errors = [float(r["run_diff_error"]) for r in completed if r.get("run_diff_error") is not None]
    avg_err = round(sum(errors) / len(errors), 2) if errors else None

    buckets_cfg = [
        ("50–60%", 50, 60),
        ("60–70%", 60, 70),
        ("70–80%", 70, 80),
        ("80–90%", 80, 90),
        ("90%+", 90, 101),
    ]
    by_confidence = []
    for label, lo, hi in buckets_cfg:
        bucket = [
            r
            for r in completed
            if lo <= max(float(r["away_win_pct"] or 0), float(r["home_win_pct"] or 0)) < hi
        ]
        if not bucket:
            continue
        b_correct = sum(int(r["correct_pick"]) for r in bucket)
        by_confidence.append(
            {
                "label": label,
                "total": len(bucket),
                "correct": b_correct,
                "pct": round(b_correct / len(bucket) * 100, 1),
            }
        )

    return {
        "total_predictions": len(all_rows),
        "results_available": len(completed),
        "correct_picks": correct,
        "accuracy_pct": accuracy_pct,
        "avg_run_diff_error": avg_err,
        "by_confidence": by_confidence,
        "recent": completed[:20],
        "all_single": all_rows,  # all single-game sims, pending + settled
    }
