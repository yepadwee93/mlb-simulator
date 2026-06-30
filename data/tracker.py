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
    confidence_grade=None,
    confidence_score=None,
    confidence_signals=None,
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
    if confidence_grade:
        row["confidence_grade"] = confidence_grade
    if confidence_score is not None:
        row["confidence_score"] = float(confidence_score)
    if confidence_signals is not None:
        row["confidence_signals"] = int(confidence_signals)

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
    all_preds = supa().table("predictions").select("*").execute()
    rows = [r for r in (all_preds.data or []) if not r.get("actual_winner")]
    updated = 0

    for row in rows:
        game_pk = row.get("game_pk")
        if not game_pk:
            continue
        try:
            import requests as _req

            live = _req.get(
                f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
                timeout=10,
            ).json()
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
        all_odds = (
            supa()
            .table("odds_history")
            .select("game_pk, game_date, actual_winner")
            .limit(500)
            .execute()
        )
        unsettled_rows = [r for r in (all_odds.data or []) if not r.get("actual_winner")]
        if not unsettled_rows:
            return 0

        game_pks = list({r["game_pk"] for r in unsettled_rows})
        all_preds = (
            supa()
            .table("predictions")
            .select("game_pk, actual_winner, actual_away_runs, actual_home_runs")
            .in_("game_pk", game_pks)
            .execute()
        )
        settled = [p for p in (all_preds.data or []) if p.get("actual_winner")]
        count = 0
        for pred in settled:
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


def get_team_trends(days=7):
    """
    Rolling accuracy by team over the last N days.
    Returns list of {team, wins, losses, pct, streak} sorted by pct desc.
    """
    from datetime import timedelta

    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    res = (
        supa()
        .table("predictions")
        .select("predicted_winner, correct_pick, away_team, home_team, game_date")
        .not_.is_("correct_pick", "null")
        .gte("game_date", cutoff)
        .order("game_date", desc=True)
        .execute()
    )
    rows = res.data or []

    teams = {}
    for r in rows:
        pw = r.get("predicted_winner", "")
        if not pw:
            continue
        correct = int(r.get("correct_pick", 0))
        if pw not in teams:
            teams[pw] = {"wins": 0, "losses": 0, "results": []}
        if correct:
            teams[pw]["wins"] += 1
        else:
            teams[pw]["losses"] += 1
        teams[pw]["results"].append(correct)

    out = []
    for team, d in teams.items():
        total = d["wins"] + d["losses"]
        if total < 2:
            continue
        streak = 0
        for r in d["results"]:
            if r == d["results"][0]:
                streak += 1
            else:
                break
        streak_label = f"W{streak}" if d["results"][0] == 1 else f"L{streak}"
        out.append(
            {
                "team": team,
                "wins": d["wins"],
                "losses": d["losses"],
                "pct": round(d["wins"] / total * 100, 1),
                "streak": streak_label,
            }
        )

    out.sort(key=lambda x: x["pct"], reverse=True)
    return out


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
    Counts all simulations (single + bulk/simulate-all).
    Returns dict with total_predictions, results_available, correct_picks,
    accuracy_pct, avg_run_diff_error, by_confidence, recent, all_single.
    """
    all_rows = get_all_predictions()
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


def get_confidence_history():
    """
    Returns confidence grade accuracy over time.
    Groups settled predictions by grade (A/B/C/D) and computes win rate for each.
    Also returns daily data for charting.
    """
    rows = (
        supa()
        .table("predictions")
        .select("*")
        .not_.is_("confidence_grade", "null")
        .not_.is_("correct_pick", "null")
        .order("game_date", desc=False)
        .execute()
    ).data or []

    by_grade = {}
    daily = {}
    for r in rows:
        grade = r.get("confidence_grade", "")
        if not grade or grade == "—":
            continue
        correct = int(r.get("correct_pick", 0))

        if grade not in by_grade:
            by_grade[grade] = {"total": 0, "correct": 0}
        by_grade[grade]["total"] += 1
        by_grade[grade]["correct"] += correct

        gd = r.get("game_date", "")
        if gd:
            if gd not in daily:
                daily[gd] = {"total": 0, "correct": 0}
            daily[gd]["total"] += 1
            daily[gd]["correct"] += correct

    grade_stats = []
    for g in ["A", "B", "C", "D"]:
        d = by_grade.get(g, {"total": 0, "correct": 0})
        pct = round(d["correct"] / d["total"] * 100, 1) if d["total"] > 0 else None
        grade_stats.append({"grade": g, "total": d["total"], "correct": d["correct"], "pct": pct})

    daily_list = []
    for dt in sorted(daily.keys()):
        d = daily[dt]
        pct = round(d["correct"] / d["total"] * 100, 1) if d["total"] > 0 else 0
        daily_list.append({"date": dt, "total": d["total"], "correct": d["correct"], "pct": pct})

    return {"by_grade": grade_stats, "daily": daily_list, "total_graded": len(rows)}


def detect_rlm(line_movement, public_betting):
    """
    Detect reverse line movement: line moves AGAINST public betting %.
    line_movement: {(team_a, team_b): {"open_away": int, "current_away": int, ...}}
    public_betting: {frozenset: {"away_pct": float, "home_pct": float}}
    Returns list of RLM alerts.
    """
    alerts = []
    for key, lm in line_movement.items():
        open_away = lm.get("open_away") or lm.get("away_open")
        curr_away = lm.get("current_away") or lm.get("away_current")
        if open_away is None or curr_away is None:
            continue

        fs_key = frozenset(key) if not isinstance(key, frozenset) else key
        pub = public_betting.get(fs_key, {})
        away_bet_pct = pub.get("away_pct", 50)

        line_moved_toward_away = curr_away < open_away
        line_moved_toward_home = curr_away > open_away
        public_on_away = away_bet_pct > 55
        public_on_home = away_bet_pct < 45

        if line_moved_toward_away and public_on_home:
            alerts.append(
                {
                    "away_team": lm.get("away_team", list(key)[0]),
                    "home_team": lm.get("home_team", list(key)[-1]),
                    "direction": "away",
                    "bet_pct": away_bet_pct,
                    "line_open": open_away,
                    "line_current": curr_away,
                    "shift": curr_away - open_away,
                }
            )
        elif line_moved_toward_home and public_on_away:
            alerts.append(
                {
                    "away_team": lm.get("away_team", list(key)[0]),
                    "home_team": lm.get("home_team", list(key)[-1]),
                    "direction": "home",
                    "bet_pct": 100 - away_bet_pct,
                    "line_open": open_away,
                    "line_current": curr_away,
                    "shift": curr_away - open_away,
                }
            )

    return alerts


def log_prop_predictions(game_pk, game_date, team, batter_props, pitcher_k_pred=None):
    """
    Log player prop predictions for a game so we can track accuracy later.
    batter_props: list of dicts from predict_batter_props (with 'name' attached).
    pitcher_k_pred: dict with projected_ks from predict_pitcher_ks (optional).
    """
    try:
        existing = (
            supa()
            .table("prop_predictions")
            .select("id")
            .eq("game_pk", str(game_pk))
            .eq("team", team)
            .limit(1)
            .execute()
        )
        if existing.data:
            return

        rows = []
        for prop in batter_props:
            name = prop.get("name", "")
            if not name or name.startswith("Batter "):
                continue
            for prop_type, pred_key, prob_key in [
                ("hits", "avg_hits", "hit_pct"),
                ("hr", "avg_hr", "hr_pct"),
                ("rbi", "avg_rbi", None),
                ("tb", "avg_tb", "tb_pct_2plus"),
            ]:
                pred_val = prop.get(pred_key)
                if pred_val is None:
                    continue
                prob = prop.get(prob_key, 0) if prob_key else None
                rows.append(
                    {
                        "game_pk": str(game_pk),
                        "game_date": str(game_date),
                        "team": team,
                        "player_name": name,
                        "slot": prop.get("slot", 0),
                        "prop_type": prop_type,
                        "predicted": round(float(pred_val), 2),
                        "over_prob": round(float(prob) / 100, 3) if prob else None,
                    }
                )

        if pitcher_k_pred and pitcher_k_pred.get("projected_ks"):
            rows.append(
                {
                    "game_pk": str(game_pk),
                    "game_date": str(game_date),
                    "team": team,
                    "player_name": pitcher_k_pred.get("name", "Pitcher"),
                    "slot": 0,
                    "prop_type": "k",
                    "predicted": round(float(pitcher_k_pred["projected_ks"]), 2),
                    "over_prob": None,
                }
            )

        if rows:
            supa().table("prop_predictions").insert(rows).execute()
    except Exception:
        pass


def settle_prop_predictions():
    """
    Settle unsettled prop predictions by checking actual box scores.
    Returns count of settled rows.
    """
    unsettled = (
        supa().table("prop_predictions").select("*").is_("actual", "null").limit(500).execute()
    )
    rows = unsettled.data or []
    if not rows:
        return 0

    game_pks = list({r["game_pk"] for r in rows})
    settled = 0

    for gpk in game_pks:
        try:
            live = _get_nocache(f"/game/{gpk}/feed/live")
            state = live.get("gameData", {}).get("status", {}).get("abstractGameState", "")
            if state != "Final":
                continue

            box = live.get("liveData", {}).get("boxscore", {}).get("teams", {})
            away_players = box.get("away", {}).get("players", {})
            home_players = box.get("home", {}).get("players", {})

            player_stats = {}
            for side_players in [away_players, home_players]:
                for pid, pdata in side_players.items():
                    name = pdata.get("person", {}).get("fullName", "")
                    if not name:
                        continue
                    batting = pdata.get("stats", {}).get("batting", {})
                    pitching = pdata.get("stats", {}).get("pitching", {})
                    player_stats[name.lower()] = {
                        "hits": int(batting.get("hits", 0)),
                        "hr": int(batting.get("homeRuns", 0)),
                        "rbi": int(batting.get("rbi", 0)),
                        "tb": int(batting.get("totalBases", 0)),
                        "k": int(pitching.get("strikeOuts", 0)),
                    }

            game_rows = [r for r in rows if r["game_pk"] == gpk]
            for r in game_rows:
                pname = (r.get("player_name") or "").lower()
                stats = player_stats.get(pname)
                if not stats:
                    for key in player_stats:
                        if pname.split()[-1] in key:
                            stats = player_stats[key]
                            break
                if not stats:
                    continue

                prop_type = r["prop_type"]
                actual_val = stats.get(prop_type, 0)
                predicted = float(r.get("predicted", 0))
                hit = 1 if actual_val >= max(predicted, 0.5) else 0

                supa().table("prop_predictions").update(
                    {
                        "actual": actual_val,
                        "hit": hit,
                        "settled_at": datetime.utcnow().isoformat(),
                    }
                ).eq("id", r["id"]).execute()
                settled += 1

        except Exception:
            continue

    return settled


def get_prop_accuracy():
    """
    Returns prop prediction accuracy stats grouped by prop type.
    """
    res = (
        supa()
        .table("prop_predictions")
        .select("*")
        .not_.is_("actual", "null")
        .order("game_date", desc=True)
        .limit(1000)
        .execute()
    )
    rows = res.data or []

    by_type = {}
    for r in rows:
        pt = r.get("prop_type", "")
        if pt not in by_type:
            by_type[pt] = {"total": 0, "correct": 0, "avg_pred": 0, "avg_actual": 0}
        by_type[pt]["total"] += 1
        by_type[pt]["correct"] += int(r.get("hit", 0))
        by_type[pt]["avg_pred"] += float(r.get("predicted", 0))
        by_type[pt]["avg_actual"] += float(r.get("actual", 0))

    type_stats = []
    for pt in ["hits", "hr", "rbi", "tb", "k"]:
        d = by_type.get(pt, {"total": 0, "correct": 0, "avg_pred": 0, "avg_actual": 0})
        if d["total"] > 0:
            d["avg_pred"] = round(d["avg_pred"] / d["total"], 2)
            d["avg_actual"] = round(d["avg_actual"] / d["total"], 2)
            d["pct"] = round(d["correct"] / d["total"] * 100, 1)
        else:
            d["pct"] = None
            d["avg_pred"] = None
            d["avg_actual"] = None
        d["prop_type"] = pt
        type_stats.append(d)

    pending = (
        supa().table("prop_predictions").select("id", count="exact").is_("actual", "null").execute()
    )

    return {
        "by_type": type_stats,
        "total_settled": len(rows),
        "total_pending": pending.count if pending.count else 0,
        "recent": rows[:30],
    }
