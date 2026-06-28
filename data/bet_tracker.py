"""
bet_tracker.py
--------------
Stores and retrieves per-user bets in Supabase.

All public functions accept user_id (Supabase users.id) to scope queries.
csv_path is accepted as a no-op keyword arg for backwards compatibility.
"""

from datetime import datetime

from data.db import supa

# ── Write ────────────────────────────────────────────────────────


def log_bet(
    game_pk,
    game_date,
    away_team,
    home_team,
    bet_on,
    bet_type="ML",
    odds=None,
    amount=100.0,
    model_edge=None,
    ev=None,
    kelly=None,
    user_id=None,
    csv_path=None,
):
    """Insert one bet row for this user."""
    if not user_id:
        raise ValueError("user_id is required to log a bet.")

    row = {
        "user_id": int(user_id),
        "game_pk": str(game_pk),
        "game_date": str(game_date),
        "away_team": away_team,
        "home_team": home_team,
        "bet_on": bet_on,
        "bet_type": bet_type,
        "odds": int(odds) if odds is not None else None,
        "amount": float(amount),
        "result": "pending",
    }
    for key, val in [("model_edge", model_edge), ("ev", ev), ("kelly", kelly)]:
        if val is not None:
            try:
                row[key] = float(val)
            except (ValueError, TypeError):
                pass

    supa().table("bets").insert(row).execute()


def settle_bets(user_id=None, csv_path=None):
    """
    Auto-settle pending bets by checking the MLB API for final scores.
    Returns count of newly settled bets.
    """
    if not user_id:
        return 0

    from data.mlb_api import _get

    res = (
        supa()
        .table("bets")
        .select("*")
        .eq("user_id", int(user_id))
        .eq("result", "pending")
        .execute()
    )
    pending = res.data or []
    settled = 0

    for bet in pending:
        game_pk = bet.get("game_pk")
        if not game_pk:
            continue
        try:
            live = _get(f"/game/{game_pk}/feed/live")
            state = live.get("gameData", {}).get("status", {}).get("abstractGameState", "")
            if state != "Final":
                continue

            ls = live["liveData"]["linescore"]["teams"]
            away = ls["away"].get("runs", 0)
            home = ls["home"].get("runs", 0)

            actual_winner = bet["away_team"] if away > home else bet["home_team"]
            bet_won = bet["bet_on"] == actual_winner

            odds = bet.get("odds") or -110
            amount = float(bet.get("amount") or 100)
            if bet_won:
                payout = (amount * odds / 100) if odds > 0 else (amount * 100 / abs(odds))
                result = "win"
            else:
                payout = -amount
                result = "loss"

            supa().table("bets").update(
                {
                    "result": result,
                    "payout": round(payout, 2),
                    "settled_at": datetime.utcnow().isoformat(),
                }
            ).eq("id", bet["id"]).execute()
            settled += 1

        except Exception:
            continue

    return settled


def update_closing_lines(odds_map, user_id=None, csv_path=None):
    """
    odds_map: {game_pk: {"away": int_odds, "home": int_odds}}
    Fills in closing_line and CLV for pending bets in those games.
    """
    if not user_id or not odds_map:
        return

    def _to_prob(o):
        o = int(o)
        return abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100)

    for game_pk, lines in odds_map.items():
        res = (
            supa()
            .table("bets")
            .select("id, bet_on, away_team, odds")
            .eq("user_id", int(user_id))
            .eq("game_pk", str(game_pk))
            .eq("result", "pending")
            .execute()
        )
        for bet in res.data or []:
            is_away = bet["bet_on"] == bet["away_team"]
            cl = lines.get("away" if is_away else "home")
            if cl is None:
                continue
            clv = None
            if bet.get("odds"):
                clv = round((_to_prob(cl) - _to_prob(bet["odds"])) * 100, 2)
            supa().table("bets").update(
                {
                    "closing_line": int(cl),
                    "clv": clv,
                }
            ).eq("id", bet["id"]).execute()


# ── Read ─────────────────────────────────────────────────────────


def get_all_bets(user_id=None, csv_path=None):
    """Return all bets for this user, newest first."""
    if not user_id:
        return []
    res = (
        supa()
        .table("bets")
        .select("*")
        .eq("user_id", int(user_id))
        .order("logged_at", desc=True)
        .execute()
    )
    return res.data or []


def get_bet_stats(user_id=None, csv_path=None):
    """Summary stats dict for the bets dashboard."""
    bets = get_all_bets(user_id=user_id)

    total_bet = total_payout = win_amount = loss_amount = 0.0
    wins = losses = pushes = pending_count = 0
    by_type: dict = {}

    for b in bets:
        amount = float(b.get("amount") or 0)
        payout = float(b.get("payout") or 0)
        result = b.get("result", "pending")
        btype = b.get("bet_type", "ML")

        if result == "pending":
            pending_count += 1
            continue

        total_bet += amount
        total_payout += payout

        if result == "win":
            wins += 1
            win_amount += payout
        elif result == "loss":
            losses += 1
            loss_amount += amount
        elif result == "push":
            pushes += 1

        bt = by_type.setdefault(btype, {"wins": 0, "losses": 0, "profit": 0.0})
        bt["profit"] += payout
        if result == "win":
            bt["wins"] += 1
        elif result == "loss":
            bt["losses"] += 1

    settled = wins + losses + pushes
    roi = round((total_payout / total_bet * 100) if total_bet else 0, 1)
    win_pct = round(wins / settled * 100, 1) if settled else None

    clv_vals = [float(b["clv"]) for b in bets if b.get("clv") is not None]
    avg_clv = round(sum(clv_vals) / len(clv_vals), 2) if clv_vals else None

    return {
        "bets": bets,
        "total_bets": len(bets),
        "settled": settled,
        "pending": pending_count,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_pct": win_pct,
        "total_wagered": round(total_bet, 2),
        "total_profit": round(total_payout, 2),
        "roi": roi,
        "avg_clv": avg_clv,
        "by_type": by_type,
    }
