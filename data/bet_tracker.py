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

    print(f"[settle_bets] found {len(pending)} pending bets")
    for bet in pending:
        game_pk = bet.get("game_pk")
        print(f"[settle_bets] checking game_pk={game_pk!r} bet_type={bet.get('bet_type')!r}")
        if not game_pk:
            print("[settle_bets] skipping — no game_pk")
            continue
        try:
            import requests as _req

            live = _req.get(
                f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live", timeout=10
            ).json()
            state = live.get("gameData", {}).get("status", {}).get("abstractGameState", "")
            print(
                f"[settle_bets] game_pk={game_pk} state={state} bet_type={bet.get('bet_type')} bet_on={bet.get('bet_on')}"
            )
            if state != "Final":
                continue

            ls = live["liveData"]["linescore"]["teams"]
            away = ls["away"].get("runs", 0)
            home = ls["home"].get("runs", 0)
            total = away + home
            margin = away - home  # positive = away won by that many

            bet_type = (bet.get("bet_type") or "ml").lower()
            bet_on = bet.get("bet_on", "")
            away_team = bet.get("away_team", "")

            # Determine result based on bet type
            if bet_type in ("rl", "run line", "runline"):
                # Run line: away -1.5 means away must win by 2+
                # bet_on is the team name; parse the line from bet_on field if present
                # Convention: away team is always -1.5 (must win by 2), home is +1.5
                is_away_side = bet_on == away_team or bet_on.endswith("-1.5")
                if is_away_side:
                    bet_won = margin >= 2  # away covers -1.5
                else:
                    bet_won = margin <= 1  # home covers +1.5 (wins outright OR loses by 1)
            elif bet_type in ("ou", "o/u", "over", "under", "total"):
                # O/U: bet_on contains "OVER X.X" or "UNDER X.X"
                bet_on_upper = bet_on.upper()
                if "OVER" in bet_on_upper or bet_on_upper.startswith("O "):
                    try:
                        line = float(bet_on_upper.replace("OVER", "").replace("O ", "").strip())
                    except ValueError:
                        line = 8.5
                    bet_won = total > line
                elif "UNDER" in bet_on_upper or bet_on_upper.startswith("U "):
                    try:
                        line = float(bet_on_upper.replace("UNDER", "").replace("U ", "").strip())
                    except ValueError:
                        line = 8.5
                    bet_won = total < line
                else:
                    continue  # can't parse, skip
            else:
                # Moneyline: bet_on is the team name that must win outright
                actual_winner = away_team if away > home else bet["home_team"]
                bet_won = bet_on == actual_winner

            if away == home:
                result = "push"
                payout = 0.0
            else:
                odds = bet.get("odds") or -110
                amount = float(bet.get("amount") or 100)
                if bet_won:
                    payout = (
                        (amount * odds / 100) if int(odds) > 0 else (amount * 100 / abs(int(odds)))
                    )
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

        except Exception as e:
            print(f"[settle_bets] game_pk={game_pk} ERROR: {e}")
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
