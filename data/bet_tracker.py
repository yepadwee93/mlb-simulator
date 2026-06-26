"""
bet_tracker.py
--------------
Tracks actual bets placed: team, bet type, odds, amount, result, P&L.
Storage: data/bets.csv

Bet types: ML (moneyline), RL (run line -1.5), OVER, UNDER, F5
"""

import csv
import os
from datetime import datetime, date as _date

from data.mlb_api import _get

CSV_PATH = os.path.join(os.path.dirname(__file__), "bets.csv")

FIELDNAMES = [
    "logged_at",
    "game_date",
    "game_pk",
    "away_team",
    "home_team",
    "bet_on",        # team name, "OVER", or "UNDER"
    "bet_type",      # ML | RL | OVER | UNDER | F5
    "odds",          # American odds as int (e.g. -110, +145)
    "amount",        # dollars wagered
    "model_edge",    # edge % at time of bet (e.g. 7.2)
    "ev",            # expected value at time of bet
    "kelly",         # Kelly % used
    # Filled after game finishes
    "result",        # "WIN" | "LOSS" | "PUSH" | ""
    "profit_loss",   # dollars won (positive) or lost (negative)
    "actual_score",  # e.g. "3-5"
]


def _ensure_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def _read_all():
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def _write_all(rows):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def log_bet(game_pk, game_date, away_team, home_team,
            bet_on, bet_type, odds, amount,
            model_edge=None, ev=None, kelly=None):
    """
    Log a new bet. Duplicate game_pk + bet_type entries are overwritten.
    """
    _ensure_csv()
    row = {
        "logged_at":  datetime.now().isoformat(timespec="seconds"),
        "game_date":  game_date,
        "game_pk":    game_pk,
        "away_team":  away_team,
        "home_team":  home_team,
        "bet_on":     bet_on,
        "bet_type":   bet_type,
        "odds":       int(odds),
        "amount":     float(amount),
        "model_edge": model_edge or "",
        "ev":         ev or "",
        "kelly":      kelly or "",
        "result":     "",
        "profit_loss": "",
        "actual_score": "",
    }
    rows = _read_all()
    for i, r in enumerate(rows):
        if str(r["game_pk"]) == str(game_pk) and r["bet_type"] == bet_type and r["bet_on"] == bet_on:
            rows[i] = row
            _write_all(rows)
            return
    rows.append(row)
    _write_all(rows)


def settle_bets():
    """
    Auto-settle open bets by checking MLB API for final scores.
    Returns count of bets settled.
    """
    _ensure_csv()
    rows    = _read_all()
    settled = 0
    today   = _date.today().isoformat()

    for row in rows:
        if row.get("result"):   # already settled
            continue
        game_pk = row.get("game_pk")
        if not game_pk:
            continue
        # Skip games from today — might not be finished
        if row.get("game_date") == today:
            continue
        try:
            live = _get(f"/game/{game_pk}/feed/live")
            state = live.get("gameData", {}).get("status", {}).get("abstractGameState", "")
            if state != "Final":
                continue
            ls = live["liveData"]["linescore"]["teams"]
            away_r = int(ls["away"].get("runs", 0))
            home_r = int(ls["home"].get("runs", 0))
            row["actual_score"] = f"{away_r}-{home_r}"

            bet_on   = row["bet_on"]
            bet_type = row["bet_type"]
            away_t   = row["away_team"]
            home_t   = row["home_team"]
            amount   = float(row["amount"])
            odds_val = int(row["odds"])

            # Determine result
            result     = "LOSS"
            profit_loss = -amount

            if bet_type in ("ML", "F5"):
                winner = away_t if away_r > home_r else home_t
                if bet_on == winner:
                    result = "WIN"
            elif bet_type == "RL":
                # bet_on wins if they cover -1.5
                if bet_on == away_t:
                    result = "WIN" if (away_r - home_r) >= 2 else "LOSS"
                else:
                    result = "WIN" if (home_r - away_r) >= 2 else "LOSS"
            elif bet_type == "OVER":
                try:
                    line = float(bet_on.replace("OVER ", ""))
                    result = "WIN" if (away_r + home_r) > line else "LOSS"
                except Exception:
                    result = "WIN" if (away_r + home_r) > 8.5 else "LOSS"
            elif bet_type == "UNDER":
                try:
                    line = float(bet_on.replace("UNDER ", ""))
                    result = "WIN" if (away_r + home_r) < line else "LOSS"
                except Exception:
                    result = "WIN" if (away_r + home_r) < 8.5 else "LOSS"

            if result == "WIN":
                if odds_val > 0:
                    profit_loss = amount * odds_val / 100
                else:
                    profit_loss = amount * 100 / abs(odds_val)
            elif result == "PUSH":
                profit_loss = 0.0

            row["result"]      = result
            row["profit_loss"] = round(profit_loss, 2)
            settled += 1
        except Exception:
            continue

    if settled:
        _write_all(rows)
    return settled


def get_bet_stats():
    """
    Full ROI dashboard stats.
    Returns dict with totals, win rate, ROI %, by bet type breakdown, recent bets.
    """
    _ensure_csv()
    rows     = _read_all()
    settled  = [r for r in rows if r.get("result") in ("WIN", "LOSS", "PUSH")]
    pending  = [r for r in rows if not r.get("result")]

    total_wagered = sum(float(r["amount"]) for r in settled)
    total_pl      = sum(float(r["profit_loss"]) for r in settled if r.get("profit_loss") not in ("", None))
    wins          = sum(1 for r in settled if r["result"] == "WIN")
    losses        = sum(1 for r in settled if r["result"] == "LOSS")
    pushes        = sum(1 for r in settled if r["result"] == "PUSH")
    non_push      = wins + losses
    win_rate      = round(wins / non_push * 100, 1) if non_push else None
    roi           = round(total_pl / total_wagered * 100, 1) if total_wagered else None

    # Breakdown by bet type
    by_type = {}
    for r in settled:
        bt = r.get("bet_type", "ML")
        if bt not in by_type:
            by_type[bt] = {"w": 0, "l": 0, "pl": 0.0, "wagered": 0.0}
        by_type[bt]["wagered"] += float(r["amount"])
        pl = float(r["profit_loss"]) if r.get("profit_loss") not in ("", None) else 0
        by_type[bt]["pl"] += pl
        if r["result"] == "WIN":
            by_type[bt]["w"] += 1
        elif r["result"] == "LOSS":
            by_type[bt]["l"] += 1

    return {
        "total_bets":    len(rows),
        "settled":       len(settled),
        "pending":       len(pending),
        "wins":          wins,
        "losses":        losses,
        "pushes":        pushes,
        "win_rate":      win_rate,
        "total_wagered": round(total_wagered, 2),
        "total_pl":      round(total_pl, 2),
        "roi":           roi,
        "by_type":       by_type,
        "recent_bets":   list(reversed(rows))[:20],
    }


def get_all_bets():
    _ensure_csv()
    return list(reversed(_read_all()))
