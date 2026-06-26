"""
my_picks.py
-----------
Stores the user's personal game picks alongside what the simulator predicted.
After games finish, fills in actual results so you can see:
  - Did YOUR pick win?
  - Did the SIMULATOR's pick win?
  - What was the actual score vs predicted?

Storage: data/my_picks.csv
"""

import csv
import os
from datetime import datetime

from data.mlb_api import _get

CSV_PATH = os.path.join(os.path.dirname(__file__), "my_picks.csv")

FIELDNAMES = [
    "logged_at",
    "game_date",
    "game_pk",
    "away_team",
    "home_team",
    # What the user picked
    "my_pick",
    "my_notes",           # optional — e.g. "gut feeling", "bullpen advantage"
    # What the simulator said
    "sim_pick",
    "sim_away_pct",
    "sim_home_pct",
    "sim_away_runs",
    "sim_home_runs",
    # Actual result (filled in later)
    "actual_away_runs",
    "actual_home_runs",
    "actual_winner",
    "my_pick_correct",    # 1/0
    "sim_pick_correct",   # 1/0
    "run_diff_error",     # abs(pred_total - actual_total)
]


def _ensure_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def add_pick(game_pk, game_date, away_team, home_team,
             my_pick, my_notes="",
             sim_away_pct=None, sim_home_pct=None,
             sim_away_runs=None, sim_home_runs=None):
    """
    Save the user's pick for a game.
    If sim data is available (game was already simulated), attach it.
    """
    _ensure_csv()

    sim_pick = None
    if sim_away_pct is not None and sim_home_pct is not None:
        sim_pick = away_team if float(sim_away_pct) >= float(sim_home_pct) else home_team

    row = {
        "logged_at":       datetime.now().isoformat(timespec="seconds"),
        "game_date":       game_date,
        "game_pk":         game_pk,
        "away_team":       away_team,
        "home_team":       home_team,
        "my_pick":         my_pick,
        "my_notes":        my_notes,
        "sim_pick":        sim_pick or "",
        "sim_away_pct":    sim_away_pct or "",
        "sim_home_pct":    sim_home_pct or "",
        "sim_away_runs":   sim_away_runs or "",
        "sim_home_runs":   sim_home_runs or "",
        "actual_away_runs": "",
        "actual_home_runs": "",
        "actual_winner":   "",
        "my_pick_correct": "",
        "sim_pick_correct": "",
        "run_diff_error":  "",
    }

    # If this game_pk is already in the file, update it instead
    existing = _read_all()
    for i, r in enumerate(existing):
        if str(r["game_pk"]) == str(game_pk):
            # Keep existing result data if already filled in
            row["actual_away_runs"]  = r.get("actual_away_runs", "")
            row["actual_home_runs"]  = r.get("actual_home_runs", "")
            row["actual_winner"]     = r.get("actual_winner", "")
            row["my_pick_correct"]   = r.get("my_pick_correct", "")
            row["sim_pick_correct"]  = r.get("sim_pick_correct", "")
            row["run_diff_error"]    = r.get("run_diff_error", "")
            existing[i] = row
            _write_all(existing)
            return

    existing.append(row)
    _write_all(existing)


def update_pick_results():
    """
    Check the MLB API for final scores on any picks without results yet.
    Returns count of updated rows.
    """
    _ensure_csv()
    rows    = _read_all()
    updated = 0

    for row in rows:
        if row.get("actual_winner"):
            continue
        game_pk = row.get("game_pk")
        if not game_pk:
            continue

        try:
            live       = _get(f"/game/{game_pk}/feed/live")
            game_state = live.get("gameData", {}).get("status", {}).get("abstractGameState", "")
            if game_state != "Final":
                continue

            away_runs = live["liveData"]["linescore"]["teams"]["away"].get("runs", 0)
            home_runs = live["liveData"]["linescore"]["teams"]["home"].get("runs", 0)

            actual_winner = row["away_team"] if away_runs > home_runs else row["home_team"]

            row["actual_away_runs"]  = away_runs
            row["actual_home_runs"]  = home_runs
            row["actual_winner"]     = actual_winner
            row["my_pick_correct"]   = 1 if row["my_pick"] == actual_winner else 0
            row["sim_pick_correct"]  = 1 if row.get("sim_pick") == actual_winner else 0

            if row.get("sim_away_runs") and row.get("sim_home_runs"):
                pred_total   = float(row["sim_away_runs"]) + float(row["sim_home_runs"])
                actual_total = away_runs + home_runs
                row["run_diff_error"] = round(abs(pred_total - actual_total), 2)

            updated += 1

        except Exception:
            continue

    if updated:
        _write_all(rows)
    return updated


def get_all_picks():
    """Returns all picks, newest first."""
    _ensure_csv()
    return list(reversed(_read_all()))


def get_pick_stats():
    """Summary stats for the My Picks page."""
    rows      = _read_all()
    completed = [r for r in rows if r.get("my_pick_correct") != ""]

    if not completed:
        return {
            "total":          len(rows),
            "completed":      0,
            "my_correct":     0,
            "my_pct":         None,
            "sim_correct":    0,
            "sim_pct":        None,
            "avg_run_error":  None,
            "all_picks":      list(reversed(rows)),
        }

    my_correct  = sum(int(r["my_pick_correct"])  for r in completed)
    sim_correct = sum(int(r["sim_pick_correct"]) for r in completed if r.get("sim_pick_correct") != "")
    sim_total   = sum(1 for r in completed if r.get("sim_pick_correct") != "")

    errors = [float(r["run_diff_error"]) for r in completed if r.get("run_diff_error") not in ("", None)]

    return {
        "total":         len(rows),
        "completed":     len(completed),
        "my_correct":    my_correct,
        "my_pct":        round(my_correct / len(completed) * 100, 1),
        "sim_correct":   sim_correct,
        "sim_pct":       round(sim_correct / sim_total * 100, 1) if sim_total else None,
        "avg_run_error": round(sum(errors) / len(errors), 2) if errors else None,
        "all_picks":     list(reversed(rows)),
    }


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
