"""
tracker.py
----------
Logs every simulation prediction and compares it to the real result
once the game is over.

Storage: a simple CSV file at data/predictions.csv
  - One row per prediction
  - Real results are filled in later by calling update_results()

This lets us answer questions like:
  - "When we say Team A has 70% chance, do they win ~70% of the time?"
  - "Are our predicted scores too high or too low on average?"
  - "Is the model getting better or worse over time?"
"""

import csv
import os
from datetime import date, datetime

from data.mlb_api import _get

# Where we store the predictions
CSV_PATH = os.path.join(os.path.dirname(__file__), "predictions.csv")

# Column order in the CSV
FIELDNAMES = [
    "logged_at",        # when we ran the simulation (ISO timestamp)
    "game_date",        # date of the game (YYYY-MM-DD)
    "game_pk",          # MLB game ID
    "away_team",
    "home_team",
    "away_win_pct",     # our model's predicted win % for away team
    "home_win_pct",     # our model's predicted win % for home team
    "away_avg_runs",    # our predicted average score
    "home_avg_runs",
    "predicted_winner", # which team our model likes
    "n_sims",           # how many simulations we ran
    # Filled in later once the game is final:
    "actual_away_runs",
    "actual_home_runs",
    "actual_winner",
    "correct_pick",     # 1 if we picked the right team, 0 if not
    "run_diff_error",   # how far off our predicted total runs were
]


def _ensure_csv():
    """Create the CSV file with headers if it doesn't exist yet."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def log_prediction(game_pk, game_date, away_team, home_team,
                   away_win_pct, home_win_pct,
                   away_avg_runs, home_avg_runs, n_sims):
    """
    Save one simulation result to the CSV.
    Called every time a single-game simulation finishes.

    If we already logged this game_pk today, we overwrite it
    (no point keeping duplicates of the same game).
    """
    _ensure_csv()

    predicted_winner = away_team if away_win_pct >= home_win_pct else home_team

    new_row = {
        "logged_at":        datetime.now().isoformat(timespec="seconds"),
        "game_date":        game_date,
        "game_pk":          game_pk,
        "away_team":        away_team,
        "home_team":        home_team,
        "away_win_pct":     away_win_pct,
        "home_win_pct":     home_win_pct,
        "away_avg_runs":    away_avg_runs,
        "home_avg_runs":    home_avg_runs,
        "predicted_winner": predicted_winner,
        "n_sims":           n_sims,
        # Results filled in later
        "actual_away_runs": "",
        "actual_home_runs": "",
        "actual_winner":    "",
        "correct_pick":     "",
        "run_diff_error":   "",
    }

    # Read existing rows, replacing any prior prediction for this game_pk
    existing = _read_all()
    updated  = False
    for i, row in enumerate(existing):
        if str(row["game_pk"]) == str(game_pk):
            # Keep the original logged_at; update everything else
            new_row["logged_at"] = row["logged_at"]
            # But preserve any actual results already filled in
            if row.get("actual_winner"):
                new_row["actual_away_runs"] = row["actual_away_runs"]
                new_row["actual_home_runs"] = row["actual_home_runs"]
                new_row["actual_winner"]    = row["actual_winner"]
                new_row["correct_pick"]     = row["correct_pick"]
                new_row["run_diff_error"]   = row["run_diff_error"]
            existing[i] = new_row
            updated = True
            break

    if not updated:
        existing.append(new_row)

    _write_all(existing)


def update_results():
    """
    For every logged prediction that doesn't have a real result yet,
    check the MLB API to see if the game is final and fill in the score.

    Returns a count of how many rows were updated.
    """
    _ensure_csv()
    rows    = _read_all()
    updated = 0

    for row in rows:
        # Skip if we already have a result
        if row.get("actual_winner"):
            continue

        game_pk = row.get("game_pk")
        if not game_pk:
            continue

        try:
            data = _get(f"/game/{game_pk}/boxscore")
            teams = data.get("teams", {})

            # Check if the game is final via linescore
            linescore = _get(f"/game/{game_pk}/linescore")
            state = linescore.get("currentInningOrdinal", "")
            is_final = "Final" in data.get("info", [{}])[0].get("value", "") \
                       if data.get("info") else False

            # Try the live feed for game status
            live = _get(f"/game/{game_pk}/feed/live")
            game_state = live.get("gameData", {}).get("status", {}).get("abstractGameState", "")

            if game_state != "Final":
                continue  # game not over yet

            away_runs = live["liveData"]["linescore"]["teams"]["away"].get("runs", 0)
            home_runs = live["liveData"]["linescore"]["teams"]["home"].get("runs", 0)

            away_team = row["away_team"]
            home_team = row["home_team"]
            actual_winner = away_team if away_runs > home_runs else home_team

            correct = 1 if actual_winner == row["predicted_winner"] else 0

            # Run diff error: how far off was our predicted total vs actual total
            pred_total   = float(row["away_avg_runs"]) + float(row["home_avg_runs"])
            actual_total = away_runs + home_runs
            run_diff_err = round(abs(pred_total - actual_total), 2)

            row["actual_away_runs"] = away_runs
            row["actual_home_runs"] = home_runs
            row["actual_winner"]    = actual_winner
            row["correct_pick"]     = correct
            row["run_diff_error"]   = run_diff_err
            updated += 1

        except Exception:
            continue  # game not available yet, skip silently

    if updated:
        _write_all(rows)

    return updated


def get_all_predictions():
    """Returns all logged predictions as a list of dicts, newest first."""
    _ensure_csv()
    rows = _read_all()
    return list(reversed(rows))


def get_accuracy_stats():
    """
    Compute overall accuracy metrics from all predictions that have real results.

    Returns a dict with:
      total_predictions  — how many games we've predicted
      results_available  — how many have real results
      correct_picks      — how many we got right
      accuracy_pct       — win/loss correct pick percentage
      avg_run_diff_error — average absolute error in predicted total runs
      by_confidence      — accuracy bucketed by how confident we were
    """
    rows = _read_all()
    completed = [r for r in rows if r.get("correct_pick") != ""]

    if not completed:
        return {
            "total_predictions": len(rows),
            "results_available": 0,
            "correct_picks":     0,
            "accuracy_pct":      None,
            "avg_run_diff_error": None,
            "by_confidence":     [],
            "recent":            [],
        }

    correct      = sum(int(r["correct_pick"]) for r in completed)
    accuracy_pct = round(correct / len(completed) * 100, 1)

    errors = [float(r["run_diff_error"]) for r in completed if r.get("run_diff_error") != ""]
    avg_err = round(sum(errors) / len(errors), 2) if errors else None

    # Bucket accuracy by confidence level of our top pick
    # e.g. "When we said 60-70% confidence, how often were we right?"
    buckets = [
        ("50–60%",  50, 60),
        ("60–70%",  60, 70),
        ("70–80%",  70, 80),
        ("80–90%",  80, 90),
        ("90%+",    90, 101),
    ]

    by_confidence = []
    for label, lo, hi in buckets:
        bucket_rows = [
            r for r in completed
            if lo <= max(float(r["away_win_pct"]), float(r["home_win_pct"])) < hi
        ]
        if not bucket_rows:
            continue
        bucket_correct = sum(int(r["correct_pick"]) for r in bucket_rows)
        by_confidence.append({
            "label":    label,
            "total":    len(bucket_rows),
            "correct":  bucket_correct,
            "pct":      round(bucket_correct / len(bucket_rows) * 100, 1),
        })

    # Last 20 predictions for the recent history table
    recent = list(reversed(completed))[:20]

    return {
        "total_predictions":  len(rows),
        "results_available":  len(completed),
        "correct_picks":      correct,
        "accuracy_pct":       accuracy_pct,
        "avg_run_diff_error": avg_err,
        "by_confidence":      by_confidence,
        "recent":             recent,
    }


# ── Internal helpers ──────────────────────────────────────────────

def _read_all():
    """Read all rows from the CSV and return as list of dicts."""
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def _write_all(rows):
    """Overwrite the CSV with a new list of row dicts."""
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
