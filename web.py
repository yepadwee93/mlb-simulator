"""
web.py
------
The Flask web app. Run this to open the simulator in your browser.

Usage:
    python web.py
Then open:  http://127.0.0.1:5000
"""

from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, render_template, abort

from data.mlb_api import (
    get_today_schedule,
    get_game_lineup,
    get_player_season_stats,
    get_batter_split_stats,
    get_pitcher_hand,
    get_ballpark_weather,
)
from simulation.engine import run_simulation

app = Flask(__name__, template_folder="app/templates")

N_SIMS_SINGLE = 100_000   # simulations for one-game view
N_SIMS_ALL    =  25_000   # simulations per game in simulate-all (faster)
PARLAY_THRESHOLD = 60.0   # min win % to include in best parlay


# ── Shared helpers ────────────────────────────────────────────────

def fetch_batter_stats_with_splits(batters, pitcher_hand):
    """
    For each batter, fetch their split stats vs the starting pitcher's hand
    (L or R). Falls back to season totals if split data is unavailable.

    This is more accurate than season totals because it accounts for
    platoon advantage — e.g. a righty batter who crushes lefty pitchers
    but struggles against righties.
    """
    stats_list = []
    for b in batters:
        if not b.get("id"):
            stats_list.append({})
            continue
        # Try split stats first
        split = get_batter_split_stats(b["id"], pitcher_hand)
        if split and split.get("plateAppearances", 0) >= 20:
            stats_list.append(split)
        else:
            # Not enough split data — fall back to full season stats
            stats_list.append(get_player_season_stats(b["id"], group="hitting"))
    return stats_list


def build_game_result(game, n_sims):
    """
    Full pipeline for one game:
      fetch lineup → get pitcher hands → fetch split stats →
      get weather → run simulation → return result dict.

    Returns None if the lineup isn't available yet.
    """
    lineup = get_game_lineup(game["gamePk"])
    if not lineup["away_batters"] or not lineup["home_batters"]:
        return None

    away_pitcher = lineup.get("away_pitcher") or {}
    home_pitcher = lineup.get("home_pitcher") or {}

    # Get which hand each pitcher throws with
    away_hand = get_pitcher_hand(away_pitcher["id"]) if away_pitcher.get("id") else "R"
    home_hand = get_pitcher_hand(home_pitcher["id"]) if home_pitcher.get("id") else "R"

    # Fetch batter stats using the correct L/R split
    # away batters face the HOME pitcher → use home_hand
    # home batters face the AWAY pitcher → use away_hand
    away_batter_stats = fetch_batter_stats_with_splits(lineup["away_batters"], home_hand)
    home_batter_stats = fetch_batter_stats_with_splits(lineup["home_batters"], away_hand)

    # Fetch pitcher season stats
    away_pitcher_stats = get_player_season_stats(away_pitcher["id"], group="pitching") if away_pitcher.get("id") else {}
    home_pitcher_stats = get_player_season_stats(home_pitcher["id"], group="pitching") if home_pitcher.get("id") else {}

    # Fetch weather for this ballpark
    weather = get_ballpark_weather(game["venue"])

    # Run the simulation
    result = run_simulation(
        away_team    = game["away_team"],
        home_team    = game["home_team"],
        away_lineup  = away_batter_stats,
        home_lineup  = home_batter_stats,
        away_pitcher = away_pitcher_stats,
        home_pitcher = home_pitcher_stats,
        weather      = weather,
        n            = n_sims,
    )

    # Attach extra context for display
    result.update({
        "gamePk":             game["gamePk"],
        "venue":              game["venue"],
        "status":             game["status"],
        "away_pitcher_name":  away_pitcher.get("name", "TBD"),
        "home_pitcher_name":  home_pitcher.get("name", "TBD"),
        "away_pitcher_hand":  away_hand,
        "home_pitcher_hand":  home_hand,
        "away_era":           away_pitcher_stats.get("era",  "N/A"),
        "away_whip":          away_pitcher_stats.get("whip", "N/A"),
        "away_wl":            f"{away_pitcher_stats.get('wins',0)}-{away_pitcher_stats.get('losses',0)}",
        "home_era":           home_pitcher_stats.get("era",  "N/A"),
        "home_whip":          home_pitcher_stats.get("whip", "N/A"),
        "home_wl":            f"{home_pitcher_stats.get('wins',0)}-{home_pitcher_stats.get('losses',0)}",
        "away_batters":       lineup["away_batters"],
        "home_batters":       lineup["home_batters"],
        "weather":            weather,
    })
    return result


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    games = get_today_schedule()
    today = date.today().strftime("%A, %B %d %Y")
    return render_template("index.html", games=games, date=today)


@app.route("/simulate/<int:game_pk>")
def simulate(game_pk):
    """Simulate a single game and show the detailed results page."""
    games = get_today_schedule()
    game  = next((g for g in games if g["gamePk"] == game_pk), None)
    if game is None:
        abort(404)

    result = build_game_result(game, n_sims=N_SIMS_SINGLE)
    if result is None:
        return render_template("index.html", games=games,
                               date=date.today().strftime("%A, %B %d %Y"),
                               error="Lineup not available yet. Try a game already in progress.")

    return render_template("result.html", **result)


@app.route("/simulate-all")
def simulate_all():
    """
    Simulate every game on today's slate in parallel, then show all
    predictions + the Best Parlay picks on one page.

    Uses ThreadPoolExecutor to run multiple games simultaneously,
    so total time ≈ time for the slowest single game (~20 seconds)
    rather than all games added together.
    """
    games = get_today_schedule()

    # Run all games in parallel (up to 5 at once to avoid rate-limiting the API)
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_game = {
            executor.submit(build_game_result, game, N_SIMS_ALL): game
            for game in games
        }
        for future in as_completed(future_to_game):
            result = future.result()
            if result:
                results.append(result)

    # Sort results by gamePk to keep a consistent order
    results.sort(key=lambda r: r["gamePk"])

    # ── Build the Best Parlay ──────────────────────────────────
    # A parlay is a bet where you pick multiple games and all must win.
    # We pick games where one team has ≥60% win probability.
    parlay_picks = []
    for r in results:
        if r["away_win_pct"] >= PARLAY_THRESHOLD:
            parlay_picks.append({
                "team":        r["away_team"],
                "opponent":    r["home_team"],
                "win_pct":     r["away_win_pct"],
                "avg_runs":    r["away_avg_runs"],
                "venue":       r["venue"],
            })
        elif r["home_win_pct"] >= PARLAY_THRESHOLD:
            parlay_picks.append({
                "team":        r["home_team"],
                "opponent":    r["away_team"],
                "win_pct":     r["home_win_pct"],
                "avg_runs":    r["home_avg_runs"],
                "venue":       r["venue"],
            })

    # Combined parlay probability = multiply all individual win chances
    combined_prob = 1.0
    for pick in parlay_picks:
        combined_prob *= pick["win_pct"] / 100.0
    combined_prob_pct = round(combined_prob * 100, 1)

    today = date.today().strftime("%A, %B %d %Y")
    return render_template(
        "all_results.html",
        results       = results,
        parlay_picks  = parlay_picks,
        combined_prob = combined_prob_pct,
        date          = today,
        n_sims        = N_SIMS_ALL,
    )


if __name__ == "__main__":
    print("\n  MLB Simulator is running!")
    print("  Open this in your browser:  http://127.0.0.1:5000\n")
    app.run(debug=True)
