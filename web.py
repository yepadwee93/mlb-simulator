"""
web.py
------
The Flask web app. Run this to open the simulator in your browser.

Usage:
    python web.py
Then open:  http://127.0.0.1:5000
"""

from datetime import date
from flask import Flask, render_template, abort

from data.mlb_api import (
    get_today_schedule,
    get_game_lineup,
    get_player_season_stats,
)
from simulation.engine import run_simulation

# Create the Flask app, pointing it to our templates folder
app = Flask(__name__, template_folder="app/templates")


# ── Home page: today's games ─────────────────────────────────────

@app.route("/")
def index():
    """Show all of today's MLB games with a 'Simulate' button on each."""
    games = get_today_schedule()
    today = date.today().strftime("%A, %B %d %Y")   # e.g. "Thursday, June 25 2026"
    return render_template("index.html", games=games, date=today)


# ── Simulate a specific game ──────────────────────────────────────

@app.route("/simulate/<int:game_pk>")
def simulate(game_pk):
    """
    Fetch the lineup for one game, run 100,000 simulations,
    and display the prediction results page.
    """

    # 1. Find the game in today's schedule so we have team names
    games = get_today_schedule()
    game  = next((g for g in games if g["gamePk"] == game_pk), None)

    if game is None:
        abort(404)  # game not found

    away = game["away_team"]
    home = game["home_team"]

    # 2. Fetch lineup
    lineup = get_game_lineup(game_pk)

    if not lineup["away_batters"] or not lineup["home_batters"]:
        # Lineup not set yet — send user back with a message
        return render_template("index.html", games=games,
                               date=date.today().strftime("%A, %B %d %Y"),
                               error="Lineup not available yet for that game. Try one that's already started.")

    # 3. Fetch batter stats (one API call per player)
    def fetch_batter_stats(batters):
        result = []
        for b in batters:
            stats = get_player_season_stats(b["id"], group="hitting") if b.get("id") else {}
            result.append(stats)
        return result

    away_batter_stats = fetch_batter_stats(lineup["away_batters"])
    home_batter_stats = fetch_batter_stats(lineup["home_batters"])

    # 4. Fetch pitcher stats
    away_pitcher = lineup.get("away_pitcher") or {}
    home_pitcher = lineup.get("home_pitcher") or {}

    away_pitcher_stats = get_player_season_stats(away_pitcher["id"], group="pitching") if away_pitcher.get("id") else {}
    home_pitcher_stats = get_player_season_stats(home_pitcher["id"], group="pitching") if home_pitcher.get("id") else {}

    # 5. Run the simulation
    results = run_simulation(
        away_team    = away,
        home_team    = home,
        away_lineup  = away_batter_stats,
        home_lineup  = home_batter_stats,
        away_pitcher = away_pitcher_stats,
        home_pitcher = home_pitcher_stats,
        n            = 100_000,
    )

    # 6. Build W-L record strings for the pitchers
    def wl(stats):
        w = stats.get("wins", 0)
        l = stats.get("losses", 0)
        return f"{w}-{l}"

    # 7. Render the results page with everything the template needs
    return render_template(
        "result.html",

        # Team names
        away_team = away,
        home_team = home,

        # Win probabilities + predicted score
        away_win_pct  = results["away_win_pct"],
        home_win_pct  = results["home_win_pct"],
        away_avg_runs = results["away_avg_runs"],
        home_avg_runs = results["home_avg_runs"],
        simulations   = results["simulations"],

        # Pitcher details
        away_pitcher_name = away_pitcher.get("name", "Unknown"),
        home_pitcher_name = home_pitcher.get("name", "Unknown"),
        away_era  = away_pitcher_stats.get("era",  "N/A"),
        away_whip = away_pitcher_stats.get("whip", "N/A"),
        away_wl   = wl(away_pitcher_stats),
        home_era  = home_pitcher_stats.get("era",  "N/A"),
        home_whip = home_pitcher_stats.get("whip", "N/A"),
        home_wl   = wl(home_pitcher_stats),

        # Lineups for display
        away_batters = lineup["away_batters"],
        home_batters = lineup["home_batters"],
    )


# ── Start the server ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  MLB Simulator is running!")
    print("  Open this in your browser:  http://127.0.0.1:5000\n")
    app.run(debug=True)
