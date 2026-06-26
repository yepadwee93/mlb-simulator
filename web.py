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

from dotenv import load_dotenv
load_dotenv()   # loads ODDS_API_KEY from .env file

from flask import Flask, render_template, abort, request

from data.odds_api import get_mlb_odds, format_odds, calc_edge
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
        # Try split stats first; fall back to season totals on any error
        try:
            split = get_batter_split_stats(b["id"], pitcher_hand)
            if split and split.get("plateAppearances", 0) >= 20:
                stats_list.append(split)
            else:
                stats_list.append(get_player_season_stats(b["id"], group="hitting"))
        except Exception:
            stats_list.append(get_player_season_stats(b["id"], group="hitting"))
    return stats_list


def build_game_result(game, n_sims, use_splits=True):
    """
    Full pipeline for one game:
      fetch lineup → get pitcher hands → fetch batter stats →
      get weather → run simulation → return result dict.

    use_splits: if True, fetch L/R split stats per batter (more accurate but slower).
                Set to False for simulate-all mode to keep things fast.

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

    if use_splits:
        # Detailed mode: use L/R split stats for each batter (more accurate)
        away_batter_stats = fetch_batter_stats_with_splits(lineup["away_batters"], home_hand)
        home_batter_stats = fetch_batter_stats_with_splits(lineup["home_batters"], away_hand)
    else:
        # Fast mode: just use season totals (good enough for parlay overview)
        away_batter_stats = [get_player_season_stats(b["id"], group="hitting") if b.get("id") else {} for b in lineup["away_batters"]]
        home_batter_stats = [get_player_season_stats(b["id"], group="hitting") if b.get("id") else {} for b in lineup["home_batters"]]

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
    # Allow browsing any date via ?date=2026-06-28
    selected = request.args.get("date", date.today().isoformat())
    try:
        from datetime import datetime
        selected_date = datetime.strptime(selected, "%Y-%m-%d").date()
    except ValueError:
        selected_date = date.today()

    from datetime import timedelta
    prev_date = (selected_date - timedelta(days=1)).isoformat()
    next_date = (selected_date + timedelta(days=1)).isoformat()
    is_today  = (selected_date == date.today())

    games = get_today_schedule(game_date=selected_date.isoformat())
    label = selected_date.strftime("%A, %B %d %Y")

    return render_template("index.html",
                           games=games,
                           date=label,
                           selected_date=selected_date.isoformat(),
                           prev_date=prev_date,
                           next_date=next_date,
                           is_today=is_today)


@app.route("/simulate/<int:game_pk>")
def simulate(game_pk):
    """Simulate a single game and show the detailed results page."""
    games = get_today_schedule()
    game  = next((g for g in games if g["gamePk"] == game_pk), None)
    if game is None:
        abort(404)

    # Read sim count from URL e.g. /simulate/822961?sims=500000
    allowed = {100_000, 500_000, 1_000_000}
    try:
        n_sims = int(request.args.get("sims", N_SIMS_SINGLE))
        if n_sims not in allowed:
            n_sims = N_SIMS_SINGLE
    except ValueError:
        n_sims = N_SIMS_SINGLE

    result = build_game_result(game, n_sims=n_sims)
    if result is None:
        return render_template("index.html", games=games,
                               date=date.today().strftime("%A, %B %d %Y"),
                               error="Lineup not available yet. Try a game already in progress.")

    # Fetch live betting odds and calculate our edge vs Vegas
    all_odds = get_mlb_odds()
    odds_key  = frozenset([game["away_team"], game["home_team"]])
    game_odds = all_odds.get(odds_key, {})

    if game_odds:
        result["away_implied_pct"] = game_odds["away_implied_pct"]
        result["home_implied_pct"] = game_odds["home_implied_pct"]
        result["away_avg_odds"]    = format_odds(game_odds["away_avg_odds"])
        result["home_avg_odds"]    = format_odds(game_odds["home_avg_odds"])
        result["away_edge"]        = calc_edge(result["away_win_pct"], game_odds["away_implied_pct"])
        result["home_edge"]        = calc_edge(result["home_win_pct"], game_odds["home_implied_pct"])
        result["books_used"]       = game_odds["books_used"]
    else:
        result["away_implied_pct"] = None
        result["home_implied_pct"] = None
        result["away_avg_odds"]    = None
        result["home_avg_odds"]    = None
        result["away_edge"]        = None
        result["home_edge"]        = None
        result["books_used"]       = 0

    return render_template("result.html", **result)


@app.route("/simulate-all")
def simulate_all():
    """
    Simulate every active game on today's slate.

    Speed strategy:
      1. Fetch all lineups in parallel
      2. Collect every unique player ID across all games
      3. Fetch ALL player stats in one big parallel batch (no duplicates)
      4. Run simulations (fast — pre-computed probs)

    This way we make the minimum number of API calls and run them
    all at the same time, cutting total time from 3+ minutes to ~20-30s.
    """
    all_games = get_today_schedule()

    # Skip finished games
    games = [g for g in all_games
             if "final" not in g["status"].lower()
             and "game over" not in g["status"].lower()]

    if not games:
        return render_template("index.html",
                               games=all_games,
                               date=date.today().strftime("%A, %B %d %Y"),
                               error="No active or upcoming games to simulate right now.")

    # ── Step 1: Fetch all lineups in parallel ─────────────────────
    lineups = {}
    def fetch_lineup(game):
        try:
            lu = get_game_lineup(game["gamePk"])
            if lu["away_batters"] and lu["home_batters"]:
                return game["gamePk"], lu
        except Exception:
            pass
        return game["gamePk"], None

    with ThreadPoolExecutor(max_workers=9) as ex:
        for gk, lu in ex.map(fetch_lineup, games):
            if lu:
                lineups[gk] = lu

    # ── Step 2: Collect all unique player IDs ─────────────────────
    batter_ids  = set()
    pitcher_ids = set()
    for lu in lineups.values():
        for b in lu["away_batters"] + lu["home_batters"]:
            if b.get("id"):
                batter_ids.add(b["id"])
        for side in ("away_pitcher", "home_pitcher"):
            p = lu.get(side)
            if p and p.get("id"):
                pitcher_ids.add(p["id"])

    # ── Step 3: Fetch all stats in one parallel batch ─────────────
    batter_cache  = {}
    pitcher_cache = {}

    def _fetch_batter(pid):
        try:
            return pid, get_player_season_stats(pid, group="hitting")
        except Exception:
            return pid, {}

    def _fetch_pitcher(pid):
        try:
            return pid, get_player_season_stats(pid, group="pitching")
        except Exception:
            return pid, {}

    with ThreadPoolExecutor(max_workers=20) as ex:
        for pid, stats in ex.map(_fetch_batter, batter_ids):
            batter_cache[pid] = stats
        for pid, stats in ex.map(_fetch_pitcher, pitcher_ids):
            pitcher_cache[pid] = stats

    # ── Step 4: Fetch weather for each venue ──────────────────────
    venue_weather = {}
    unique_venues = {g["venue"] for g in games}
    def _fetch_weather(venue):
        try:
            return venue, get_ballpark_weather(venue)
        except Exception:
            return venue, {}

    with ThreadPoolExecutor(max_workers=9) as ex:
        for venue, w in ex.map(_fetch_weather, unique_venues):
            venue_weather[venue] = w

    # ── Step 5: Run simulations ───────────────────────────────────
    results = []
    for game in games:
        lu = lineups.get(game["gamePk"])
        if not lu:
            continue

        away_pitcher = lu.get("away_pitcher") or {}
        home_pitcher = lu.get("home_pitcher") or {}
        away_ps = pitcher_cache.get(away_pitcher.get("id"), {})
        home_ps = pitcher_cache.get(home_pitcher.get("id"), {})
        weather = venue_weather.get(game["venue"], {})

        away_stats = [batter_cache.get(b["id"], {}) for b in lu["away_batters"]]
        home_stats = [batter_cache.get(b["id"], {}) for b in lu["home_batters"]]

        result = run_simulation(
            away_team    = game["away_team"],
            home_team    = game["home_team"],
            away_lineup  = away_stats,
            home_lineup  = home_stats,
            away_pitcher = away_ps,
            home_pitcher = home_ps,
            weather      = weather,
            n            = N_SIMS_ALL,
        )
        result.update({
            "gamePk":            game["gamePk"],
            "venue":             game["venue"],
            "status":            game["status"],
            "away_pitcher_name": away_pitcher.get("name", "TBD"),
            "home_pitcher_name": home_pitcher.get("name", "TBD"),
            "away_pitcher_hand": "R",
            "home_pitcher_hand": "R",
            "away_era":  away_ps.get("era",  "N/A"),
            "away_whip": away_ps.get("whip", "N/A"),
            "away_wl":   f"{away_ps.get('wins',0)}-{away_ps.get('losses',0)}",
            "home_era":  home_ps.get("era",  "N/A"),
            "home_whip": home_ps.get("whip", "N/A"),
            "home_wl":   f"{home_ps.get('wins',0)}-{home_ps.get('losses',0)}",
            "away_batters": lu["away_batters"],
            "home_batters": lu["home_batters"],
            "weather":   weather,
        })
        results.append(result)

    results.sort(key=lambda r: r["gamePk"])

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
