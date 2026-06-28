"""
simulate.py
-----------
Fetches real MLB data for a game and runs the Monte Carlo simulation.

Usage:
    python simulate.py
"""

import time

from data.mlb_api import (
    get_game_lineup,
    get_player_season_stats,
    get_today_schedule,
)
from simulation.engine import run_simulation


def fetch_batter_stats(batters: list) -> list:
    """
    Fetch season hitting stats for all batters in a lineup.
    Returns a list of stat dicts (one per batter, in batting order).
    """
    stats_list = []
    for b in batters:
        if b.get("id"):
            stats = get_player_season_stats(b["id"], group="hitting")
        else:
            stats = {}   # empty = simulation uses league-average fallback
        stats_list.append(stats)
    return stats_list


def print_banner(title):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)


def main():
    # ── 1. Show today's games ──────────────────────────────────
    print_banner("MLB GAME SIMULATOR")
    print("\n  Fetching today's schedule...")

    games = get_today_schedule()
    if not games:
        print("  No games scheduled today.")
        return

    print("\n  TODAY'S GAMES:\n")
    for i, g in enumerate(games):
        status = g["status"]
        print(f"    [{i}] {g['away_team']:28s} @ {g['home_team']:28s}  ({status})")

    # ── 2. Pick a game ─────────────────────────────────────────
    print()
    try:
        raw = input("  Enter a game number to simulate: ").strip()
        choice = int(raw)
        if choice < 0 or choice >= len(games):
            raise ValueError
        game = games[choice]
    except (ValueError, EOFError):
        print("  Invalid choice — defaulting to game 0.")
        game = games[0]

    away = game["away_team"]
    home = game["home_team"]

    # ── 3. Fetch lineup ────────────────────────────────────────
    print(f"\n  Loading lineup for {away} @ {home}...")
    lineup = get_game_lineup(game["gamePk"])

    if not lineup["away_batters"] or not lineup["home_batters"]:
        print("\n  Lineup not set yet (game may not have started).")
        print("  Try again closer to first pitch, or pick a game already in progress.")
        return

    # Show the lineups
    for side in ("away", "home"):
        team = away if side == "away" else home
        print(f"\n  {side.upper()} LINEUP — {team}")
        for b in lineup[f"{side}_batters"]:
            slot = (b["batting_order"] // 100) if b["batting_order"] else "?"
            print(f"    {slot}. {b['name']}  ({b['position']})")
        p = lineup[f"{side}_pitcher"]
        if p:
            print(f"  ▶ Starting pitcher: {p['name']}")

    # ── 4. Fetch player stats ──────────────────────────────────
    print("\n  Fetching season stats for all players...")

    away_batter_stats = fetch_batter_stats(lineup["away_batters"])
    home_batter_stats = fetch_batter_stats(lineup["home_batters"])

    away_pitcher_stats = {}
    home_pitcher_stats = {}

    if lineup["away_pitcher"] and lineup["away_pitcher"].get("id"):
        away_pitcher_stats = get_player_season_stats(
            lineup["away_pitcher"]["id"], group="pitching"
        )
    if lineup["home_pitcher"] and lineup["home_pitcher"].get("id"):
        home_pitcher_stats = get_player_season_stats(
            lineup["home_pitcher"]["id"], group="pitching"
        )

    away_p = (lineup["away_pitcher"] or {}).get("name", "Unknown pitcher")
    home_p = (lineup["home_pitcher"] or {}).get("name", "Unknown pitcher")

    print("\n  Pitching matchup:")
    print(f"    {away_p:30s}  ERA: {away_pitcher_stats.get('era', 'N/A'):>5}  "
          f"WHIP: {away_pitcher_stats.get('whip', 'N/A'):>5}")
    print(f"    {home_p:30s}  ERA: {home_pitcher_stats.get('era', 'N/A'):>5}  "
          f"WHIP: {home_pitcher_stats.get('whip', 'N/A'):>5}")

    # ── 5. Run the simulation ──────────────────────────────────
    n = 100_000
    print(f"\n  Running {n:,} simulations...")
    print("  (this takes about 15–30 seconds)\n")

    start = time.time()

    results = run_simulation(
        away_team    = away,
        home_team    = home,
        away_lineup  = away_batter_stats,
        home_lineup  = home_batter_stats,
        away_pitcher = away_pitcher_stats,
        home_pitcher = home_pitcher_stats,
        n            = n,
    )

    elapsed = time.time() - start

    # ── 6. Print results ───────────────────────────────────────
    print_banner("SIMULATION RESULTS")
    print()

    away_pct = results["away_win_pct"]
    home_pct = results["home_win_pct"]
    away_runs = results["away_avg_runs"]
    home_runs = results["home_avg_runs"]

    # Visual probability bar (50 chars wide)
    bar_width  = 50
    away_chars = round(away_pct / 100 * bar_width)
    home_chars = bar_width - away_chars
    bar = "█" * away_chars + "░" * home_chars

    print(f"  {away:>28s}  |  {home}")
    print(f"  {'WIN PROBABILITY':>28s}  |")
    print(f"  {away_pct:>27.1f}%  |  {home_pct:.1f}%")
    print()
    print(f"  [{bar}]")
    print()
    print(f"  Predicted score:  {away} {away_runs:.1f}  —  {home} {home_runs:.1f}")
    print()
    print(f"  Based on {results['simulations']:,} simulated games  |  Ran in {elapsed:.1f}s")
    print_banner("DONE")
    print()


if __name__ == "__main__":
    main()
