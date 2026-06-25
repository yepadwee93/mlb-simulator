"""
main.py
-------
Entry point for the MLB Simulator — Step 1.
Run this file to confirm that real MLB data is flowing correctly.

Usage:
    python main.py
"""

import json
from data.mlb_api import (
    get_today_schedule,
    get_game_lineup,
    get_player_season_stats,
    get_player_recent_stats,
    get_player_info,
)


def print_section(title):
    """Prints a nice section header so the output is easy to read."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():

    # ── 1. TODAY'S SCHEDULE ──────────────────────────────────────
    print_section("TODAY'S MLB SCHEDULE")

    games = get_today_schedule()

    if not games:
        print("No games scheduled today.")
        return

    for i, g in enumerate(games):
        print(f"  [{i}] {g['away_team']} @ {g['home_team']}")
        print(f"       Status: {g['status']} | Venue: {g['venue']}")
        print(f"       gamePk: {g['gamePk']}")
        print()

    # ── 2. PICK THE FIRST GAME AND PULL LINEUPS ──────────────────
    # We'll use the first game in the list as our sample.
    # If it's pre-game, lineups may not be set yet — we'll say so.
    sample_game = games[0]
    print_section(f"SAMPLE GAME: {sample_game['away_team']} @ {sample_game['home_team']}")
    print(f"  gamePk: {sample_game['gamePk']}  |  Status: {sample_game['status']}")

    lineup = get_game_lineup(sample_game["gamePk"])

    # Print the batting lineups
    for side in ("away", "home"):
        team_name = sample_game[f"{side}_team"]
        print(f"\n  {side.upper()} LINEUP — {team_name}")

        batters = lineup[f"{side}_batters"]
        if batters:
            for b in batters:
                print(f"    {b['batting_order'] or '?':>3}. {b['name']} ({b['position']})")
        else:
            print("    (lineup not yet available)")

        pitcher = lineup[f"{side}_pitcher"]
        if pitcher:
            print(f"\n  {side.upper()} STARTING PITCHER: {pitcher['name']} (id={pitcher['id']})")
        else:
            print(f"\n  {side.upper()} STARTING PITCHER: not yet announced")

    # ── 3. SEASON STATS FOR THE BATTERS ──────────────────────────
    print_section("SEASON BATTING STATS (sample: first 3 hitters from each team)")

    for side in ("away", "home"):
        team_name = sample_game[f"{side}_team"]
        batters   = lineup[f"{side}_batters"][:3]   # just first 3 to keep output manageable

        print(f"\n  {team_name}")
        for b in batters:
            if not b["id"]:
                continue
            stats = get_player_season_stats(b["id"], group="hitting")
            if stats:
                # avg/obp/slg come back as strings like ".244" — print as-is
                print(f"    {b['name']:25s}  "
                      f"AVG:{stats.get('avg','N/A'):>5}  "
                      f"OBP:{stats.get('obp','N/A'):>5}  "
                      f"SLG:{stats.get('slg','N/A'):>5}  "
                      f"HR:{stats.get('homeRuns',0):>3}  "
                      f"RBI:{stats.get('rbi',0):>3}  "
                      f"K:{stats.get('strikeOuts',0):>3}  "
                      f"BB:{stats.get('baseOnBalls',0):>3}")
            else:
                print(f"    {b['name']:25s}  (no stats yet)")

    # ── 4. SEASON STATS FOR BOTH STARTING PITCHERS ───────────────
    print_section("SEASON PITCHING STATS — STARTING PITCHERS")

    for side in ("away", "home"):
        pitcher = lineup[f"{side}_pitcher"]
        if not pitcher or not pitcher["id"]:
            print(f"  {side.upper()} pitcher: not available")
            continue

        stats = get_player_season_stats(pitcher["id"], group="pitching")
        if stats:
            print(f"\n  {pitcher['name']} ({sample_game[f'{side}_team']})")
            print(f"    ERA:  {stats.get('era', 'N/A')}")
            print(f"    WHIP: {stats.get('whip', 'N/A')}")
            print(f"    W-L:  {stats.get('wins',0)}-{stats.get('losses',0)}")
            print(f"    IP:   {stats.get('inningsPitched','N/A')}")
            print(f"    K:    {stats.get('strikeOuts',0)}")
            print(f"    BB:   {stats.get('baseOnBalls',0)}")
            print(f"    HR allowed: {stats.get('homeRuns',0)}")
        else:
            print(f"  {pitcher['name']}: no stats yet")

    # ── 5. RECENT FORM (LAST 10 GAMES) FOR ONE PLAYER ────────────
    # We'll use the leadoff hitter from the away team as our demo subject.
    print_section("RECENT FORM — LAST 10 GAMES (leadoff hitter, away team)")

    demo_batter = (lineup["away_batters"] or lineup["home_batters"] or [None])[0]
    if demo_batter and demo_batter["id"]:
        info   = get_player_info(demo_batter["id"])
        recent = get_player_recent_stats(demo_batter["id"], group="hitting", num_games=10)

        print(f"\n  Player: {info.get('name')}  |  Bats: {info.get('bats')}  |  Team: {info.get('current_team')}")
        print(f"  Games sampled: {recent.get('games_found', 0)}")
        print()

        # Key recent stats
        keys_to_show = [
            ("atBats",           "AB"),
            ("hits",             "H"),
            ("homeRuns",         "HR"),
            ("rbi",              "RBI"),
            ("strikeOuts",       "K"),
            ("baseOnBalls",      "BB"),
            ("avg_recent",       "AVG (recent)"),
            ("obp_recent",       "OBP (recent)"),
            ("slg_recent",       "SLG (recent)"),
        ]
        for key, label in keys_to_show:
            val = recent.get(key, "N/A")
            print(f"    {label:<18} {val}")
    else:
        print("  No batter available for recent-form demo.")

    # ── DONE ──────────────────────────────────────────────────────
    print_section("DONE — Data pipeline confirmed working!")
    print("  Next step: build the Monte Car