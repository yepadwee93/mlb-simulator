"""
mlb_api.py
----------
All the code for talking to MLB's free public Stats API.
Base URL: https://statsapi.mlb.com/api/v1/

No API key needed — MLB makes this data public.
"""

import requests
from datetime import date

# The root URL for every API call we make
BASE_URL = "https://statsapi.mlb.com/api/v1"


def _get(path, params=None):
    """
    Internal helper: send a GET request and return the JSON response.
    Raises an exception if the request fails so we hear about it immediately.
    """
    url = f"{BASE_URL}{path}"
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()   # throws an error if status != 200
    return response.json()


# ──────────────────────────────────────────────
# 1. TODAY'S SCHEDULE
# ──────────────────────────────────────────────

def get_today_schedule(game_date=None):
    """
    Returns a list of today's MLB games.
    Each game dict has: gamePk (unique game ID), teams, status, venue, time.

    game_date: optional string like '2026-06-25'. Defaults to today.
    """
    if game_date is None:
        game_date = date.today().isoformat()   # e.g. "2026-06-25"

    data = _get("/schedule", params={
        "sportId": 1,          # sportId=1 means MLB (as opposed to minor leagues etc.)
        "date": game_date,
        "hydrate": "team"      # include full team names in the response
    })

    games = []
    # The API wraps results in dates[] -> games[]
    for day in data.get("dates", []):
        for game in day.get("games", []):
            games.append({
                "gamePk":       game["gamePk"],          # unique game ID used in other calls
                "status":       game["status"]["detailedState"],
                "home_team":    game["teams"]["home"]["team"]["name"],
                "away_team":    game["teams"]["away"]["team"]["name"],
                "home_id":      game["teams"]["home"]["team"]["id"],
                "away_id":      game["teams"]["away"]["team"]["id"],
                "game_time":    game.get("gameDate", ""),   # UTC time string
                "venue":        game["venue"]["name"],
            })
    return games


# ──────────────────────────────────────────────
# 2. LINEUPS & STARTING PITCHERS FOR ONE GAME
# ──────────────────────────────────────────────

def get_game_lineup(game_pk):
    """
    Given a gamePk, returns the batting lineups and starting pitchers
    for both teams. Works for live, completed, or scheduled games
    (though pre-game lineups may not be available until ~30 min before first pitch).

    Returns a dict with keys: home_batters, away_batters,
                               home_pitcher, away_pitcher
    Each batter is {id, name, position, batting_order}
    Each pitcher is {id, name}
    """
    # The boxscore endpoint has the full lineup once it's set
    data = _get(f"/game/{game_pk}/boxscore")

    result = {
        "home_batters": [],
        "away_batters": [],
        "home_pitcher": None,
        "away_pitcher": None,
    }

    for side in ("home", "away"):
        team_data = data["teams"][side]
        players   = team_data.get("players", {})  # keyed like "ID123456"

        # Each player object has a "battingOrder" string: "100" = 1st, "200" = 2nd, etc.
        # We collect all players that have one, then sort by that number.
        batters = []
        starters = []   # pitchers with gamesStarted = 1 in their box stats

        for pid_key, p in players.items():
            person = p.get("person", {})
            pos    = p.get("position", {})

            # If they have a battingOrder, they appeared in the lineup
            bat_order = p.get("battingOrder")
            if bat_order:
                batters.append({
                    "id":            person.get("id"),
                    "name":          person.get("fullName"),
                    "position":      pos.get("abbreviation"),
                    "batting_order": int(bat_order),  # "100" → 100 (so we can sort)
                })

            # If their box-score pitching line shows gamesStarted=1, they started
            pitch_stats = p.get("stats", {}).get("pitching", {})
            if pitch_stats.get("gamesStarted", 0) == 1:
                starters.append({
                    "id":   person.get("id"),
                    "name": person.get("fullName"),
                })

        # Sort batters by their order slot (100, 200, 300, ...)
        batters.sort(key=lambda b: b["batting_order"])
        result[f"{side}_batters"] = batters

        # Starter: prefer gamesStarted flag; fall back to first in pitchers list
        if starters:
            result[f"{side}_pitcher"] = starters[0]
        else:
            pitchers = team_data.get("pitchers", [])
            if pitchers:
                p_data = players.get(f"ID{pitchers[0]}", {})
                person = p_data.get("person", {})
                result[f"{side}_pitcher"] = {
                    "id":   person.get("id"),
                    "name": person.get("fullName"),
                }

    return result


# ──────────────────────────────────────────────
# 3. SEASON STATS FOR A PLAYER
# ──────────────────────────────────────────────

def get_player_season_stats(player_id, group="hitting", season=None):
    """
    Returns season-long stats for one player.

    group: "hitting" or "pitching"
    season: year as int or string, defaults to current year

    Returns a flat dict of stat fields, e.g.:
      Hitting: avg, obp, slg, ops, homeRuns, rbi, strikeOuts, walks, ...
      Pitching: era, whip, strikeOuts, wins, losses, inningsPitched, ...
    """
    if season is None:
        season = date.today().year

    data = _get(f"/people/{player_id}/stats", params={
        "stats":    "season",       # season totals (vs. career, etc.)
        "group":    group,          # hitting or pitching
        "season":   season,
        "gameType": "R",            # R = Regular Season only (not playoffs)
    })

    stats_list = data.get("stats", [])
    if not stats_list:
        return {}   # no stats available (player hasn't played or just called up)

    splits = stats_list[0].get("splits", [])
    if not splits:
        return {}

    return splits[0].get("stat", {})


def get_player_info(player_id):
    """
    Returns basic bio info for a player (name, position, bats, throws, team).
    """
    data = _get(f"/people/{player_id}")
    people = data.get("people", [])
    if not people:
        return {}
    p = people[0]
    return {
        "id":           p.get("id"),
        "name":         p.get("fullName"),
        "position":     p.get("primaryPosition", {}).get("abbreviation"),
        "bats":         p.get("batSide", {}).get("code"),       # R, L, or S (switch)
        "throws":       p.get("pitchHand", {}).get("code"),     # R or L
        "current_team": p.get("currentTeam", {}).get("name"),
    }


# ──────────────────────────────────────────────
# 4. LAST-N-GAMES ("RECENT FORM") STATS
# ──────────────────────────────────────────────

def get_player_recent_stats(player_id, group="hitting", num_games=10, season=None):
    """
    Returns a rolling average of stats over the player's last N games.
    This is how we capture "recent form" — someone who's been hot or cold lately.

    Uses the 'gameLog' stat type, which gives one row per game played,
    then we sum up the last num_games entries.

    Returns: a dict of aggregated stats over the last N games, plus
             a 'games_found' count so we know how many games were available.
    """
    if season is None:
        season = date.today().year

    data = _get(f"/people/{player_id}/stats", params={
        "stats":    "gameLog",      # one row per game (most recent first)
        "group":    group,
        "season":   season,
        "gameType": "R",
    })

    splits = data.get("stats", [{}])[0].get("splits", [])
    if not splits:
        return {"games_found": 0}

    # gameLog is sorted oldest→newest, so take the LAST num_games entries
    recent = splits[-num_games:]

    # Aggregate the key stats by summing them up
    totals = {}
    for game in recent:
        for key, val in game.get("stat", {}).items():
            if isinstance(val, (int, float)):
                totals[key] = totals.get(key, 0) + val

    # Compute derived rates (avoid division by zero)
    ab   = totals.get("atBats", 0)
    pa   = totals.get("plateAppearances", 0)
    hits = totals.get("hits", 0)
    bb   = totals.get("baseOnBalls", 0)
    hbp  = totals.get("hitByPitch", 0)
    tb   = totals.get("totalBases", 0)

    if ab > 0:
        totals["avg_recent"]  = round(hits / ab, 3)
        totals["slg_recent"]  = round(tb / ab, 3)
    if (ab + bb + hbp) > 0:
        totals["obp_recent"]  = round((hits + bb + hbp) / (ab + bb + hbp), 3)

    totals["games_found"] = len(recent)
    return totals
