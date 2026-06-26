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

# ── Ballpark coordinates for weather lookups ─────────────────────
# (latitude, longitude) for each MLB venue name
BALLPARK_COORDS = {
    "Tropicana Field":      (27.768,  -82.653),
    "PNC Park":             (40.447,  -80.006),
    "Oracle Park":          (37.779,  -122.389),
    "Comerica Park":        (42.339,  -83.049),
    "Nationals Park":       (38.873,  -77.007),
    "Rogers Centre":        (43.641,  -79.389),
    "Citi Field":           (40.757,  -73.846),
    "Fenway Park":          (42.347,  -71.097),
    "Busch Stadium":        (38.623,  -90.193),
    "Yankee Stadium":       (40.829,  -73.926),
    "Dodger Stadium":       (34.074,  -118.240),
    "Wrigley Field":        (41.948,  -87.656),
    "Great American Ball Park": (39.097, -84.507),
    "Truist Park":          (33.891,  -84.468),
    "American Family Field":(43.028,  -87.971),
    "Guaranteed Rate Field":(41.830,  -87.634),
    "Progressive Field":    (41.496,  -81.685),
    "Kauffman Stadium":     (39.051,  -94.480),
    "Target Field":         (44.982,  -93.278),
    "Globe Life Field":     (32.747,  -97.083),
    "Minute Maid Park":     (29.757,  -95.355),
    "Angel Stadium":        (33.800,  -117.883),
    "T-Mobile Park":        (47.591,  -122.333),
    "Oakland Coliseum":     (37.752,  -122.201),
    "Sutter Health Park":   (38.580,  -121.500),
    "Chase Field":          (33.445,  -112.067),
    "Coors Field":          (39.756,  -104.994),
    "Petco Park":           (32.708,  -117.157),
    "loanDepot park":       (25.778,  -80.220),
    "Citizens Bank Park":   (39.906,  -75.166),
    "Daikin Park":          (29.757,  -95.355),
    "Oriole Park at Camden Yards": (39.284, -76.622),
    "Sahlen Field":         (42.886,  -78.872),
}


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
        "hydrate": "team,probablePitcher",   # also grab probable starters
    })

    games = []
    # The API wraps results in dates[] -> games[]
    for day in data.get("dates", []):
        for game in day.get("games", []):
            away = game["teams"]["away"]
            home = game["teams"]["home"]

            # probablePitcher is available days before the game
            away_prob = away.get("probablePitcher", {})
            home_prob = home.get("probablePitcher", {})

            games.append({
                "gamePk":             game["gamePk"],
                "status":             game["status"]["detailedState"],
                "home_team":          home["team"]["name"],
                "away_team":          away["team"]["name"],
                "home_id":            home["team"]["id"],
                "away_id":            away["team"]["id"],
                "game_time":          game.get("gameDate", ""),
                "venue":              game["venue"]["name"],
                # Probable pitchers (empty string if not yet announced)
                "away_probable":      away_prob.get("fullName", "TBD"),
                "home_probable":      home_prob.get("fullName", "TBD"),
                "away_probable_id":   away_prob.get("id"),
                "home_probable_id":   home_prob.get("id"),
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

def get_player_recent_stats(player_id, group="hitting", num_games=20, season=None):
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


# ──────────────────────────────────────────────
# 5. BATTER VS LEFT / RIGHT PITCHER SPLITS
# ──────────────────────────────────────────────

def get_pitcher_hand(pitcher_id):
    """
    Returns 'L' or 'R' — which hand the pitcher throws with.
    We use this to decide which batter split stats to pull.
    """
    info = get_player_info(pitcher_id)
    return info.get("throws", "R")   # default to R if unknown


def get_batter_split_stats(player_id, pitcher_hand, season=None):
    """
    Returns a batter's stats specifically against left-handed or right-handed pitchers.

    pitcher_hand: 'L' or 'R'

    This is much more accurate than season totals because:
    - Some batters hit .300 vs righties but only .200 vs lefties
    - Knowing who's on the mound means we can use the right split

    sitCodes: 'vl' = vs left-handed pitchers, 'vr' = vs right-handed pitchers
    """
    sit_code = "vl" if pitcher_hand == "L" else "vr"
    return get_batter_sitcode_stats(player_id, sit_code, season=season)


def get_batter_sitcode_stats(player_id, sit_code, season=None):
    """
    Returns a batter's stats for any MLB situation code.

    Common sitCodes:
      'vl'  — vs left-handed pitchers
      'vr'  — vs right-handed pitchers
      'vsp' — vs starting pitchers
      'vrp' — vs relief pitchers (bullpen)

    Useful for showing whether a batter performs better early (vs SP) or
    later in the game when the bullpen comes in (vs RP).
    """
    if season is None:
        season = date.today().year

    data = _get(f"/people/{player_id}/stats", params={
        "stats":    "statSplits",
        "group":    "hitting",
        "season":   season,
        "gameType": "R",
        "sitCodes": sit_code,
    })

    stats_list = data.get("stats", [])
    if not stats_list:
        return {}

    splits = stats_list[0].get("splits", [])
    if not splits:
        return {}

    return splits[0].get("stat", {})


# ──────────────────────────────────────────────
# 6. WEATHER FOR A BALLPARK
# ──────────────────────────────────────────────

def get_ballpark_weather(venue_name):
    """
    Fetches the current weather forecast for a ballpark using Open-Meteo.
    Free API, no key needed.

    Returns a dict with:
      temp_f        — temperature in Fahrenheit
      wind_mph      — wind speed in mph
      wind_deg      — wind direction in degrees (0=N, 90=E, 180=S, 270=W)
      wind_label    — human-readable wind direction (e.g. "15 mph NE")
      condition     — short description

    If the venue isn't in our coords list, returns empty dict.
    """
    coords = BALLPARK_COORDS.get(venue_name)
    if not coords:
        return {}

    lat, lon = coords

    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":        lat,
                "longitude":       lon,
                "current":         "temperature_2m,wind_speed_10m,wind_direction_10m,weather_code",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone":        "auto",
            },
            timeout=5,
        )
        resp.raise_for_status()
        data    = resp.json()
        current = data.get("current", {})

        temp_f   = current.get("temperature_2m",    70)
        wind_mph = current.get("wind_speed_10m",     0)
        wind_deg = current.get("wind_direction_10m", 0)

        # Convert wind degrees to compass label
        dirs   = ["N","NE","E","SE","S","SW","W","NW"]
        label  = dirs[round(wind_deg / 45) % 8]

        return {
            "temp_f":     round(temp_f),
            "wind_mph":   round(wind_mph),
            "wind_deg":   wind_deg,
            "wind_label": f"{round(wind_mph)} mph {label}",
        }

    except Exception:
        return {}   # weather unavailable — simulation still runs without it
