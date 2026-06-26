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

def get_team_bullpen_stats(team_id, season=None):
    """
    Returns the aggregated ERA and WHIP for a team's relief pitchers (bullpen).

    We fetch all pitchers on the team's roster, pull their season stats,
    filter to those with gamesStarted < 3 (relief pitchers), and average
    their ERA and WHIP weighted by innings pitched.

    This replaces the league-average bullpen modifier we used before,
    so teams with dominant bullpens (e.g. Dodgers) or terrible ones
    actually affect late-inning probabilities.

    Returns a dict like: {"era": "3.21", "whip": "1.18"}
    Falls back to league average if data unavailable.
    """
    if season is None:
        season = date.today().year

    LEAGUE_AVG = {"era": "4.20", "whip": "1.30"}

    try:
        # Get team roster — pitchers only
        roster_data = _get(f"/teams/{team_id}/roster", params={
            "rosterType": "active",
            "season":     season,
        })
        pitchers = [
            p["person"]["id"]
            for p in roster_data.get("roster", [])
            if p.get("position", {}).get("code") == "1"   # position code 1 = pitcher
        ]

        if not pitchers:
            return LEAGUE_AVG

        # Fetch stats for all pitchers (limit to avoid too many calls)
        total_ip  = 0.0
        era_sum   = 0.0
        whip_sum  = 0.0

        for pid in pitchers[:25]:   # cap at 25 pitchers
            try:
                stats = get_player_season_stats(pid, group="pitching", season=season)
                gs   = stats.get("gamesStarted", 0)
                ip   = float(stats.get("inningsPitched", 0) or 0)

                # Only include relief pitchers (< 3 starts)
                if gs >= 3 or ip < 5:
                    continue

                try:
                    era  = float(stats.get("era",  0) or 0)
                    whip = float(stats.get("whip", 0) or 0)
                except (ValueError, TypeError):
                    continue

                if era == 0 or whip == 0:
                    continue

                # Weight by innings pitched so a closer with 60 IP matters more
                # than a mop-up guy with 5 IP
                era_sum  += era  * ip
                whip_sum += whip * ip
                total_ip += ip
            except Exception:
                continue

        if total_ip < 10:
            return LEAGUE_AVG

        return {
            "era":  str(round(era_sum  / total_ip, 2)),
            "whip": str(round(whip_sum / total_ip, 2)),
        }

    except Exception:
        return LEAGUE_AVG


def get_pitcher_game_log(pitcher_id, num_games=10, season=None):
    """
    Returns per-start stats for a pitcher's last N starts.
    Used to build a fatigue curve — which innings does this pitcher tend to
    get hit hard, and when do they typically get pulled?

    Returns a list of dicts, each with:
      ip  — innings pitched as a float  (6.1 → 6.33 innings, 4.2 → 4.67 innings)
      er  — earned runs allowed in that start

    If a pitcher was a reliever in a game (gamesStarted=0) that game is skipped,
    so short outings from bullpen appearances don't distort the fatigue profile.
    """
    if season is None:
        season = date.today().year

    try:
        data = _get(f"/people/{pitcher_id}/stats", params={
            "stats":    "gameLog",
            "group":    "pitching",
            "season":   season,
            "gameType": "R",
        })

        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return []

        # Only include starts (gamesStarted >= 1), not relief appearances
        starts = [s for s in splits if s.get("stat", {}).get("gamesStarted", 0) >= 1]

        # Take the most recent num_games starts
        recent = starts[-num_games:]

        games = []
        for game in recent:
            stat   = game.get("stat", {})
            ip_str = str(stat.get("inningsPitched", "0") or "0")
            try:
                # MLB stores IP as "6.1" = 6 full innings + 1 out (not decimals!)
                # .1 = 1 out = 1/3 inning, .2 = 2 outs = 2/3 inning
                parts        = ip_str.split(".")
                full_innings = int(parts[0])
                partial_outs = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                ip           = full_innings + partial_outs / 3.0
            except (ValueError, IndexError):
                ip = 0.0

            games.append({
                "ip":   round(ip, 2),
                "er":   int(stat.get("earnedRuns", 0) or 0),
                "date": game.get("date", ""),   # YYYY-MM-DD of the start
            })

        return games

    except Exception:
        return []


def get_batter_vs_pitcher(batter_id, pitcher_id):
    """
    Returns a batter's CAREER stats against one specific pitcher.

    Uses the vsPlayer stat type — no season filter, so we get the full career
    history. Only meaningful with 20+ career plate appearances; smaller samples
    are too noisy to be useful.

    When a batter has dominated a pitcher (e.g. .380 career AVG in 50 PA),
    that edge should show up in the simulation — this function gives us the data.

    Returns a stat dict with plateAppearances, hits, homeRuns, etc.
    Returns {} if insufficient data or API error.
    """
    if not batter_id or not pitcher_id:
        return {}
    try:
        data = _get(f"/people/{batter_id}/stats", params={
            "stats":             "vsPlayer",
            "group":             "hitting",
            "opposingPlayerId":  pitcher_id,
            "gameType":          "R",
        })
        stats_list = data.get("stats", [])
        if not stats_list:
            return {}
        splits = stats_list[0].get("splits", [])
        if not splits:
            return {}
        return splits[0].get("stat", {})
    except Exception:
        return {}


def get_batter_risp_stats(player_id, season=None):
    """
    Returns a batter's stats specifically with runners in scoring position
    (runner on 2nd or 3rd base).

    sitCode "risp" = Runners In Scoring Position.

    This is the single most important clutch stat — some batters hit .340
    with RISP while others choke at .210. The difference directly translates
    to whether runs score when the game is on the line.

    When the simulation has a runner on 2nd or 3rd, we switch to these probs
    instead of the batter's regular stats.
    """
    return get_batter_sitcode_stats(player_id, "risp", season=season)


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
