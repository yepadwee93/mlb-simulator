"""
mlb_api.py
----------
All the code for talking to MLB's free public Stats API.
Base URL: https://statsapi.mlb.com/api/v1/

No API key needed — MLB makes this data public.
"""

import requests
import time
from datetime import date

# The root URL for every API call we make
BASE_URL = "https://statsapi.mlb.com/api/v1"

# ── In-memory TTL cache for all MLB API calls ─────────────────────
# Stats/lineups don't change mid-sim, so caching for 10 min is safe.
_API_CACHE = {}          # key -> (timestamp, data)
_API_CACHE_TTL = 600     # seconds (10 min)

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

# Direction wind must blow TO push balls toward CF/RF for HR boost
# Degrees (meteorological: 0=N, 90=E) — if wind_deg is within 45deg of this, it's blowing OUT
PARK_CF_DIRECTION = {
    "Coors Field":              315,   # NW (blowing out toward CF)
    "Wrigley Field":            270,   # W (famous wind blowing out)
    "Yankee Stadium":           270,
    "Fenway Park":              315,
    "Citizens Bank Park":       270,
    "Great American Ball Park": 315,
    "Kauffman Stadium":         315,
    "Minute Maid Park":         270,
    "Oracle Park":              0,     # N (rare — usually blows IN from bay)
    "T-Mobile Park":            0,
    "Petco Park":               270,
    "Dodger Stadium":           270,
    "Target Field":             315,
    "Progressive Field":        315,
    "Globe Life Field":         0,     # indoors — wind direction irrelevant
    "Chase Field":              0,     # retractable roof
    "Tropicana Field":          0,     # dome
}


def _get(path, params=None):
    """
    Internal helper: GET with in-memory TTL cache (10 min).
    Identical params → same cache key, so repeated batter stat fetches in one sim are free.
    """
    import hashlib, json as _json
    key = path + (("?" + _json.dumps(params, sort_keys=True)) if params else "")
    now = time.time()
    cached = _API_CACHE.get(key)
    if cached and (now - cached[0]) < _API_CACHE_TTL:
        return cached[1]
    url = f"{BASE_URL}{path}"
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    _API_CACHE[key] = (now, data)
    return data


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

        # Starter: prefer gamesStarted flag; fall back to first pitcher in list
        # Note: gamesStarted=1 may not be set during live games, so always
        # fall back to the first pitcher listed (they entered first = starter)
        if starters:
            result[f"{side}_pitcher"] = starters[0]
        else:
            pitchers = team_data.get("pitchers", [])
            if pitchers:
                p_data = players.get(f"ID{pitchers[0]}", {})
                person = p_data.get("person", {})
                if person.get("id"):
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
                "ip":          round(ip, 2),
                "er":          int(stat.get("earnedRuns",     0) or 0),
                "pitches":     int(stat.get("numberOfPitches", 0) or 0),
                "strikes":     int(stat.get("strikes",         0) or 0),
                "ks":          int(stat.get("strikeOuts",      0) or 0),
                "bb":          int(stat.get("baseOnBalls",     0) or 0),
                "date":        game.get("date", ""),
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


def get_pitcher_arsenal(pitcher_id, season=None) -> dict:
    """
    Build a pitcher's stat-based 'arsenal' profile from season stats.
    Returns computed rates: K%, BB%, HR/9, GB%, FIP, and qualitative grades.
    """
    if season is None:
        season = date.today().year

    raw = get_player_season_stats(pitcher_id, group="pitching", season=season)
    if not raw:
        return {}

    try:
        ip_str = str(raw.get("inningsPitched", "0") or "0")
        parts  = ip_str.split(".")
        full   = int(parts[0])
        outs   = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        ip     = full + outs / 3.0
    except Exception:
        ip = 0.0

    if ip < 1:
        return {}

    bf   = int(raw.get("battersFaced", 0) or 0) or max(int(ip * 4.3), 1)
    ks   = int(raw.get("strikeOuts",   0) or 0)
    bb   = int(raw.get("baseOnBalls",  0) or 0)
    hrs  = int(raw.get("homeRuns",     0) or 0)
    er   = int(raw.get("earnedRuns",   0) or 0)
    go   = int(raw.get("groundOuts",   0) or 0)
    fo   = int(raw.get("flyOuts",      0) or 0)
    hits = int(raw.get("hits",         0) or 0)

    k_pct  = round(ks / bf * 100, 1)   if bf  > 0 else None
    bb_pct = round(bb / bf * 100, 1)   if bf  > 0 else None
    hr9    = round(hrs / ip * 9,  2)   if ip  > 0 else None
    gb_pct = round(go / (go + fo) * 100, 1) if (go + fo) > 0 else None
    era    = round(er / ip * 9,   2)   if ip  > 0 else None
    # FIP = (13*HR + 3*BB - 2*K) / IP + 3.10 (constant)
    fip    = round((13 * hrs + 3 * bb - 2 * ks) / ip + 3.10, 2) if ip > 0 else None
    whip   = round((bb + hits) / ip, 2) if ip > 0 else None

    def grade_k(v):
        if v is None: return "—"
        if v >= 28: return "Elite"
        if v >= 23: return "Above Avg"
        if v >= 18: return "Average"
        return "Below Avg"

    def grade_bb(v):
        if v is None: return "—"
        if v <= 5:  return "Elite"
        if v <= 7:  return "Above Avg"
        if v <= 9:  return "Average"
        return "Below Avg"

    def grade_hr9(v):
        if v is None: return "—"
        if v <= 0.9: return "Elite"
        if v <= 1.2: return "Above Avg"
        if v <= 1.5: return "Average"
        return "Below Avg"

    def grade_gb(v):
        if v is None: return "—"
        if v >= 52: return "Elite GB"
        if v >= 46: return "Above Avg"
        if v >= 40: return "Average"
        return "Fly Ball"

    wins   = int(raw.get("wins",   0) or 0)
    losses = int(raw.get("losses", 0) or 0)
    gs     = int(raw.get("gamesStarted", 0) or 0)
    g      = int(raw.get("gamesPitched", 0) or 0)

    return {
        "ip":           round(ip, 1),
        "gs":           gs,
        "g":            g,
        "wins":         wins,
        "losses":       losses,
        "era":          era,
        "fip":          fip,
        "whip":         whip,
        "k_pct":        k_pct,
        "bb_pct":       bb_pct,
        "hr9":          hr9,
        "gb_pct":       gb_pct,
        "k_grade":      grade_k(k_pct),
        "bb_grade":     grade_bb(bb_pct),
        "hr9_grade":    grade_hr9(hr9),
        "gb_grade":     grade_gb(gb_pct),
        "k_bb_ratio":   round(k_pct / bb_pct, 2) if (k_pct and bb_pct and bb_pct > 0) else None,
    }


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

def get_ballpark_weather(venue_name, game_time_utc: str = None):
    """
    Fetches the weather forecast for a ballpark at game time using Open-Meteo.
    Free API, no key needed.

    If game_time_utc is provided (ISO string, e.g. "2025-06-15T23:10:00Z"),
    fetches the hourly forecast for that specific hour so the sim uses
    game-time conditions, not current conditions.

    Returns a dict with:
      temp_f         — temperature in Fahrenheit at game time
      wind_mph       — wind speed in mph
      wind_deg       — wind direction in degrees (0=N, 90=E, 180=S, 270=W)
      wind_label     — human-readable direction (e.g. "15 mph NE")
      wind_hr_boost  — HR multiplier based on wind direction vs CF
      wind_effect    — "out" / "in" / "neutral"
      precip_pct     — precipitation probability 0-100 (rain chance)
      condition      — short label ("Clear", "Partly Cloudy", "Rain", etc.)
      condition_icon — emoji icon for condition
      is_forecast    — True if this is game-time forecast vs current conditions

    If the venue isn't in our coords list, returns empty dict.
    """
    # Open-Meteo WMO weather code → condition label + emoji
    WMO_CONDITIONS = {
        0:  ("Clear",          "☀️"),
        1:  ("Mostly Clear",   "🌤"),
        2:  ("Partly Cloudy",  "⛅"),
        3:  ("Overcast",       "☁️"),
        45: ("Foggy",          "🌫"),
        48: ("Foggy",          "🌫"),
        51: ("Drizzle",        "🌦"),
        53: ("Drizzle",        "🌦"),
        55: ("Drizzle",        "🌦"),
        61: ("Light Rain",     "🌧"),
        63: ("Rain",           "🌧"),
        65: ("Heavy Rain",     "🌧"),
        71: ("Snow",           "❄️"),
        73: ("Snow",           "❄️"),
        75: ("Heavy Snow",     "🌨"),
        80: ("Showers",        "🌦"),
        81: ("Showers",        "🌦"),
        82: ("Heavy Showers",  "⛈"),
        95: ("Thunderstorm",   "⛈"),
        96: ("Thunderstorm",   "⛈"),
        99: ("Thunderstorm",   "⛈"),
    }

    coords = BALLPARK_COORDS.get(venue_name)
    if not coords:
        return {}

    lat, lon = coords

    # Parse game time to find the target hour
    target_hour_str = None
    is_forecast = False
    if game_time_utc:
        try:
            from datetime import datetime, timezone
            gt = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
            # Round down to the hour
            target_hour_str = gt.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")
            is_forecast = True
        except Exception:
            pass

    try:
        params = {
            "latitude":         lat,
            "longitude":        lon,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit":  "mph",
            "timezone":         "auto",
        }

        if target_hour_str:
            # Fetch hourly data for ±2 days to cover the game time
            from datetime import date, timedelta
            today = date.today()
            params["hourly"]     = "temperature_2m,wind_speed_10m,wind_direction_10m,weather_code,precipitation_probability"
            params["start_date"] = today.isoformat()
            params["end_date"]   = (today + timedelta(days=2)).isoformat()
        else:
            params["current"] = "temperature_2m,wind_speed_10m,wind_direction_10m,weather_code,precipitation_probability"

        resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        temp_f = wind_mph = wind_deg = 70, 0, 0
        precip_pct = 0
        wmo_code = 0

        if target_hour_str and "hourly" in data:
            hourly = data["hourly"]
            times  = hourly.get("time", [])
            # Find the closest hour to game time
            idx = next((i for i, t in enumerate(times) if t == target_hour_str), None)
            if idx is None:
                # Fall back to nearest
                idx = 0
            temp_f     = hourly["temperature_2m"][idx]       if idx < len(hourly.get("temperature_2m", [])) else 70
            wind_mph   = hourly["wind_speed_10m"][idx]       if idx < len(hourly.get("wind_speed_10m", [])) else 0
            wind_deg   = hourly["wind_direction_10m"][idx]   if idx < len(hourly.get("wind_direction_10m", [])) else 0
            wmo_code   = hourly["weather_code"][idx]         if idx < len(hourly.get("weather_code", [])) else 0
            precip_pct = hourly.get("precipitation_probability", [0]*100)[idx] if idx < len(hourly.get("precipitation_probability", [0]*100)) else 0
        else:
            current  = data.get("current", {})
            temp_f   = current.get("temperature_2m",    70)
            wind_mph = current.get("wind_speed_10m",     0)
            wind_deg = current.get("wind_direction_10m", 0)
            wmo_code = current.get("weather_code",       0)
            precip_pct = current.get("precipitation_probability", 0)

        # Condition label and icon
        condition, condition_icon = WMO_CONDITIONS.get(int(wmo_code), ("Unknown", "🌡"))

        # Convert wind degrees to compass label
        dirs  = ["N","NE","E","SE","S","SW","W","NW"]
        label = dirs[round(wind_deg / 45) % 8]

        # Wind direction HR boost: if wind blows OUT toward CF and >= 10 mph
        cf_dir = PARK_CF_DIRECTION.get(venue_name)
        wind_hr_boost = 1.0
        wind_effect   = "neutral"
        if cf_dir is not None and wind_mph >= 10:
            diff = abs((wind_deg - cf_dir + 180) % 360 - 180)
            if diff <= 45:
                boost_factor  = min(1.15, 1.0 + (wind_mph - 10) * 0.005)
                wind_hr_boost = round(boost_factor, 3)
                wind_effect   = "out"
            elif diff >= 135:
                suppress_factor = max(0.90, 1.0 - (wind_mph - 10) * 0.004)
                wind_hr_boost   = round(suppress_factor, 3)
                wind_effect     = "in"

        # Scoring environment summary
        hr_boost_pct = round((wind_hr_boost - 1.0) * 100, 1)
        if temp_f >= 85:
            temp_effect = "hot air boosts HRs"
        elif temp_f <= 50:
            temp_effect = "cold air suppresses HRs"
        else:
            temp_effect = None

        return {
            "temp_f":         round(temp_f),
            "wind_mph":       round(wind_mph),
            "wind_deg":       wind_deg,
            "wind_label":     f"{round(wind_mph)} mph {label}",
            "wind_hr_boost":  wind_hr_boost,
            "wind_effect":    wind_effect,
            "precip_pct":     int(precip_pct),
            "condition":      condition,
            "condition_icon": condition_icon,
            "is_forecast":    is_forecast,
            "hr_boost_pct":   hr_boost_pct,
            "temp_effect":    temp_effect,
        }

    except Exception:
        return {}   # weather unavailable — simulation still runs without it


def get_team_rest_days(team_name, today=None):
    """
    Return days since this team last played a completed game (looks back 5 days).
    1 = normal rest, 2+ = extra rest, 3+ = well rested.
    Returns 1 as fallback if lookup fails.
    """
    from datetime import date as _date, timedelta
    today_dt = _date.fromisoformat(today[:10]) if today else _date.today()

    for days_back in range(1, 6):
        check_date = (today_dt - timedelta(days=days_back)).isoformat()
        try:
            games = get_today_schedule(game_date=check_date)
            for g in games:
                if team_name in (g["away_team"], g["home_team"]):
                    status = g.get("status", "").lower()
                    if any(w in status for w in ("final", "game over", "completed")):
                        return days_back
        except Exception:
            pass

    return 1


# ── Baseball Savant Statcast data ────────────────────────────────────────────
# Fetches barrel rate and hard-hit % for all batters from Baseball Savant's
# free leaderboard CSV. Cached in memory for the session.

_SAVANT_CACHE = {"data": {}, "year": None}

SAVANT_URL = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
    "?type=batter&year={year}&position=&team=&min=10&csv=true"
)

# Known column name variants across Savant CSV versions
_BARREL_COLS   = ("barrel_batted_rate", "barrel_bip_rate", "barrel_rate")
_HARDHIT_COLS  = ("hard_hit_percent", "hard_hit_rate", "hard_hit_pct")
_EV_COLS       = ("exit_velocity_avg", "avg_exit_velocity", "launch_speed")
_PID_COLS      = ("player_id", "mlbam_id", "batter")


def get_savant_stats_all(year=None):
    """
    Fetch all batters' Statcast barrel rate and hard-hit % from Baseball Savant.
    Returns dict keyed by MLB player_id (int):
      {"barrel_pct": float, "hard_hit_pct": float, "exit_velo_avg": float}
    Returns empty dict if the fetch fails -- sim runs fine without it.
    """
    import csv, io
    from datetime import date as _date

    if year is None:
        year = _date.today().year

    if _SAVANT_CACHE["year"] == year and _SAVANT_CACHE["data"]:
        return _SAVANT_CACHE["data"]

    url = SAVANT_URL.format(year=year)
    try:
        import requests as _req
        resp = _req.get(url, timeout=15,
                        headers={"User-Agent": "Mozilla/5.0 (compatible)"})
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        headers = reader.fieldnames or []

        def pick(cols):
            for c in cols:
                if c in headers:
                    return c
            return None

        col_pid  = pick(_PID_COLS)
        col_brl  = pick(_BARREL_COLS)
        col_hh   = pick(_HARDHIT_COLS)
        col_ev   = pick(_EV_COLS)

        if not col_pid:
            return {}

        result = {}
        for row in reader:
            try:
                pid = int(row[col_pid])
                barrel   = float(row[col_brl]  or 0) if col_brl  else 7.0
                hard_hit = float(row[col_hh]   or 0) if col_hh   else 38.0
                exit_velo = float(row[col_ev]  or 0) if col_ev   else 88.5
                if pid:
                    result[pid] = {
                        "barrel_pct":    barrel,
                        "hard_hit_pct":  hard_hit,
                        "exit_velo_avg": exit_velo,
                    }
            except (ValueError, TypeError):
                continue

        _SAVANT_CACHE["data"] = result
        _SAVANT_CACHE["year"] = year
        return result

    except Exception:
        return {}


def get_game_umpire(game_pk):
    """
    Return the home plate umpire name for a game, or None if not yet assigned.
    Uses the boxscore officials list (same endpoint as get_game_lineup).
    """
    try:
        data = _get(f"/game/{game_pk}/boxscore")
        for official in data.get("officials", []):
            if official.get("officialType", "").lower() == "home plate":
                return official.get("official", {}).get("fullName")
    except Exception:
        pass
    return None


def get_team_bullpen_usage(team_id: int, days_back: int = 3) -> dict:
    """
    Return the total bullpen innings pitched over the last `days_back` days
    for a given team, plus a fatigue label.

    Strategy: fetch the team schedule for recent dates, pull boxscores for
    completed games, sum IP from all non-starting pitchers.

    Returns:
      {
        "total_bp_ip":  float,   # total bullpen IP in the window
        "games_played": int,     # games found in the window
        "fatigue":      str,     # "fresh" | "normal" | "tired" | "gassed"
        "era_modifier": float,   # multiply against bullpen ERA (1.0 = no change)
        "whip_modifier": float,
      }
    """
    from datetime import date as _date, timedelta
    today = _date.today()
    total_bp_ip = 0.0
    games_played = 0

    for d in range(1, days_back + 1):
        check_date = (today - timedelta(days=d)).isoformat()
        try:
            sched = _get("/schedule", params={
                "sportId": 1,
                "date": check_date,
                "teamId": team_id,
                "hydrate": "team",
            })
            for day in sched.get("dates", []):
                for game in day.get("games", []):
                    status = game.get("status", {}).get("detailedState", "").lower()
                    if not any(w in status for w in ("final", "game over", "completed")):
                        continue
                    gp = game["gamePk"]
                    # Determine which side this team is
                    home_id = game["teams"]["home"]["team"]["id"]
                    side = "home" if home_id == team_id else "away"
                    try:
                        box = _get(f"/game/{gp}/boxscore")
                        team_data = box["teams"][side]
                        pitchers   = team_data.get("pitchers", [])   # list of player IDs
                        players    = team_data.get("players", {})
                        if not pitchers:
                            continue
                        starter_id = pitchers[0]   # first pitcher listed = starter
                        for pid in pitchers[1:]:   # all others = bullpen
                            p_stats = players.get(f"ID{pid}", {}).get("stats", {}).get("pitching", {})
                            ip_str  = str(p_stats.get("inningsPitched", "0"))
                            # IP is stored as "1.2" meaning 1 full inning + 2 outs
                            try:
                                parts = ip_str.split(".")
                                full  = int(parts[0])
                                outs  = int(parts[1]) if len(parts) > 1 else 0
                                total_bp_ip += full + outs / 3
                            except (ValueError, IndexError):
                                pass
                        games_played += 1
                    except Exception:
                        pass
        except Exception:
            pass

    # Classify fatigue level
    # League average: ~3.5 BP IP per game
    # Gassed: 5+ IP in window (deep into pen multiple days)
    if total_bp_ip >= 5.0 or (games_played >= 2 and total_bp_ip >= 8.0):
        fatigue, era_mod, whip_mod = "gassed",  1.18, 1.12
    elif total_bp_ip >= 3.5 or (games_played >= 2 and total_bp_ip >= 6.0):
        fatigue, era_mod, whip_mod = "tired",   1.10, 1.06
    elif total_bp_ip >= 1.5:
        fatigue, era_mod, whip_mod = "normal",  1.00, 1.00
    else:
        fatigue, era_mod, whip_mod = "fresh",   0.93, 0.97

    return {
        "total_bp_ip":   round(total_bp_ip, 1),
        "games_played":  games_played,
        "fatigue":       fatigue,
        "era_modifier":  era_mod,
        "whip_modifier": whip_mod,
    }


def get_team_il_players(team_id: int) -> list:
    """
    Returns list of IL players for a team.
    Each entry: {"id": int, "name": str, "il_type": str, "since": str}
    Uses the 40-man roster endpoint — IL players have status.description set.
    """
    try:
        data = _get(f"/teams/{team_id}/roster", params={"rosterType": "40Man"})
        il = []
        for p in data.get("roster", []):
            status = p.get("status", {}).get("description", "")
            if "IL" in status or "Injured" in status:
                person = p.get("person", {})
                il.append({
                    "id":      person.get("id"),
                    "name":    person.get("fullName", "Unknown"),
                    "il_type": status,   # e.g. "10-Day IL", "60-Day IL"
                })
        return il
    except Exception:
        return []


def check_lineup_il_overlap(lineup_players: list, il_players: list) -> list:
    """
    Given a list of lineup player dicts (with 'id') and IL player dicts (with 'id'),
    returns IL entries for any player in the lineup who is on the IL.
    Useful to warn if a starter slipped through a stale lineup.
    """
    il_ids = {p["id"] for p in il_players if p.get("id")}
    return [p for p in il_players if p.get("id") in il_ids
            and any(b.get("id") == p["id"] for b in lineup_players)]


def get_recent_transactions(team_id: int, days_back: int = 3) -> list:
    """
    Returns recent IL placements and activations for a team.
    Uses MLB transactions endpoint — filtered by team and date range.
    Returns list of {"date", "player", "type"} dicts.
    """
    from datetime import date, timedelta
    end   = date.today().isoformat()
    start = (date.today() - timedelta(days=days_back)).isoformat()
    try:
        data = _get("/transactions", params={
            "teamId":    team_id,
            "startDate": start,
            "endDate":   end,
        })
        txns = []
        for t in data.get("transactions", []):
            ttype = t.get("typeDesc", "")
            if any(k in ttype for k in ("Placed", "Activated", "Transfer", "IL")):
                txns.append({
                    "date":   t.get("date", ""),
                    "player": t.get("person", {}).get("fullName", ""),
                    "type":   ttype,
                })
        return txns
    except Exception:
        return []


def get_series_game_number(game_pk: int, away_team_id: int, home_team_id: int) -> int:
    """
    Returns the game number within the current series (1, 2, 3, or 4).
    Looks back up to 4 days at the home team's schedule to find consecutive
    games between the same two teams.
    Returns 1 if this is the series opener or data is unavailable.
    """
    from datetime import date, timedelta
    try:
        today = date.today()
        # Fetch home team schedule for last 5 days (covers a full 3-4 game series)
        start = (today - timedelta(days=5)).isoformat()
        end   = today.isoformat()
        data  = _get(f"/schedule", params={
            "teamId":    home_team_id,
            "startDate": start,
            "endDate":   end,
            "sportId":   1,
            "fields":    "dates,date,games,gamePk,teams,away,home,team,id",
        })
        # Collect games between these two teams in date order
        series_games = []
        for day in data.get("dates", []):
            for g in day.get("games", []):
                away_id = g.get("teams", {}).get("away", {}).get("team", {}).get("id")
                h_id    = g.get("teams", {}).get("home", {}).get("team", {}).get("id")
                if {away_id, h_id} == {away_team_id, home_team_id}:
                    series_games.append((day["date"], g["gamePk"]))

        series_games.sort()
        for i, (_, pk) in enumerate(series_games, 1):
            if pk == game_pk:
                return i
        return len(series_games) + 1  # today's game is next in series
    except Exception:
        return 1


def get_catcher_cs_rate(team_id: int) -> float:
    """
    Returns the opposing catcher's caught-stealing percentage.
    Fetches the active catcher from the team roster, then their season catching stats.
    Falls back to league average (0.28) on any error.

    League average CS% is ~28% (2023-2025 MLB).
    A catcher above 35% is elite; below 20% is a green light for base stealers.
    """
    LEAGUE_AVG_CS = 0.28
    try:
        # Get 25-man roster and find the catcher(s)
        data = _get(f"/teams/{team_id}/roster", params={"rosterType": "active"})
        catchers = [
            p["person"]["id"]
            for p in data.get("roster", [])
            if p.get("position", {}).get("abbreviation") == "C"
        ]
        if not catchers:
            return LEAGUE_AVG_CS

        # Use the first (primary) catcher
        catcher_id = catchers[0]
        stats = _get(f"/people/{catcher_id}/stats", params={
            "stats": "season",
            "group": "catching",
            "season": __import__("datetime").date.today().year,
        })

        splits = (stats.get("stats") or [{}])[0].get("splits") or []
        if not splits:
            return LEAGUE_AVG_CS

        s = splits[0].get("stat", {})
        sb  = int(s.get("stolenBases",     0) or 0)   # stolen bases allowed
        cs  = int(s.get("caughtStealing",  0) or 0)   # caught stealing by this catcher
        attempts = sb + cs
        if attempts < 10:
            return LEAGUE_AVG_CS

        cs_rate = cs / attempts
        return round(min(max(cs_rate, 0.10), 0.55), 3)
    except Exception:
        return LEAGUE_AVG_CS


def get_bullpen_depth_score(team_id: int, season: int = None) -> dict:
    """
    Score a team's bullpen depth on a 0-100 scale.
    Counts the number of "reliable" relief arms (ERA < 3.80, WHIP < 1.30, IP >= 15).
    Also tracks the number of "elite" arms (ERA < 3.00).

    Returns:
      {
        "score":         int,   # 0-100
        "grade":         str,   # "Elite" / "Strong" / "Average" / "Weak" / "Thin"
        "reliable_arms": int,   # count of arms with ERA < 3.80
        "elite_arms":    int,   # count of arms with ERA < 3.00
        "avg_era":       float,
        "avg_whip":      float,
      }
    """
    if season is None:
        from datetime import date as _d
        season = _d.today().year
    try:
        roster_data = _get(f"/teams/{team_id}/roster", params={
            "rosterType": "active", "season": season,
        })
        pitchers = [
            p["person"]["id"]
            for p in roster_data.get("roster", [])
            if p.get("position", {}).get("code") == "1"
        ]
        reliable = 0
        elite    = 0
        total_ip = 0.0
        era_sum  = 0.0
        whip_sum = 0.0

        for pid in pitchers[:25]:
            try:
                stats = get_player_season_stats(pid, group="pitching", season=season)
                gs   = int(stats.get("gamesStarted", 0) or 0)
                ip   = float(stats.get("inningsPitched", 0) or 0)
                if gs >= 3 or ip < 10:
                    continue
                era  = float(stats.get("era",  99) or 99)
                whip = float(stats.get("whip", 99) or 99)
                if era <= 0 or whip <= 0:
                    continue
                era_sum  += era  * ip
                whip_sum += whip * ip
                total_ip += ip
                if ip >= 15 and era < 3.80 and whip < 1.30:
                    reliable += 1
                if ip >= 15 and era < 3.00:
                    elite += 1
            except Exception:
                continue

        avg_era  = round(era_sum  / total_ip, 2) if total_ip else 4.20
        avg_whip = round(whip_sum / total_ip, 2) if total_ip else 1.30

        # Score: base 50, +8 per reliable arm (max 6 arms), +10 per elite arm (max 2)
        score = 50 + min(reliable, 6) * 8 + min(elite, 2) * 10
        # Adjust for avg ERA: sub 3.50 → +5, above 4.50 → -10
        if avg_era < 3.50:
            score += 5
        elif avg_era > 4.50:
            score -= 10
        score = max(0, min(100, score))

        if score >= 85:
            grade = "Elite"
        elif score >= 70:
            grade = "Strong"
        elif score >= 55:
            grade = "Average"
        elif score >= 40:
            grade = "Weak"
        else:
            grade = "Thin"

        return {
            "score":         score,
            "grade":         grade,
            "reliable_arms": reliable,
            "elite_arms":    elite,
            "avg_era":       avg_era,
            "avg_whip":      avg_whip,
        }
    except Exception:
        return {"score": 50, "grade": "Average", "reliable_arms": 0,
                "elite_arms": 0, "avg_era": 4.20, "avg_whip": 1.30}


def get_lineup_status(game_pk: int) -> str:
    """
    Returns "confirmed", "projected", or "unknown" for a game's lineup.
    MLB API gameData.status.detailedState gives us the game state.
    If boxscore already has batting order data with batters listed as "confirmed",
    we consider it confirmed. Otherwise projected.
    """
    try:
        data = _get(f"/game/{game_pk}/feed/live")
        state = data.get("gameData", {}).get("status", {}).get("detailedState", "")
        # If game has started or lineup is out, it's confirmed
        if any(s in state.lower() for s in ["in progress", "final", "game over", "pre-game"]):
            # Check if batting orders are populated
            away_batters = data.get("liveData", {}).get("boxscore", {}).get(
                "teams", {}).get("away", {}).get("battingOrder", [])
            if away_batters:
                return "confirmed"
        return "projected"
    except Exception:
        return "unknown"


def get_team_streak(team_id: int, days_back: int = 10) -> dict:
    """
    Returns a team's current win/loss streak over the last N days.
    Returns {"streak": int, "type": "W"|"L"|"", "label": str, "hot": bool, "cold": bool}
    Positive streak = win streak, negative = loss streak.
    hot = 5+ game win streak, cold = 5+ game loss streak.
    """
    from datetime import date, timedelta
    try:
        end   = date.today().isoformat()
        start = (date.today() - timedelta(days=days_back)).isoformat()
        data  = _get("/schedule", params={
            "teamId":    team_id,
            "startDate": start,
            "endDate":   end,
            "sportId":   1,
            "fields":    "dates,date,games,gamePk,status,teams,away,home,team,id,score,isWinner",
        })
        results = []
        for day in sorted(data.get("dates", []), key=lambda d: d["date"]):
            for game in day.get("games", []):
                state = game.get("status", {}).get("abstractGameState", "")
                if state != "Final":
                    continue
                teams = game.get("teams", {})
                away_id = teams.get("away", {}).get("team", {}).get("id")
                home_id = teams.get("home", {}).get("team", {}).get("id")
                if team_id not in (away_id, home_id):
                    continue
                side = "away" if team_id == away_id else "home"
                won  = teams.get(side, {}).get("isWinner", False)
                results.append("W" if won else "L")

        if not results:
            return {"streak": 0, "type": "", "label": "", "hot": False, "cold": False}

        # Count current streak from most recent game backwards
        current = results[-1]
        streak  = 0
        for r in reversed(results):
            if r == current:
                streak += 1
            else:
                break

        if current == "W":
            label = f"W{streak}"
        else:
            label = f"L{streak}"
            streak = -streak

        return {
            "streak": streak,
            "type":   current,
            "label":  label,
            "hot":    streak >= 5,
            "cold":   streak <= -5,
        }
    except Exception:
        return {"streak": 0, "type": "", "label": "", "hot": False, "cold": False}


# ──────────────────────────────────────────────
# LIVE SCORES — scoreboard strip
# ──────────────────────────────────────────────

def get_live_scores(game_date=None):
    """
    Returns live/final/scheduled scores for today's games.
    Used for the scoreboard strip at the top of the main page.
    Each dict has:
      gamePk, away_team, home_team, away_abbr, home_abbr,
      away_score, home_score, inning, inning_half,
      status, abstract_state, game_time_utc
    """
    from datetime import date as _date
    if game_date is None:
        game_date = _date.today().isoformat()

    try:
        data = _get("/schedule", params={
            "sportId": 1,
            "date":    game_date,
            "hydrate": "team,linescore",
        })
    except Exception:
        return []

    # Short team name → abbreviation mapping (common ones)
    _ABBR = {
        "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
        "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
        "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
        "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
        "Colorado Rockies": "COL", "Detroit Tigers": "DET",
        "Houston Astros": "HOU", "Kansas City Royals": "KC",
        "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
        "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
        "Minnesota Twins": "MIN", "New York Mets": "NYM",
        "New York Yankees": "NYY", "Oakland Athletics": "OAK",
        "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
        "San Diego Padres": "SD", "San Francisco Giants": "SF",
        "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
        "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
        "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
        "Athletics": "OAK",
    }

    scores = []
    for day in data.get("dates", []):
        for game in day.get("games", []):
            away   = game["teams"]["away"]
            home   = game["teams"]["home"]
            ls     = game.get("linescore", {})
            status = game.get("status", {})
            abstract = status.get("abstractGameState", "Preview")  # Preview/Live/Final
            detailed = status.get("detailedState", "Scheduled")

            away_name = away["team"]["name"]
            home_name = home["team"]["name"]

            # Linescore scores (0 if not started)
            away_score = ls.get("teams", {}).get("away", {}).get("runs", 0) or 0
            home_score = ls.get("teams", {}).get("home", {}).get("runs", 0) or 0

            # Inning info
            inning      = ls.get("currentInning", 0)
            inning_half = ls.get("inningHalf", "")   # "Top" | "Bottom" | ""
            outs        = ls.get("outs", 0)

            scores.append({
                "gamePk":       game["gamePk"],
                "away_team":    away_name,
                "home_team":    home_name,
                "away_abbr":    _ABBR.get(away_name, away_name[:3].upper()),
                "home_abbr":    _ABBR.get(home_name, home_name[:3].upper()),
                "away_score":   away_score,
                "home_score":   home_score,
                "inning":       inning,
                "inning_half":  inning_half,
                "outs":         outs,
                "status":       detailed,
                "abstract_state": abstract,
                "game_time_utc": game.get("gameDate", ""),
            })


def compute_injury_impact(il_players: list) -> dict:
    """
    Score how much a team's offense is hurt by their current IL players.

    For each IL player, fetch their season hitting stats and compute an
    offensive weight based on wRC+ proxy (OPS relative to league average).
    Returns:
      {
        "score":      int,        # 0-100 (0 = no impact, 100 = devastating)
        "grade":      str,        # "None" / "Minor" / "Moderate" / "Significant" / "Severe"
        "color":      str,        # hex color for UI badge
        "key_players": list[dict] # top impacted players with their stats
      }
    """
    LEAGUE_AVG_OPS = 0.720

    if not il_players:
        return {"score": 0, "grade": "None", "color": "#555870", "key_players": []}

    scored = []
    for player in il_players:
        pid = player.get("id")
        if not pid:
            continue
        try:
            stats = get_player_season_stats(pid, group="hitting")
        except Exception:
            stats = {}

        # If no hitting stats this season, they're a pitcher or not a contributor
        ab = int(stats.get("atBats", 0) or 0)
        if ab < 50:
            # Try pitching — a pitcher going to IL also matters
            try:
                pstats = get_player_season_stats(pid, group="pitching")
                era = float(pstats.get("era", 5.0) or 5.0)
                gs  = int(pstats.get("gamesStarted", 0) or 0)
                if gs >= 3:
                    # Starting pitcher — moderate impact regardless
                    weight = 15
                    scored.append({
                        "name":    player["name"],
                        "il_type": player.get("il_type", "IL"),
                        "role":    "SP",
                        "stat":    f"ERA {era:.2f}",
                        "weight":  weight,
                    })
            except Exception:
                pass
            continue

        try:
            obp = float(stats.get("obp", "0") or 0)
            slg = float(stats.get("slg", "0") or 0)
            hr  = int(stats.get("homeRuns", 0) or 0)
            rbi = int(stats.get("rbi", 0) or 0)
        except (ValueError, TypeError):
            continue

        ops = obp + slg
        # Weight = how much above/below league average this bat is
        # A .900 OPS player = 25% above avg = high impact
        ops_delta = (ops - LEAGUE_AVG_OPS) / LEAGUE_AVG_OPS
        # Scale: 0.25 delta → weight ~20, -0.1 delta → weight ~5
        weight = max(2, min(30, int(15 + ops_delta * 50)))

        scored.append({
            "name":    player["name"],
            "il_type": player.get("il_type", "IL"),
            "role":    "Bat",
            "stat":    f".{str(stats.get('avg','000')).replace('.','')[:3]} OPS {ops:.3f}",
            "weight":  weight,
        })

    if not scored:
        return {"score": 0, "grade": "None", "color": "#555870", "key_players": []}

    # Total impact score: sum weights, cap at 100
    # A full lineup (~9 players × 15 avg weight = 135) would be 100
    raw_score = sum(p["weight"] for p in scored)
    score = min(100, int(raw_score / 1.35))

    # Sort by weight descending, keep top 5 for display
    scored.sort(key=lambda x: x["weight"], reverse=True)

    if score == 0:
        grade, color = "None", "#555870"
    elif score < 15:
        grade, color = "Minor", "#81c784"
    elif score < 35:
        grade, color = "Moderate", "#ffd54f"
    elif score < 60:
        grade, color = "Significant", "#ffb74d"
    else:
        grade, color = "Severe", "#ef5350"

    return {
        "score":       score,
        "grade":       grade,
        "color":       color,
        "key_players": scored[:5],
    }

    return scores
