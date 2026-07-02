"""
nfl_api.py
----------
Fetches NFL data from ESPN's free public API.
No API key needed — ESPN makes this data public.

Base URLs:
  - Scoreboard: site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard
  - Teams:      site.api.espn.com/apis/site/v2/sports/football/nfl/teams
  - Team stats: sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{year}/types/2/teams/{id}/statistics
"""

import time
from datetime import date, datetime, timedelta

import requests

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

# ── In-memory TTL cache ─────────────────────────────────────────
_API_CACHE = {}
_API_CACHE_TTL = 600  # 10 min


def _cached_get(url, params=None, ttl=None):
    ttl = ttl or _API_CACHE_TTL
    key = url + str(sorted((params or {}).items()))
    now = time.time()
    if key in _API_CACHE:
        ts, data = _API_CACHE[key]
        if now - ts < ttl:
            return data
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        _API_CACHE[key] = (now, data)
        return data
    except Exception:
        return {}


# ── Team ID mapping ─────────────────────────────────────────────
# ESPN team IDs for all 32 NFL teams
NFL_TEAMS = {
    "ARI": {"id": 22, "name": "Arizona Cardinals", "short": "Cardinals"},
    "ATL": {"id": 1, "name": "Atlanta Falcons", "short": "Falcons"},
    "BAL": {"id": 33, "name": "Baltimore Ravens", "short": "Ravens"},
    "BUF": {"id": 2, "name": "Buffalo Bills", "short": "Bills"},
    "CAR": {"id": 29, "name": "Carolina Panthers", "short": "Panthers"},
    "CHI": {"id": 3, "name": "Chicago Bears", "short": "Bears"},
    "CIN": {"id": 4, "name": "Cincinnati Bengals", "short": "Bengals"},
    "CLE": {"id": 5, "name": "Cleveland Browns", "short": "Browns"},
    "DAL": {"id": 6, "name": "Dallas Cowboys", "short": "Cowboys"},
    "DEN": {"id": 7, "name": "Denver Broncos", "short": "Broncos"},
    "DET": {"id": 8, "name": "Detroit Lions", "short": "Lions"},
    "GB": {"id": 9, "name": "Green Bay Packers", "short": "Packers"},
    "HOU": {"id": 34, "name": "Houston Texans", "short": "Texans"},
    "IND": {"id": 11, "name": "Indianapolis Colts", "short": "Colts"},
    "JAX": {"id": 30, "name": "Jacksonville Jaguars", "short": "Jaguars"},
    "KC": {"id": 12, "name": "Kansas City Chiefs", "short": "Chiefs"},
    "LV": {"id": 13, "name": "Las Vegas Raiders", "short": "Raiders"},
    "LAC": {"id": 24, "name": "Los Angeles Chargers", "short": "Chargers"},
    "LAR": {"id": 14, "name": "Los Angeles Rams", "short": "Rams"},
    "MIA": {"id": 15, "name": "Miami Dolphins", "short": "Dolphins"},
    "MIN": {"id": 16, "name": "Minnesota Vikings", "short": "Vikings"},
    "NE": {"id": 17, "name": "New England Patriots", "short": "Patriots"},
    "NO": {"id": 18, "name": "New Orleans Saints", "short": "Saints"},
    "NYG": {"id": 19, "name": "New York Giants", "short": "Giants"},
    "NYJ": {"id": 20, "name": "New York Jets", "short": "Jets"},
    "PHI": {"id": 21, "name": "Philadelphia Eagles", "short": "Eagles"},
    "PIT": {"id": 23, "name": "Pittsburgh Steelers", "short": "Steelers"},
    "SF": {"id": 25, "name": "San Francisco 49ers", "short": "49ers"},
    "SEA": {"id": 26, "name": "Seattle Seahawks", "short": "Seahawks"},
    "TB": {"id": 27, "name": "Tampa Bay Buccaneers", "short": "Buccaneers"},
    "TEN": {"id": 10, "name": "Tennessee Titans", "short": "Titans"},
    "WAS": {"id": 28, "name": "Washington Commanders", "short": "Commanders"},
}

# Reverse lookup: ESPN ID -> abbreviation
_ID_TO_ABBR = {v["id"]: k for k, v in NFL_TEAMS.items()}


def _espn_id_to_abbr(team_id):
    return _ID_TO_ABBR.get(int(team_id), f"TM{team_id}")


# ── Stadium coordinates for weather lookups ─────────────────────
NFL_STADIUM_COORDS = {
    "State Farm Stadium": (33.528, -112.263),
    "Mercedes-Benz Stadium": (33.755, -84.401),
    "M&T Bank Stadium": (39.278, -76.623),
    "Highmark Stadium": (42.774, -78.787),
    "Bank of America Stadium": (35.226, -80.853),
    "Soldier Field": (41.862, -87.617),
    "Paycor Stadium": (39.095, -84.516),
    "Cleveland Browns Stadium": (41.506, -81.700),
    "AT&T Stadium": (32.748, -97.093),
    "Empower Field at Mile High": (39.744, -105.020),
    "Ford Field": (42.340, -83.046),
    "Lambeau Field": (44.501, -88.062),
    "NRG Stadium": (29.685, -95.411),
    "Lucas Oil Stadium": (39.760, -86.164),
    "EverBank Stadium": (30.324, -81.637),
    "GEHA Field at Arrowhead Stadium": (39.049, -94.484),
    "Allegiant Stadium": (36.091, -115.184),
    "SoFi Stadium": (33.953, -118.339),
    "Hard Rock Stadium": (25.958, -80.239),
    "U.S. Bank Stadium": (44.974, -93.258),
    "Gillette Stadium": (42.091, -71.264),
    "Caesars Superdome": (29.951, -90.081),
    "MetLife Stadium": (40.814, -74.074),
    "Lincoln Financial Field": (39.901, -75.168),
    "Acrisure Stadium": (40.447, -80.016),
    "Levi's Stadium": (37.403, -121.970),
    "Lumen Field": (47.595, -122.332),
    "Raymond James Stadium": (27.976, -82.503),
    "Nissan Stadium": (36.166, -86.771),
    "Northwest Stadium": (38.908, -76.864),
}

# Domed stadiums (no weather impact)
DOMED_STADIUMS = {
    "State Farm Stadium",
    "Mercedes-Benz Stadium",
    "Ford Field",
    "Lucas Oil Stadium",
    "Allegiant Stadium",
    "SoFi Stadium",
    "U.S. Bank Stadium",
    "Caesars Superdome",
    "AT&T Stadium",
    "NRG Stadium",
}


# ── Schedule / Scoreboard ───────────────────────────────────────


def get_nfl_schedule(week=None, season_type=2, year=None):
    """
    Get NFL schedule for a given week.
    season_type: 1=preseason, 2=regular, 3=postseason
    Returns list of game dicts.
    """
    if year is None:
        today = date.today()
        year = today.year if today.month >= 3 else today.year - 1

    params = {"dates": str(year)}
    if week:
        params["week"] = str(week)
    params["seasontype"] = str(season_type)

    data = _cached_get(f"{ESPN_BASE}/scoreboard", params=params)
    events = data.get("events", [])

    games = []
    for ev in events:
        comp = ev.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = away = None
        for c in competitors:
            team_data = {
                "id": int(c.get("id", 0)),
                "abbr": c.get("team", {}).get("abbreviation", ""),
                "name": c.get("team", {}).get("displayName", ""),
                "short_name": c.get("team", {}).get("shortDisplayName", ""),
                "logo": c.get("team", {}).get("logo", ""),
                "score": int(c.get("score", 0)) if c.get("score") else None,
                "record": c.get("records", [{}])[0].get("summary", "") if c.get("records") else "",
                "rank": int(c.get("curatedRank", {}).get("current", 99)),
            }
            if c.get("homeAway") == "home":
                home = team_data
            else:
                away = team_data

        if not home or not away:
            continue

        status = comp.get("status", {})
        venue_info = comp.get("venue", {})
        odds_info = comp.get("odds", [{}])[0] if comp.get("odds") else {}
        weather_info = comp.get("weather", {})

        game_date_str = ev.get("date", "")
        try:
            game_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
        except Exception:
            game_dt = None

        games.append(
            {
                "game_id": ev.get("id", ""),
                "game_date": game_dt.strftime("%Y-%m-%d") if game_dt else "",
                "game_time": game_dt.strftime("%I:%M %p ET") if game_dt else "",
                "week": ev.get("week", {}).get("number", 0)
                if isinstance(ev.get("week"), dict)
                else 0,
                "season_type": season_type,
                "status": status.get("type", {}).get("name", ""),
                "status_detail": status.get("type", {}).get("shortDetail", ""),
                "period": status.get("period", 0),
                "clock": status.get("displayClock", ""),
                "home": home,
                "away": away,
                "venue": venue_info.get("fullName", ""),
                "city": venue_info.get("address", {}).get("city", ""),
                "state": venue_info.get("address", {}).get("state", ""),
                "is_dome": venue_info.get("fullName", "") in DOMED_STADIUMS,
                "spread": odds_info.get("details", ""),
                "over_under": odds_info.get("overUnder"),
                "weather_temp": weather_info.get("temperature"),
                "weather_condition": weather_info.get("displayValue", ""),
                "broadcast": ev.get("competitions", [{}])[0]
                .get("broadcasts", [{}])[0]
                .get("names", [""])[0]
                if comp.get("broadcasts")
                else "",
            }
        )

    return games


def get_current_week():
    """Determine the current NFL week based on today's date."""
    data = _cached_get(f"{ESPN_BASE}/scoreboard")
    week = data.get("week", {})
    if isinstance(week, dict):
        return week.get("number", 1)
    return 1


# ── Team Statistics ─────────────────────────────────────────────


def get_team_stats(team_abbr, year=None):
    """
    Get season statistics for a team.
    Returns dict with offensive and defensive stats.
    """
    if team_abbr not in NFL_TEAMS:
        return {}

    if year is None:
        today = date.today()
        year = today.year if today.month >= 3 else today.year - 1

    team_id = NFL_TEAMS[team_abbr]["id"]
    url = f"{ESPN_CORE}/seasons/{year}/types/2/teams/{team_id}/statistics"
    data = _cached_get(url)

    stats = {}
    splits = data.get("splits", {}).get("categories", [])
    for cat in splits:
        cat_name = cat.get("name", "")
        for stat in cat.get("stats", []):
            key = f"{cat_name}_{stat.get('name', '')}"
            stats[key] = stat.get("value", 0)

    return stats


def get_team_record(team_abbr, year=None):
    """Get team W-L record."""
    if team_abbr not in NFL_TEAMS:
        return {"wins": 0, "losses": 0, "ties": 0}

    if year is None:
        today = date.today()
        year = today.year if today.month >= 3 else today.year - 1

    team_id = NFL_TEAMS[team_abbr]["id"]
    data = _cached_get(f"{ESPN_BASE}/teams/{team_id}")
    team = data.get("team", {})
    record = team.get("record", {}).get("items", [{}])[0] if team.get("record") else {}
    stats = {s.get("name"): s.get("value") for s in record.get("stats", [])}

    return {
        "wins": int(stats.get("wins", 0)),
        "losses": int(stats.get("losses", 0)),
        "ties": int(stats.get("ties", 0)),
        "pct": float(stats.get("winPercent", 0)),
    }


# ── Roster / Depth Chart ────────────────────────────────────────


def get_team_roster(team_abbr):
    """Get team roster with key player stats."""
    if team_abbr not in NFL_TEAMS:
        return []

    team_id = NFL_TEAMS[team_abbr]["id"]
    data = _cached_get(f"{ESPN_BASE}/teams/{team_id}/roster")

    players = []
    for group in data.get("athletes", []):
        position_group = group.get("position", "")
        for athlete in group.get("items", []):
            players.append(
                {
                    "id": athlete.get("id", ""),
                    "name": athlete.get("fullName", ""),
                    "position": athlete.get("position", {}).get("abbreviation", ""),
                    "position_group": position_group,
                    "jersey": athlete.get("jersey", ""),
                    "height": athlete.get("displayHeight", ""),
                    "weight": athlete.get("displayWeight", ""),
                    "age": athlete.get("age", 0),
                    "experience": athlete.get("experience", {}).get("years", 0),
                }
            )

    return players


def get_team_depth_chart(team_abbr):
    """Get depth chart for a team — returns starters at each position."""
    if team_abbr not in NFL_TEAMS:
        return {}

    team_id = NFL_TEAMS[team_abbr]["id"]
    data = _cached_get(f"{ESPN_BASE}/teams/{team_id}/depthcharts")

    depth = {}
    for item in data.get("items", []):
        positions = item.get("positions", {})
        for pos_name, pos_data in positions.items():
            athletes = pos_data.get("athletes", [])
            if athletes:
                starter = athletes[0].get("athlete", {})
                depth[pos_name] = {
                    "name": starter.get("displayName", ""),
                    "id": starter.get("id", ""),
                    "rank": athletes[0].get("rank", 1),
                }

    return depth


# ── Game-specific data ──────────────────────────────────────────


def get_game_details(game_id):
    """Get detailed game information including play-by-play if available."""
    data = _cached_get(f"{ESPN_BASE}/summary", params={"event": str(game_id)}, ttl=120)

    result = {
        "game_id": game_id,
        "boxscore": {},
        "leaders": {},
        "predictor": {},
    }

    boxscore = data.get("boxscore", {})
    if boxscore:
        for team_data in boxscore.get("teams", []):
            side = team_data.get("team", {}).get("abbreviation", "")
            stats = {}
            for stat_group in team_data.get("statistics", []):
                for stat in stat_group.get("stats", []):
                    stats[stat.get("label", "")] = stat.get("displayValue", "")
            result["boxscore"][side] = stats

    predictor = data.get("predictor", {})
    if predictor:
        result["predictor"] = {
            "home_pct": predictor.get("homeTeam", {}).get("gameProjection", 0),
            "away_pct": predictor.get("awayTeam", {}).get("gameProjection", 0),
        }

    return result


def get_nfl_odds(game_date=None):
    """
    Fetch NFL odds from The Odds API.
    Uses the same API key as MLB odds.
    """
    from data.odds_api import (
        ODDS_API_KEY,
        ODDS_BASE,
        PREFERRED_BOOKS,
        _file_cache_get,
        _file_cache_set,
    )

    if not ODDS_API_KEY:
        return {}

    cache_key = f"nfl_odds_{game_date or 'current'}"
    cached = _file_cache_get(cache_key, ttl=300)
    if cached:
        return cached

    try:
        r = requests.get(
            f"{ODDS_BASE}/sports/americanfootball_nfl/odds/",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
            },
            timeout=15,
        )
        r.raise_for_status()
        events = r.json()
    except Exception:
        return {}

    odds_by_game = {}
    for ev in events:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")

        mls = {"home": [], "away": []}
        spreads = {"home": [], "away": []}
        totals = []

        for bk in ev.get("bookmakers", []):
            if bk.get("key") not in PREFERRED_BOOKS:
                continue
            for mkt in bk.get("markets", []):
                if mkt["key"] == "h2h":
                    for o in mkt.get("outcomes", []):
                        side = "home" if o["name"] == home else "away"
                        mls[side].append(o.get("price", 0))
                elif mkt["key"] == "spreads":
                    for o in mkt.get("outcomes", []):
                        side = "home" if o["name"] == home else "away"
                        spreads[side].append(
                            {"spread": o.get("point", 0), "price": o.get("price", 0)}
                        )
                elif mkt["key"] == "totals":
                    for o in mkt.get("outcomes", []):
                        if o["name"] == "Over":
                            totals.append(o.get("point", 0))

        def avg_ml(lst):
            return int(sum(lst) / len(lst)) if lst else None

        avg_spread = (
            round(sum(s["spread"] for s in spreads["home"]) / len(spreads["home"]), 1)
            if spreads["home"]
            else None
        )
        avg_total = round(sum(totals) / len(totals), 1) if totals else None

        odds_by_game[f"{away} @ {home}"] = {
            "home_team": home,
            "away_team": away,
            "home_ml": avg_ml(mls["home"]),
            "away_ml": avg_ml(mls["away"]),
            "spread": avg_spread,
            "over_under": avg_total,
            "commence_time": ev.get("commence_time", ""),
        }

    try:
        _file_cache_set(cache_key, odds_by_game)
    except Exception:
        pass

    return odds_by_game
