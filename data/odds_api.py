"""
odds_api.py
-----------
Fetches real moneyline odds from The Odds API (https://the-odds-api.com)
and compares them to our simulation's win probabilities.

Free tier: 500 requests/month — plenty for daily use.
"""

import os
import time

import requests

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_BASE    = "https://api.the-odds-api.com/v4"

# Sportsbooks to average across (all available on free tier)
PREFERRED_BOOKS = {"draftkings", "fanduel", "betmgm", "bovada", "williamhill_us"}

# ── File-based persistent cache ──────────────────────────────────────────────
# Survives server restarts — saves quota during development.
import json as _json_mod
import os as _os

_CACHE_DIR = _os.path.join(_os.path.dirname(__file__), "_odds_cache")

def _file_cache_get(key: str, ttl: float) -> dict | None:
    """Return cached data if still fresh, else None."""
    path = _os.path.join(_CACHE_DIR, f"{key}.json")
    try:
        with open(path) as f:
            obj = _json_mod.load(f)
        if time.time() - obj.get("ts", 0) < ttl:
            return obj.get("data")
    except Exception:
        pass
    return None

def _file_cache_set(key: str, data):
    """Write data + timestamp to disk cache."""
    try:
        _os.makedirs(_CACHE_DIR, exist_ok=True)
        path = _os.path.join(_CACHE_DIR, f"{key}.json")
        with open(path, "w") as f:
            _json_mod.dump({"ts": time.time(), "data": data}, f)
    except Exception:
        pass

# ── In-memory cache for moneyline odds ──────────────────────────────
# Odds barely move inning to inning, so we cache for 30 minutes.
# This cuts API usage from 1 request per simulation down to
# ~1 request per 30-minute window regardless of how many games you run.
_ODDS_CACHE = {"data": {}, "ts": 0.0}
_CACHE_TTL  = 60 * 60   # 60 minutes in seconds

# Requests remaining — updated from response headers after every API call
_requests_remaining = None


def get_requests_remaining() -> int | None:
    """Returns how many Odds API requests you have left this month, or None if unknown."""
    return _requests_remaining


def _update_remaining(resp):
    """Read the x-requests-remaining header and store it globally."""
    global _requests_remaining
    val = resp.headers.get("x-requests-remaining")
    if val is not None:
        try:
            _requests_remaining = int(val)
        except ValueError:
            pass


def _american_to_prob(odds: int) -> float:
    """
    Convert American moneyline odds to implied win probability (0.0–1.0).

    Examples:
      -150  →  0.60  (60% implied)
      +130  →  0.435 (43.5% implied)
    """
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def get_mlb_odds() -> dict:
    """
    Fetch today's MLB moneyline odds from all available US sportsbooks.
    Results are cached in memory for 30 minutes to conserve API quota.

    Returns a dict keyed by a frozenset of {away_team, home_team} so we
    can look up a game regardless of which team is home or away.

    Each value is a dict:
      {
        away_team:        str,
        home_team:        str,
        away_implied_pct: float,   # Vegas implied win % for away team
        home_implied_pct: float,   # Vegas implied win % for home team
        away_avg_odds:    int,     # averaged American odds for away team
        home_avg_odds:    int,     # averaged American odds for home team
        books_used:       int,     # how many books were averaged
      }

    Returns empty dict if the API call fails or key is missing.
    """
    if not ODDS_API_KEY:
        return {}

    # Low-quota guard: if we have fewer than 50 requests left, extend TTL to 4 hours
    effective_ttl = _CACHE_TTL
    if _requests_remaining is not None and _requests_remaining < 50:
        effective_ttl = 4 * 60 * 60  # 4 hours

    # Return in-memory cache if still fresh
    if time.time() - _ODDS_CACHE["ts"] < effective_ttl and _ODDS_CACHE["data"]:
        return _ODDS_CACHE["data"]

    # Check file cache (survives restarts — saves API quota during development)
    file_cached = _file_cache_get("moneyline", effective_ttl)
    if file_cached is not None:
        _ODDS_CACHE["data"] = {frozenset(k.split("|")): v for k, v in file_cached.items()}
        _ODDS_CACHE["ts"] = time.time()
        return _ODDS_CACHE["data"]

    try:
        resp = requests.get(
            f"{ODDS_BASE}/sports/baseball_mlb/odds/",
            params={
                "apiKey":      ODDS_API_KEY,
                "regions":     "us",
                "markets":     "h2h",          # h2h = moneyline
                "oddsFormat":  "american",
            },
            timeout=8,
        )
        resp.raise_for_status()
        _update_remaining(resp)
        games = resp.json()
    except Exception:
        # Return stale cache rather than empty dict — better to show old odds than nothing
        if _ODDS_CACHE["data"]:
            return _ODDS_CACHE["data"]
        return {}

    result = {}

    for game in games:
        away = game.get("away_team", "")
        home = game.get("home_team", "")

        # Collect odds from each bookmaker
        away_odds_list = []
        home_odds_list = []
        best_away_odds = None
        best_home_odds = None
        best_away_book = ""
        best_home_book = ""

        for book in game.get("bookmakers", []):
            book_name = book.get("title", book.get("key", ""))
            for market in book.get("markets", []):
                if market["key"] != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    price = outcome["price"]
                    if outcome["name"] == away:
                        away_odds_list.append(price)
                        # Best away odds = highest American odds (most favorable to bettor)
                        if best_away_odds is None or price > best_away_odds:
                            best_away_odds = price
                            best_away_book = book_name
                    elif outcome["name"] == home:
                        home_odds_list.append(price)
                        if best_home_odds is None or price > best_home_odds:
                            best_home_odds = price
                            best_home_book = book_name

        if not away_odds_list or not home_odds_list:
            continue

        # Average the odds across all books
        avg_away = round(sum(away_odds_list) / len(away_odds_list))
        avg_home = round(sum(home_odds_list) / len(home_odds_list))

        # Convert to implied probabilities
        away_prob = _american_to_prob(avg_away)
        home_prob = _american_to_prob(avg_home)

        # Normalize to remove the vig (bookmaker's margin)
        # Raw implied probs sum to >100% because books take a cut
        total = away_prob + home_prob
        away_prob_clean = round(away_prob / total * 100, 1)
        home_prob_clean = round(home_prob / total * 100, 1)

        key = frozenset([away, home])
        result[key] = {
            "away_team":        away,
            "home_team":        home,
            "away_implied_pct": away_prob_clean,
            "home_implied_pct": home_prob_clean,
            "away_avg_odds":    avg_away,
            "home_avg_odds":    avg_home,
            "books_used":       len(away_odds_list),
            "best_away_odds":   best_away_odds or avg_away,
            "best_away_book":   best_away_book,
            "best_home_odds":   best_home_odds or avg_home,
            "best_home_book":   best_home_book,
        }

    # Store in cache with current timestamp
    _ODDS_CACHE["data"] = result
    _ODDS_CACHE["ts"]   = time.time()
    # Persist to file so cache survives server restarts
    _file_cache_set("moneyline", {"|".join(sorted(k)): v for k, v in result.items()})
    return result


def get_mlb_events() -> dict:
    """
    Fetch today's MLB event IDs from The Odds API.

    Returns a dict keyed by frozenset({away_team, home_team}) → event_id string.
    We need the event ID to fetch player props for a specific game.
    """
    if not ODDS_API_KEY:
        return {}

    try:
        resp = requests.get(
            f"{ODDS_BASE}/sports/baseball_mlb/events",
            params={"apiKey": ODDS_API_KEY},
            timeout=8,
        )
        resp.raise_for_status()
        _update_remaining(resp)
        events = resp.json()
    except Exception:
        return {}

    result = {}
    for ev in events:
        away = ev.get("away_team", "")
        home = ev.get("home_team", "")
        if away and home:
            result[frozenset([away, home])] = ev["id"]
    return result


def get_player_props(event_id: str) -> dict:
    """
    Fetch player prop odds for one MLB game from The Odds API.

    Markets fetched:
      batter_home_runs    — will this batter hit a HR? (yes/no line)
      batter_hits         — hits over/under (typically 0.5)
      batter_total_bases  — total bases over/under
      pitcher_strikeouts  — pitcher Ks over/under

    Returns a dict keyed by player name → list of prop dicts:
      {
        "market":      "HR",          # short display label
        "description": "Over 0.5",    # the specific line
        "odds":        "+320",        # American odds formatted
        "book":        "DraftKings",  # sportsbook name
      }

    Returns empty dict if props aren't available (common on free tier / early in day).
    """
    if not ODDS_API_KEY or not event_id:
        return {}

    # Markets to fetch — player props
    prop_markets = [
        "batter_home_runs",
        "batter_hits",
        "batter_total_bases",
        "pitcher_strikeouts",
        "batter_rbis",
    ]

    # Short display labels for each market
    MARKET_LABELS = {
        "batter_home_runs":   "HR",
        "batter_hits":        "Hits",
        "batter_total_bases": "TB",
        "pitcher_strikeouts": "K",
        "batter_rbis":        "RBI",
    }

    try:
        resp = requests.get(
            f"{ODDS_BASE}/sports/baseball_mlb/events/{event_id}/odds",
            params={
                "apiKey":      ODDS_API_KEY,
                "regions":     "us",
                "markets":     ",".join(prop_markets),
                "oddsFormat":  "american",
            },
            timeout=10,
        )
        resp.raise_for_status()
        _update_remaining(resp)
        data = resp.json()
    except Exception:
        return {}

    # Build player → props mapping
    # We only take the BEST (most favorable) odds for each player+market combo
    # across all books, so we show the highest-paying option
    player_props = {}   # player_name → {market_key → {desc, odds, book}}

    for book in data.get("bookmakers", []):
        book_name = book.get("title", "")
        for market in book.get("markets", []):
            mkey = market.get("key", "")
            label = MARKET_LABELS.get(mkey, mkey)
            for outcome in market.get("outcomes", []):
                player = outcome.get("description", "")  # player name is in "description"
                if not player:
                    player = outcome.get("name", "")
                point  = outcome.get("point", "")
                name   = outcome.get("name", "")   # "Over" or "Under"
                price  = outcome.get("price", 0)

                # Only show Over lines for most props (more actionable)
                if name == "Under":
                    continue

                desc = f"O {point}" if point != "" else name

                if player not in player_props:
                    player_props[player] = {}

                # Keep the best (highest) odds for this market across books
                existing = player_props[player].get(mkey)
                if existing is None or price > existing["raw_odds"]:
                    player_props[player][mkey] = {
                        "market":   label,
                        "desc":     desc,
                        "odds":     format_odds(price),
                        "raw_odds": price,
                        "book":     book_name,
                    }

    # Convert to final format: player_name → sorted list of props
    result = {}
    for player, markets in player_props.items():
        result[player] = sorted(markets.values(), key=lambda x: x["market"])

    return result


def format_odds(american: int) -> str:
    """Format American odds for display: -150 → '-150', 130 → '+130'"""
    return f"+{american}" if american > 0 else str(american)


def calc_edge(our_pct: float, vegas_pct: float) -> float:
    """
    Edge = our model's win % minus Vegas implied win %.
    Positive = we think the team is MORE likely to win than Vegas does → value bet.
    Negative = we agree or think they're less likely.
    """
    return round(our_pct - vegas_pct, 1)


# ── In-memory cache for totals (O/U) odds ──────────────────────────
_TOTALS_CACHE = {"data": {}, "ts": 0.0}


def get_mlb_totals() -> dict:
    """
    Fetch today's MLB over/under (totals) lines from The Odds API.

    Returns a dict keyed by frozenset({away_team, home_team}).
    Each value:
      {
        "line":        float,   # e.g. 8.5
        "over_odds":   int,     # American odds for Over
        "under_odds":  int,     # American odds for Under
        "over_implied": float,  # implied probability of Over (0-100)
        "under_implied": float,
      }
    Returns {} if API key missing or call fails.
    """
    if not ODDS_API_KEY:
        return {}

    if time.time() - _TOTALS_CACHE["ts"] < _CACHE_TTL and _TOTALS_CACHE["data"]:
        return _TOTALS_CACHE["data"]

    file_cached_t = _file_cache_get("totals", _CACHE_TTL)
    if file_cached_t is not None:
        _TOTALS_CACHE["data"] = {frozenset(k.split("|")): v for k, v in file_cached_t.items()}
        _TOTALS_CACHE["ts"] = time.time()
        return _TOTALS_CACHE["data"]

    try:
        resp = requests.get(
            f"{ODDS_BASE}/sports/baseball_mlb/odds/",
            params={
                "apiKey":     ODDS_API_KEY,
                "regions":    "us",
                "markets":    "totals",
                "oddsFormat": "american",
            },
            timeout=8,
        )
        resp.raise_for_status()
        _update_remaining(resp)
        games = resp.json()
    except Exception:
        return {}

    result = {}
    for game in games:
        away = game.get("away_team", "")
        home = game.get("home_team", "")
        over_lines  = []
        under_lines = []
        over_odds_list  = []
        under_odds_list = []

        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] != "totals":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "").lower()
                    point = outcome.get("point", 0)
                    price = outcome.get("price", 0)
                    if name == "over":
                        over_lines.append(point)
                        over_odds_list.append(price)
                    elif name == "under":
                        under_lines.append(point)
                        under_odds_list.append(price)

        if not over_lines:
            continue

        avg_line       = round(sum(over_lines) / len(over_lines), 1)
        avg_over_odds  = round(sum(over_odds_list) / len(over_odds_list))
        avg_under_odds = round(sum(under_odds_list) / len(under_odds_list)) if under_odds_list else -110

        over_impl  = _american_to_prob(avg_over_odds)
        under_impl = _american_to_prob(avg_under_odds)
        total_impl = over_impl + under_impl
        over_impl_clean  = round(over_impl  / total_impl * 100, 1)
        under_impl_clean = round(under_impl / total_impl * 100, 1)

        key = frozenset([away, home])
        result[key] = {
            "line":          avg_line,
            "over_odds":     avg_over_odds,
            "under_odds":    avg_under_odds,
            "over_implied":  over_impl_clean,
            "under_implied": under_impl_clean,
        }

    _TOTALS_CACHE["data"] = result
    _TOTALS_CACHE["ts"]   = time.time()
    _file_cache_set("totals", {"|".join(sorted(k)): v for k, v in result.items()})
    return result


def get_mlb_runline() -> dict:
    """
    Fetch MLB run line (spread, always ±1.5) odds.
    Returns dict keyed by frozenset({away, home}):
      { "away_rl_odds": int, "home_rl_odds": int,
        "away_rl_implied": float, "home_rl_implied": float }
    """
    if not ODDS_API_KEY:
        return {}
    # Use same cache window as moneyline
    cache_key = "_rl"
    if not hasattr(get_mlb_runline, "_cache"):
        get_mlb_runline._cache = {"data": {}, "ts": 0.0}
    c = get_mlb_runline._cache
    if time.time() - c["ts"] < _CACHE_TTL and c["data"]:
        return c["data"]
    try:
        resp = requests.get(
            f"{ODDS_BASE}/sports/baseball_mlb/odds/",
            params={"apiKey": ODDS_API_KEY, "regions": "us",
                    "markets": "spreads", "oddsFormat": "american"},
            timeout=8,
        )
        resp.raise_for_status()
        _update_remaining(resp)
        games = resp.json()
    except Exception:
        return {}
    result = {}
    for game in games:
        away = game.get("away_team", "")
        home = game.get("home_team", "")
        away_list, home_list = [], []
        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] != "spreads":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome.get("point", 0) < 0:   # negative spread = favorite covers
                        if outcome["name"] == away:
                            away_list.append(outcome["price"])
                        else:
                            home_list.append(outcome["price"])
                    else:
                        if outcome["name"] == away:
                            away_list.append(outcome["price"])
                        else:
                            home_list.append(outcome["price"])
        if not away_list or not home_list:
            continue
        avg_away = round(sum(away_list) / len(away_list))
        avg_home = round(sum(home_list) / len(home_list))
        ai = _american_to_prob(avg_away); hi = _american_to_prob(avg_home)
        t = ai + hi
        result[frozenset([away, home])] = {
            "away_rl_odds":    avg_away,
            "home_rl_odds":    avg_home,
            "away_rl_implied": round(ai / t * 100, 1),
            "home_rl_implied": round(hi / t * 100, 1),
        }
    c["data"] = result; c["ts"] = time.time()
    return result


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal (European) odds."""
    if odds > 0:
        return odds / 100 + 1.0
    else:
        return 100 / abs(odds) + 1.0


def calc_ev(model_prob: float, american_odds: int) -> float:
    """
    Expected value per $100 bet.
    model_prob: 0-100 (model's win probability %)
    Returns EV in dollars (positive = +EV, negative = -EV).
    """
    p = model_prob / 100
    dec = american_to_decimal(american_odds)
    return round(p * (dec - 1) * 100 - (1 - p) * 100, 2)


def calc_kelly(model_prob: float, american_odds: int, fraction: float = 0.5) -> float:
    """
    Fractional Kelly Criterion — recommended bet size as % of bankroll.
    fraction: 0.5 = half Kelly (safer, reduces variance).
    Returns % of bankroll to bet (e.g. 4.2 means bet 4.2% of your roll).
    Clamped to 0-25%.
    """
    p = model_prob / 100
    q = 1 - p
    b = american_to_decimal(american_odds) - 1   # net profit per $1 risked
    if b <= 0:
        return 0.0
    kelly = (b * p - q) / b
    kelly_f = kelly * fraction
    return round(max(0.0, min(kelly_f * 100, 25.0)), 1)   # as %


# ── Line movement tracker ────────────────────────────────────────────────────
# Snapshots the moneyline odds to a JSON file once per session day.
# On subsequent fetches, compares the current line to the opening snapshot
# and flags games where the line moved 10+ cents (sharp money signal).

import json as _json

_MOVEMENT_PATH = os.path.join(os.path.dirname(__file__), "line_movement.json")


def _load_snapshot() -> dict:
    try:
        with open(_MOVEMENT_PATH) as f:
            return _json.load(f)
    except Exception:
        return {}


def _save_snapshot(data: dict):
    try:
        with open(_MOVEMENT_PATH, "w") as f:
            _json.dump(data, f)
    except Exception:
        pass


def get_line_movement() -> dict:
    """
    Compare current moneyline odds to today's opening snapshot.

    Returns a dict keyed by frozenset({away, home}) with:
      {
        "away_open":    int,    # opening American odds for away
        "home_open":    int,    # opening American odds for home
        "away_current": int,    # current odds
        "home_current": int,
        "away_move":    int,    # current - open (positive = line moved toward away)
        "home_move":    int,
        "sharp_away":   bool,   # True if away line moved 10+ cents toward away (sharp money)
        "sharp_home":   bool,
      }
    Returns {} if no current odds available.
    """
    current = get_mlb_odds()
    if not current:
        return {}

    today = __import__("datetime").date.today().isoformat()
    snap  = _load_snapshot()

    # If snapshot is from a different day, reset it
    if snap.get("date") != today:
        snap = {"date": today, "games": {}}

    result   = {}
    snap_games = snap.get("games", {})

    for key, odds in current.items():
        away = odds["away_team"]
        home = odds["home_team"]
        game_key = f"{away}|{home}"

        away_cur = odds["away_avg_odds"]
        home_cur = odds["home_avg_odds"]

        if game_key not in snap_games:
            # First time seeing this game today — save as opening line
            snap_games[game_key] = {
                "away_open": away_cur,
                "home_open": home_cur,
            }
            away_open = away_cur
            home_open = home_cur
        else:
            away_open = snap_games[game_key]["away_open"]
            home_open = snap_games[game_key]["home_open"]

        away_move = away_cur - away_open
        home_move = home_cur - home_open

        # "Sharp move toward away" = away odds got shorter (negative move for away fav)
        # A moneyline moving from +120 to +105 = move of -15 = sharp on away
        # A moneyline moving from -130 to -145 = move of -15 = sharp on away (favorite getting heavier)
        # Simple rule: sharp_away if away line moved 10+ points toward being shorter (away implied prob went up)
        away_impl_open = _american_to_prob(away_open) * 100
        away_impl_cur  = _american_to_prob(away_cur)  * 100
        home_impl_open = _american_to_prob(home_open) * 100
        home_impl_cur  = _american_to_prob(home_cur)  * 100

        away_impl_move = away_impl_cur - away_impl_open   # positive = sharp on away
        home_impl_move = home_impl_cur - home_impl_open

        fkey = frozenset([away, home])
        result[fkey] = {
            "away_open":       away_open,
            "home_open":       home_open,
            "away_current":    away_cur,
            "home_current":    home_cur,
            "away_impl_move":  round(away_impl_move, 1),
            "home_impl_move":  round(home_impl_move, 1),
            "sharp_away":      away_impl_move >= 3.0,   # 3+ pp implies sharp money
            "sharp_home":      home_impl_move >= 3.0,
        }

    snap["games"] = snap_games
    _save_snapshot(snap)
    return result


# ── Public Betting % (Action Network) ───────────────────────────────────────

_PUBLIC_BET_CACHE: dict = {}
_PUBLIC_BET_TTL = 60 * 30   # 30 minutes


def get_public_betting_pcts() -> dict:
    """
    Fetch today's MLB public betting percentages from Action Network.
    Returns a dict keyed by frozenset({away_team, home_team}):
      {
        "away_bet_pct":   int,   # % of bets on away ML
        "home_bet_pct":   int,   # % of bets on home ML
        "away_money_pct": int,   # % of money on away ML
        "home_money_pct": int,   # % of money on home ML
        "sharp_indicator": str,  # "away" / "home" / None
      }
    """
    cache_key = "public_bet_pcts"
    cached = _file_cache_get(cache_key, _PUBLIC_BET_TTL)
    if cached:
        return {frozenset(k.split("|")): v for k, v in cached.items()}

    from datetime import date
    today = date.today().strftime("%Y%m%d")

    try:
        resp = requests.get(
            "https://api.actionnetwork.com/web/v1/games",
            params={"sport": "mlb", "date": today},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}

    # Action Network team name → MLB short name mapping
    AN_TEAMS = {
        "Arizona Diamondbacks": "Arizona Diamondbacks",
        "Atlanta Braves": "Atlanta Braves",
        "Baltimore Orioles": "Baltimore Orioles",
        "Boston Red Sox": "Boston Red Sox",
        "Chicago Cubs": "Chicago Cubs",
        "Chicago White Sox": "Chicago White Sox",
        "Cincinnati Reds": "Cincinnati Reds",
        "Cleveland Guardians": "Cleveland Guardians",
        "Colorado Rockies": "Colorado Rockies",
        "Detroit Tigers": "Detroit Tigers",
        "Houston Astros": "Houston Astros",
        "Kansas City Royals": "Kansas City Royals",
        "Los Angeles Angels": "Los Angeles Angels",
        "Los Angeles Dodgers": "Los Angeles Dodgers",
        "Miami Marlins": "Miami Marlins",
        "Milwaukee Brewers": "Milwaukee Brewers",
        "Minnesota Twins": "Minnesota Twins",
        "New York Mets": "New York Mets",
        "New York Yankees": "New York Yankees",
        "Oakland Athletics": "Oakland Athletics",
        "Philadelphia Phillies": "Philadelphia Phillies",
        "Pittsburgh Pirates": "Pittsburgh Pirates",
        "San Diego Padres": "San Diego Padres",
        "San Francisco Giants": "San Francisco Giants",
        "Seattle Mariners": "Seattle Mariners",
        "St. Louis Cardinals": "St. Louis Cardinals",
        "Tampa Bay Rays": "Tampa Bay Rays",
        "Texas Rangers": "Texas Rangers",
        "Toronto Blue Jays": "Toronto Blue Jays",
        "Washington Nationals": "Washington Nationals",
    }

    result = {}
    teams_lookup = data.get("teams", {})

    for game in data.get("games", []):
        try:
            away_id = game.get("away_team_id")
            home_id = game.get("home_team_id")

            away_name = teams_lookup.get(str(away_id), {}).get("full_name", "")
            home_name = teams_lookup.get(str(home_id), {}).get("full_name", "")

            away_name = AN_TEAMS.get(away_name, away_name)
            home_name = AN_TEAMS.get(home_name, home_name)

            if not away_name or not home_name:
                continue

            # Moneyline bet/money %
            odds_list = game.get("odds", [])
            away_bet = home_bet = away_money = home_money = None

            for o in odds_list:
                if o.get("book_id") in (15, 16, 17, 138):  # popular books on AN
                    ml = o.get("ml_away_bet_pct") or o.get("away_ml_bet_pct")
                    if ml is not None:
                        away_bet  = int(ml)
                        home_bet  = 100 - away_bet
                    mm = o.get("ml_away_money_pct") or o.get("away_ml_money_pct")
                    if mm is not None:
                        away_money = int(mm)
                        home_money = 100 - away_money
                    if away_bet is not None:
                        break

            # Fallback: use first odds entry with any bet pct
            if away_bet is None:
                for o in odds_list:
                    for key in ("ml_away_bet_pct", "away_ml_bet_pct", "away_bet_pct"):
                        if o.get(key) is not None:
                            away_bet  = int(o[key])
                            home_bet  = 100 - away_bet
                            break
                    if away_bet is not None:
                        break

            if away_bet is None:
                continue

            # Sharp indicator: money % diverges significantly from bet %
            # "Reverse line movement" — lots of public bets on one side
            # but more money on the other = sharp money on the other side
            sharp = None
            if away_money is not None:
                if away_money - away_bet >= 15:
                    sharp = "away"    # sharp money on away despite public on home
                elif home_money - home_bet >= 15:
                    sharp = "home"

            fkey = frozenset([away_name, home_name])
            result[fkey] = {
                "away_bet_pct":    away_bet,
                "home_bet_pct":    home_bet,
                "away_money_pct":  away_money,
                "home_money_pct":  home_money,
                "sharp_indicator": sharp,
            }
        except Exception:
            continue

    # Cache with string keys (frozensets can't be JSON-serialized)
    serializable = {"|".join(sorted(k)): v for k, v in result.items()}
    _file_cache_set(cache_key, serializable)
    return result
