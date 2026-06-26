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

# ── In-memory cache for moneyline odds ──────────────────────────────
# Odds barely move inning to inning, so we cache for 30 minutes.
# This cuts API usage from 1 request per simulation down to
# ~1 request per 30-minute window regardless of how many games you run.
_ODDS_CACHE = {"data": {}, "ts": 0.0}
_CACHE_TTL  = 30 * 60   # 30 minutes in seconds

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

    # Return cached data if it's still fresh
    if time.time() - _ODDS_CACHE["ts"] < _CACHE_TTL and _ODDS_CACHE["data"]:
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
        return {}

    result = {}

    for game in games:
        away = game.get("away_team", "")
        home = game.get("home_team", "")

        # Collect odds from each bookmaker
        away_odds_list = []
        home_odds_list = []

        for book in game.get("bookmakers", []):
            # Optionally filter to preferred books only
            # (comment this out to use all available books)
            # if book["key"] not in PREFERRED_BOOKS:
            #     continue

            for market in book.get("markets", []):
                if market["key"] != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome["name"] == away:
                        away_odds_list.append(outcome["price"])
                    elif outcome["name"] == home:
                        home_odds_list.append(outcome["price"])

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
        }

    # Store in cache with current timestamp
    _ODDS_CACHE["data"] = result
    _ODDS_CACHE["ts"]   = time.time()
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
