"""
odds_api.py
-----------
Fetches real moneyline odds from The Odds API (https://the-odds-api.com)
and compares them to our simulation's win probabilities.

Free tier: 500 requests/month — plenty for daily use.
"""

import os
import requests

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_BASE    = "https://api.the-odds-api.com/v4"

# Sportsbooks to average across (all available on free tier)
PREFERRED_BOOKS = {"draftkings", "fanduel", "betmgm", "bovada", "williamhill_us"}


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
