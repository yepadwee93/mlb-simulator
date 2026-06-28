"""Quick debug script — shows what the Odds API returns for team names."""

from dotenv import load_dotenv

load_dotenv()
from data.mlb_api import get_today_schedule
from data.odds_api import get_mlb_odds

print("\n=== ODDS API TEAM NAMES ===")
odds = get_mlb_odds()
for key, o in odds.items():
    print(
        f"  '{o['away_team']}' @ '{o['home_team']}'  →  {o['away_implied_pct']}% / {o['home_implied_pct']}%"
    )

print("\n=== MLB API TEAM NAMES ===")
games = get_today_schedule()
for g in games:
    print(f"  '{g['away_team']}' @ '{g['home_team']}'")
