"""
engine.py
---------
The Monte Carlo simulation engine.
Plays out one baseball game at-bat by at-bat, then repeats N times.

How it works:
  1. Convert each batter's real stats into outcome probabilities
     (what % of the time do they single, double, homer, walk, strike out, etc.)
  2. Adjust those probabilities based on who the starting pitcher is
     (a great pitcher shifts probability away from hits toward outs)
  3. Simulate each at-bat by drawing a random outcome
  4. Track runners on base and count runs
  5. Repeat for all 9 innings → one complete game result
  6. Run that 100,000 times → win probability
"""

import random
from bisect import bisect
from itertools import accumulate

# ── Outcome labels ──────────────────────────────────────────────
SINGLE    = "single"
DOUBLE    = "double"
TRIPLE    = "triple"
HR        = "home_run"
WALK      = "walk"
STRIKEOUT = "strikeout"
OUT       = "out"

# Order matters — this is the fixed order we use for probability arrays
OUTCOME_ORDER = [WALK, STRIKEOUT, SINGLE, DOUBLE, TRIPLE, HR, OUT]

# League average benchmarks (2025 MLB season approximations)
LEAGUE_AVG_ERA  = 4.00
LEAGUE_AVG_WHIP = 1.28

# ── Ballpark factors ─────────────────────────────────────────────
#
# Each park has multipliers for:
#   hr_factor   — how much the park boosts or suppresses home runs
#                 (1.0 = league average, 1.30 = 30% more HRs, 0.75 = 25% fewer)
#   hit_factor  — affects singles and doubles (line drives, gaps)
#   run_factor  — overall run environment scaling
#
# Sources: multi-year MLB park factors (Statcast/Baseball Reference)
#
# OUTCOME_ORDER = [WALK, STRIKEOUT, SINGLE, DOUBLE, TRIPLE, HR, OUT]
#                    0        1        2       3       4      5   6

BALLPARK_FACTORS = {
    # ── Hitter-friendly parks ──────────────────────────────────────
    "Coors Field":              {"hr": 1.30, "hit": 1.15, "run": 1.20},  # altitude, thin air
    "Great American Ball Park": {"hr": 1.18, "hit": 1.05, "run": 1.10},  # small park, Ohio wind
    "Globe Life Field":         {"hr": 1.12, "hit": 1.04, "run": 1.06},  # hitter-friendly indoors
    "Minute Maid Park":         {"hr": 1.10, "hit": 1.03, "run": 1.05},  # short left field
    "Yankee Stadium":           {"hr": 1.15, "hit": 1.02, "run": 1.07},  # short porch in right
    "Citizens Bank Park":       {"hr": 1.12, "hit": 1.04, "run": 1.07},  # Philly wind, small gaps
    "American Family Field":    {"hr": 1.08, "hit": 1.03, "run": 1.05},
    "Truist Park":              {"hr": 1.06, "hit": 1.02, "run": 1.04},
    "Kauffman Stadium":         {"hr": 1.05, "hit": 1.02, "run": 1.03},
    "Angel Stadium":            {"hr": 1.04, "hit": 1.01, "run": 1.02},
    "Daikin Park":              {"hr": 1.04, "hit": 1.01, "run": 1.03},
    "Oriole Park at Camden Yards": {"hr": 1.08, "hit": 1.03, "run": 1.05},

    # ── Neutral parks ─────────────────────────────────────────────
    "Dodger Stadium":           {"hr": 1.00, "hit": 1.00, "run": 1.00},
    "Wrigley Field":            {"hr": 1.02, "hit": 1.01, "run": 1.01},  # varies with wind
    "Target Field":             {"hr": 0.98, "hit": 1.00, "run": 0.99},
    "Busch Stadium":            {"hr": 0.97, "hit": 0.99, "run": 0.98},
    "Progressive Field":        {"hr": 0.98, "hit": 1.00, "run": 0.99},
    "Comerica Park":            {"hr": 0.96, "hit": 1.00, "run": 0.98},
    "Rogers Centre":            {"hr": 1.01, "hit": 1.00, "run": 1.00},
    "Nationals Park":           {"hr": 0.99, "hit": 1.00, "run": 0.99},
    "Chase Field":              {"hr": 1.01, "hit": 1.01, "run": 1.01},  # retractable roof
    "loanDepot park":           {"hr": 0.97, "hit": 0.99, "run": 0.98},  # pitcher-friendly

    # ── Pitcher-friendly parks ─────────────────────────────────────
    "Oracle Park":              {"hr": 0.82, "hit": 0.96, "run": 0.88},  # sea breeze kills HRs
    "T-Mobile Park":            {"hr": 0.88, "hit": 0.97, "run": 0.92},  # marine layer
    "Petco Park":               {"hr": 0.86, "hit": 0.97, "run": 0.90},  # large dimensions
    "PNC Park":                 {"hr": 0.90, "hit": 0.98, "run": 0.93},
    "Fenway Park":              {"hr": 0.92, "hit": 1.02, "run": 0.97},  # Green Monster = hits not HRs
    "Citi Field":               {"hr": 0.88, "hit": 0.97, "run": 0.91},
    "Tropicana Field":          {"hr": 0.90, "hit": 0.97, "run": 0.92},
    "Guaranteed Rate Field":    {"hr": 0.93, "hit": 0.98, "run": 0.95},
    "Sutter Health Park":       {"hr": 0.95, "hit": 0.99, "run": 0.97},
    "Sahlen Field":             {"hr": 0.95, "hit": 0.99, "run": 0.97},
}

# Default for any park not in the list (league average)
DEFAULT_PARK_FACTOR = {"hr": 1.0, "hit": 1.0, "run": 1.0}


def apply_ballpark_factor(probs: list, venue: str) -> list:
    """
    Adjust batter outcome probabilities based on the ballpark.

    Coors Field: thin air at 5,280 ft — balls carry further → +30% HR, +15% hits
    Oracle Park: ocean breeze blows balls back in → -18% HR, -4% hits
    Fenway Park: Green Monster turns HRs into doubles → lower HR, higher hits
    etc.

    OUTCOME_ORDER = [WALK, K, 1B, 2B, 3B, HR, OUT]
                       0   1   2   3   4   5   6
    """
    factors = BALLPARK_FACTORS.get(venue, DEFAULT_PARK_FACTOR)

    adjusted = list(probs)
    adjusted[2] *= factors["hit"]   # singles
    adjusted[3] *= factors["hit"]   # doubles
    adjusted[4] *= factors["hit"]   # triples
    adjusted[5] *= factors["hr"]    # home runs

    # Normalize so probabilities still sum to 1.0
    total = sum(adjusted)
    return [p / total for p in adjusted]


# ── STEP 1: Build batter probability distribution ───────────────

def build_batter_probs(stats: dict) -> list:
    """
    Convert a batter's season stats into a list of probabilities,
    one per outcome, in OUTCOME_ORDER.

    For example, a .300 hitter with 30 HR might look like:
      [walk=8%, K=20%, single=15%, double=5%, triple=1%, HR=5%, out=46%]

    If we have no stats (e.g. player didn't play yet), we fall back
    to a generic league-average hitter.
    """
    pa = stats.get("plateAppearances", 0)

    if pa < 10:
        # Not enough data — use a league-average hitter as fallback
        return [0.082, 0.224, 0.148, 0.048, 0.004, 0.033, 0.461]

    hits    = stats.get("hits", 0)
    bb      = stats.get("baseOnBalls", 0)
    k       = stats.get("strikeOuts", 0)
    hr      = stats.get("homeRuns", 0)
    doubles = stats.get("doubles", 0)
    triples = stats.get("triples", 0)
    singles = max(0, hits - doubles - triples - hr)

    p_walk = bb      / pa
    p_k    = k       / pa
    p_1b   = singles / pa
    p_2b   = doubles / pa
    p_3b   = triples / pa
    p_hr   = hr      / pa
    p_out  = max(0.0, 1.0 - p_walk - p_k - p_1b - p_2b - p_3b - p_hr)

    # Return in OUTCOME_ORDER: WALK, K, 1B, 2B, 3B, HR, OUT
    return [p_walk, p_k, p_1b, p_2b, p_3b, p_hr, p_out]


def apply_weather_modifier(probs: list, weather: dict) -> list:
    """
    Adjust HR probability based on ballpark weather conditions.

    Effects:
    - Cold air (< 50°F): ball doesn't carry as well → fewer HRs
    - Hot air (> 85°F): ball carries better → more HRs
    - Wind blowing OUT (speed > 10 mph, roughly toward CF): more HRs
    - Wind blowing IN  (speed > 10 mph, roughly toward home): fewer HRs

    We simplify wind direction to: if wind_deg is between 45–135° (roughly
    blowing from left/right field toward the infield) it's blowing IN.
    Otherwise it's blowing OUT or across.
    """
    if not weather:
        return probs   # no weather data, no change

    adjusted = list(probs)

    # Temperature effect on HR (index 5 = HR in OUTCOME_ORDER)
    temp = weather.get("temp_f", 72)
    if temp < 50:
        hr_temp_factor = 0.85    # cold air: 15% fewer HRs
    elif temp > 85:
        hr_temp_factor = 1.10    # hot air: 10% more HRs
    else:
        hr_temp_factor = 1.0

    # Wind effect on HR
    wind_mph = weather.get("wind_mph", 0)
    wind_deg = weather.get("wind_deg", 0)

    # Blowing IN = wind coming from the outfield toward home plate (roughly 45–135°)
    # Blowing OUT = coming from behind home plate toward outfield (roughly 225–315°)
    if wind_mph > 10:
        if 45 <= wind_deg <= 135:
            hr_wind_factor = max(0.75, 1.0 - (wind_mph - 10) * 0.015)  # blowing in
        elif 225 <= wind_deg <= 315:
            hr_wind_factor = min(1.35, 1.0 + (wind_mph - 10) * 0.015)  # blowing out
        else:
            hr_wind_factor = 1.0   # crosswind
    else:
        hr_wind_factor = 1.0

    total_hr_factor = hr_temp_factor * hr_wind_factor
    adjusted[5] *= total_hr_factor   # index 5 = HR

    # Normalize
    total = sum(adjusted)
    return [p / total for p in adjusted]


def apply_pitcher_modifier(probs: list, pitcher_stats: dict) -> list:
    """
    Adjust batter probabilities for the quality of the pitcher they're facing.

    Logic:
    - A pitcher with ERA below the league average (4.00) is better → fewer hits
    - A pitcher with ERA above average → more hits
    - We scale hit probabilities up/down, then re-normalize to 1.0

    We also adjust walk probability based on WHIP.
    """
    # Get pitcher ERA and WHIP (they come from the API as strings like "3.31")
    try:
        era  = float(pitcher_stats.get("era",  LEAGUE_AVG_ERA))
    except (ValueError, TypeError):
        era  = LEAGUE_AVG_ERA

    try:
        whip = float(pitcher_stats.get("whip", LEAGUE_AVG_WHIP))
    except (ValueError, TypeError):
        whip = LEAGUE_AVG_WHIP

    # Hit scale: ERA 2.00 → hits × 0.50, ERA 4.00 → hits × 1.00, ERA 6.00 → hits × 1.50
    hit_scale  = min(max(era  / LEAGUE_AVG_ERA,  0.50), 1.50)
    walk_scale = min(max(whip / LEAGUE_AVG_WHIP, 0.50), 1.50)

    # Indices in OUTCOME_ORDER: WALK=0, K=1, 1B=2, 2B=3, 3B=4, HR=5, OUT=6
    adjusted = list(probs)
    adjusted[0] *= walk_scale          # walk
    adjusted[2] *= hit_scale           # single
    adjusted[3] *= hit_scale           # double
    adjusted[4] *= hit_scale           # triple
    adjusted[5] *= hit_scale           # home run

    # Normalize so probabilities still sum to 1.0
    total = sum(adjusted)
    return [p / total for p in adjusted]


def blend_probs(season_probs: list, recent_probs: list, recent_weight: float = 0.4) -> list:
    """
    Blend season-long probabilities with recent form probabilities.

    recent_weight=0.4 means:
      40% of the prediction comes from last 20 games
      60% comes from the full season

    This way a player on a hot streak gets a meaningful boost,
    but one good week doesn't completely override who they are
    as a hitter over 150+ games.
    """
    season_weight = 1.0 - recent_weight
    blended = [season_weight * s + recent_weight * r
               for s, r in zip(season_probs, recent_probs)]
    total = sum(blended)
    return [p / total for p in blended]


def precompute_lineup(lineup_stats: list, pitcher_stats: dict,
                      weather: dict = None, recent_stats_list: list = None,
                      venue: str = None) -> list:
    """
    Pre-calculate cumulative probability arrays for every batter in a lineup.
    Doing this ONCE before the simulation loop (instead of inside it) is the
    key performance optimization — we avoid repeating the same math 100,000 times.

    Applies (in order):
      1. Recent form blend (season 60% + last-20-games 40%)
      2. Pitcher modifier (ERA/WHIP)
      3. Ballpark factor (Coors, Oracle Park, etc.)
      4. Weather modifier (wind, temperature)

    Ballpark is applied before weather so that park-specific conditions
    (altitude, dimensions) are set first, then today's specific weather
    fine-tunes on top of that.
    """
    result = []
    for i, stats in enumerate(lineup_stats):
        # Start with season stats
        probs = build_batter_probs(stats)

        # Blend in recent form if available (last 20 games)
        if recent_stats_list and i < len(recent_stats_list):
            recent = recent_stats_list[i]
            if recent and recent.get("plateAppearances", 0) >= 20:
                recent_probs = build_batter_probs(recent)
                probs = blend_probs(probs, recent_probs, recent_weight=0.4)

        # Pitcher quality modifier
        probs = apply_pitcher_modifier(probs, pitcher_stats)

        # Ballpark factor (dimensions, altitude, park tendencies)
        if venue:
            probs = apply_ballpark_factor(probs, venue)

        # Today's weather on top of ballpark baseline
        if weather:
            probs = apply_weather_modifier(probs, weather)

        cum_weights = list(accumulate(probs))
        result.append(cum_weights)
    return result


# ── STEP 2: Simulate one at-bat ─────────────────────────────────

def simulate_at_bat(cum_weights: list) -> str:
    """
    Draw one random at-bat outcome.
    Uses a random float + binary search (bisect) — faster than random.choices().
    """
    r = random.random()                      # random float 0.0–1.0
    i = bisect(cum_weights, r)               # find which bucket it lands in
    i = min(i, len(OUTCOME_ORDER) - 1)      # safety clamp
    return OUTCOME_ORDER[i]


# ── STEP 3: Advance base runners ─────────────────────────────────

def advance_runners(b1: bool, b2: bool, b3: bool, outcome: str) -> tuple:
    """
    Given who's on base and what just happened, return:
      (new_b1, new_b2, new_b3, runs_scored)

    Simplified base-running rules:
      - Single:  3rd scores, 2nd scores, 1st→2nd, batter→1st
      - Double:  3rd scores, 2nd scores, 1st scores, batter→2nd
      - Triple:  all score, batter→3rd
      - HR:      all score + batter
      - Walk:    force advance only
      - Out/K:   no movement
    """
    if outcome == WALK:
        if b1 and b2 and b3:
            return True,  True,  True,  1     # 3rd pushed home
        if b1 and b2:
            return True,  True,  True,  0     # load bases
        if b1:
            return True,  True,  b3,    0     # 1st pushes to 2nd
        return True,  b2,   b3,   0           # batter takes 1st

    if outcome in (STRIKEOUT, OUT):
        return b1, b2, b3, 0                  # no movement

    if outcome == SINGLE:
        runs = int(b3) + int(b2)              # 3rd and 2nd score
        return True, b1, False, runs          # batter→1st, old 1st→2nd

    if outcome == DOUBLE:
        runs = int(b3) + int(b2) + int(b1)   # everyone scores (simplified)
        return False, True, False, runs       # batter→2nd

    if outcome == TRIPLE:
        runs = int(b1) + int(b2) + int(b3)
        return False, False, True, runs       # batter→3rd

    if outcome == HR:
        runs = int(b1) + int(b2) + int(b3) + 1  # everyone + batter
        return False, False, False, runs

    return b1, b2, b3, 0


# ── STEP 4: Simulate one half-inning ────────────────────────────

def simulate_half_inning(precomp_lineup: list, lineup_pos: int) -> tuple:
    """
    Simulate one half-inning (3 outs) for one team.

    precomp_lineup: pre-computed cumulative weight arrays (one per batter)
    lineup_pos:     which slot in the batting order is up first

    Returns: (runs_scored, new_lineup_pos)
    The lineup_pos return value lets the next inning pick up where this one left off.
    """
    outs = 0
    runs = 0
    b1 = b2 = b3 = False  # nobody on base
    pos = lineup_pos

    while outs < 3:
        cum_weights = precomp_lineup[pos % 9]
        outcome     = simulate_at_bat(cum_weights)

        if outcome in (STRIKEOUT, OUT):
            outs += 1
        else:
            b1, b2, b3, new_runs = advance_runners(b1, b2, b3, outcome)
            runs += new_runs

        pos += 1

    return runs, pos % 9


# ── STEP 5: Simulate one full game ──────────────────────────────

def simulate_game(
    away_precomp_early: list,
    home_precomp_early: list,
    away_precomp_late:  list = None,
    home_precomp_late:  list = None,
    innings: int = 9,
    bullpen_start: int = 5,
) -> tuple:
    """
    Simulate one complete 9-inning game.

    Innings 1 through bullpen_start (default: 5) use the starting pitcher lineup probs.
    Innings bullpen_start+1 onward switch to bullpen lineup probs (if provided).
    This means batters who are strong vs relievers get a boost late in the game,
    and batters who only hit starters take a hit when the bullpen comes in.

    Returns: (away_runs, home_runs)
    """
    away_runs = 0
    home_runs = 0
    away_pos  = 0
    home_pos  = 0

    for inning in range(innings):
        # Switch from starter to bullpen probs after bullpen_start innings
        if inning >= bullpen_start and away_precomp_late:
            away_cur = away_precomp_late
            home_cur = home_precomp_late
        else:
            away_cur = away_precomp_early
            home_cur = home_precomp_early

        # Away team bats
        runs, away_pos = simulate_half_inning(away_cur, away_pos)
        away_runs += runs

        # Walk-off rule: home team wins in 9th without finishing if already ahead
        if inning == innings - 1 and home_runs > away_runs:
            break

        # Home team bats
        runs, home_pos = simulate_half_inning(home_cur, home_pos)
        home_runs += runs

    # Extra innings if tied (up to 3 extra) — use bullpen probs
    extra_away = away_precomp_late or away_precomp_early
    extra_home = home_precomp_late or home_precomp_early
    extra = 0
    while away_runs == home_runs and extra < 3:
        runs, away_pos = simulate_half_inning(extra_away, away_pos)
        away_runs += runs
        runs, home_pos = simulate_half_inning(extra_home, home_pos)
        home_runs += runs
        extra += 1

    # If still tied, random winner (very rare)
    if away_runs == home_runs:
        if random.random() < 0.5:
            away_runs += 1
        else:
            home_runs += 1

    return away_runs, home_runs


# ── STEP 6: Run N simulations ────────────────────────────────────

def run_simulation(
    away_team:      str,
    home_team:      str,
    away_lineup:    list,        # batter stat dicts (L/R split) for away team
    home_lineup:    list,        # batter stat dicts (L/R split) for home team
    away_pitcher:   dict,        # away starting pitcher's season stats
    home_pitcher:   dict,        # home starting pitcher's season stats
    weather:        dict = None, # ballpark weather
    away_recent:    list = None, # last-20-game stats for away batters
    home_recent:    list = None, # last-20-game stats for home batters
    away_rp_stats:  list = None, # away batters' stats vs relief pitchers (bullpen)
    home_rp_stats:  list = None, # home batters' stats vs relief pitchers (bullpen)
    away_bullpen:   dict = None, # away team's actual bullpen ERA/WHIP (for innings 6-9)
    home_bullpen:   dict = None, # home team's actual bullpen ERA/WHIP (for innings 6-9)
    venue:          str  = None, # ballpark name for park factor adjustment
    n:              int = 100_000,
) -> dict:
    """
    Run N Monte Carlo simulations and return aggregated results.

    Factors in:
      - L/R pitcher splits (via away_lineup / home_lineup)
      - Recent form: 60% season stats + 40% last-20-game stats
      - Weather: wind and temperature affect HR probability
      - SP vs RP: innings 1-5 use starting pitcher probs,
                  innings 6-9 switch to bullpen probs (if rp_stats provided)
    """
    # League-average pitcher stats used as the "generic bullpen" modifier
    # (we don't know which relievers will pitch, so we assume league average)
    LEAGUE_AVG_PITCHER = {"era": "4.20", "whip": "1.30"}

    # ── Early game: batters vs the starting pitcher ──────────────────
    away_precomp_early = precompute_lineup(away_lineup, home_pitcher, weather, away_recent, venue)
    home_precomp_early = precompute_lineup(home_lineup, away_pitcher, weather, home_recent, venue)

    # ── Late game: batters vs the bullpen ────────────────────────────
    # Use real team bullpen ERA/WHIP if available, otherwise league average.
    # A team with a 3.00 bullpen ERA gets a real edge in innings 6-9.
    away_bp_pitcher = away_bullpen if away_bullpen else LEAGUE_AVG_PITCHER
    home_bp_pitcher = home_bullpen if home_bullpen else LEAGUE_AVG_PITCHER

    away_precomp_late = None
    home_precomp_late = None

    if away_rp_stats and any(s for s in away_rp_stats if s):
        away_precomp_late = precompute_lineup(
            away_rp_stats, home_bp_pitcher, weather, away_recent, venue
        )
    if home_rp_stats and any(s for s in home_rp_stats if s):
        home_precomp_late = precompute_lineup(
            home_rp_stats, away_bp_pitcher, weather, home_recent, venue
        )

    # ── Run the simulation loop ───────────────────────────────────────
    away_wins  = 0
    home_wins  = 0
    away_total = 0
    home_total = 0

    for _ in range(n):
        a, h = simulate_game(
            away_precomp_early, home_precomp_early,
            away_precomp_late,  home_precomp_late,
        )
        away_total += a
        home_total += h
        if a > h:
            away_wins += 1
        else:
            home_wins += 1

    raw_away_pct = away_wins / n * 100
    raw_home_pct = home_wins / n * 100

    # ── Home field advantage adjustment ──────────────────────────────
    # MLB home teams win ~54% of games historically, even controlling for
    # team quality. This captures crowd noise, familiarity, no travel fatigue,
    # and the psychological edge of playing at home.
    # We apply a small shift: nudge home team +2% and away team -2%,
    # then re-normalize so the two still sum to 100%.
    HOME_FIELD_BOOST = 2.0   # percentage points
    adj_home = raw_home_pct + HOME_FIELD_BOOST
    adj_away = raw_away_pct - HOME_FIELD_BOOST
    # Clamp so neither goes below 1% or above 99%
    adj_home = min(max(adj_home, 1.0), 99.0)
    adj_away = min(max(adj_away, 1.0), 99.0)
    # Re-normalize to exactly 100
    total    = adj_away + adj_home
    adj_away = round(adj_away / total * 100, 1)
    adj_home = round(adj_home / total * 100, 1)

    return {
        "away_team":         away_team,
        "home_team":         home_team,
        "simulations":       n,
        "away_win_pct":      adj_away,
        "home_win_pct":      adj_home,
        "away_win_pct_raw":  round(raw_away_pct, 1),  # pre-HFA, for display
        "home_win_pct_raw":  round(raw_home_pct, 1),
        "away_avg_runs":     round(away_total / n, 2),
        "home_avg_runs":     round(home_total / n, 2),
        "uses_bullpen_data": away_precomp_late is not None,
    }
