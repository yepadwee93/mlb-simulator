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
from collections import Counter

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


def apply_rest_modifier(pitcher_stats: dict, rest_days: int) -> dict:
    """
    Adjust a pitcher's ERA and WHIP based on days of rest since their last start.

    Rest day effects on pitcher performance (MLB research consensus):
      ≤ 3 days  — short rest: significantly worse, arm not recovered
        4 days  — one day short: mild penalty
       5-6 days — ideal window: no adjustment
       7-9 days — slightly rusty from the extra days off
      10+ days  — extended break (possible injury return or schedule gap): notable rust

    This gets applied once in run_simulation BEFORE any fatigue or precompute,
    so it stacks correctly with fatigue (tired + short-rest = double danger).

    Returns a copy of pitcher_stats with adjusted era and whip.
    """
    if rest_days is None:
        return pitcher_stats

    if rest_days <= 3:
        era_factor  = 1.18    # Short rest — very taxing on the arm
        whip_factor = 1.11
    elif rest_days == 4:
        era_factor  = 1.08    # One day short of optimal
        whip_factor = 1.05
    elif rest_days <= 6:
        era_factor  = 1.0     # Ideal rest — no change
        whip_factor = 1.0
    elif rest_days <= 9:
        era_factor  = 1.04    # Slightly rusty
        whip_factor = 1.02
    else:
        era_factor  = 1.10    # Extended break — possible injury return / rustiness
        whip_factor = 1.06

    try:
        era  = float(pitcher_stats.get("era",  LEAGUE_AVG_ERA))
    except (ValueError, TypeError):
        era  = LEAGUE_AVG_ERA
    try:
        whip = float(pitcher_stats.get("whip", LEAGUE_AVG_WHIP))
    except (ValueError, TypeError):
        whip = LEAGUE_AVG_WHIP

    return {
        **pitcher_stats,                               # keep wins, losses, etc.
        "era":  str(round(era  * era_factor,  2)),
        "whip": str(round(whip * whip_factor, 2)),
    }


def build_fatigue_curve(pitcher_game_log: list) -> dict:
    """
    Analyze a pitcher's last 10 starts to build inning-phase fatigue multipliers.

    Returns a dict with ERA/WHIP scaling factors for 3 phases:
      phase1 (innings 1-2): always 1.0 — pitcher is always fresh early
      phase2 (innings 3-4): moderate fatigue based on how long they typically last
      phase3 (innings 5):   deep fatigue, last inning before bullpen takes over

    Multiplier > 1.0 means pitcher is WORSE (ERA scaled up by that factor).

    Real-world logic:
      - Jacob deGrom averaging 7 IP → barely fatigues: {1.0, 1.03, 1.07}
      - League-average starter, 5 IP avg → {1.0, 1.08, 1.18}
      - Struggling short-starter avg 3.5 IP → {1.0, 1.18, 1.38}
      - Pitcher who blows up in inning 4 consistently → phase2 amplified further
    """
    if not pitcher_game_log:
        # Default: typical league-average fatigue
        return {"phase1": 1.0, "phase2": 1.07, "phase3": 1.16}

    n = len(pitcher_game_log)
    if n == 0:
        return {"phase1": 1.0, "phase2": 1.07, "phase3": 1.16}

    avg_ip   = sum(g["ip"] for g in pitcher_game_log) / n
    total_ip = sum(g["ip"] for g in pitcher_game_log)
    total_er = sum(g["er"] for g in pitcher_game_log)

    # ER per inning pitched — rough in-game ERA proxy
    er_per_ip = total_er / total_ip if total_ip > 0 else 0.45   # 0.45 ≈ 4.05 ERA

    # How often does the pitcher get knocked out before inning 3?
    early_blowup_rate = sum(1 for g in pitcher_game_log if g["ip"] < 3.0) / n

    # ── Phase 2 multiplier (innings 3-4) ──────────────────────────────────
    # Depends on how long they typically last:
    if avg_ip >= 6.5:
        phase2 = 1.03   # workhorse ace — barely shows fatigue before inning 5
    elif avg_ip >= 5.0:
        phase2 = 1.07   # solid starter
    elif avg_ip >= 3.5:
        phase2 = 1.14   # often pulled in middle innings — already tiring by inning 3
    else:
        phase2 = 1.22   # short-starter type, frequently in trouble early

    # Amplify if they get knocked out early a lot (bad sign by inning 3)
    if early_blowup_rate >= 0.30:
        phase2 = round(phase2 * 1.07, 3)

    # ── Phase 3 multiplier (inning 5, last inning before bullpen) ─────────
    if avg_ip >= 6.5:
        phase3 = phase2 * 1.05    # still effective, just a tiny drop-off
    elif avg_ip >= 5.0:
        phase3 = phase2 * 1.11    # noticeably worse by inning 5
    elif avg_ip >= 3.5:
        phase3 = phase2 * 1.18    # really falling apart
    else:
        phase3 = phase2 * 1.25    # rarely makes it here — very dangerous inning

    # Amplify or reduce based on overall run-prevention quality
    if er_per_ip > 0.60:     # > 5.4 ERA-equivalent — struggling pitcher
        phase2 *= 1.04
        phase3 *= 1.07
    elif er_per_ip < 0.25:   # < 2.25 ERA-equivalent — ace
        phase2 *= 0.93
        phase3 *= 0.90

    return {
        "phase1": 1.0,
        "phase2": round(min(phase2, 1.50), 3),
        "phase3": round(min(phase3, 1.90), 3),
    }


def build_fatigued_pitcher_stats(base_stats: dict, fatigue_factor: float) -> dict:
    """
    Scale a pitcher's ERA and WHIP up by the fatigue factor.

    fatigue_factor = 1.0  → no change (fresh pitcher, innings 1-2)
    fatigue_factor = 1.12 → ERA and WHIP 12% worse (innings 3-4)
    fatigue_factor = 1.28 → 28% worse (inning 5 for a struggling pitcher)

    We scale WHIP slightly less aggressively than ERA because walks/hits
    don't degrade as quickly as run-prevention in late innings.
    """
    try:
        era  = float(base_stats.get("era",  LEAGUE_AVG_ERA))
    except (ValueError, TypeError):
        era  = LEAGUE_AVG_ERA
    try:
        whip = float(base_stats.get("whip", LEAGUE_AVG_WHIP))
    except (ValueError, TypeError):
        whip = LEAGUE_AVG_WHIP

    return {
        "era":  str(round(era  * fatigue_factor, 2)),
        # WHIP degrades at ~75% the rate of ERA
        "whip": str(round(whip * (1 + (fatigue_factor - 1) * 0.75), 2)),
    }


def apply_batter_rest_modifier(probs: list, rest_days: int) -> list:
    """
    Adjust batter hit probabilities based on days of rest since last game.

    Back-to-back (1 day rest): small hit penalty, batters tire over 162 games.
    Extra rest (3+ days): small hit boost, batters are fresh.
    Normal (2 days): no change.

    OUTCOME_ORDER = [WALK, K, 1B, 2B, 3B, HR, OUT]
    """
    if rest_days is None or rest_days == 2:
        return probs

    adjusted = list(probs)
    if rest_days == 1:
        hit_factor = 0.96   # back-to-back: 4% hit reduction
    elif rest_days == 3:
        hit_factor = 1.01   # one extra day off
    else:
        hit_factor = 1.02   # 4+ days rest: fresh

    for i in (2, 3, 4, 5):   # 1B, 2B, 3B, HR
        adjusted[i] *= hit_factor

    total = sum(adjusted)
    return [p / total for p in adjusted]


# Statcast league averages (2024 MLB)
LEAGUE_AVG_BARREL   = 7.0    # 7% barrel rate
LEAGUE_AVG_HARD_HIT = 38.0   # 38% hard-hit rate (95+ mph)


def apply_statcast_modifier(probs: list, barrel_pct: float, hard_hit_pct: float) -> list:
    """
    Adjust HR and extra-base hit probabilities using Statcast quality-of-contact data.

    Barrel rate and hard-hit % both correlate strongly with power output.
    A 14% barrel rate (2x league avg) boosts HR probability meaningfully.
    A 3% barrel rate (below avg) reduces it.

    We use dampened power functions to avoid overcorrecting on small samples.
    OUTCOME_ORDER = [WALK, K, 1B, 2B, 3B, HR, OUT]
                       0   1   2   3   4   5   6
    """
    if barrel_pct <= 0 or hard_hit_pct <= 0:
        return probs

    barrel_factor   = (barrel_pct   / LEAGUE_AVG_BARREL)   ** 0.35
    hard_hit_factor = (hard_hit_pct / LEAGUE_AVG_HARD_HIT) ** 0.25
    combined = barrel_factor * 0.65 + hard_hit_factor * 0.35

    adjusted = list(probs)
    adjusted[5] *= combined                          # HR: full effect
    adjusted[3] *= (1 + (combined - 1) * 0.40)      # 2B: 40% of effect

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
                      venue: str = None, matchup_stats_list: list = None,
                      daynight_stats_list: list = None,
                      batter_rest_days: int = None,
                      savant_list: list = None,
                      umpire_name: str = None) -> list:
    """
    Pre-calculate cumulative probability arrays for every batter in a lineup.
    Doing this ONCE before the simulation loop (instead of inside it) is the
    key performance optimization — we avoid repeating the same math 100,000 times.

    Applies (in order):
      1. Recent form blend (season 60% + last-20-games 40%)
      2. Day/night split blend (25% weight if batter has 50+ PA in this game type)
      3. Matchup history blend (30% weight if batter has 20+ career PA vs this pitcher)
      4. Pitcher modifier (ERA/WHIP)
      5. Ballpark factor (Coors, Oracle Park, etc.)
      6. Weather modifier (wind, temperature)
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

        # Blend in day/night split — some batters are dramatically better in day or night games.
        # Require 50+ PA in this game type to keep the sample meaningful.
        if daynight_stats_list and i < len(daynight_stats_list):
            dn = daynight_stats_list[i]
            if dn and dn.get("plateAppearances", 0) >= 50:
                dn_probs = build_batter_probs(dn)
                probs = blend_probs(probs, dn_probs, recent_weight=0.25)

        # Blend in matchup history if batter has 20+ career PA vs this specific pitcher.
        # 30% weight: meaningful enough to matter, small enough not to override
        # everything else from a 25-PA career sample.
        if matchup_stats_list and i < len(matchup_stats_list):
            matchup = matchup_stats_list[i]
            if matchup and matchup.get("plateAppearances", 0) >= 20:
                matchup_probs = build_batter_probs(matchup)
                probs = blend_probs(probs, matchup_probs, recent_weight=0.30)

        # Pitcher quality modifier
        probs = apply_pitcher_modifier(probs, pitcher_stats)

        # Lineup protection: pitchers issue more walks to good hitters when
        # the next batter is significantly better (don't want to face the
        # cleanup guy with a runner on). Conversely, they attack the zone
        # more aggressively when a weak hitter follows.
        # OUTCOME_ORDER index 0 = WALK
        try:
            n_batters = len(lineup_stats)
            curr_ops = float(stats.get('ops', '0') or '0')
            next_stats = lineup_stats[(i + 1) % n_batters]
            next_ops = float(next_stats.get('ops', '0') or '0')
            ops_diff = next_ops - curr_ops
            if ops_diff >= 0.120:
                probs[0] *= 1.18   # next batter is a monster -- lots of nibbling
            elif ops_diff >= 0.060:
                probs[0] *= 1.10   # next batter is clearly better
            elif ops_diff >= 0.030:
                probs[0] *= 1.05   # next batter is somewhat better
            elif ops_diff <= -0.060:
                probs[0] *= 0.94   # next batter is weaker -- pitcher attacks zone
            total = sum(probs)
            probs = [p / total for p in probs]
        except (ValueError, TypeError, ZeroDivisionError):
            pass   # missing OPS data -- skip protection modifier

        # Ballpark factor (dimensions, altitude, park tendencies)
        if venue:
            probs = apply_ballpark_factor(probs, venue)

        # Today's weather on top of ballpark baseline
        if weather:
            probs = apply_weather_modifier(probs, weather)

        # Batter rest modifier
        if batter_rest_days is not None:
            probs = apply_batter_rest_modifier(probs, batter_rest_days)

        # Statcast: barrel rate + hard-hit % from Baseball Savant
        if savant_list and i < len(savant_list) and savant_list[i]:
            sv = savant_list[i]
            probs = apply_statcast_modifier(
                probs,
                sv.get('barrel_pct',   0),
                sv.get('hard_hit_pct', 0),
            )

        # Umpire tendencies: wide/tight zone shifts K and BB rates
        if umpire_name:
            probs = apply_umpire_modifier(probs, umpire_name)

        cum_weights = list(accumulate(probs))
        result.append(cum_weights)
    return result


# ── Batter situational tendency rates ────────────────────────────

# League-average rates used as fallback when a batter lacks enough sample
LEAGUE_GDP_RATE    = 0.09   # % of non-K outs with runner on 1st that become DPs
LEAGUE_SF_RATE     = 0.04   # % of outs with runner on 3rd, <2 out that score as sac flies
LEAGUE_SB_RATE     = 0.04   # % of at-bats with runner on 1st only where steal is attempted
LEAGUE_SB_SUCCESS  = 0.72   # % of steal attempts that succeed (league average)


def compute_batter_rates(lineup_stats: list) -> list:
    """
    Extract per-batter situational tendency rates from season stats.

    These rates are used inside simulate_half_inning to apply:
      - GIDP  (ground into double play): when runner on 1st, <2 outs, non-K out
      - SF    (sacrifice fly): when runner on 3rd, <2 outs, non-K out
      - SB    (stolen base attempt): when runner on 1st only, before each at-bat

    Returns a list of dicts (one per batter):
      {
        "gdp_rate":   float,   # probability a non-K out becomes a GIDP (runner on 1st)
        "sf_rate":    float,   # probability a non-K out becomes a sac fly (runner on 3rd)
        "sb_rate":    float,   # probability of a steal attempt (runner on 1st, 2nd open)
        "sb_success": float,   # probability the steal succeeds if attempted
      }
    """
    rates = []
    for stats in lineup_stats:
        ab  = int(stats.get("atBats",              0) or 0)
        gdp = int(stats.get("groundIntoDoublePlay", 0) or 0)
        sf  = int(stats.get("sacFlies",            0) or 0)
        sb  = int(stats.get("stolenBases",         0) or 0)
        cs  = int(stats.get("caughtStealing",      0) or 0)
        hits = int(stats.get("hits",               0) or 0)
        k    = int(stats.get("strikeOuts",         0) or 0)
        pa   = int(stats.get("plateAppearances",   0) or 0)

        if ab >= 100:
            # GIDP rate: GDP / (non-K outs) — these are the outs where a DP is even possible
            non_k_outs = max(ab - hits - k, 1)
            gdp_rate   = gdp / non_k_outs

            # Sac fly rate: SF / (estimated PA with runner on 3rd, <2 out)
            # ~20% of PA come with runner on 3rd and <2 out (rough MLB estimate)
            sf_eligible = max(pa * 0.20, 1)
            sf_rate     = sf / sf_eligible

            # SB: attempt rate per PA where runner on 1st only (~30% of PA)
            sb_attempts = sb + cs
            sb_eligible = max(pa * 0.30, 1)
            sb_rate     = sb_attempts / sb_eligible
            sb_success  = (sb / sb_attempts) if sb_attempts >= 5 else LEAGUE_SB_SUCCESS
        else:
            # Insufficient sample — use league averages
            gdp_rate   = LEAGUE_GDP_RATE
            sf_rate    = LEAGUE_SF_RATE
            sb_rate    = LEAGUE_SB_RATE
            sb_success = LEAGUE_SB_SUCCESS

        rates.append({
            "gdp_rate":   round(min(max(gdp_rate,   0.01), 0.25), 3),
            "sf_rate":    round(min(max(sf_rate,    0.005), 0.15), 3),
            "sb_rate":    round(min(max(sb_rate,    0.005), 0.20), 3),
            "sb_success": round(min(max(sb_success, 0.45), 0.92), 3),
        })

    return rates


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

# ── Umpire tendencies ────────────────────────────────────────────────────────
# k_factor  > 1.0 = wider strike zone = more strikeouts
# bb_factor > 1.0 = tighter zone = more walks
# Both factors are applied to batter probs via apply_umpire_modifier().
# Updated periodically based on UmpScorecards data.
UMP_TENDENCIES = {
    # Wide zone (more Ks, fewer BBs)
    "Laz Diaz":             {"k": 1.14, "bb": 0.86},
    "Alfonso Marquez":      {"k": 1.10, "bb": 0.91},
    "Hunter Wendelstedt":   {"k": 1.08, "bb": 0.92},
    "Jim Reynolds":         {"k": 1.08, "bb": 0.93},
    "Mike Winters":         {"k": 1.07, "bb": 0.93},
    "Marvin Hudson":        {"k": 1.06, "bb": 0.94},
    "Brian Knight":         {"k": 1.05, "bb": 0.95},
    "Fieldin Culbreth":     {"k": 1.05, "bb": 0.95},
    "Doug Eddings":         {"k": 1.04, "bb": 0.96},
    "Scott Barry":          {"k": 1.03, "bb": 0.97},
    "Adam Beck":            {"k": 1.03, "bb": 0.97},
    "Tripp Gibson":         {"k": 1.02, "bb": 0.98},
    "Gabe Morales":         {"k": 1.02, "bb": 0.98},
    "Chris Segal":          {"k": 1.02, "bb": 0.98},
    "Stu Scheurwater":      {"k": 1.02, "bb": 0.98},
    # Neutral / accurate
    "Pat Hoberg":           {"k": 1.00, "bb": 1.00},
    "Will Little":          {"k": 1.01, "bb": 0.99},
    "Ben May":              {"k": 1.01, "bb": 0.99},
    "Ryan Blakney":         {"k": 1.01, "bb": 0.99},
    "Todd Tichenor":        {"k": 1.01, "bb": 0.99},
    "Roberto Ortiz":        {"k": 1.00, "bb": 1.00},
    "James Hoye":           {"k": 1.00, "bb": 1.00},
    "Mark Carlson":         {"k": 0.99, "bb": 1.01},
    "Erich Bacchus":        {"k": 0.99, "bb": 1.01},
    "DJ Reyburn":           {"k": 1.01, "bb": 0.99},
    "John Tumpane":         {"k": 0.99, "bb": 1.01},
    "Paul Nauert":          {"k": 1.00, "bb": 1.00},
    # Tight zone (fewer Ks, more BBs)
    "Andy Fletcher":        {"k": 0.98, "bb": 1.02},
    "Mark Wegner":          {"k": 0.98, "bb": 1.02},
    "Chad Fairchild":       {"k": 0.98, "bb": 1.02},
    "Kerwin Danley":        {"k": 0.98, "bb": 1.02},
    "Lance Barksdale":      {"k": 0.97, "bb": 1.03},
    "Sam Holbrook":         {"k": 0.96, "bb": 1.04},
    "Dan Iassogna":         {"k": 0.95, "bb": 1.06},
    "Ted Barrett":          {"k": 0.94, "bb": 1.07},
    "Phil Cuzzi":           {"k": 0.93, "bb": 1.08},
    "Vic Carapazza":        {"k": 0.93, "bb": 1.08},
    "Tom Hallion":          {"k": 0.91, "bb": 1.10},
    "Bill Miller":          {"k": 0.90, "bb": 1.12},
    "CB Bucknor":           {"k": 0.87, "bb": 1.16},
}


def apply_umpire_modifier(probs: list, umpire_name: str) -> list:
    """
    Adjust strikeout and walk probabilities based on the home plate umpire.

    Wide-zone umps (Laz Diaz, Alfonso Marquez) call more strikes, leading to
    more Ks and fewer BBs. Tight-zone umps (CB Bucknor, Bill Miller) do the
    opposite. This is one of the more predictable game-day edges.

    OUTCOME_ORDER = [WALK, K, 1B, 2B, 3B, HR, OUT]
                       0   1   2   3   4   5   6
    """
    if not umpire_name:
        return probs
    tend = UMP_TENDENCIES.get(umpire_name)
    if not tend:
        return probs

    adjusted = list(probs)
    adjusted[0] *= tend["bb"]   # WALK
    adjusted[1] *= tend["k"]    # STRIKEOUT
    total = sum(adjusted)
    return [p / total for p in adjusted]


# League-average error and wild pitch rates (2024 MLB season)
# Error: ~0.50 errors per team per game / 27 outs = ~1.85% per non-K out
# Wild pitch/PB: ~0.40 WP+PB per team per game = ~1.5% per AB with runners on
ERROR_RATE      = 0.018
WILD_PITCH_RATE = 0.015


def simulate_half_inning(precomp_lineup: list, lineup_pos: int,
                         risp_precomp: list = None,
                         batter_rates: list = None) -> tuple:
    """
    Simulate one half-inning (3 outs) for one team.

    precomp_lineup: pre-computed cumulative weight arrays (one per batter)
    lineup_pos:     which slot in the batting order is up first
    risp_precomp:   optional — RISP (runners in scoring position) probs.
                    When a runner is on 2nd or 3rd, we swap to these probs
                    instead of the regular ones. Clutch batters get a boost;
                    chokers take a hit.
    batter_rates:   optional — per-batter situational tendency rates from
                    compute_batter_rates(). Enables:
                      • GIDP  — runner on 1st, <2 outs, non-K out → double play
                      • SF    — runner on 3rd, <2 outs, non-K out → run scores
                      • SB    — runner on 1st only → steal attempt before each AB

    Returns: (runs_scored, new_lineup_pos)
    The lineup_pos return value lets the next inning pick up where this one left off.
    """
    outs = 0
    runs = 0
    b1 = b2 = b3 = False  # nobody on base
    pos = lineup_pos

    while outs < 3:
        # ── Stolen base attempt (between at-bats, runner on 1st only) ────────
        # We check BEFORE the current batter faces a pitch.
        # Uses the previous batter's (the runner's) steal tendency.
        if batter_rates and b1 and not b2 and outs < 2:
            runner_idx = (pos - 1) % 9   # runner on 1st was the prior batter
            r = batter_rates[runner_idx]
            if random.random() < r["sb_rate"]:
                if random.random() < r["sb_success"]:
                    b2, b1 = True, False    # successful steal of 2nd
                else:
                    b1 = False              # caught stealing — runner is out
                    outs += 1
                    if outs >= 3:
                        break

        # ── Select probability array ──────────────────────────────────────────
        # Switch to RISP stats when runner is on 2nd or 3rd (scoring position).
        if risp_precomp and (b2 or b3):
            cum_weights = risp_precomp[pos % 9]
        else:
            cum_weights = precomp_lineup[pos % 9]

        outcome = simulate_at_bat(cum_weights)

        if outcome in (STRIKEOUT, OUT):
            if outcome == OUT and batter_rates and outs < 2:
                batter_idx = pos % 9

                # ── GIDP: runner on 1st, <2 outs, non-K out ──────────────────
                # Ground ball turns into a double play — two outs, runner erased.
                if b1:
                    if random.random() < batter_rates[batter_idx]["gdp_rate"]:
                        outs += 2       # batter AND runner are both out
                        b1 = False      # runner on 1st is erased
                        pos += 1
                        continue

                # ── Sac fly: runner on 3rd, <2 outs, non-K out ───────────────
                # Ball is caught in the outfield — runner tags and scores.
                # (Only checked if GIDP didn't fire above)
                if b3:
                    if random.random() < batter_rates[batter_idx]["sf_rate"]:
                        runs += 1       # run scores on the sac fly
                        b3 = False      # runner on 3rd scores
                        outs += 1
                        pos += 1
                        continue

            # -- Error: on a non-K out, small chance batter reaches base
            if outcome == OUT and random.random() < ERROR_RATE:
                # Fielding error -- batter reaches 1st, runners advance
                b1, b2, b3, new_runs = advance_runners(b1, b2, b3, SINGLE)
                runs += new_runs
                pos += 1
                continue   # no out recorded

            outs += 1

            # -- Wild pitch / passed ball: advance all runners one base
            # Only matters when there are runners on base.
            if (b1 or b2 or b3) and random.random() < WILD_PITCH_RATE:
                if b3: runs += 1
                b3 = b2
                b2 = b1
                b1 = False

        else:
            # -- Wild pitch before the hit: runners advance on passed ball
            if (b1 or b2 or b3) and random.random() < WILD_PITCH_RATE:
                if b3: runs += 1
                b3 = b2
                b2 = b1
                b1 = False

            b1, b2, b3, new_runs = advance_runners(b1, b2, b3, outcome)
            runs += new_runs

        pos += 1

    return runs, pos % 9


# ── STEP 5: Simulate one full game ──────────────────────────────

def simulate_game(
    away_precomp_early: list,           # innings 1-2: fresh starter
    home_precomp_early: list,
    away_precomp_mid:   list = None,    # innings 3-5: starter showing fatigue
    home_precomp_mid:   list = None,
    away_precomp_late:  list = None,    # innings 6+: bullpen
    home_precomp_late:  list = None,
    away_precomp_risp:  list = None,    # RISP probs for away batters (b2 or b3 occupied)
    home_precomp_risp:  list = None,    # RISP probs for home batters
    away_batter_rates:  list = None,    # per-batter GIDP/SF/SB rates for away batters
    home_batter_rates:  list = None,    # per-batter GIDP/SF/SB rates for home batters
    innings: int = 9,
    bullpen_start: int = 5,             # 0-indexed: inning 6 in human terms
    mid_start:     int = 2,             # 0-indexed: inning 3 in human terms
    resolve_ties:  bool = True,         # False = return tied score as-is (for F3/F5/F7 push calc)
) -> tuple:
    """
    Simulate one complete 9-inning game with 3 pitching phases + RISP clutch stats:

      Phase 1 (innings 1-2):   Starter is fresh. Uses early probs.
      Phase 2 (innings 3-5):   Starter tires. Uses mid probs (ERA/WHIP scaled up).
      Phase 3 (innings 6-9+):  Bullpen takes over. Uses late probs.

      RISP: whenever a runner reaches 2nd or 3rd, the current batter's probs
            switch to their RISP (runners in scoring position) stats. Clutch
            hitters get a boost; weak RISP batters get a penalty.

    Returns: (away_runs, home_runs)
    """
    away_runs = 0
    home_runs = 0
    away_pos  = 0
    home_pos  = 0

    for inning in range(innings):
        # Select which precomputed lineup to use based on inning
        if inning >= bullpen_start and away_precomp_late:
            # Innings 6+ — bullpen is pitching
            away_cur = away_precomp_late
            home_cur = home_precomp_late
        elif inning >= mid_start and away_precomp_mid:
            # Innings 3-5 — starter is tiring (fatigue kicks in)
            away_cur = away_precomp_mid
            home_cur = home_precomp_mid
        else:
            # Innings 1-2 — starter is fresh
            away_cur = away_precomp_early
            home_cur = home_precomp_early

        # Away team bats (RISP probs swap in when b2/b3; GIDP/SF/SB from batter_rates)
        runs, away_pos = simulate_half_inning(away_cur, away_pos,
                                              away_precomp_risp, away_batter_rates)
        away_runs += runs

        # Walk-off rule: home team wins in 9th without finishing if already ahead
        if inning == innings - 1 and home_runs > away_runs:
            break

        # Home team bats
        runs, home_pos = simulate_half_inning(home_cur, home_pos,
                                              home_precomp_risp, home_batter_rates)
        home_runs += runs

    # Extra innings if tied (up to 3 extra) — use bullpen probs
    # Skip if resolve_ties=False (segment bets: tied score = push)
    if resolve_ties:
        extra_away = away_precomp_late or away_precomp_mid or away_precomp_early
        extra_home = home_precomp_late or home_precomp_mid or home_precomp_early
        extra = 0
        while away_runs == home_runs and extra < 3:
            runs, away_pos = simulate_half_inning(extra_away, away_pos,
                                                  away_precomp_risp, away_batter_rates)
            away_runs += runs
            runs, home_pos = simulate_half_inning(extra_home, home_pos,
                                                  home_precomp_risp, home_batter_rates)
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
    away_team:         str,
    home_team:         str,
    away_lineup:       list,        # batter stat dicts (L/R split) for away team
    home_lineup:       list,        # batter stat dicts (L/R split) for home team
    away_pitcher:      dict,        # away starting pitcher's season stats
    home_pitcher:      dict,        # home starting pitcher's season stats
    weather:           dict = None, # ballpark weather
    away_recent:       list = None, # last-20-game stats for away batters
    home_recent:       list = None, # last-20-game stats for home batters
    away_rp_stats:     list = None, # away batters' stats vs relief pitchers (bullpen)
    home_rp_stats:     list = None, # home batters' stats vs relief pitchers (bullpen)
    away_bullpen:      dict = None, # away team's actual bullpen ERA/WHIP (for innings 6-9)
    home_bullpen:      dict = None, # home team's actual bullpen ERA/WHIP (for innings 6-9)
    venue:             str  = None, # ballpark name for park factor adjustment
    away_pitcher_log:  list = None, # away starter's last 10 game logs (ip, er per start)
    home_pitcher_log:  list = None, # home starter's last 10 game logs
    away_rest_days:    int  = None, # days since away starter's last start
    home_rest_days:    int  = None, # days since home starter's last start
    away_risp_stats:    list = None, # away batters' RISP stats (runners on 2nd/3rd)
    home_risp_stats:    list = None, # home batters' RISP stats
    away_matchup_stats:  list = None, # away batters' career stats vs home starter
    home_matchup_stats:  list = None, # home batters' career stats vs away starter
    away_daynight_stats: list = None, # away batters' day or night game splits
    home_daynight_stats: list = None, # home batters' day or night game splits
    away_batter_rest:    int  = None,
    home_batter_rest:    int  = None,
    away_savant:         list = None,  # Statcast data per away batter
    home_savant:         list = None,  # Statcast data per home batter
    umpire_name:         str  = None,  # home plate umpire for K/BB modifier
    n:                   int = 100_000,
) -> dict:
    """
    Run N Monte Carlo simulations and return aggregated results.

    Factors in:
      - L/R pitcher splits (via away_lineup / home_lineup)
      - Recent form: 60% season stats + 40% last-20-game stats
      - Pitcher fatigue: innings 3-5 use degraded pitcher ERA/WHIP based
        on how long this specific pitcher has lasted in recent starts
      - Weather: wind and temperature affect HR probability
      - SP vs RP: innings 1-5 use starting pitcher probs,
                  innings 6-9 switch to real team bullpen probs
    """
    # League-average pitcher stats used as the "generic bullpen" modifier
    LEAGUE_AVG_PITCHER = {"era": "4.20", "whip": "1.30"}

    # ── Apply rest day modifier to starter stats ─────────────────────
    # Short rest = arm not recovered = worse ERA/WHIP across all innings.
    # This is applied BEFORE fatigue so both stack (tired + short-rest = double hit).
    away_pitcher = apply_rest_modifier(away_pitcher, away_rest_days)
    home_pitcher = apply_rest_modifier(home_pitcher, home_rest_days)

    # ── Build pitcher fatigue curves from last 10 starts ─────────────
    # away_fatigue affects home batters (they're facing the away starter)
    # home_fatigue affects away batters (they're facing the home starter)
    away_fatigue = build_fatigue_curve(away_pitcher_log)
    home_fatigue = build_fatigue_curve(home_pitcher_log)

    # ── Phase 1: innings 1-2 (starter is fresh) ───────────────────────
    away_precomp_early = precompute_lineup(
        away_lineup, home_pitcher, weather, away_recent, venue,
        away_matchup_stats, away_daynight_stats, away_batter_rest, away_savant, umpire_name)
    home_precomp_early = precompute_lineup(
        home_lineup, away_pitcher, weather, home_recent, venue,
        home_matchup_stats, home_daynight_stats, home_batter_rest, home_savant, umpire_name)

    # ── Phase 2: innings 3-5 (starter showing fatigue) ───────────────
    home_pitcher_mid = build_fatigued_pitcher_stats(home_pitcher, home_fatigue["phase2"])
    away_pitcher_mid = build_fatigued_pitcher_stats(away_pitcher, away_fatigue["phase2"])
    away_precomp_mid = precompute_lineup(
        away_lineup, home_pitcher_mid, weather, away_recent, venue,
        away_matchup_stats, away_daynight_stats, away_batter_rest, away_savant, umpire_name)
    home_precomp_mid = precompute_lineup(
        home_lineup, away_pitcher_mid, weather, home_recent, venue,
        home_matchup_stats, home_daynight_stats, home_batter_rest, home_savant, umpire_name)

    # ── RISP: clutch stats when runner on 2nd or 3rd ─────────────────
    away_precomp_risp = None
    home_precomp_risp = None

    if away_risp_stats and any(s for s in away_risp_stats if s and s.get("plateAppearances", 0) >= 20):
        away_precomp_risp = precompute_lineup(
            away_risp_stats, home_pitcher, weather, away_recent, venue,
            away_matchup_stats, away_daynight_stats, away_batter_rest, away_savant, umpire_name
        )
    if home_risp_stats and any(s for s in home_risp_stats if s and s.get("plateAppearances", 0) >= 20):
        home_precomp_risp = precompute_lineup(
            home_risp_stats, away_pitcher, weather, home_recent, venue,
            home_matchup_stats, home_daynight_stats, home_batter_rest, home_savant, umpire_name
        )

    # ── Late game: batters vs the bullpen (innings 6+) ───────────────
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

    # ── Compute per-batter situational rates (GIDP, sac fly, stolen base) ──────
    # Derived from season stats already in memory — no extra API calls needed.
    # away_batter_rates describes away BATTERS (how they run, hit with runners on)
    # home_batter_rates describes home BATTERS
    away_batter_rates = compute_batter_rates(away_lineup)
    home_batter_rates = compute_batter_rates(home_lineup)

    # ── Run the simulation loop ───────────────────────────────────────
    away_wins  = 0
    home_wins  = 0
    away_total = 0
    home_total = 0
    away_cover = 0      # wins by 2+ (covers -1.5 run line)
    home_cover = 0
    total_runs_hist = Counter()  # {total_runs: count} for O/U model

    def _sim(**kw):
        return simulate_game(
            away_precomp_early, home_precomp_early,
            away_precomp_mid,   home_precomp_mid,
            away_precomp_late,  home_precomp_late,
            away_precomp_risp,  home_precomp_risp,
            away_batter_rates,  home_batter_rates,
            **kw
        )

    for _ in range(n):
        a, h = _sim()
        away_total += a
        home_total += h
        total_runs_hist[a + h] += 1
        if a > h:
            away_wins += 1
            if a - h >= 2:
                away_cover += 1
        else:
            home_wins += 1
            if h - a >= 2:
                home_cover += 1

    # ── F3 / F5 / F7 inning-segment simulations ──────────────────────
    # Simulate a smaller batch to get win probs through each inning cutoff.
    # Ties count as a push (neither team wins the bet), matching how F5
    # betting markets work.
    n_seg = max(n // 4, 10_000)
    seg_results = {}
    for label, inn in (("f3", 3), ("f5", 5), ("f7", 7)):
        aw = hw = at = ht = ties = 0
        for _ in range(n_seg):
            a, h = _sim(innings=inn, resolve_ties=False)
            at += a; ht += h
            if a > h:
                aw += 1
            elif h > a:
                hw += 1
            else:
                ties += 1
        # Express win% among non-tied games (matches push-excluded betting market)
        non_tie = n_seg - ties
        seg_results[label] = {
            "away_win_pct": round(aw / non_tie * 100, 1) if non_tie else 50.0,
            "home_win_pct": round(hw / non_tie * 100, 1) if non_tie else 50.0,
            "tie_pct":      round(ties / n_seg * 100, 1),
            "away_avg_runs": round(at / n_seg, 2),
            "home_avg_runs": round(ht / n_seg, 2),
        }

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

    # ── Summarize fatigue profile for display in the pitcher card ────────
    def _fatigue_label(log, fatigue):
        if not log:
            return "Typical", None
        avg_ip = sum(g["ip"] for g in log) / len(log)
        p2     = fatigue["phase2"]
        if avg_ip >= 6.5 and p2 <= 1.04:
            label = "Durable"
        elif avg_ip >= 5.0 and p2 <= 1.10:
            label = "Moderate"
        elif avg_ip >= 3.5:
            label = "Tires in middle innings"
        else:
            label = "Short starts — early danger"
        return label, round(avg_ip, 1)

    away_fatigue_label, away_avg_ip = _fatigue_label(away_pitcher_log, away_fatigue)
    home_fatigue_label, home_avg_ip = _fatigue_label(home_pitcher_log, home_fatigue)

    # Rest day label for display
    def _rest_label(days):
        if days is None:
            return None, None
        if days <= 3:
            return days, "short"
        elif days <= 6:
            return days, "ideal"
        elif days <= 9:
            return days, "rusty"
        else:
            return days, "extended"

    away_rest_label, away_rest_type = _rest_label(away_rest_days)
    home_rest_label, home_rest_type = _rest_label(home_rest_days)

    return {
        "away_team":           away_team,
        "home_team":           home_team,
        "simulations":         n,
        "away_win_pct":        adj_away,
        "home_win_pct":        adj_home,
        "away_win_pct_raw":    round(raw_away_pct, 1),  # pre-HFA, for display
        "home_win_pct_raw":    round(raw_home_pct, 1),
        "away_avg_runs":       round(away_total / n, 2),
        "home_avg_runs":       round(home_total / n, 2),
        "uses_bullpen_data":   away_precomp_late is not None,
        # Pitcher fatigue profile
        "away_fatigue_label":  away_fatigue_label,
        "away_avg_ip":         away_avg_ip,
        "home_fatigue_label":  home_fatigue_label,
        "home_avg_ip":         home_avg_ip,
        # Pitcher rest days
        "away_rest_days":      away_rest_label,
        "away_rest_type":      away_rest_type,
        "home_rest_days":      home_rest_label,
        "home_rest_type":      home_rest_type,
        # Run line coverage
        "away_cover_pct":      round(away_cover / n * 100, 1),
        "home_cover_pct":      round(home_cover / n * 100, 1),
        # Over/Under: full distribution so web layer can calc P(total > any line)
        "total_runs_hist":     dict(total_runs_hist),
        "avg_total_runs":      round((away_total + home_total) / n, 2),
        # Inning-segment results (F3/F5/F7)
        "f3":                  seg_results["f3"],
        "f5":                  seg_results["f5"],
        "f7":                  seg_results["f7"],
    }
