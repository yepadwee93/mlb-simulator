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

import math as _math
import random
from bisect import bisect
from collections import Counter
from itertools import accumulate

# ── Outcome labels ──────────────────────────────────────────────
SINGLE = "single"
DOUBLE = "double"
TRIPLE = "triple"
HR = "home_run"
WALK = "walk"
STRIKEOUT = "strikeout"
OUT = "out"

# Order matters — this is the fixed order we use for probability arrays
OUTCOME_ORDER = [WALK, STRIKEOUT, SINGLE, DOUBLE, TRIPLE, HR, OUT]

# League average benchmarks (2025 MLB season approximations)
LEAGUE_AVG_ERA = 4.00
LEAGUE_AVG_WHIP = 1.28
LEAGUE_AVG_FIP = 4.00  # FIP constant ~3.10 bakes in; league FIP ≈ ERA by construction
LEAGUE_AVG_K_PCT = 22.5  # % of PA ending in strikeout
LEAGUE_AVG_BB_PCT = 8.5  # % of PA ending in walk
LEAGUE_AVG_HR9 = 1.30  # home runs per 9 innings

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
    "Coors Field": {"hr": 1.30, "hit": 1.15, "run": 1.20},  # altitude, thin air
    "Great American Ball Park": {"hr": 1.18, "hit": 1.05, "run": 1.10},  # small park, Ohio wind
    "Globe Life Field": {"hr": 1.12, "hit": 1.04, "run": 1.06},  # hitter-friendly indoors
    "Minute Maid Park": {"hr": 1.10, "hit": 1.03, "run": 1.05},  # short left field
    "Yankee Stadium": {"hr": 1.15, "hit": 1.02, "run": 1.07},  # short porch in right
    "Citizens Bank Park": {"hr": 1.12, "hit": 1.04, "run": 1.07},  # Philly wind, small gaps
    "American Family Field": {"hr": 1.08, "hit": 1.03, "run": 1.05},
    "Truist Park": {"hr": 1.06, "hit": 1.02, "run": 1.04},
    "Kauffman Stadium": {"hr": 1.05, "hit": 1.02, "run": 1.03},
    "Angel Stadium": {"hr": 1.04, "hit": 1.01, "run": 1.02},
    "Daikin Park": {"hr": 1.04, "hit": 1.01, "run": 1.03},
    "Oriole Park at Camden Yards": {"hr": 1.08, "hit": 1.03, "run": 1.05},
    # ── Neutral parks ─────────────────────────────────────────────
    "Dodger Stadium": {"hr": 1.00, "hit": 1.00, "run": 1.00},
    "Wrigley Field": {"hr": 1.02, "hit": 1.01, "run": 1.01},  # varies with wind
    "Target Field": {"hr": 0.98, "hit": 1.00, "run": 0.99},
    "Busch Stadium": {"hr": 0.97, "hit": 0.99, "run": 0.98},
    "Progressive Field": {"hr": 0.98, "hit": 1.00, "run": 0.99},
    "Comerica Park": {"hr": 0.96, "hit": 1.00, "run": 0.98},
    "Rogers Centre": {"hr": 1.01, "hit": 1.00, "run": 1.00},
    "Nationals Park": {"hr": 0.99, "hit": 1.00, "run": 0.99},
    "Chase Field": {"hr": 1.01, "hit": 1.01, "run": 1.01},  # retractable roof
    "loanDepot park": {"hr": 0.97, "hit": 0.99, "run": 0.98},  # pitcher-friendly
    # ── Pitcher-friendly parks ─────────────────────────────────────
    "Oracle Park": {"hr": 0.82, "hit": 0.96, "run": 0.88},  # sea breeze kills HRs
    "T-Mobile Park": {"hr": 0.88, "hit": 0.97, "run": 0.92},  # marine layer
    "Petco Park": {"hr": 0.86, "hit": 0.97, "run": 0.90},  # large dimensions
    "PNC Park": {"hr": 0.90, "hit": 0.98, "run": 0.93},
    "Fenway Park": {"hr": 0.92, "hit": 1.02, "run": 0.97},  # Green Monster = hits not HRs
    "Citi Field": {"hr": 0.88, "hit": 0.97, "run": 0.91},
    "Tropicana Field": {"hr": 0.90, "hit": 0.97, "run": 0.92},
    "Guaranteed Rate Field": {"hr": 0.93, "hit": 0.98, "run": 0.95},
    "Sutter Health Park": {"hr": 0.95, "hit": 0.99, "run": 0.97},
    "Sahlen Field": {"hr": 0.95, "hit": 0.99, "run": 0.97},
}

# Default for any park not in the list (league average)
DEFAULT_PARK_FACTOR = {"hr": 1.0, "hit": 1.0, "run": 1.0}

# Monthly modifiers — some parks play very differently by season
# Factor multiplied onto the park's HR factor for that month
# Cold weather = fewer HRs (ball doesn't carry); hot summer = more HRs
MONTHLY_HR_MODIFIER = {
    # month: {venue: multiplier}  — only override parks that vary significantly
    4: {
        "Coors Field": 0.88,
        "Wrigley Field": 0.85,
        "Target Field": 0.82,
        "Kauffman Stadium": 0.90,
        "Progressive Field": 0.88,
        "Guaranteed Rate Field": 0.87,
    },
    5: {
        "Coors Field": 0.93,
        "Wrigley Field": 0.92,
        "Target Field": 0.90,
        "Kauffman Stadium": 0.94,
        "Progressive Field": 0.93,
    },
    6: {},  # approaching average — no overrides
    7: {
        "Coors Field": 1.08,
        "Great American Ball Park": 1.06,
        "Globe Life Field": 0.94,  # AC keeps it cool
        "Chase Field": 0.93,
    },  # retractable roof vs heat
    8: {
        "Coors Field": 1.10,
        "Great American Ball Park": 1.07,
        "Citizens Bank Park": 1.05,
        "Yankee Stadium": 1.04,
    },
    9: {"Coors Field": 1.05, "Wrigley Field": 0.94, "T-Mobile Park": 0.92},
    10: {"Wrigley Field": 0.88, "Target Field": 0.85, "Progressive Field": 0.87},
}


def apply_ballpark_factor(probs: list, venue: str, month: int = None) -> list:
    """
    Adjust batter outcome probabilities based on the ballpark.

    Coors Field: thin air at 5,280 ft — balls carry further → +30% HR, +15% hits
    Oracle Park: ocean breeze blows balls back in → -18% HR, -4% hits
    Fenway Park: Green Monster turns HRs into doubles → lower HR, higher hits
    etc.

    OUTCOME_ORDER = [WALK, K, 1B, 2B, 3B, HR, OUT]
                       0   1   2   3   4   5   6
    """
    base_factors = BALLPARK_FACTORS.get(venue, DEFAULT_PARK_FACTOR)
    # Apply monthly modifier to HR factor if available
    import copy

    factors = copy.copy(base_factors)
    if month:
        mo_mod = MONTHLY_HR_MODIFIER.get(month, {}).get(venue)
        if mo_mod is not None:
            factors = dict(factors)
            factors["hr"] = round(factors["hr"] * mo_mod, 3)

    adjusted = list(probs)
    adjusted[2] *= factors["hit"]  # singles
    adjusted[3] *= factors["hit"]  # doubles
    adjusted[4] *= factors["hit"]  # triples
    adjusted[5] *= factors["hr"]  # home runs

    # Normalize so probabilities still sum to 1.0
    total = sum(adjusted)
    return [p / total for p in adjusted]


# ── STEP 1: Build batter probability distribution ───────────────


# League-average outcome rates (2023-2024 MLB, per plate appearance)
# OUTCOME_ORDER = [WALK, K, 1B, 2B, 3B, HR, OUT]
LEAGUE_AVG_PROBS = [0.082, 0.224, 0.148, 0.048, 0.004, 0.033, 0.461]

# wOBA linear weights (2024 MLB run values per outcome, per Tango/FanGraphs)
# These reflect how many runs each outcome is worth on average.
# OUTCOME_ORDER = [WALK,  K,     1B,    2B,    3B,    HR,    OUT]
WOBA_WEIGHTS = [0.690, 0.000, 0.888, 1.271, 1.616, 2.101, 0.000]

# League-average wOBA (2024 MLB)
LEAGUE_AVG_WOBA = 0.315

# Realistic wOBA bounds: below/above these → compounding factors have
# pushed probs to an implausible extreme; clamp back toward league avg.
WOBA_MIN = 0.200  # ~replacement level hitter
WOBA_MAX = 0.450  # ~peak Barry Bonds / outlier season


def compute_implied_woba(probs: list) -> float:
    """Compute the implied wOBA from a probability vector (OUTCOME_ORDER)."""
    return sum(p * w for p, w in zip(probs, WOBA_WEIGHTS))


def clamp_woba(probs: list) -> list:
    """
    Soft-clamp the probability vector so its implied wOBA stays within
    realistic MLB bounds [WOBA_MIN, WOBA_MAX].

    Why this matters: when multiple adjustments stack (hot streak + favorable
    park + weak pitcher + strong home split), the compounded probs can push
    a batter's implied wOBA above 0.450 — a level that has never been
    sustained over a full season by anyone in modern MLB. Similarly, a cold
    batter facing a great pitcher in a pitcher's park might drop below 0.200.

    The fix: if the wOBA is outside the realistic range, scale all offensive
    outcome probs (walk, 1B, 2B, 3B, HR) toward the boundary. K and OUT
    absorb the difference to keep the vector summing to 1.
    """
    woba = compute_implied_woba(probs)
    if WOBA_MIN <= woba <= WOBA_MAX:
        return probs  # already in realistic range — no change

    target = max(WOBA_MIN, min(WOBA_MAX, woba))
    if woba < 0.001:
        return probs  # degenerate vector, skip

    scale = target / woba
    new_probs = list(probs)
    # Scale offensive outcomes (indices 0,2,3,4,5 = walk,1B,2B,3B,HR)
    for i in (0, 2, 3, 4, 5):
        new_probs[i] = probs[i] * scale
    # OUT absorbs the remainder to keep sum = 1
    new_probs[6] = max(0.0, 1.0 - sum(new_probs[i] for i in range(6)))
    return new_probs


# Stabilization points (PA needed before a stat is 50% reliable).
# Source: Tango/MGL "The Book" + Pizza Cutter stabilization research.
# Lower = stat stabilizes faster (K rate is quick; HR rate takes all season).
# Formula: shrunk = (observed_count + stab * league_avg) / (pa + stab)
# At pa=0: pure league avg. At pa=stab: 50/50 blend. At pa>>stab: trust observed.
_STAB = {
    "walk": 120,  # BB% stabilizes around 120 PA
    "k": 60,  # K%  stabilizes fastest — pitchers control this
    "1b": 460,  # Singles (BABIP-driven) — very slow to stabilize
    "2b": 290,  # Doubles — moderate
    "3b": 1200,  # Triples — almost never stabilize (speed/park dependent)
    "hr": 170,  # HR rate — mid-range
}


def _shrink(observed_count: float, pa: int, league_avg_rate: float, stab: int) -> float:
    """
    Bayesian shrinkage toward league average.

    Blends the observed rate with the league average rate, weighted by
    how much data we have vs how much data we'd need to fully trust it.

      shrunk = (observed_count + stab * league_avg) / (pa + stab)

    At pa=0:    returns league_avg (no data, use prior entirely)
    At pa=stab: returns midpoint between observed and league avg (50/50)
    At pa=500 with stab=120: returns ~80% observed, 20% league avg
    """
    return (observed_count + stab * league_avg_rate) / (pa + stab)


def build_batter_probs(stats: dict) -> list:
    """
    Convert a batter's season stats into a list of probabilities,
    one per outcome, in OUTCOME_ORDER.

    Applies Bayesian shrinkage (Empirical Bayes) toward league average
    for each outcome, weighted by how many PA the player has vs the
    stabilization point for that stat.

    Why this matters:
      A batter hitting .380 in 50 PA in April is almost certainly not
      a .380 true-talent hitter — it's a hot start with luck involved.
      A batter hitting .380 in 500 PA is almost certainly the real deal.
      This function knows the difference; the old version did not.

    Example: batter with 80 PA, 5 HR (6.3% HR rate vs 3.3% league avg)
      Old: HR prob = 6.3% (takes hot start at face value)
      New: HR prob = (5 + 170*0.033) / (80 + 170) = 10.6/250 = 4.2%
           (pulls toward league avg because 80 PA is well below the 170 PA
            needed for HR rate to stabilize — likely some luck involved)
    """
    pa = stats.get("plateAppearances", 0)

    if pa < 10:
        return list(LEAGUE_AVG_PROBS)

    hits = stats.get("hits", 0)
    bb = stats.get("baseOnBalls", 0)
    k = stats.get("strikeOuts", 0)
    hr = stats.get("homeRuns", 0)
    doubles = stats.get("doubles", 0)
    triples = stats.get("triples", 0)
    singles = max(0, hits - doubles - triples - hr)

    # Apply shrinkage per outcome using its individual stabilization point
    p_walk = _shrink(bb, pa, LEAGUE_AVG_PROBS[0], _STAB["walk"])
    p_k = _shrink(k, pa, LEAGUE_AVG_PROBS[1], _STAB["k"])
    p_1b = _shrink(singles, pa, LEAGUE_AVG_PROBS[2], _STAB["1b"])
    p_2b = _shrink(doubles, pa, LEAGUE_AVG_PROBS[3], _STAB["2b"])
    p_3b = _shrink(triples, pa, LEAGUE_AVG_PROBS[4], _STAB["3b"])
    p_hr = _shrink(hr, pa, LEAGUE_AVG_PROBS[5], _STAB["hr"])

    p_out = max(0.0, 1.0 - p_walk - p_k - p_1b - p_2b - p_3b - p_hr)

    # ── Batted ball profile adjustment ───────────────────────────────────────
    # Ground ball hitters: more singles (weak grounders fall in), fewer HRs,
    #   but more GIDP exposure. FB hitters: more HRs, fewer singles.
    # We use groundOuts/flyOuts from the MLB API (already in season stats).
    # League average GB% ≈ 44%. Each 5pp above/below avg shifts outcomes slightly.
    go = int(stats.get("groundOuts", 0) or 0)
    fo = int(stats.get("flyOuts", 0) or 0)
    if go + fo >= 50:  # need enough sample to trust the profile
        gb_pct = go / (go + fo) * 100
        gb_delta = (gb_pct - 44.0) / 5.0  # units of 5pp above/below league avg
        # Ground ball bias: boosts singles slightly, suppresses HRs
        # Fly ball bias: boosts HRs, suppresses singles
        gb_1b_boost = gb_delta * 0.005  # +0.5% singles per 5pp GB above avg
        gb_hr_suppress = gb_delta * 0.003  # -0.3% HR per 5pp GB above avg
        p_1b = max(0.0, p_1b + gb_1b_boost)
        p_hr = max(0.0, p_hr - gb_hr_suppress)
        p_out = max(0.0, 1.0 - p_walk - p_k - p_1b - p_2b - p_3b - p_hr)

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
        return probs  # no weather data, no change

    adjusted = list(probs)

    # Temperature effect on HR (index 5 = HR in OUTCOME_ORDER)
    temp = weather.get("temp_f", 72)
    if temp < 50:
        hr_temp_factor = 0.85  # cold air: 15% fewer HRs
    elif temp > 85:
        hr_temp_factor = 1.10  # hot air: 10% more HRs
    else:
        hr_temp_factor = 1.0

    # Wind effect on HR — uses park-specific CF direction from weather API
    wind_mph = weather.get("wind_mph", 0)
    # If weather API already computed directional boost, use it; else fall back to generic
    if "wind_hr_boost" in weather:
        hr_wind_factor = weather["wind_hr_boost"]
    else:
        wind_deg = weather.get("wind_deg", 0)
        if wind_mph > 10:
            if 45 <= wind_deg <= 135:
                hr_wind_factor = max(0.80, 1.0 - (wind_mph - 10) * 0.015)
            elif 225 <= wind_deg <= 315:
                hr_wind_factor = min(1.30, 1.0 + (wind_mph - 10) * 0.015)
            else:
                hr_wind_factor = 1.0
        else:
            hr_wind_factor = 1.0

    # Barometric pressure effect on HR.
    # Lower pressure = less air resistance = ball carries further.
    # Sea level standard = 1013.25 hPa. Coors Field sits at ~843 hPa.
    # Research shows ~1% HR increase per 17 hPa below sea level standard.
    pressure_hpa = weather.get("pressure_hpa")
    if pressure_hpa and pressure_hpa > 0:
        pressure_delta = 1013.25 - pressure_hpa  # positive = below sea level
        hr_pressure_factor = 1.0 + (pressure_delta / 17.0) * 0.01
        hr_pressure_factor = min(1.25, max(0.90, hr_pressure_factor))
    else:
        hr_pressure_factor = 1.0

    # Humidity effect on HR.
    # Humid air is less dense than dry air (water vapor displaces heavier N2/O2).
    # High humidity (>70%) adds ~2-3% to HR carry; dry air (<30%) suppresses slightly.
    humidity_pct = weather.get("humidity_pct", 50)
    if humidity_pct > 70:
        hr_humidity_factor = 1.0 + (humidity_pct - 70) / 30 * 0.03
    elif humidity_pct < 30:
        hr_humidity_factor = 1.0 - (30 - humidity_pct) / 30 * 0.02
    else:
        hr_humidity_factor = 1.0
    hr_humidity_factor = min(1.05, max(0.97, hr_humidity_factor))

    total_hr_factor = hr_temp_factor * hr_wind_factor * hr_pressure_factor * hr_humidity_factor
    adjusted[5] *= total_hr_factor  # index 5 = HR

    # Normalize
    total = sum(adjusted)
    return [p / total for p in adjusted]


def apply_pitcher_modifier(probs: list, pitcher_stats: dict) -> list:
    """
    Adjust batter probabilities for the quality of the pitcher they're facing.

    Upgraded to use FIP-based quality signal plus per-pitch-type rates:

    OLD approach: ERA/WHIP ratio scales all hits uniformly.
    NEW approach:
      1. FIP (Fielding Independent Pitching) replaces ERA as the quality baseline.
         FIP strips out luck and defense — it only counts strikeouts, walks, and
         home runs allowed. A pitcher with a 4.50 ERA but 3.20 FIP is being
         unlucky; his true skill is closer to the FIP number.
         We blend FIP 70% / ERA 30% so we don't completely ignore recent results.

      2. K% adjusts strikeout probability directly from the pitcher's actual
         strikeout rate, not inferred from ERA. A high-K pitcher (DeGrom, Cole)
         gets more Ks simulated regardless of his ERA.

      3. BB% adjusts walk probability directly from the pitcher's actual walk rate.

      4. HR/9 adjusts home run probability. A fly-ball pitcher with high HR/9
         gives up more HRs even if his ERA looks decent. A groundball pitcher
         with low HR/9 suppresses HRs even on hard contact.

      5. GB% (groundball %) further modulates HR: groundball pitchers convert
         would-be HRs/doubles into groundouts at a higher rate.

    OUTCOME_ORDER = [WALK, K, 1B, 2B, 3B, HR, OUT]
                       0   1   2   3   4   5   6
    """
    # ── 1. FIP-blended quality signal ──────────────────────────────────────
    try:
        era = float(pitcher_stats.get("era") or LEAGUE_AVG_ERA)
    except (ValueError, TypeError):
        era = LEAGUE_AVG_ERA

    try:
        fip = float(pitcher_stats.get("fip") or era)  # fall back to ERA if no FIP
    except (ValueError, TypeError):
        fip = era

    try:
        whip = float(pitcher_stats.get("whip") or LEAGUE_AVG_WHIP)
    except (ValueError, TypeError):
        whip = LEAGUE_AVG_WHIP

    # Blend: 70% FIP, 30% ERA — trust skill over luck, but don't ignore results
    blended_era = 0.70 * fip + 0.30 * era

    # Overall hit scale: blended ERA 2.00 → 0.55 (fewer hits), 4.00 → 1.00, 6.00 → 1.45
    hit_scale = min(max(blended_era / LEAGUE_AVG_FIP, 0.55), 1.45)

    # ── 2. K% — adjust strikeout probability directly ──────────────────────
    try:
        k_pct = float(pitcher_stats.get("k_pct") or LEAGUE_AVG_K_PCT)
    except (ValueError, TypeError):
        k_pct = LEAGUE_AVG_K_PCT
    # Scale relative to league average; cap at ±50%
    k_scale = min(max(k_pct / LEAGUE_AVG_K_PCT, 0.50), 1.50)

    # ── 3. BB% — adjust walk probability directly ──────────────────────────
    try:
        bb_pct = float(pitcher_stats.get("bb_pct") or LEAGUE_AVG_BB_PCT)
    except (ValueError, TypeError):
        bb_pct = LEAGUE_AVG_BB_PCT
    bb_scale = min(max(bb_pct / LEAGUE_AVG_BB_PCT, 0.50), 1.75)

    # ── 4. HR/9 — adjust home run probability ──────────────────────────────
    try:
        hr9 = float(pitcher_stats.get("hr9") or LEAGUE_AVG_HR9)
    except (ValueError, TypeError):
        hr9 = LEAGUE_AVG_HR9
    hr_scale = min(max(hr9 / LEAGUE_AVG_HR9, 0.40), 1.80)

    # ── 5. GB% — groundball pitchers suppress HRs further ─────────────────
    try:
        gb_pct = float(pitcher_stats.get("gb_pct") or 44.0)
    except (ValueError, TypeError):
        gb_pct = 44.0
    # High GB% (52%+) = fewer HRs; Low GB% (35%-) = more HRs
    gb_hr_modifier = min(max(1.0 - (gb_pct - 44.0) * 0.008, 0.75), 1.25)

    # ── 6. Stuff grade composite ───────────────────────────────────────────
    # A pitcher who excels in ALL of K%, BB%, and GB% simultaneously has
    # elite "stuff" that adds a small but real additional suppression beyond
    # what each factor contributes individually.
    #
    # Formula: stuff_score = (k_scale - 1) - (bb_scale - 1) + (gb_hr_modifier - 1)
    #   Positive = above-average stuff; negative = below-average.
    # Effect: caps at ±5% additional hit suppression across singles/doubles.
    stuff_score = (k_scale - 1.0) - (bb_scale - 1.0) * 0.5 + (1.0 - gb_hr_modifier) * 0.5
    stuff_hit_mod = min(1.05, max(0.95, 1.0 - stuff_score * 0.05))

    # ── Apply all adjustments ──────────────────────────────────────────────
    adjusted = list(probs)
    adjusted[0] *= bb_scale  # walk — driven by BB%
    adjusted[1] *= k_scale  # strikeout — driven by K%
    adjusted[2] *= hit_scale * stuff_hit_mod  # single
    adjusted[3] *= hit_scale * stuff_hit_mod  # double
    adjusted[4] *= hit_scale * stuff_hit_mod  # triple
    adjusted[5] *= hit_scale * hr_scale * gb_hr_modifier  # HR — three-factor adjustment

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
        era_factor = 1.18  # Short rest — very taxing on the arm
        whip_factor = 1.11
    elif rest_days == 4:
        era_factor = 1.08  # One day short of optimal
        whip_factor = 1.05
    elif rest_days <= 6:
        era_factor = 1.0  # Ideal rest — no change
        whip_factor = 1.0
    elif rest_days <= 9:
        era_factor = 1.04  # Slightly rusty
        whip_factor = 1.02
    else:
        era_factor = 1.10  # Extended break — possible injury return / rustiness
        whip_factor = 1.06

    def _f(key, default):
        try:
            return float(pitcher_stats.get(key) or default)
        except (ValueError, TypeError):
            return default

    era = _f("era", LEAGUE_AVG_ERA)
    fip = _f("fip", era)
    whip = _f("whip", LEAGUE_AVG_WHIP)
    k_pct = _f("k_pct", LEAGUE_AVG_K_PCT)
    bb_pct = _f("bb_pct", LEAGUE_AVG_BB_PCT)
    hr9 = _f("hr9", LEAGUE_AVG_HR9)

    # Short rest / rust affects all quality signals proportionally
    return {
        **pitcher_stats,
        "era": round(era * era_factor, 2),
        "fip": round(fip * era_factor, 2),
        "whip": round(whip * whip_factor, 2),
        "k_pct": round(k_pct * (2.0 - era_factor), 1),  # K% drops on short rest
        "bb_pct": round(bb_pct * whip_factor, 1),  # BB% rises
        "hr9": round(hr9 * era_factor, 2),  # HR rate rises
    }


def build_fatigue_curve(pitcher_game_log: list) -> dict:
    """
    Analyze a pitcher's last 10 starts to build inning-phase fatigue multipliers.

    Now uses pitch count data (if available) to refine the curve:
      - High-pitch-count pitcher (105+ avg) → tires later
      - Low-pitch-count pitcher (80-90 avg) → short-arm type, tires early
      - If last start had 115+ pitches → carry-over fatigue this start

    Returns a dict with ERA/WHIP scaling factors for 3 phases:
      phase1 (innings 1-2): always 1.0 — pitcher is always fresh early
      phase2 (innings 3-4): moderate fatigue based on how long they typically last
      phase3 (innings 5):   deep fatigue, last inning before bullpen takes over

    Multiplier > 1.0 means pitcher is WORSE (ERA scaled up by that factor).
    """
    if not pitcher_game_log:
        return {
            "phase1": 1.0,
            "phase2": 1.07,
            "phase3": 1.16,
            "avg_pitches": None,
            "last_pitches": None,
            "pitch_carryover": False,
        }

    n = len(pitcher_game_log)
    if n == 0:
        return {
            "phase1": 1.0,
            "phase2": 1.07,
            "phase3": 1.16,
            "avg_pitches": None,
            "last_pitches": None,
            "pitch_carryover": False,
        }

    avg_ip = sum(g["ip"] for g in pitcher_game_log) / n
    total_ip = sum(g["ip"] for g in pitcher_game_log)
    total_er = sum(g["er"] for g in pitcher_game_log)

    # ── Pitch count analysis ──────────────────────────────────────────────
    pitch_counts = [g.get("pitches", 0) for g in pitcher_game_log if g.get("pitches", 0) > 0]
    avg_pitches = round(sum(pitch_counts) / len(pitch_counts), 1) if pitch_counts else None
    last_pitches = pitcher_game_log[-1].get("pitches", 0) if pitcher_game_log else 0

    # Carry-over fatigue: if last start was 115+ pitches, arm not fully recovered
    pitch_carryover = bool(last_pitches and last_pitches >= 115)

    # ER per inning pitched — rough in-game ERA proxy
    er_per_ip = total_er / total_ip if total_ip > 0 else 0.45

    # How often does the pitcher get knocked out before inning 3?
    early_blowup_rate = sum(1 for g in pitcher_game_log if g["ip"] < 3.0) / n

    # ── Phase 2 multiplier (innings 3-4) ──────────────────────────────────
    # Use pitch count to refine the IP-based estimate when available
    if avg_pitches is not None:
        if avg_pitches >= 105:
            phase2 = 1.03  # high-volume arm — tires slowly
        elif avg_pitches >= 95:
            phase2 = 1.07  # normal workload
        elif avg_pitches >= 85:
            phase2 = 1.12  # kept on a short leash — tires faster
        else:
            phase2 = 1.20  # short-arm type, frequently managed out early
    else:
        # Fall back to IP-based estimate
        if avg_ip >= 6.5:
            phase2 = 1.03
        elif avg_ip >= 5.0:
            phase2 = 1.07
        elif avg_ip >= 3.5:
            phase2 = 1.14
        else:
            phase2 = 1.22

    # Amplify if they get knocked out early a lot
    if early_blowup_rate >= 0.30:
        phase2 = round(phase2 * 1.07, 3)

    # Carry-over penalty: arm slightly fatigued from big outing last start
    if pitch_carryover:
        phase2 = round(phase2 * 1.04, 3)

    # ── Phase 3 multiplier (inning 5+) ────────────────────────────────────
    if avg_ip >= 6.5 or (avg_pitches and avg_pitches >= 105):
        phase3 = phase2 * 1.05
    elif avg_ip >= 5.0 or (avg_pitches and avg_pitches >= 95):
        phase3 = phase2 * 1.11
    elif avg_ip >= 3.5 or (avg_pitches and avg_pitches >= 85):
        phase3 = phase2 * 1.18
    else:
        phase3 = phase2 * 1.25

    # Quality modifier
    if er_per_ip > 0.60:
        phase2 *= 1.04
        phase3 *= 1.07
    elif er_per_ip < 0.25:
        phase2 *= 0.93
        phase3 *= 0.90

    return {
        "phase1": 1.0,
        "phase2": round(min(phase2, 1.50), 3),
        "phase3": round(min(phase3, 1.90), 3),
        "avg_pitches": avg_pitches,
        "last_pitches": last_pitches if last_pitches else None,
        "pitch_carryover": pitch_carryover,
    }


def build_fatigued_pitcher_stats(base_stats: dict, fatigue_factor: float) -> dict:
    """
    Scale a pitcher's stats by the fatigue factor as innings accumulate.

    fatigue_factor = 1.0  → no change (innings 1-2, pitcher is fresh)
    fatigue_factor = 1.12 → 12% worse (innings 3-4)
    fatigue_factor = 1.28 → 28% worse (inning 5+ for a struggling pitcher)

    Degradation rates by stat (research-backed: K rate drops most as pitch
    count rises; walk rate and HR rate rise; hit rate rises modestly):
      ERA/FIP: full fatigue factor (run prevention degrades fastest)
      WHIP:    75% of factor (walks/hits degrade somewhat slower)
      K%:      K rate falls as pitch count rises — pitchers lose velo/movement
      BB%:     walk rate rises as command fades with fatigue
      HR/9:    pitchers leave more pitches up in zone when tired
      GB%:     groundball rate stays fairly stable (mechanics don't change much)
    """
    fade = fatigue_factor - 1.0  # how much worse, as a fraction (e.g. 0.12 for 12%)

    def _f(key, default):
        try:
            return float(base_stats.get(key) or default)
        except (ValueError, TypeError):
            return default

    era = _f("era", LEAGUE_AVG_ERA)
    fip = _f("fip", era)
    whip = _f("whip", LEAGUE_AVG_WHIP)
    k_pct = _f("k_pct", LEAGUE_AVG_K_PCT)
    bb_pct = _f("bb_pct", LEAGUE_AVG_BB_PCT)
    hr9 = _f("hr9", LEAGUE_AVG_HR9)
    gb_pct = _f("gb_pct", 44.0)

    return {
        **base_stats,
        "era": round(era * fatigue_factor, 2),
        "fip": round(fip * fatigue_factor, 2),
        "whip": round(whip * (1 + fade * 0.75), 2),
        "k_pct": round(k_pct * (1 - fade * 0.50), 1),  # K rate drops with fatigue
        "bb_pct": round(bb_pct * (1 + fade * 0.80), 1),  # walk rate rises
        "hr9": round(hr9 * (1 + fade * 0.60), 2),  # more HRs allowed when tired
        "gb_pct": gb_pct,  # groundball rate is stable
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
        hit_factor = 0.96  # back-to-back: 4% hit reduction
    elif rest_days == 3:
        hit_factor = 1.01  # one extra day off
    else:
        hit_factor = 1.02  # 4+ days rest: fresh

    for i in (2, 3, 4, 5):  # 1B, 2B, 3B, HR
        adjusted[i] *= hit_factor

    total = sum(adjusted)
    return [p / total for p in adjusted]


# Statcast league averages (2024 MLB)
LEAGUE_AVG_BARREL = 7.0  # 7% barrel rate
LEAGUE_AVG_HARD_HIT = 38.0  # 38% hard-hit rate (95+ mph)


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

    barrel_factor = (barrel_pct / LEAGUE_AVG_BARREL) ** 0.35
    hard_hit_factor = (hard_hit_pct / LEAGUE_AVG_HARD_HIT) ** 0.25
    combined = barrel_factor * 0.65 + hard_hit_factor * 0.35

    adjusted = list(probs)
    adjusted[5] *= combined  # HR: full effect
    adjusted[3] *= 1 + (combined - 1) * 0.40  # 2B: 40% of effect

    total = sum(adjusted)
    return [p / total for p in adjusted]


def blend_probs(season_probs: list, recent_probs: list, recent_weight: float = 0.4) -> list:
    """
    Blend season-long probabilities with recent form probabilities.

    recent_weight=0.4 means:
      40% of the prediction comes from recent games
      60% comes from the full season

    This way a player on a hot streak gets a meaningful boost,
    but one good week doesn't completely override who they are
    as a hitter over 150+ games.
    """
    season_weight = 1.0 - recent_weight
    blended = [season_weight * s + recent_weight * r for s, r in zip(season_probs, recent_probs)]
    total = sum(blended)
    return [p / total for p in blended]


def adaptive_recent_weight(
    season_probs: list, recent_probs: list, base_weight: float = 0.35
) -> float:
    """
    Scale the recent-form blend weight based on how much the recent stats
    actually deviate from the season average.

    If a player is performing very close to their season norm recently,
    there's little signal in the recent data — use a lower weight.
    If they're on a big hot or cold streak (large deviation), trust the
    recent data more because something real is happening.

    Deviation is measured as the sum of absolute differences across all
    outcome probabilities (hit rate, HR rate, K rate, etc.).

    Examples:
      - Player hitting near his season average recently → weight stays ~0.25
      - Player 2-for-30 cold streak (K% way up, hit% way down) → weight → 0.45
      - Player 18-for-40 hot streak → weight → 0.45

    Caps: minimum 0.20 (always some recent signal), maximum 0.50
    (season sample of 400+ PA always outweighs ~20-game window).
    """
    total_deviation = sum(abs(r - s) for r, s in zip(recent_probs, season_probs))
    # total_deviation ranges roughly 0.0 (identical) to ~0.30+ (massive streak)
    # Scale: each 0.05 of deviation adds ~0.05 to the weight
    deviation_boost = min(total_deviation * 1.0, 0.15)
    weight = base_weight + deviation_boost
    return min(max(weight, 0.20), 0.50)


def precompute_lineup(
    lineup_stats: list,
    pitcher_stats: dict,
    weather: dict = None,
    recent_stats_list: list = None,
    venue: str = None,
    matchup_stats_list: list = None,
    daynight_stats_list: list = None,
    batter_rest_days: int = None,
    savant_list: list = None,
    umpire_name: str = None,
    homeaway_stats_list: list = None,
    opp_catcher_cs: float = None,
    bat_sides: list = None,
    pitch_hand: str = None,
    opp_team_babip: float = None,
    opp_team_oaa: float = None,
) -> list:
    """
    Pre-calculate cumulative probability arrays for every batter in a lineup.
    Doing this ONCE before the simulation loop (instead of inside it) is the
    key performance optimization — we avoid repeating the same math 100,000 times.

    Applies (in order):
      1. Recent form blend (season 60% + last-20-games 40%)
      2. Day/night split blend (25% weight if batter has 50+ PA in this game type)
      3. Home/away split blend (25% weight if batter has 80+ PA in this context)
      4. Matchup history blend (30% weight if batter has 20+ career PA vs this pitcher)
      5. Pitcher modifier (ERA/WHIP)
      6. Ballpark factor (Coors, Oracle Park, etc.)
      7. Weather modifier (wind, temperature)
    """
    result = []
    for i, stats in enumerate(lineup_stats):
        # Start with season stats — use pre-blended split probs if already computed
        if stats.get("_split_blended") and stats.get("_blended_probs"):
            probs = list(stats["_blended_probs"])
        else:
            probs = build_batter_probs(stats)

        # James-Stein shrinkage: pull noisy small-sample rates toward
        # population mean. Players with 500+ PA barely shrink; 80 PA
        # players get pulled substantially toward league average.
        pa = int(stats.get("plateAppearances", 0) or 0)
        if pa < 400:
            shrunk = build_shrunk_probs(stats)
            # Blend: more shrinkage for fewer PA
            shrink_weight = max(0.10, min(0.60, 1.0 - pa / 500.0))
            probs = [p * (1.0 - shrink_weight) + s * shrink_weight for p, s in zip(probs, shrunk)]
            total = sum(probs)
            probs = [p / total for p in probs]

        # Blend in recent form if available (last 20 games).
        # Use adaptive weighting: bigger hot/cold streaks get more weight.
        # Lower PA threshold to 10 — even a week of data is a signal.
        if recent_stats_list and i < len(recent_stats_list):
            recent = recent_stats_list[i]
            recent_pa = recent.get("plateAppearances", 0) if recent else 0
            if recent and recent_pa >= 10:
                recent_probs = build_batter_probs(recent)
                # Scale weight by sample size: 10 PA → 60% of full weight, 20+ PA → 100%
                pa_scale = min(recent_pa / 20.0, 1.0)
                weight = adaptive_recent_weight(probs, recent_probs, base_weight=0.35) * pa_scale
                probs = blend_probs(probs, recent_probs, recent_weight=weight)

        # Blend in day/night split — some batters are dramatically better in day or night games.
        # Require 50+ PA in this game type to keep the sample meaningful.
        if daynight_stats_list and i < len(daynight_stats_list):
            dn = daynight_stats_list[i]
            if dn and dn.get("plateAppearances", 0) >= 50:
                dn_probs = build_batter_probs(dn)
                probs = blend_probs(probs, dn_probs, recent_weight=0.25)

        # Blend in home/away split — some batters perform very differently at home vs on road.
        # Classic examples: Coors hitters who lose ~60 OPS pts away, or road warriors
        # who see better pitches in unfamiliar parks.
        # Require 80+ PA in this context (home or road) for a stable enough sample.
        # Weight is PA-scaled: 80 PA → 12%, 200+ PA → 25% (max).
        if homeaway_stats_list and i < len(homeaway_stats_list):
            ha = homeaway_stats_list[i]
            ha_pa = ha.get("plateAppearances", 0) if ha else 0
            if ha and ha_pa >= 80:
                ha_probs = build_batter_probs(ha)
                ha_weight = min(0.12 + (ha_pa / 200.0) * 0.13, 0.25)
                probs = blend_probs(probs, ha_probs, recent_weight=ha_weight)

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

        # Enhanced lineup protection: affects BB, HR, and contact quality
        # based on surrounding hitters (on-deck + preceding batter)
        try:
            prot_list = compute_lineup_protection(lineup_stats)
            probs = apply_lineup_protection(probs, prot_list[i])
        except (ValueError, TypeError, ZeroDivisionError, IndexError):
            pass

        # Platoon split modifier (L/R matchup advantage)
        if bat_sides and i < len(bat_sides) and pitch_hand:
            probs = apply_platoon_modifier(probs, bat_sides[i], pitch_hand)

        # Defensive quality modifier (opposing team's fielding)
        if opp_team_babip is not None or opp_team_oaa is not None:
            try:
                gb_pct = float(pitcher_stats.get("gb_pct", "44") or "44")
            except (ValueError, TypeError):
                gb_pct = 44.0
            def_mod = compute_team_defense_mod(opp_team_babip, gb_pct, opp_team_oaa)
            probs = apply_defense_modifier(probs, def_mod)

        # Ballpark factor (dimensions, altitude, park tendencies)
        if venue:
            probs = apply_ballpark_factor(
                probs, venue, month=__import__("datetime").date.today().month
            )

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
                sv.get("barrel_pct", 0),
                sv.get("hard_hit_pct", 0),
            )

        # Umpire tendencies: wide/tight zone shifts K and BB rates
        if umpire_name:
            probs = apply_umpire_modifier(probs, umpire_name)

        # Catcher framing: elite framers expand zone → more Ks, fewer BBs
        if opp_catcher_cs is not None:
            probs = apply_catcher_framing(probs, opp_catcher_cs)

        # Spray angle / defensive positioning BABIP modifier
        sv = savant_list[i] if (savant_list and i < len(savant_list)) else None
        spray_mod = compute_spray_babip_mod(stats, sv)
        if spray_mod != 1.0:
            probs = apply_spray_modifier(probs, spray_mod)

        # Count-based PA model: patient hitters get more favorable counts
        count_prof = compute_count_profile(stats)
        probs = apply_count_model(probs, count_prof)

        # Pitcher arsenal contact profile modifiers
        # Build batter's contact profile and adjust for pitcher archetype
        contact_prof = build_contact_profile(stats, sv)
        contact_prof = apply_arsenal_to_contact(contact_prof, pitcher_stats)
        # Use contact profile to refine HR and XBH probabilities
        barrel_rate = contact_prof.get("barrel", 0.07)
        topped_rate = contact_prof.get("topped", 0.315)
        # Barrel rate above average boosts HR; high topped suppresses HR
        barrel_effect = (barrel_rate / 0.07) ** 0.20
        topped_effect = (0.315 / max(topped_rate, 0.15)) ** 0.10
        combined_contact = barrel_effect * topped_effect
        if abs(combined_contact - 1.0) > 0.005:
            probs[5] *= combined_contact  # HR
            probs[3] *= 1.0 + (combined_contact - 1.0) * 0.2  # 2B partial effect
            total = sum(probs)
            probs = [p / total for p in probs]

        # wOBA sanity clamp: if compounding factors pushed the implied wOBA
        # outside the realistic MLB range [0.200, 0.450], scale it back.
        probs = clamp_woba(probs)

        cum_weights = list(accumulate(probs))
        result.append(cum_weights)
    return result


# ── Batter situational tendency rates ────────────────────────────

# League-average rates used as fallback when a batter lacks enough sample
LEAGUE_GDP_RATE = 0.09  # % of non-K outs with runner on 1st that become DPs
LEAGUE_SF_RATE = 0.04  # % of outs with runner on 3rd, <2 out that score as sac flies
LEAGUE_SB_RATE = 0.04  # % of at-bats with runner on 1st only where steal is attempted
LEAGUE_SB_SUCCESS = 0.72  # % of steal attempts that succeed (league average)


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
        ab = int(stats.get("atBats", 0) or 0)
        gdp = int(stats.get("groundIntoDoublePlay", 0) or 0)
        sf = int(stats.get("sacFlies", 0) or 0)
        sb = int(stats.get("stolenBases", 0) or 0)
        cs = int(stats.get("caughtStealing", 0) or 0)
        hits = int(stats.get("hits", 0) or 0)
        k = int(stats.get("strikeOuts", 0) or 0)
        pa = int(stats.get("plateAppearances", 0) or 0)

        go = int(stats.get("groundOuts", 0) or 0)
        fo = int(stats.get("flyOuts", 0) or 0)

        if ab >= 100:
            # GIDP rate: GDP / (non-K outs), scaled by GB% profile.
            # High-GB% batters hit into more DPs (more balls on the ground).
            # Each 5pp of GB% above 44% adds ~5% to the raw GIDP rate.
            non_k_outs = max(ab - hits - k, 1)
            gdp_rate = gdp / non_k_outs
            if go + fo >= 50:
                gb_pct = go / (go + fo) * 100
                gb_scale = 1.0 + (gb_pct - 44.0) / 5.0 * 0.05
                gdp_rate = gdp_rate * max(0.5, min(1.5, gb_scale))

            # Sac fly rate: SF / (estimated PA with runner on 3rd, <2 out)
            # ~20% of PA come with runner on 3rd and <2 out (rough MLB estimate)
            sf_eligible = max(pa * 0.20, 1)
            sf_rate = sf / sf_eligible

            # SB: attempt rate per PA where runner on 1st only (~30% of PA)
            sb_attempts = sb + cs
            sb_eligible = max(pa * 0.30, 1)
            sb_rate = sb_attempts / sb_eligible
            sb_success = (sb / sb_attempts) if sb_attempts >= 5 else LEAGUE_SB_SUCCESS
        else:
            # Insufficient sample — use league averages
            gdp_rate = LEAGUE_GDP_RATE
            sf_rate = LEAGUE_SF_RATE
            sb_rate = LEAGUE_SB_RATE
            sb_success = LEAGUE_SB_SUCCESS

        rates.append(
            {
                "gdp_rate": round(min(max(gdp_rate, 0.01), 0.25), 3),
                "sf_rate": round(min(max(sf_rate, 0.005), 0.15), 3),
                "sb_rate": round(min(max(sb_rate, 0.005), 0.20), 3),
                "sb_success": round(min(max(sb_success, 0.45), 0.92), 3),
                "speed": classify_runner_speed(stats),
            }
        )

    return rates


# ── STEP 2: Simulate one at-bat ─────────────────────────────────


def simulate_at_bat(cum_weights: list) -> str:
    """
    Draw one random at-bat outcome.
    Uses a random float + binary search (bisect) — faster than random.choices().
    """
    r = random.random()  # random float 0.0–1.0
    i = bisect(cum_weights, r)  # find which bucket it lands in
    i = min(i, len(OUTCOME_ORDER) - 1)  # safety clamp
    return OUTCOME_ORDER[i]


# ── STEP 3: Advance base runners ─────────────────────────────────


def advance_runners(
    b1: bool,
    b2: bool,
    b3: bool,
    outcome: str,
    speed_b1: str = "avg",
    speed_b2: str = "avg",
    speed_b3: str = "avg",
) -> tuple:
    """
    Given who's on base and what just happened, return:
      (new_b1, new_b2, new_b3, runs_scored)

    Uses speed-graded MLB baserunning probabilities. Each runner's speed tier
    (elite/fast/avg/slow/plod) determines their advancement probability on
    singles and doubles. Triples and HRs always score everyone.
    """
    r = random.random  # local alias for speed inside hot loop

    if outcome == WALK:
        if b1 and b2 and b3:
            return True, True, True, 1
        if b1 and b2:
            return True, True, True, 0
        if b1:
            return True, True, b3, 0
        return True, b2, b3, 0

    if outcome in (STRIKEOUT, OUT):
        return b1, b2, b3, 0

    if outcome == SINGLE:
        runs = 0
        new_b1 = True
        new_b2 = False
        new_b3 = False

        st = SPEED_TRANSITIONS["single"]
        if b3:
            if r() < st["from_3rd"].get(speed_b3, 0.87):
                runs += 1
            else:
                new_b3 = True

        if b2:
            if r() < st["from_2nd"].get(speed_b2, 0.62):
                runs += 1
            else:
                new_b3 = True

        if b1:
            if r() < st["from_1st_to_3rd"].get(speed_b1, 0.27):
                new_b3 = True
            else:
                new_b2 = True

        return new_b1, new_b2, new_b3, runs

    if outcome == DOUBLE:
        runs = 0
        new_b1 = False
        new_b2 = True
        new_b3 = False

        st = SPEED_TRANSITIONS["double"]
        if b3:
            if r() < st["from_3rd"].get(speed_b3, 0.95):
                runs += 1
            else:
                new_b3 = True

        if b2:
            if r() < st["from_2nd"].get(speed_b2, 0.87):
                runs += 1
            else:
                new_b3 = True

        if b1:
            if r() < st["from_1st"].get(speed_b1, 0.52):
                runs += 1
            else:
                new_b3 = True

        return new_b1, new_b2, new_b3, runs

    if outcome == TRIPLE:
        runs = int(b1) + int(b2) + int(b3)
        return False, False, True, runs

    if outcome == HR:
        runs = int(b1) + int(b2) + int(b3) + 1
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
    "Laz Diaz": {"k": 1.14, "bb": 0.86},
    "Alfonso Marquez": {"k": 1.10, "bb": 0.91},
    "Hunter Wendelstedt": {"k": 1.08, "bb": 0.92},
    "Jim Reynolds": {"k": 1.08, "bb": 0.93},
    "Mike Winters": {"k": 1.07, "bb": 0.93},
    "Marvin Hudson": {"k": 1.06, "bb": 0.94},
    "Brian Knight": {"k": 1.05, "bb": 0.95},
    "Fieldin Culbreth": {"k": 1.05, "bb": 0.95},
    "Doug Eddings": {"k": 1.04, "bb": 0.96},
    "Scott Barry": {"k": 1.03, "bb": 0.97},
    "Adam Beck": {"k": 1.03, "bb": 0.97},
    "Tripp Gibson": {"k": 1.02, "bb": 0.98},
    "Gabe Morales": {"k": 1.02, "bb": 0.98},
    "Chris Segal": {"k": 1.02, "bb": 0.98},
    "Stu Scheurwater": {"k": 1.02, "bb": 0.98},
    # Neutral / accurate
    "Pat Hoberg": {"k": 1.00, "bb": 1.00},
    "Will Little": {"k": 1.01, "bb": 0.99},
    "Ben May": {"k": 1.01, "bb": 0.99},
    "Ryan Blakney": {"k": 1.01, "bb": 0.99},
    "Todd Tichenor": {"k": 1.01, "bb": 0.99},
    "Roberto Ortiz": {"k": 1.00, "bb": 1.00},
    "James Hoye": {"k": 1.00, "bb": 1.00},
    "Mark Carlson": {"k": 0.99, "bb": 1.01},
    "Erich Bacchus": {"k": 0.99, "bb": 1.01},
    "DJ Reyburn": {"k": 1.01, "bb": 0.99},
    "John Tumpane": {"k": 0.99, "bb": 1.01},
    "Paul Nauert": {"k": 1.00, "bb": 1.00},
    # Tight zone (fewer Ks, more BBs)
    "Andy Fletcher": {"k": 0.98, "bb": 1.02},
    "Mark Wegner": {"k": 0.98, "bb": 1.02},
    "Chad Fairchild": {"k": 0.98, "bb": 1.02},
    "Kerwin Danley": {"k": 0.98, "bb": 1.02},
    "Lance Barksdale": {"k": 0.97, "bb": 1.03},
    "Sam Holbrook": {"k": 0.96, "bb": 1.04},
    "Dan Iassogna": {"k": 0.95, "bb": 1.06},
    "Ted Barrett": {"k": 0.94, "bb": 1.07},
    "Phil Cuzzi": {"k": 0.93, "bb": 1.08},
    "Vic Carapazza": {"k": 0.93, "bb": 1.08},
    "Tom Hallion": {"k": 0.91, "bb": 1.10},
    "Bill Miller": {"k": 0.90, "bb": 1.12},
    "CB Bucknor": {"k": 0.87, "bb": 1.16},
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
    adjusted[0] *= tend["bb"]  # WALK
    adjusted[1] *= tend["k"]  # STRIKEOUT
    total = sum(adjusted)
    return [p / total for p in adjusted]


# Calibrated error and wild pitch rates.
#
# These are NOT the raw MLB rates — they represent the MARGINAL probability of
# an error or wild pitch that adds a run BEYOND what's already captured in
# batter statistics. Batter stats record "out" for error plate appearances
# (no hit credited), which means some error-to-base events are already
# implicitly baked into the out probability. Similarly, GIDP and sac fly
# rates (applied via batter_rates) suppress runs below the raw probability
# distribution, so errors and WPs provide the corrective balance.
#
# Calibration target: 9.10 runs/game (2024 MLB average, both teams combined).
# Verified with 50K simulations using league-average inputs:
#   ERROR_RATE=0.005 + WILD_PITCH_RATE=0.004 → 9.13 runs/game (within noise).
ERROR_RATE = 0.005
WILD_PITCH_RATE = 0.004

# ── Times-through-order (TTO) penalty ────────────────────────────────────────
# Batters see the starter better each time through the order. Research from
# Tango/MGL shows:
#   1st time through (PA 1-9):   baseline — starter has the info advantage
#   2nd time through (PA 10-18): +6% wOBA boost to batters (~batters learning)
#   3rd time through (PA 19+):   +12% wOBA boost — why managers pull at 6 innings
#
# We apply this as a multiplier on cumulative probs (blended toward offense):
#   TTO_FACTORS[n] = factor to blend probs toward league-avg offense (hitter-friendly)
# 0 = no change, 0.06 = 6% blend toward league-average batters
TTO_FACTORS = {1: 0.0, 2: 0.06, 3: 0.12}


def apply_tto(precomp: list, tto: int) -> list:
    """
    Adjust cumulative weight arrays for the times-through-order effect.
    tto=1: no change. tto=2: 6% blend toward league-avg. tto=3: 12%.
    """
    factor = TTO_FACTORS.get(tto, 0.12)
    if factor == 0 or not precomp:
        return precomp
    from itertools import accumulate as _acc

    lg_cum = list(_acc(LEAGUE_AVG_PROBS))
    return [
        [c * (1 - factor) + lg * factor for c, lg in zip(batter_cum, lg_cum)]
        for batter_cum in precomp
    ]


# ── Catcher framing modifier ──────────────────────────────────────────────────
# Elite framers (CS% >> 28%) effectively expand the strike zone, generating
# more called strikes → higher K rate and lower BB rate for their pitchers.
# We model this from CS% as a proxy for overall receiving quality, since
# framing and blocking ability correlate strongly with arm strength.
#
# Effect size: ±0.03 K modifier and ∓0.03 BB modifier per 10pp above/below avg.
# Capped at ±12% to avoid overcorrecting on small samples.
LEAGUE_AVG_CS_PCT = 0.28


def apply_catcher_framing(probs: list, catcher_cs_pct: float) -> list:
    """
    Adjust K and BB probabilities based on the opposing catcher's CS%.
    Elite catchers (high CS%) frame pitches better → more Ks, fewer BBs.

    OUTCOME_ORDER = [WALK, K, 1B, 2B, 3B, HR, OUT]
                       0   1   2   3   4   5   6
    """
    if catcher_cs_pct is None:
        return probs
    delta = catcher_cs_pct - LEAGUE_AVG_CS_PCT  # e.g. +0.08 for elite framer
    k_mod = 1.0 + min(0.12, max(-0.12, delta * 0.30))
    bb_mod = 1.0 + min(0.12, max(-0.12, -delta * 0.30))
    adjusted = list(probs)
    adjusted[0] *= bb_mod  # walk
    adjusted[1] *= k_mod  # strikeout
    total = sum(adjusted)
    return [p / total for p in adjusted]


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED MATH MODULE: Batted Ball Physics, Pitcher Repertoire, Situational
# Hitting, Spray Angle, and Count-Based Plate Appearances
# ══════════════════════════════════════════════════════════════════════════════


# ── 1. BATTED BALL PHYSICS MODEL ─────────────────────────────────────────────
#
# Instead of flat outcome probabilities, this models what happens when bat
# meets ball using launch angle (LA) and exit velocity (EV) distributions.
#
# The key insight: a barrel (LA 25-30°, EV 98+ mph) is a HR ~70% of the time.
# A topped ball (LA < 0°, any EV) is a grounder. The distribution of LA × EV
# a batter produces determines their TRUE power profile better than HR/FB%.
#
# Statcast buckets (2022-2024 MLB averages):
#   Barrel:      LA 26-30°, EV 98+ → .822 xBA, .534 HR rate
#   Solid:       LA 10-25°, EV 95+ → .672 xBA, .086 HR rate
#   Flare/Burner: LA 20-40°, EV <95 → .581 xBA, .014 HR rate
#   Topped:      LA < 10°, any EV   → .234 xBA, .001 HR rate (grounders)
#   Under:       LA > 40°, any EV   → .132 xBA, .023 HR rate (popups/weak fly)
#   Weak:        EV < 85            → .219 xBA, .002 HR rate

# Contact quality distribution (fraction of batted balls in each bucket)
# These are MLB-wide averages; individual batters deviate significantly.
CONTACT_QUALITY_LEAGUE_AVG = {
    "barrel": 0.070,  # 7% of batted balls
    "solid": 0.180,  # 18%
    "flare": 0.215,  # 21.5%
    "topped": 0.315,  # 31.5% (most common — grounders)
    "under": 0.135,  # 13.5% (popups)
    "weak": 0.085,  # 8.5%
}

# Outcome probability given contact type: [1B, 2B, 3B, HR, OUT]
CONTACT_OUTCOMES = {
    "barrel": [0.178, 0.228, 0.016, 0.534, 0.044],
    "solid": [0.375, 0.186, 0.025, 0.086, 0.328],
    "flare": [0.438, 0.112, 0.017, 0.014, 0.419],
    "topped": [0.216, 0.012, 0.005, 0.001, 0.766],
    "under": [0.058, 0.028, 0.023, 0.023, 0.868],
    "weak": [0.182, 0.024, 0.011, 0.002, 0.781],
}


def build_contact_profile(stats: dict, savant: dict = None) -> dict:
    """
    Build a batter's contact quality distribution from Statcast data.

    If Statcast data is available (barrel_pct, hard_hit_pct, gb_pct),
    we reconstruct the full contact profile. Otherwise fall back to
    league average modified by the batter's power indicators.

    Returns a dict with the same keys as CONTACT_QUALITY_LEAGUE_AVG.
    """
    profile = dict(CONTACT_QUALITY_LEAGUE_AVG)

    if savant:
        barrel_pct = savant.get("barrel_pct", 0) or 0
        hard_hit_pct = savant.get("hard_hit_pct", 0) or 0
        gb_pct_sv = savant.get("gb_pct", 0) or 0

        if barrel_pct > 0:
            # Scale barrel rate relative to league avg
            barrel_ratio = barrel_pct / 7.0
            profile["barrel"] = min(0.18, 0.070 * barrel_ratio)

            # Hard hit drives solid contact
            if hard_hit_pct > 0:
                solid_ratio = hard_hit_pct / 38.0
                profile["solid"] = min(0.30, 0.180 * solid_ratio)

            # GB% drives topped ball rate
            if gb_pct_sv > 0:
                gb_ratio = gb_pct_sv / 44.0
                profile["topped"] = min(0.50, 0.315 * gb_ratio)
                # Fly ball hitters trade topped for under/flare
                profile["under"] = max(0.05, 0.135 * (2.0 - gb_ratio) * 0.5 + 0.135 * 0.5)

    elif stats:
        # Infer from traditional stats
        go = int(stats.get("groundOuts", 0) or 0)
        fo = int(stats.get("flyOuts", 0) or 0)
        hr = int(stats.get("homeRuns", 0) or 0)
        pa = int(stats.get("plateAppearances", 0) or 0)

        if pa >= 100 and go + fo >= 50:
            gb_pct = go / (go + fo)
            hr_rate = hr / pa if pa > 0 else 0.033

            # Power hitters have more barrels
            barrel_est = min(0.16, hr_rate * 2.5)
            profile["barrel"] = max(0.02, barrel_est)
            profile["topped"] = min(0.45, 0.315 * (gb_pct / 0.44))

    # Normalize so all buckets sum to 1.0
    total = sum(profile.values())
    return {k: v / total for k, v in profile.items()}


def batted_ball_outcome(contact_profile: dict, park_hr_factor: float = 1.0) -> tuple:
    """
    Given a batter's contact profile, simulate one batted ball event.

    Returns (outcome_type, is_hard_hit) where outcome_type is one of
    SINGLE, DOUBLE, TRIPLE, HR, OUT and is_hard_hit indicates quality.
    """
    r = random.random()
    cumulative = 0.0
    contact_type = "topped"  # default
    for ctype, prob in contact_profile.items():
        cumulative += prob
        if r < cumulative:
            contact_type = ctype
            break

    # Now determine outcome within that contact type
    outcomes = CONTACT_OUTCOMES[contact_type]
    # Apply park HR factor to the HR probability within this contact type
    adj_outcomes = list(outcomes)
    adj_outcomes[3] *= park_hr_factor  # HR boosted/suppressed by park
    # Normalize
    total = sum(adj_outcomes)
    adj_outcomes = [p / total for p in adj_outcomes]

    r2 = random.random()
    cum = 0.0
    # outcomes order: [1B, 2B, 3B, HR, OUT]
    outcome_map = [SINGLE, DOUBLE, TRIPLE, HR, OUT]
    for idx, p in enumerate(adj_outcomes):
        cum += p
        if r2 < cum:
            is_hard = contact_type in ("barrel", "solid")
            return outcome_map[idx], is_hard

    return OUT, False


# ── 2. PITCHER STUFF+ / REPERTOIRE MODEL ─────────────────────────────────────
#
# Different pitch types produce different contact quality distributions.
# A pitcher's arsenal composition determines what batters can do against them.
#
# Pitch archetypes and their effects on contact:
#   Power fastball (95+ mph): high K rate, but when hit → barrel-prone
#   Sinker/2-seam: induces grounders (topped), suppresses barrels
#   Slider: induces weak contact (whiffs + weak), good 2-strike pitch
#   Curveball: induces under/popup contact, low EV
#   Changeup: induces topped + flare, keeps hitters off-balance
#   Cutter: hybrid — some GB induction, moderate K rate
#
# We classify pitchers into archetypes and modify the batter's contact
# profile accordingly.

PITCH_ARSENAL_EFFECTS = {
    "power": {
        # Power arms: higher K rate offset by harder contact when batters connect
        "barrel": 1.15,
        "solid": 1.08,
        "flare": 0.90,
        "topped": 0.88,
        "under": 1.05,
        "weak": 0.92,
    },
    "sinker": {
        # Sinker/GB pitchers: tons of grounders, suppress barrels
        "barrel": 0.72,
        "solid": 0.85,
        "flare": 0.90,
        "topped": 1.40,
        "under": 0.78,
        "weak": 1.10,
    },
    "breaking": {
        # Slider/curveball dominant: weak contact + whiffs
        "barrel": 0.82,
        "solid": 0.88,
        "flare": 1.05,
        "topped": 0.95,
        "under": 1.25,
        "weak": 1.20,
    },
    "offspeed": {
        # Changeup/splitter dominant: induces topped + foul
        "barrel": 0.85,
        "solid": 0.92,
        "flare": 1.15,
        "topped": 1.18,
        "under": 0.90,
        "weak": 1.05,
    },
    "balanced": {
        # No dominant pitch type — league average contact profile
        "barrel": 1.00,
        "solid": 1.00,
        "flare": 1.00,
        "topped": 1.00,
        "under": 1.00,
        "weak": 1.00,
    },
}


def classify_pitcher_arsenal(pitcher_stats: dict) -> str:
    """
    Classify a pitcher's archetype from their stat profile.

    Uses GB%, K%, and pitch velocity indicators to determine which
    contact-quality modifiers to apply.

    Returns one of: "power", "sinker", "breaking", "offspeed", "balanced"
    """
    try:
        k_pct = float(pitcher_stats.get("k_pct") or 22.5)
    except (ValueError, TypeError):
        k_pct = 22.5
    try:
        gb_pct = float(pitcher_stats.get("gb_pct") or 44.0)
    except (ValueError, TypeError):
        gb_pct = 44.0
    try:
        bb_pct = float(pitcher_stats.get("bb_pct") or 8.5)
    except (ValueError, TypeError):
        bb_pct = 8.5

    # High K + low GB = power (throws hard, gets whiffs)
    if k_pct >= 28.0 and gb_pct < 42.0:
        return "power"
    # High GB + moderate/low K = sinker type
    if gb_pct >= 50.0:
        return "sinker"
    # High K + high GB = breaking ball specialist
    if k_pct >= 26.0 and gb_pct >= 44.0:
        return "breaking"
    # Low K + low BB + moderate GB = offspeed/finesse
    if k_pct < 20.0 and bb_pct < 7.0:
        return "offspeed"
    return "balanced"


def apply_arsenal_to_contact(contact_profile: dict, pitcher_stats: dict) -> dict:
    """
    Modify a batter's contact quality distribution based on pitcher archetype.

    A sinker-baller produces more topped balls (grounders) from any batter.
    A power pitcher produces more barrels when contact is made.
    """
    archetype = classify_pitcher_arsenal(pitcher_stats)
    effects = PITCH_ARSENAL_EFFECTS[archetype]

    modified = {}
    for ctype, base_prob in contact_profile.items():
        modified[ctype] = base_prob * effects.get(ctype, 1.0)

    # Normalize
    total = sum(modified.values())
    return {k: v / total for k, v in modified.items()}


# ── 3. SITUATIONAL HITTING (LEVERAGE / 2-OUT / RISP) ─────────────────────────
#
# Batters change approach in high-leverage situations. Research shows:
#   - With RISP and 2 outs: K rate +8%, contact quality drops (pressing)
#   - With runners on, <2 outs: more topped balls (trying to drive, goes GB)
#   - Late & close (innings 7+, within 2 runs): K rate +4% (adrenaline)
#   - Blowout (5+ runs lead): lower effort → more topped/flare contact
#
# These are applied dynamically DURING the half-inning simulation based on
# the current game state.

SITUATIONAL_MODS = {
    "risp_2out": {
        # Pressing with 2 outs, runner in scoring position
        "k_mult": 1.08,
        "contact_shift": {"barrel": 0.88, "topped": 1.15, "weak": 1.10},
    },
    "risp_0out": {
        # Runner on 3rd, no outs — batter tries to lift ball (sac fly approach)
        "k_mult": 0.94,
        "contact_shift": {"under": 1.30, "topped": 0.85, "flare": 1.15},
    },
    "late_close": {
        # High leverage: innings 7+, game within 2 runs
        "k_mult": 1.04,
        "contact_shift": {"barrel": 1.06, "weak": 1.08},
    },
    "blowout_leading": {
        # Up 5+: lower adrenaline, less aggressive swings
        "k_mult": 0.97,
        "contact_shift": {"barrel": 0.90, "flare": 1.12, "topped": 1.05},
    },
}


def get_situation_key(outs: int, b2: bool, b3: bool, inning: int, run_diff: int) -> str:
    """Determine the situational modifier key from current game state."""
    if abs(run_diff) >= 5 and inning >= 5:
        return "blowout_leading" if run_diff > 0 else ""
    if inning >= 6 and abs(run_diff) <= 2:
        return "late_close"
    if (b2 or b3) and outs == 2:
        return "risp_2out"
    if b3 and outs == 0:
        return "risp_0out"
    return ""


def apply_situational_mod(probs: list, situation: str) -> list:
    """
    Apply situational K-rate modifier to outcome probabilities.

    OUTCOME_ORDER = [WALK, K, 1B, 2B, 3B, HR, OUT]
                       0   1   2   3   4   5   6
    """
    if not situation or situation not in SITUATIONAL_MODS:
        return probs
    mod = SITUATIONAL_MODS[situation]
    adjusted = list(probs)
    adjusted[1] *= mod["k_mult"]
    total = sum(adjusted)
    return [p / total for p in adjusted]


# ── 4. SPRAY ANGLE + DEFENSIVE POSITIONING ──────────────────────────────────
#
# Batters have directional tendencies (pull, center, opposite field) that
# interact with defensive alignment. A dead-pull hitter facing a shift has
# lower BABIP on grounders; a spray hitter beats shifts easily.
#
# Pull%/Center%/Oppo% distributions from Statcast (2022-2024):
#   MLB average: Pull 40%, Center 34%, Oppo 26%
#   Extreme pull hitter (Judge, Alonso): Pull 52%+, Oppo 18%
#   Spray hitter (Arraez, Betts): Pull 32%, Center 38%, Oppo 30%
#
# Defensive shift effect on BABIP:
#   Standard alignment: BABIP = batter's natural rate
#   Pull-heavy vs shift: BABIP -0.020 on grounders (2022 shift ban helped, but
#     teams still position optimally within traditional rules)
#   Oppo/spray hitter: BABIP +0.010 (naturally avoids defensive clustering)

LEAGUE_AVG_PULL = 0.40
LEAGUE_AVG_CENTER = 0.34
LEAGUE_AVG_OPPO = 0.26


def compute_spray_babip_mod(stats: dict, savant: dict = None) -> float:
    """
    Compute a BABIP modifier based on spray angle tendency vs defense.

    Returns a multiplier on hit probability (1B, 2B, 3B):
      > 1.0 = spray hitter, beats positioning
      < 1.0 = extreme pull hitter, loses hits to positioning
      = 1.0 = league average directional profile
    """
    pull_pct = None

    if savant:
        pull_pct = savant.get("pull_pct")

    if pull_pct is None and stats:
        # Estimate from GB/FB ratio: high GB% hitters tend to pull more
        go = int(stats.get("groundOuts", 0) or 0)
        fo = int(stats.get("flyOuts", 0) or 0)
        if go + fo >= 80:
            gb_ratio = go / (go + fo)
            # Empirical: each 5pp GB% above average → +2pp pull tendency
            pull_pct = 0.40 + (gb_ratio - 0.44) * 0.40
            pull_pct = max(0.28, min(0.58, pull_pct))

    if pull_pct is None:
        return 1.0

    # Extreme pull hitters lose BABIP to positioning
    # Spray hitters gain BABIP from beating alignment
    deviation = pull_pct - LEAGUE_AVG_PULL
    # Each 1pp above average pull = -0.15% BABIP; below = +0.10% BABIP
    if deviation > 0:
        modifier = 1.0 - deviation * 0.15
    else:
        modifier = 1.0 - deviation * 0.10  # negative * negative = positive

    return max(0.92, min(1.08, modifier))


def apply_spray_modifier(probs: list, spray_mod: float) -> list:
    """
    Apply spray angle BABIP modifier to singles, doubles, triples.

    OUTCOME_ORDER = [WALK, K, 1B, 2B, 3B, HR, OUT]
                       0   1   2   3   4   5   6
    """
    if spray_mod == 1.0:
        return probs
    adjusted = list(probs)
    adjusted[2] *= spray_mod  # singles most affected
    adjusted[3] *= 1.0 + (spray_mod - 1.0) * 0.6  # doubles 60% effect
    adjusted[4] *= 1.0 + (spray_mod - 1.0) * 0.4  # triples 40% effect
    total = sum(adjusted)
    return [p / total for p in adjusted]


# ── 5. COUNT-BASED PLATE APPEARANCE MODEL ────────────────────────────────────
#
# The current sim draws ONE outcome per PA. Reality: each PA is a sequence of
# pitches where the count shifts outcome probabilities dramatically.
#
# Key research (Tango/MGL, 2020-2024 Statcast):
#   0-0: Batter sees fastball 62% of the time, swings 28%, BA .340 on contact
#   0-2: Pitcher throws offspeed 55%, batter chases 33%, K rate 33%
#   3-0: Batter takes 95%, walk rate 52% (on next pitches)
#   3-1: Hitter's count — fastball 72%, HR rate 2.5x average
#   3-2: Full count — walk 22%, K 22%, HR rate 1.4x
#
# Rather than simulating pitch-by-pitch (too slow for 100K games), we model
# the COUNT DISTRIBUTION a batter reaches and weight outcomes accordingly.
#
# Each batter has a "count profile" — how often they reach each count state.
# Then we apply count-specific outcome multipliers to the base probabilities.

# Count-specific outcome multipliers relative to PA average
# Format: (K_mult, BB_mult, HR_mult, hit_mult)
COUNT_MULTIPLIERS = {
    "ahead": (0.62, 1.45, 0.85, 1.12),  # 1-0, 2-0, 2-1, 3-1 (hitter's counts)
    "behind": (1.55, 0.40, 1.25, 0.78),  # 0-1, 0-2, 1-2 (pitcher's counts)
    "even": (1.00, 1.00, 1.00, 1.00),  # 0-0, 1-1, 2-2 (neutral)
    "full": (1.10, 1.80, 1.40, 0.95),  # 3-2 (high variance)
    "deep_behind": (1.90, 0.20, 1.50, 0.65),  # 0-2 specifically
}

# How often an average batter reaches each count state (% of PAs that pass
# through this state before resolution). Derived from pitch-level data.
LEAGUE_COUNT_DIST = {
    "ahead": 0.32,  # ~32% of PA time is in hitter-favorable counts
    "behind": 0.35,  # ~35% in pitcher-favorable counts
    "even": 0.22,  # ~22% in neutral counts (0-0 start + even)
    "full": 0.11,  # ~11% reach full count
}


def compute_count_profile(stats: dict) -> dict:
    """
    Estimate how a batter's PA count distribution differs from average.

    Patient hitters (high BB%, low chase rate) spend more time in
    hitter's counts. Free swingers (high K%, low BB%) spend more
    time in pitcher's counts.

    Returns a dict with same keys as LEAGUE_COUNT_DIST.
    """
    pa = int(stats.get("plateAppearances", 0) or 0)
    if pa < 50:
        return dict(LEAGUE_COUNT_DIST)

    bb = int(stats.get("baseOnBalls", 0) or 0)
    k = int(stats.get("strikeOuts", 0) or 0)

    bb_rate = bb / pa
    k_rate = k / pa

    # Discipline score: high BB + low K = patient, works counts
    # Low BB + high K = free swinger, falls behind
    discipline = (bb_rate - 0.085) * 2.0 - (k_rate - 0.224) * 1.5
    # discipline > 0 = patient; < 0 = aggressive

    profile = dict(LEAGUE_COUNT_DIST)

    # Patient hitters shift time from "behind" to "ahead"
    shift = min(0.08, max(-0.08, discipline * 0.15))
    profile["ahead"] += shift
    profile["behind"] -= shift

    # Very patient hitters see more full counts
    if bb_rate > 0.10:
        fc_boost = min(0.04, (bb_rate - 0.10) * 0.5)
        profile["full"] += fc_boost
        profile["even"] -= fc_boost

    # Normalize
    total = sum(profile.values())
    return {k: v / total for k, v in profile.items()}


def apply_count_model(probs: list, count_profile: dict) -> list:
    """
    Weight outcome probabilities by the batter's count distribution.

    Instead of a single set of probs, we compute a weighted average of
    count-specific outcomes based on how often this batter reaches each state.

    This naturally produces:
    - Higher K rates for free swingers (more time in 0-2, 1-2)
    - Higher BB rates for patient hitters (more time in 3-1, 3-2)
    - Higher HR rates for patient power hitters (3-1 fastball hunting)
    - Lower HR rates for aggressive hitters who expand the zone

    OUTCOME_ORDER = [WALK, K, 1B, 2B, 3B, HR, OUT]
                       0   1   2   3   4   5   6
    """
    # Compute weighted multipliers across all count states
    k_mult_w = 0.0
    bb_mult_w = 0.0
    hr_mult_w = 0.0
    hit_mult_w = 0.0

    for state, weight in count_profile.items():
        k_m, bb_m, hr_m, hit_m = COUNT_MULTIPLIERS.get(state, (1.0, 1.0, 1.0, 1.0))
        k_mult_w += k_m * weight
        bb_mult_w += bb_m * weight
        hr_mult_w += hr_m * weight
        hit_mult_w += hit_m * weight

    # Compute the league-average baseline so the model is ZERO-centered:
    # a league-average batter with league-average count profile produces mult=1.0
    k_base = bb_base = hr_base = hit_base = 0.0
    for state, weight in LEAGUE_COUNT_DIST.items():
        k_m, bb_m, hr_m, hit_m = COUNT_MULTIPLIERS.get(state, (1.0, 1.0, 1.0, 1.0))
        k_base += k_m * weight
        bb_base += bb_m * weight
        hr_base += hr_m * weight
        hit_base += hit_m * weight

    # Relative to baseline: only the DIFFERENCE from league average matters
    DAMPEN = 0.35
    k_adj = 1.0 + (k_mult_w - k_base) * DAMPEN
    bb_adj = 1.0 + (bb_mult_w - bb_base) * DAMPEN
    hr_adj = 1.0 + (hr_mult_w - hr_base) * DAMPEN
    hit_adj = 1.0 + (hit_mult_w - hit_base) * DAMPEN

    adjusted = list(probs)
    adjusted[0] *= bb_adj  # walk
    adjusted[1] *= k_adj  # strikeout
    adjusted[2] *= hit_adj  # single
    adjusted[3] *= hit_adj  # double
    adjusted[4] *= hit_adj  # triple
    adjusted[5] *= hr_adj  # HR (separate from generic hits)
    total = sum(adjusted)
    return [p / total for p in adjusted]


# ── 6. PLATOON SPLIT MODEL (L/R MATCHUP) ─────────────────────────────────────
# Lefty vs lefty and righty vs righty (same-side) matchups dramatically change
# outcomes. Empirical MLB data (2019-2024 averages):
#
#   Same-side (LHP vs LHB or RHP vs RHB):
#     K rate +12%, BB rate -8%, HR rate -15%, BABIP -18 points
#   Opposite-side (LHP vs RHB or RHP vs LHB):
#     K rate -6%, BB rate +5%, HR rate +10%, BABIP +12 points
#   Switch hitter vs any: ~60% of the opposite-side advantage
#
# These factors are applied on top of the split-blended probs from web.py.
# We dampen to 40% because web.py already blends platoon split stats into
# the base probabilities — this model captures the RESIDUAL effect that
# individual split stats miss (small samples, structural platoon advantage).
PLATOON_EFFECTS = {
    "same": {"k_mult": 1.12, "bb_mult": 0.92, "hr_mult": 0.85, "hit_mult": 0.94},
    "opposite": {"k_mult": 0.94, "bb_mult": 1.05, "hr_mult": 1.10, "hit_mult": 1.06},
    "switch_opp": {"k_mult": 0.96, "bb_mult": 1.03, "hr_mult": 1.06, "hit_mult": 1.04},
    "neutral": {"k_mult": 1.0, "bb_mult": 1.0, "hr_mult": 1.0, "hit_mult": 1.0},
}

PLATOON_DAMPEN = 0.40


def get_platoon_key(bat_side: str, pitch_hand: str) -> str:
    if not bat_side or not pitch_hand:
        return "neutral"
    bat_side = bat_side.upper()
    pitch_hand = pitch_hand.upper()
    if bat_side == "S":
        return "switch_opp"
    if bat_side == pitch_hand:
        return "same"
    return "opposite"


def apply_platoon_modifier(probs: list, bat_side: str, pitch_hand: str) -> list:
    key = get_platoon_key(bat_side, pitch_hand)
    effects = PLATOON_EFFECTS[key]
    if key == "neutral":
        return probs

    k_adj = 1.0 + (effects["k_mult"] - 1.0) * PLATOON_DAMPEN
    bb_adj = 1.0 + (effects["bb_mult"] - 1.0) * PLATOON_DAMPEN
    hr_adj = 1.0 + (effects["hr_mult"] - 1.0) * PLATOON_DAMPEN
    hit_adj = 1.0 + (effects["hit_mult"] - 1.0) * PLATOON_DAMPEN

    adjusted = list(probs)
    adjusted[0] *= bb_adj
    adjusted[1] *= k_adj
    adjusted[2] *= hit_adj
    adjusted[3] *= hit_adj
    adjusted[4] *= hit_adj
    adjusted[5] *= hr_adj
    total = sum(adjusted)
    return [p / total for p in adjusted]


# ── 7. RUNNER SPEED GRADES + MARKOV BASE-STATE TRANSITIONS ───────────────────
# Instead of fixed advancement probabilities for all runners, we grade runners
# into speed tiers based on sprint speed / stolen base data and use tier-specific
# transition probabilities.
#
# Speed tiers (derived from Statcast sprint speed, ft/sec):
#   "elite"  (≥29.0): Billy Hamilton, Trea Turner — extra-base taking machines
#   "fast"   (≥28.0): above-average speed, aggressive baserunning
#   "avg"    (≥27.0): league average
#   "slow"   (≥26.0): below average, careful baserunning
#   "plod"   (<26.0): catchers, DHs, aging sluggers — station-to-station
#
# Transition probabilities: P(scores | on base X, outcome Y) by speed tier
SPEED_TRANSITIONS = {
    "single": {
        "from_3rd": {"elite": 0.95, "fast": 0.92, "avg": 0.87, "slow": 0.82, "plod": 0.75},
        "from_2nd": {"elite": 0.78, "fast": 0.70, "avg": 0.62, "slow": 0.52, "plod": 0.40},
        "from_1st_to_3rd": {"elite": 0.45, "fast": 0.35, "avg": 0.27, "slow": 0.18, "plod": 0.10},
    },
    "double": {
        "from_3rd": {"elite": 0.98, "fast": 0.97, "avg": 0.95, "slow": 0.93, "plod": 0.90},
        "from_2nd": {"elite": 0.95, "fast": 0.92, "avg": 0.87, "slow": 0.82, "plod": 0.75},
        "from_1st": {"elite": 0.72, "fast": 0.62, "avg": 0.52, "slow": 0.40, "plod": 0.28},
    },
}

LEAGUE_AVG_SPRINT_SPEED = 27.0


def classify_runner_speed(stats: dict, savant: dict = None) -> str:
    sprint = None
    if savant:
        sprint = savant.get("sprint_speed")
    if sprint is None and stats:
        sb = stats.get("stolenBases", 0) or 0
        cs = stats.get("caughtStealing", 0) or 0
        pa = stats.get("plateAppearances", 1) or 1
        sb_rate = sb / pa
        if sb_rate > 0.06:
            sprint = 29.0
        elif sb_rate > 0.03:
            sprint = 28.0
        elif sb_rate > 0.01:
            sprint = 27.0
        elif sb + cs == 0 and pa > 100:
            sprint = 25.5
        else:
            sprint = 26.5
    if sprint is None:
        return "avg"
    if sprint >= 29.0:
        return "elite"
    if sprint >= 28.0:
        return "fast"
    if sprint >= 27.0:
        return "avg"
    if sprint >= 26.0:
        return "slow"
    return "plod"


# ── 8. ENHANCED LINEUP PROTECTION ────────────────────────────────────────────
# The existing protection model only modifies walk rate. Real lineup protection
# also affects:
#   - Pitch quality: pitchers throw more strikes to batters before strong hitters
#     → better contact quality (fewer weak contacts, more barrels)
#   - HR rate: when protection is strong, batter sees more fastballs in zone
#     → slight HR boost (the "protection effect" in sabermetrics)
#   - Conversely, weak protection means more nibbling → worse contact but more BBs
#
# This replaces the simple OPS-diff walk modifier in precompute_lineup.
def compute_lineup_protection(lineup_stats: list) -> list:
    """Return per-batter protection multipliers based on surrounding lineup quality."""
    n = len(lineup_stats)
    protection = []
    for i in range(n):
        curr_ops = float(lineup_stats[i].get("ops", "0") or "0")
        next_ops = float(lineup_stats[(i + 1) % n].get("ops", "0") or "0")
        prev_ops = float(lineup_stats[(i - 1) % n].get("ops", "0") or "0")

        # On-deck hitter drives the primary effect
        ondeck_diff = next_ops - curr_ops
        # Preceding hitter has a smaller effect (pitcher already faced them)
        prev_diff = prev_ops - curr_ops

        # Weighted protection score: on-deck is 75% of the signal
        prot_score = ondeck_diff * 0.75 + prev_diff * 0.25

        # Translate to multipliers
        # Positive score = strong protection → more strikes → better contact, fewer BBs
        # Negative score = weak protection → more nibbling → worse contact, more BBs
        bb_mult = 1.0
        hr_mult = 1.0
        contact_mult = 1.0

        if prot_score >= 0.120:
            bb_mult = 1.20
            hr_mult = 0.96
            contact_mult = 0.97
        elif prot_score >= 0.060:
            bb_mult = 1.12
            hr_mult = 0.98
            contact_mult = 0.98
        elif prot_score >= 0.030:
            bb_mult = 1.06
            hr_mult = 0.99
            contact_mult = 0.99
        elif prot_score <= -0.090:
            bb_mult = 0.88
            hr_mult = 1.06
            contact_mult = 1.04
        elif prot_score <= -0.060:
            bb_mult = 0.92
            hr_mult = 1.04
            contact_mult = 1.02
        elif prot_score <= -0.030:
            bb_mult = 0.96
            hr_mult = 1.02
            contact_mult = 1.01

        protection.append(
            {
                "bb_mult": bb_mult,
                "hr_mult": hr_mult,
                "contact_mult": contact_mult,
            }
        )
    return protection


def apply_lineup_protection(probs: list, prot: dict) -> list:
    adjusted = list(probs)
    adjusted[0] *= prot["bb_mult"]
    adjusted[5] *= prot["hr_mult"]
    adjusted[2] *= prot["contact_mult"]
    adjusted[3] *= prot["contact_mult"]
    adjusted[4] *= prot["contact_mult"]
    total = sum(adjusted)
    return [p / total for p in adjusted]


# ── 9. CONTINUOUS PITCHER STAMINA DECAY ──────────────────────────────────────
# The existing pitch-count model uses 4 fixed thresholds (60/80/100 pitches).
# This replaces it with a continuous decay function that models:
#   - Velocity loss: ~0.3 mph per 15 pitches after pitch 45
#   - Command degradation: walk rate climbs exponentially after pitch 75
#   - Barrel rate increase: fatigue → less movement → more hard contact
#
# The continuous model captures the gradual decline rather than step functions.
def compute_stamina_decay(estimated_pitches: float, pitcher_stats: dict = None) -> dict:
    """
    Compute continuous stamina decay multipliers based on pitch count.

    Returns dict of multipliers for K, BB, HR, and hit rates.
    All multipliers are 1.0 at pitch 0 and degrade continuously.
    """
    if estimated_pitches < 30:
        return {"k_mult": 1.0, "bb_mult": 1.0, "hr_mult": 1.0, "hit_mult": 1.0}

    # Velocity decay: starts at pitch 45, accelerates after 80
    # Each 0.5 mph lost ≈ 2% less K rate, 1.5% more barrel rate
    if estimated_pitches <= 45:
        velo_loss = 0.0
    elif estimated_pitches <= 80:
        velo_loss = (estimated_pitches - 45) * 0.009
    else:
        velo_loss = 35 * 0.009 + (estimated_pitches - 80) * 0.018

    # Command decay: walk rate climbs after pitch 60
    if estimated_pitches <= 60:
        cmd_decay = 0.0
    elif estimated_pitches <= 90:
        cmd_decay = (estimated_pitches - 60) * 0.004
    else:
        cmd_decay = 30 * 0.004 + (estimated_pitches - 90) * 0.008

    # Convert to multipliers
    k_mult = max(0.75, 1.0 - velo_loss * 0.04)
    bb_mult = min(1.30, 1.0 + cmd_decay)
    hr_mult = min(1.25, 1.0 + velo_loss * 0.03)
    hit_mult = min(1.15, 1.0 + velo_loss * 0.015 + cmd_decay * 0.5)

    # Pitcher durability: high-IP pitchers handle pitch counts better
    if pitcher_stats:
        try:
            ip = float(pitcher_stats.get("inningsPitched", "0") or "0")
            if ip >= 180:
                durability = 0.70
            elif ip >= 160:
                durability = 0.80
            elif ip >= 140:
                durability = 0.90
            else:
                durability = 1.0
            k_mult = 1.0 + (k_mult - 1.0) * durability
            bb_mult = 1.0 + (bb_mult - 1.0) * durability
            hr_mult = 1.0 + (hr_mult - 1.0) * durability
            hit_mult = 1.0 + (hit_mult - 1.0) * durability
        except (ValueError, TypeError):
            pass

    return {"k_mult": k_mult, "bb_mult": bb_mult, "hr_mult": hr_mult, "hit_mult": hit_mult}


def apply_stamina_decay(probs: list, decay: dict) -> list:
    adjusted = list(probs)
    adjusted[0] *= decay["bb_mult"]
    adjusted[1] *= decay["k_mult"]
    adjusted[2] *= decay["hit_mult"]
    adjusted[3] *= decay["hit_mult"]
    adjusted[4] *= decay["hit_mult"]
    adjusted[5] *= decay["hr_mult"]
    total = sum(adjusted)
    return [p / total for p in adjusted]


# ── 10. LEVERAGE INDEX (LI) MODEL ───────────────────────────────────────────
# Real-time game leverage affects pitcher performance and manager decisions.
# High-leverage situations (tying run at bat in late innings) produce measurably
# different outcomes than low-leverage situations.
#
# Leverage Index (LI) measures how much a single PA affects the game outcome:
#   LI = 1.0: average leverage (typical 5th inning, no runners)
#   LI > 2.0: high leverage (late, close, runners on)
#   LI < 0.5: low leverage (blowout, early innings)
#
# Effects of high leverage (observed in MLB data):
#   - Relievers are higher quality (manager deploys best arms)
#   - K rate increases (pitchers throw harder, batters expand zone)
#   - Walk rate increases (pitchers nibble more carefully)
#   - HR rate per contact decreases slightly (fewer pitches in zone)
#   - BABIP drops slightly (outfielders play more carefully)
LI_TABLE = {
    (0, 0, 0, 0): 0.9,
    (0, 0, 0, 1): 1.3,
    (0, 0, 0, 2): 1.1,
    (0, 0, 1, 0): 1.1,
    (0, 0, 1, 1): 1.6,
    (0, 0, 1, 2): 1.2,
    (0, 1, 0, 0): 1.2,
    (0, 1, 0, 1): 1.7,
    (0, 1, 0, 2): 1.3,
    (0, 1, 1, 0): 1.5,
    (0, 1, 1, 1): 2.1,
    (0, 1, 1, 2): 1.5,
    (1, 0, 0, 0): 1.0,
    (1, 0, 0, 1): 1.5,
    (1, 0, 0, 2): 1.4,
    (1, 0, 1, 0): 1.3,
    (1, 0, 1, 1): 1.9,
    (1, 0, 1, 2): 1.7,
    (1, 1, 0, 0): 1.4,
    (1, 1, 0, 1): 2.0,
    (1, 1, 0, 2): 1.8,
    (1, 1, 1, 0): 1.7,
    (1, 1, 1, 1): 2.5,
    (1, 1, 1, 2): 2.2,
    (2, 0, 0, 0): 1.1,
    (2, 0, 0, 1): 1.6,
    (2, 0, 0, 2): 2.0,
    (2, 0, 1, 0): 1.4,
    (2, 0, 1, 1): 2.1,
    (2, 0, 1, 2): 2.6,
    (2, 1, 0, 0): 1.5,
    (2, 1, 0, 1): 2.3,
    (2, 1, 0, 2): 2.8,
    (2, 1, 1, 0): 1.9,
    (2, 1, 1, 1): 2.8,
    (2, 1, 1, 2): 3.5,
}


def compute_leverage_index(outs: int, b1: bool, b2: bool, inning: int, run_diff: int) -> float:
    """
    Compute an approximate Leverage Index for the current game state.

    Uses a simplified base-out state table scaled by inning and score proximity.
    """
    # Base LI from runners/outs state
    b1_int = int(b1)
    b2_int = int(b2)
    close = 1 if abs(run_diff) <= 2 else 0
    base_li = LI_TABLE.get((outs, b1_int, b2_int, close), 1.0)

    # Scale by inning: late innings amplify leverage
    if inning >= 8:
        inning_mult = 1.4
    elif inning >= 6:
        inning_mult = 1.2
    elif inning >= 4:
        inning_mult = 1.0
    else:
        inning_mult = 0.8

    # Score proximity: blowouts kill leverage regardless of state
    if abs(run_diff) >= 6:
        score_mult = 0.3
    elif abs(run_diff) >= 4:
        score_mult = 0.5
    elif abs(run_diff) >= 3:
        score_mult = 0.7
    else:
        score_mult = 1.0

    return min(5.0, base_li * inning_mult * score_mult)


def apply_leverage_modifier(probs: list, leverage: float) -> list:
    """
    Adjust outcome probabilities based on leverage index.

    High leverage: K rate up (pitchers throw harder), walk rate up (nibble more),
    BABIP down (defense sharpens), HR per contact slightly down.
    Low leverage: opposite effects.
    """
    if 0.9 <= leverage <= 1.1:
        return probs

    # Scale effects relative to average leverage (1.0)
    li_diff = leverage - 1.0
    dampen = 0.03

    k_adj = 1.0 + li_diff * dampen * 1.5
    bb_adj = 1.0 + li_diff * dampen * 1.0
    hr_adj = 1.0 - li_diff * dampen * 0.5
    hit_adj = 1.0 - li_diff * dampen * 0.8

    adjusted = list(probs)
    adjusted[0] *= bb_adj
    adjusted[1] *= k_adj
    adjusted[2] *= hit_adj
    adjusted[3] *= hit_adj
    adjusted[5] *= hr_adj
    total = sum(adjusted)
    return [p / total for p in adjusted]


# ── 11. PITCH-BY-PITCH PLATE APPEARANCE SIMULATION ──────────────────────────
# Instead of drawing one outcome per PA, we simulate individual pitches.
# Each pitch resolves as: ball, called strike, swinging strike, foul, or
# ball in play. The count progresses naturally, and the final outcome
# (K, BB, or batted ball → hit/out) emerges from first principles.
#
# Key parameters derived from MLB pitch-level data (2022-2024):
#   - Zone rate: % of pitches thrown in the strike zone (~45% league avg)
#   - Swing rate: P(swing | zone/chase) varies by count and batter discipline
#   - Contact rate: P(contact | swing) — higher on zone pitches, lower on chases
#   - Whiff rate: 1 - contact rate
#   - Foul rate: P(foul | contact, 2 strikes) — fouls extend the AB
#   - In-play rate: P(in play | contact) = 1 - foul rate
#
# The batted ball outcome (hit type) is determined by the existing contact
# quality model when the ball is put in play.

# Zone rate adjustments by count: pitchers throw more strikes when behind
COUNT_ZONE_RATES = {
    (0, 0): 0.48,  # first pitch — slightly above avg zone rate
    (1, 0): 0.52,
    (2, 0): 0.57,
    (3, 0): 0.62,  # behind → must throw strikes
    (0, 1): 0.43,
    (0, 2): 0.38,  # ahead → can nibble/waste
    (1, 1): 0.46,
    (1, 2): 0.40,
    (2, 1): 0.50,
    (2, 2): 0.44,
    (3, 1): 0.56,
    (3, 2): 0.50,  # full count — competitive
}

# Swing rates: P(swing) given zone vs chase, by count pressure
# Format: (balls, strikes) → (swing_at_zone, swing_at_chase)
COUNT_SWING_RATES = {
    (0, 0): (0.62, 0.22),  # first pitch — selective
    (1, 0): (0.60, 0.20),
    (2, 0): (0.55, 0.16),
    (3, 0): (0.48, 0.12),
    (0, 1): (0.68, 0.26),
    (0, 2): (0.75, 0.34),  # 2 strikes — protect
    (1, 1): (0.66, 0.24),
    (1, 2): (0.73, 0.32),
    (2, 1): (0.64, 0.22),
    (2, 2): (0.72, 0.30),
    (3, 1): (0.58, 0.18),
    (3, 2): (0.70, 0.28),
}

# Contact rate on swings: zone pitches are easier to hit
ZONE_CONTACT_RATE = 0.87
CHASE_CONTACT_RATE = 0.62

# When contact is made, probability of fouling it off vs putting in play
# Foul rate is higher with 2 strikes (batters shorten up / protect)
FOUL_RATE_NORMAL = 0.42
FOUL_RATE_TWO_STRIKES = 0.52


def build_pitch_profile(batter_stats: dict, pitcher_stats: dict = None) -> dict:
    """
    Build a batter's pitch-level profile from their plate discipline stats.

    Aggressive hitters: higher swing rates, lower discipline, more chases
    Patient hitters: lower swing rates, better zone judgment, fewer chases
    """
    pa = max(int(batter_stats.get("plateAppearances", 1) or 1), 1)
    bb = int(batter_stats.get("baseOnBalls", 0) or 0)
    k = int(batter_stats.get("strikeOuts", 0) or 0)

    bb_rate = bb / pa
    k_rate = k / pa

    # Discipline index: high = patient, low = aggressive
    # BB% and K% together reveal approach
    discipline = bb_rate - k_rate * 0.5  # range roughly -0.10 to +0.05

    # Zone swing modifier: patient hitters swing less at everything
    zone_swing_mod = 1.0 - discipline * 2.0  # patient → less swinging
    zone_swing_mod = max(0.85, min(1.15, zone_swing_mod))

    # Chase rate modifier: disciplined hitters chase less
    chase_mod = 1.0 - discipline * 4.0
    chase_mod = max(0.65, min(1.40, chase_mod))

    # Contact ability: low-K hitters make more contact
    contact_mod = 1.0 - (k_rate - 0.22) * 1.5  # avg K rate ~22%
    contact_mod = max(0.88, min(1.12, contact_mod))

    # Pitcher adjustments: high-K pitchers reduce contact and increase chases
    if pitcher_stats:
        try:
            p_k_pct = float(pitcher_stats.get("k_pct", "22") or "22") / 100.0
            pitcher_stuff = (p_k_pct - 0.22) * 2.0  # how far above average
            contact_mod *= max(0.90, 1.0 - pitcher_stuff * 0.3)
            chase_mod *= min(1.20, 1.0 + pitcher_stuff * 0.5)
        except (ValueError, TypeError):
            pass

    return {
        "zone_swing_mod": zone_swing_mod,
        "chase_mod": chase_mod,
        "contact_mod": contact_mod,
    }


def simulate_pitch_pa(
    pitch_profile: dict,
    batter_probs: list,
    contact_profile: dict = None,
    park_hr_factor: float = 1.0,
) -> str:
    """
    Simulate a full plate appearance pitch by pitch.

    Returns one of: WALK, STRIKEOUT, SINGLE, DOUBLE, TRIPLE, HR, OUT
    matching OUTCOME_ORDER.
    """
    balls = 0
    strikes = 0
    zsm = pitch_profile["zone_swing_mod"]
    cm = pitch_profile["chase_mod"]
    ctm = pitch_profile["contact_mod"]

    while True:
        # Get count-specific rates
        count = (balls, strikes)
        zone_rate = COUNT_ZONE_RATES.get(count, 0.45)
        zone_swing, chase_swing = COUNT_SWING_RATES.get(count, (0.65, 0.25))

        # Apply batter's discipline modifiers
        zone_swing = min(0.95, zone_swing * zsm)
        chase_swing = min(0.60, chase_swing * cm)

        # Is this pitch in the zone?
        in_zone = random.random() < zone_rate

        if in_zone:
            # Zone pitch
            if random.random() < zone_swing:
                # Batter swings at zone pitch
                contact_rate = min(0.98, ZONE_CONTACT_RATE * ctm)
                if random.random() < contact_rate:
                    # Contact made on zone pitch
                    foul_rate = FOUL_RATE_TWO_STRIKES if strikes == 2 else FOUL_RATE_NORMAL
                    if random.random() < foul_rate:
                        # Foul ball — strike if < 2 strikes
                        if strikes < 2:
                            strikes += 1
                    else:
                        # Ball in play — use batted ball physics if available
                        if contact_profile:
                            return batted_ball_outcome(contact_profile, park_hr_factor)[0]
                        else:
                            return _resolve_bip(batter_probs)
                else:
                    # Swinging strike
                    strikes += 1
                    if strikes >= 3:
                        return STRIKEOUT
            else:
                # Takes zone pitch — called strike
                strikes += 1
                if strikes >= 3:
                    return STRIKEOUT
        else:
            # Out of zone
            if random.random() < chase_swing:
                # Batter chases
                contact_rate = min(0.85, CHASE_CONTACT_RATE * ctm)
                if random.random() < contact_rate:
                    # Contact on chase — usually weak
                    foul_rate = FOUL_RATE_TWO_STRIKES if strikes == 2 else FOUL_RATE_NORMAL
                    if random.random() < foul_rate:
                        if strikes < 2:
                            strikes += 1
                    else:
                        # Weak contact on chase — shift toward ground outs
                        if contact_profile:
                            # Degrade contact quality for chase contact
                            degraded = dict(contact_profile)
                            degraded["barrel"] = degraded.get("barrel", 0.07) * 0.4
                            degraded["weak"] = degraded.get("weak", 0.085) * 1.6
                            degraded["topped"] = degraded.get("topped", 0.315) * 1.3
                            total_c = sum(degraded.values())
                            degraded = {k: v / total_c for k, v in degraded.items()}
                            return batted_ball_outcome(degraded, park_hr_factor)[0]
                        else:
                            return _resolve_bip_weak(batter_probs)
                else:
                    # Swing and miss on chase
                    strikes += 1
                    if strikes >= 3:
                        return STRIKEOUT
            else:
                # Takes ball
                balls += 1
                if balls >= 4:
                    return WALK


def _resolve_bip(probs: list) -> str:
    """Resolve a ball in play using the batter's probability distribution."""
    # Remove K and BB from the distribution, renormalize
    bip_probs = [0.0, 0.0, probs[2], probs[3], probs[4], probs[5], probs[6]]
    total = sum(bip_probs)
    if total <= 0:
        return OUT
    bip_probs = [p / total for p in bip_probs]
    cum = list(accumulate(bip_probs))
    r = random.random()
    i = bisect(cum, r)
    i = min(i, len(OUTCOME_ORDER) - 1)
    return OUTCOME_ORDER[i]


def _resolve_bip_weak(probs: list) -> str:
    """Resolve a weakly-hit ball in play (chase contact)."""
    # Shift distribution toward outs: halve XBH, boost OUT
    bip_probs = [
        0.0,
        0.0,
        probs[2] * 0.7,  # single reduced
        probs[3] * 0.4,  # double reduced more
        probs[4] * 0.3,  # triple very rare
        probs[5] * 0.25,  # HR very rare on chase
        probs[6] * 1.5,  # out boosted
    ]
    total = sum(bip_probs)
    if total <= 0:
        return OUT
    bip_probs = [p / total for p in bip_probs]
    cum = list(accumulate(bip_probs))
    r = random.random()
    i = bisect(cum, r)
    i = min(i, len(OUTCOME_ORDER) - 1)
    return OUTCOME_ORDER[i]


# ── 12. BAYESIAN IN-GAME PITCHER QUALITY UPDATING ───────────────────────────
# Before the game, we estimate pitcher quality from season stats. But within
# a game, we observe actual results and can update our estimate.
#
# Model: pitcher quality follows a Beta(α, β) distribution where:
#   α = "good outcome units" (outs recorded, Ks)
#   β = "bad outcome units" (hits, walks, runs allowed)
#
# Prior is set from the pitcher's season K/BB/HR rates. As batters face
# the pitcher, each PA outcome updates the posterior. If the pitcher gives
# up 3 hits in the 1st inning, his posterior quality drops, making future
# PAs more hitter-friendly — capturing "he doesn't have it today."
#
# The update magnitude is controlled by a "game weight" parameter that
# determines how much one game's evidence shifts the prior. Too high and
# a single lucky hit collapses the estimate; too low and it doesn't react.
GAME_EVIDENCE_WEIGHT = 0.15  # how much one PA shifts the prior (0-1)


class InGamePitcherState:
    """Track pitcher quality within a simulated game using Bayesian updating."""

    __slots__ = ("prior_quality", "alpha", "beta", "pa_count", "runs_allowed")

    def __init__(self, pitcher_stats: dict = None):
        # Prior quality from season stats: 0.0 = terrible, 1.0 = elite
        if pitcher_stats:
            try:
                era = float(pitcher_stats.get("era", "4.00") or "4.00")
                fip = float(pitcher_stats.get("fip", era) or era)
                blended = era * 0.4 + fip * 0.6
                self.prior_quality = max(0.1, min(0.95, 1.0 - (blended - 2.0) / 5.0))
            except (ValueError, TypeError):
                self.prior_quality = 0.50
        else:
            self.prior_quality = 0.50

        # Beta distribution parameters: higher α = more good outcomes expected
        # Start with moderate confidence (α+β=20 → ~20 PA worth of prior)
        prior_strength = 20.0
        self.alpha = self.prior_quality * prior_strength
        self.beta = (1.0 - self.prior_quality) * prior_strength
        self.pa_count = 0
        self.runs_allowed = 0

    def update(self, outcome_good: bool):
        """Update posterior after observing a PA outcome."""
        self.pa_count += 1
        if outcome_good:
            self.alpha += GAME_EVIDENCE_WEIGHT
        else:
            self.beta += GAME_EVIDENCE_WEIGHT

    def record_run(self):
        """Record a run allowed — stronger negative signal."""
        self.runs_allowed += 1
        self.beta += GAME_EVIDENCE_WEIGHT * 2.0

    @property
    def current_quality(self) -> float:
        """Current posterior mean quality estimate."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def quality_multiplier(self) -> float:
        """
        Ratio of current quality to prior quality.
        > 1.0 = pitcher is better than expected today
        < 1.0 = pitcher is worse than expected today
        """
        if self.prior_quality <= 0:
            return 1.0
        return self.current_quality / self.prior_quality

    def get_adjustment(self) -> dict:
        """
        Convert quality shift into probability adjustments.

        When the pitcher is performing worse than expected:
          - K rate decreases
          - BB rate increases
          - Hit rate increases
          - HR rate increases

        Capped at ±15% to avoid wild swings from small samples.
        """
        qm = self.quality_multiplier
        # Dampen: don't let a few PAs dominate
        diff = (qm - 1.0) * 0.6
        diff = max(-0.15, min(0.15, diff))

        return {
            "k_mult": 1.0 + diff * 1.2,  # quality → more Ks
            "bb_mult": 1.0 - diff * 1.0,  # quality → fewer BBs
            "hit_mult": 1.0 - diff * 0.8,  # quality → fewer hits
            "hr_mult": 1.0 - diff * 0.6,  # quality → fewer HRs
        }


def apply_ingame_pitcher_adj(probs: list, adj: dict) -> list:
    """Apply in-game pitcher quality adjustment to outcome probs."""
    adjusted = list(probs)
    adjusted[0] *= adj["bb_mult"]
    adjusted[1] *= adj["k_mult"]
    adjusted[2] *= adj["hit_mult"]
    adjusted[3] *= adj["hit_mult"]
    adjusted[4] *= adj["hit_mult"]
    adjusted[5] *= adj["hr_mult"]
    total = sum(adjusted)
    return [p / total for p in adjusted]


# ── 13. DEFENSIVE WAR / FIELDER RANGE MODEL ─────────────────────────────────
# Individual fielder quality affects BABIP. Elite defenders convert more
# batted balls into outs; poor defenders leak hits.
#
# We model this using OAA (Outs Above Average) or UZR as a team-level
# defensive adjustment. Since we don't have individual fielder data in
# the simulation inputs, we estimate team defensive quality from:
#   - Team BABIP vs league average (available from team stats)
#   - Pitcher GB% (ground ball pitchers benefit more from good IF defense)
#   - Catcher framing (already modeled separately)
#
# OAA ranges (per team, per season):
#   Elite defense:  +30 to +50 OAA  → BABIP suppressed by ~12-18 points
#   Average:        -10 to +10 OAA  → neutral
#   Poor defense:   -30 to -50 OAA  → BABIP inflated by ~12-18 points
#
# Each OAA is worth ~0.4 BABIP points over a full season.
LEAGUE_AVG_BABIP = 0.298

# Position-specific defensive impact weights:
# CF and SS have the most range; 1B and DH have the least
POSITION_DEF_WEIGHT = {
    "CF": 1.5,
    "SS": 1.4,
    "2B": 1.2,
    "3B": 1.1,
    "LF": 0.9,
    "RF": 1.0,
    "1B": 0.5,
    "C": 0.8,
    "DH": 0.0,
}


def compute_team_defense_mod(
    team_babip: float = None,
    pitcher_gb_pct: float = None,
    team_oaa: float = None,
) -> float:
    """
    Compute a BABIP modifier based on team defensive quality.

    Returns a multiplier on BABIP (hit probability on balls in play):
      < 1.0 = good defense (suppresses hits)
      > 1.0 = bad defense (leaks hits)
    """
    if team_oaa is not None:
        # Direct OAA: each OAA point = ~0.4 BABIP points over 600 BIP
        babip_shift = -team_oaa * 0.0004
        mod = 1.0 + babip_shift / LEAGUE_AVG_BABIP
    elif team_babip is not None and team_babip > 0:
        # Infer from team BABIP vs league average
        babip_diff = team_babip - LEAGUE_AVG_BABIP
        # Only attribute 60% of BABIP diff to defense (rest is pitching/luck)
        mod = 1.0 + (babip_diff * 0.6) / LEAGUE_AVG_BABIP
    else:
        return 1.0

    # GB pitchers are more affected by infield defense quality
    if pitcher_gb_pct is not None:
        gb_scale = 1.0 + (pitcher_gb_pct - 44.0) / 100.0 * 0.5
        mod = 1.0 + (mod - 1.0) * max(0.7, min(1.4, gb_scale))

    return max(0.88, min(1.14, mod))


def apply_defense_modifier(probs: list, def_mod: float) -> list:
    """
    Apply team defensive quality modifier to hit probabilities.

    Adjusts singles and doubles (balls in play) but not HR (over the fence)
    or K/BB (not affected by defense).
    """
    if 0.99 <= def_mod <= 1.01:
        return probs
    adjusted = list(probs)
    adjusted[2] *= def_mod  # single
    adjusted[3] *= def_mod * 0.7 + 0.3  # double (less affected — gap hits)
    adjusted[4] *= def_mod * 0.5 + 0.5  # triple (least affected — deep gaps)
    # Compensate: if defense is good (mod < 1), more balls become outs
    out_change = (probs[2] - adjusted[2]) + (probs[3] - adjusted[3]) + (probs[4] - adjusted[4])
    adjusted[6] += out_change
    total = sum(adjusted)
    return [p / total for p in adjusted]


# ── 14. MOMENTUM / HOT-INNING CORRELATION ───────────────────────────────────
# Real baseball exhibits inning-to-inning correlation that pure Monte Carlo
# misses. After a big inning (3+ runs), multiple effects compound:
#
#   1. Pitcher confidence drops (Bayesian update handles this)
#   2. Batting team has "seeing the ball well" momentum
#   3. Bullpen may not be warm yet (rushed entry)
#   4. Opposing pitcher may be rattled even after removal
#
# MLB data shows the team that scores 3+ runs in an inning scores ~0.3
# more runs in the next inning than expected from base rates.
#
# We model this as a temporary probability shift that decays:
#   - Immediately after a 3+ run inning: +4% blend toward offense
#   - After a 4+ run inning: +6% blend
#   - After a 5+ run inning: +8% blend
#   - Effect halves each subsequent inning
MOMENTUM_THRESHOLDS = [
    (5, 0.08),  # 5+ run inning → 8% offensive boost
    (4, 0.06),  # 4+ run inning → 6%
    (3, 0.04),  # 3+ run inning → 4%
]


class MomentumTracker:
    """Track inning-to-inning momentum for each team."""

    __slots__ = ("away_momentum", "home_momentum")

    def __init__(self):
        self.away_momentum = 0.0
        self.home_momentum = 0.0

    def update_after_half(self, is_away: bool, runs_scored: int):
        """Update momentum after a half-inning."""
        # Decay existing momentum
        if is_away:
            self.away_momentum *= 0.5
        else:
            self.home_momentum *= 0.5

        # Check if the scoring team hit a momentum threshold
        boost = 0.0
        for threshold, effect in MOMENTUM_THRESHOLDS:
            if runs_scored >= threshold:
                boost = effect
                break

        if is_away:
            self.away_momentum = min(0.12, self.away_momentum + boost)
            # Big inning by away team gives opponent slight negative momentum
            if boost > 0:
                self.home_momentum = max(0.0, self.home_momentum - boost * 0.3)
        else:
            self.home_momentum = min(0.12, self.home_momentum + boost)
            if boost > 0:
                self.away_momentum = max(0.0, self.away_momentum - boost * 0.3)

    def get_momentum(self, is_away: bool) -> float:
        return self.away_momentum if is_away else self.home_momentum


def apply_momentum(precomp: list, momentum: float) -> list:
    """Blend cumulative-weight arrays toward league avg offense (momentum boost)."""
    if momentum < 0.005:
        return precomp
    from itertools import accumulate as _acc

    lg_cum = list(_acc(LEAGUE_AVG_PROBS))
    return [
        [c * (1 - momentum) + lg * momentum for c, lg in zip(batter_cum, lg_cum)]
        for batter_cum in precomp
    ]


# ── 15. WIN PROBABILITY MARKOV CHAIN ────────────────────────────────────────
# Build a full state-space model that computes exact win probability at every
# game state. This enables:
#   - More accurate run line probabilities (not just counting outcomes)
#   - Better F5/F7 segment predictions
#   - Precise O/U probability distributions
#   - "Live" win probability updates during simulation
#
# State: (inning, half, outs, bases, score_diff)
# We pre-compute WP from simulation results rather than analytically,
# since our transition probabilities are too complex for closed-form.

# Pre-computed from 2022-2024 MLB data: WP by (inning, score_diff) for the
# home team. These serve as interpolation anchors.
# score_diff = home_runs - away_runs (positive = home leading)
HOME_WP_TABLE = {
    # (inning_0indexed, score_diff) → home win probability
    # Inning 0 (top 1st, before game starts)
    (0, -5): 0.08,
    (0, -4): 0.12,
    (0, -3): 0.18,
    (0, -2): 0.28,
    (0, -1): 0.38,
    (0, 0): 0.54,
    (0, 1): 0.68,
    (0, 2): 0.77,
    (0, 3): 0.84,
    (0, 4): 0.90,
    (0, 5): 0.94,
    # Inning 2 (3rd inning)
    (2, -5): 0.06,
    (2, -4): 0.10,
    (2, -3): 0.16,
    (2, -2): 0.25,
    (2, -1): 0.36,
    (2, 0): 0.54,
    (2, 1): 0.70,
    (2, 2): 0.80,
    (2, 3): 0.87,
    (2, 4): 0.92,
    (2, 5): 0.96,
    # Inning 4 (5th inning)
    (4, -5): 0.04,
    (4, -4): 0.07,
    (4, -3): 0.12,
    (4, -2): 0.21,
    (4, -1): 0.33,
    (4, 0): 0.54,
    (4, 1): 0.72,
    (4, 2): 0.83,
    (4, 3): 0.90,
    (4, 4): 0.95,
    (4, 5): 0.97,
    # Inning 6 (7th inning)
    (6, -5): 0.02,
    (6, -4): 0.04,
    (6, -3): 0.08,
    (6, -2): 0.16,
    (6, -1): 0.29,
    (6, 0): 0.54,
    (6, 1): 0.75,
    (6, 2): 0.87,
    (6, 3): 0.94,
    (6, 4): 0.97,
    (6, 5): 0.99,
    # Inning 8 (9th inning)
    (8, -5): 0.01,
    (8, -4): 0.02,
    (8, -3): 0.04,
    (8, -2): 0.09,
    (8, -1): 0.20,
    (8, 0): 0.54,
    (8, 1): 0.82,
    (8, 2): 0.93,
    (8, 3): 0.97,
    (8, 4): 0.99,
    (8, 5): 1.00,
}


def get_win_probability(inning: int, score_diff: int) -> float:
    """
    Get home team win probability for a given game state.

    Uses bilinear interpolation between the pre-computed anchor points.
    score_diff = home_runs - away_runs (positive = home leading)
    """
    # Clamp score diff
    sd = max(-5, min(5, score_diff))

    # Find bounding innings in the table
    table_innings = [0, 2, 4, 6, 8]
    inn = max(0, min(8, inning))

    # Find surrounding innings for interpolation
    lower_inn = 0
    upper_inn = 8
    for ti in table_innings:
        if ti <= inn:
            lower_inn = ti
        if ti >= inn and upper_inn == 8:
            upper_inn = ti
            break
    if lower_inn == upper_inn or inn >= 8:
        return HOME_WP_TABLE.get((min(inn, 8), sd), 0.54)

    # Linear interpolation between innings
    wp_lower = HOME_WP_TABLE.get((lower_inn, sd), 0.54)
    wp_upper = HOME_WP_TABLE.get((upper_inn, sd), 0.54)
    frac = (inn - lower_inn) / max(upper_inn - lower_inn, 1)
    return wp_lower + (wp_upper - wp_lower) * frac


def compute_run_distribution(score_pairs: list) -> dict:
    """
    From simulation score pairs, compute detailed run distribution metrics:
      - Exact run line probabilities (away/home -1.5, -2.5, etc.)
      - Over/under probability curve
      - Most likely final score
      - Standard deviation of runs
    """
    if not score_pairs:
        return {}

    n = len(score_pairs)
    away_runs_list = [p[0] for p in score_pairs]
    home_runs_list = [p[1] for p in score_pairs]
    total_runs_list = [a + h for a, h in score_pairs]

    # Run line probabilities
    run_lines = {}
    for spread in [-1.5, -2.5, -3.5, 1.5, 2.5, 3.5]:
        away_cover = sum(1 for a, h in score_pairs if (a - h) > spread) / n
        home_cover = sum(1 for a, h in score_pairs if (h - a) > -spread) / n
        run_lines[f"away_{spread:+.1f}"] = round(away_cover * 100, 1)
        run_lines[f"home_{spread:+.1f}"] = round(home_cover * 100, 1)

    # Over/under curve
    ou_probs = {}
    for total in [x * 0.5 for x in range(12, 24)]:  # 6.0 to 11.5
        over_pct = sum(1 for t in total_runs_list if t > total) / n
        ou_probs[f"o{total:.1f}"] = round(over_pct * 100, 1)

    # Score distribution
    from collections import Counter

    score_counts = Counter((a, h) for a, h in score_pairs)
    most_common_score = score_counts.most_common(1)[0]

    # Standard deviation
    import math

    mean_total = sum(total_runs_list) / n
    variance = sum((t - mean_total) ** 2 for t in total_runs_list) / n
    std_dev = math.sqrt(variance)

    # Win probability at each inning checkpoint (for live WP curve)
    wp_checkpoints = {}
    for inn in range(9):
        # Average WP at this inning across all simulations would require
        # tracking per-inning scores. Instead, use the final margin
        # distribution to estimate mid-game WP.
        pass  # Populated during actual simulation if needed

    return {
        "run_lines": run_lines,
        "ou_curve": ou_probs,
        "most_likely_score": {
            "away": most_common_score[0][0],
            "home": most_common_score[0][1],
            "frequency_pct": round(most_common_score[1] / n * 100, 1),
        },
        "total_runs_std_dev": round(std_dev, 2),
        "mean_total": round(mean_total, 2),
    }


# ── 16. HIDDEN MARKOV MODEL (HMM) FOR PITCHER LATENT STATES ─────────────────
# The pitcher exists in one of 3 unobserved (hidden) states at any point in
# the game. We can't directly observe the state, but we observe outcomes
# (K, BB, hit, out) and infer the most likely state using the forward algorithm.
#
# States:
#   0 = "locked_in"  — pitcher is dominant, fastball command is sharp
#   1 = "normal"     — baseline performance matching season stats
#   2 = "struggling" — losing command, elevated HR risk, fewer Ks
#
# Transition matrix A[i][j] = P(state_j | state_i) per batter faced:
#   - "locked_in" tends to persist (0.80) but can slip to "normal" (0.18)
#   - "normal" is the attractor state (0.75 self-loop)
#   - "struggling" can recover to "normal" (0.25) but often persists (0.70)
#
# Emission probabilities B[state][outcome] define outcome distributions per state.
# The forward variable α[t][s] = P(o_1..o_t, S_t=s) is updated each PA.

HMM_STATES = ["locked_in", "normal", "struggling"]
HMM_N_STATES = 3

HMM_TRANSITION = [
    [0.80, 0.18, 0.02],  # locked_in → locked_in/normal/struggling
    [0.10, 0.75, 0.15],  # normal → locked_in/normal/struggling
    [0.05, 0.25, 0.70],  # struggling → locked_in/normal/struggling
]

# Emission probabilities: P(outcome_category | state)
# Categories: "dominant_out" (K), "normal_out" (ground/fly out),
#             "hit" (single/double), "extra" (HR/triple), "walk"
HMM_EMISSIONS = {
    "locked_in": {
        "dominant_out": 0.30,
        "normal_out": 0.38,
        "hit": 0.16,
        "extra": 0.04,
        "walk": 0.05,
    },
    "normal": {"dominant_out": 0.22, "normal_out": 0.34, "hit": 0.22, "extra": 0.06, "walk": 0.08},
    "struggling": {
        "dominant_out": 0.14,
        "normal_out": 0.28,
        "hit": 0.28,
        "extra": 0.10,
        "walk": 0.14,
    },
}

# Initial state distribution (prior): most pitchers start "normal"
HMM_INITIAL = [0.20, 0.65, 0.15]

# Outcome-to-emission category mapping
_OUTCOME_TO_EMISSION = {
    STRIKEOUT: "dominant_out",
    OUT: "normal_out",
    SINGLE: "hit",
    DOUBLE: "hit",
    TRIPLE: "extra",
    HR: "extra",
    WALK: "walk",
}

# State-to-probability multipliers (how each state modifies base probs)
HMM_STATE_MULTS = {
    "locked_in": {"k_mult": 1.25, "bb_mult": 0.65, "hr_mult": 0.70, "hit_mult": 0.80},
    "normal": {"k_mult": 1.00, "bb_mult": 1.00, "hr_mult": 1.00, "hit_mult": 1.00},
    "struggling": {"k_mult": 0.72, "bb_mult": 1.45, "hr_mult": 1.40, "hit_mult": 1.25},
}


class PitcherHMM:
    """Hidden Markov Model tracking pitcher latent state within a game."""

    __slots__ = ("alpha", "state_probs")

    def __init__(self, pitcher_stats: dict = None):
        # Initialize forward variables with prior distribution
        # Adjust prior based on pitcher quality
        if pitcher_stats:
            try:
                era = float(pitcher_stats.get("era", "4.00") or "4.00")
                if era < 3.00:
                    self.alpha = [0.35, 0.55, 0.10]
                elif era > 5.00:
                    self.alpha = [0.08, 0.52, 0.40]
                else:
                    self.alpha = list(HMM_INITIAL)
            except (ValueError, TypeError):
                self.alpha = list(HMM_INITIAL)
        else:
            self.alpha = list(HMM_INITIAL)
        self.state_probs = list(self.alpha)

    def observe(self, outcome: str):
        """
        Update state probabilities after observing a PA outcome.
        Uses the forward algorithm: α'[j] = Σ_i α[i] * A[i][j] * B[j][obs]
        Then normalize to get P(state | observations).
        """
        emission_cat = _OUTCOME_TO_EMISSION.get(outcome, "normal_out")

        new_alpha = [0.0] * HMM_N_STATES
        for j in range(HMM_N_STATES):
            # Sum over all possible previous states
            trans_sum = sum(self.alpha[i] * HMM_TRANSITION[i][j] for i in range(HMM_N_STATES))
            # Multiply by emission probability
            state_name = HMM_STATES[j]
            emit_prob = HMM_EMISSIONS[state_name].get(emission_cat, 0.20)
            new_alpha[j] = trans_sum * emit_prob

        # Normalize to prevent underflow and get state probabilities
        total = sum(new_alpha)
        if total > 0:
            self.alpha = [a / total for a in new_alpha]
        self.state_probs = list(self.alpha)

    def get_state_adjustment(self) -> dict:
        """
        Compute probability-weighted adjustment from current state distribution.
        Returns multipliers for K, BB, HR, hit rates.
        """
        k_mult = sum(
            self.state_probs[i] * HMM_STATE_MULTS[HMM_STATES[i]]["k_mult"]
            for i in range(HMM_N_STATES)
        )
        bb_mult = sum(
            self.state_probs[i] * HMM_STATE_MULTS[HMM_STATES[i]]["bb_mult"]
            for i in range(HMM_N_STATES)
        )
        hr_mult = sum(
            self.state_probs[i] * HMM_STATE_MULTS[HMM_STATES[i]]["hr_mult"]
            for i in range(HMM_N_STATES)
        )
        hit_mult = sum(
            self.state_probs[i] * HMM_STATE_MULTS[HMM_STATES[i]]["hit_mult"]
            for i in range(HMM_N_STATES)
        )

        # Dampen the effect — HMM state is uncertain early in the game
        dampen = 0.50
        return {
            "k_mult": 1.0 + (k_mult - 1.0) * dampen,
            "bb_mult": 1.0 + (bb_mult - 1.0) * dampen,
            "hr_mult": 1.0 + (hr_mult - 1.0) * dampen,
            "hit_mult": 1.0 + (hit_mult - 1.0) * dampen,
        }

    @property
    def most_likely_state(self) -> str:
        idx = self.state_probs.index(max(self.state_probs))
        return HMM_STATES[idx]


def apply_hmm_adjustment(probs: list, adj: dict) -> list:
    """Apply HMM state-weighted adjustments to outcome probs."""
    adjusted = list(probs)
    adjusted[0] *= adj["bb_mult"]
    adjusted[1] *= adj["k_mult"]
    adjusted[2] *= adj["hit_mult"]
    adjusted[3] *= adj["hit_mult"]
    adjusted[4] *= adj["hit_mult"]
    adjusted[5] *= adj["hr_mult"]
    total = sum(adjusted)
    return [p / total for p in adjusted]


# ── 17. GAUSSIAN COPULA FOR INTRA-INNING PA CORRELATION ─────────────────────
# Standard Monte Carlo treats each PA as independent. But real baseball
# exhibits intra-inning correlation: if one batter reaches base, the next
# batter faces a pitcher who's working from the stretch, may be rattled,
# and the defense is shifted. Empirically, consecutive hits are ~15% more
# likely than independence would predict.
#
# We model this using a Gaussian copula with correlation parameter ρ:
#   1. Each PA draws a uniform random u ~ U(0,1)
#   2. Transform to normal: z = Φ⁻¹(u)
#   3. Apply correlation: z' = ρ * z_prev + √(1-ρ²) * z_new
#   4. Transform back: u' = Φ(z')
#   5. Use u' to draw the outcome from cumulative weights
#
# ρ = 0: independent (standard MC)
# ρ > 0: positive correlation (rallies cluster)
# ρ < 0: negative correlation (unlikely in baseball)
#
# We use ρ = 0.12 as the base correlation, increasing to 0.20 when runners
# are on base (pitcher in the stretch) and 0.08 with bases empty.

COPULA_RHO_BASE = 0.12
COPULA_RHO_RUNNERS = 0.20
COPULA_RHO_EMPTY = 0.08

# Pre-compute standard normal CDF/inverse CDF using rational approximation
# (Abramowitz & Stegun, formula 26.2.17 for CDF; Beasley-Springer-Moro for inverse)
_SQRT2 = _math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / _math.sqrt(2.0 * _math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + _math.erf(x / _SQRT2))


def _norm_inv_cdf(p: float) -> float:
    """
    Inverse standard normal CDF (quantile function).
    Uses rational approximation (Beasley-Springer-Moro algorithm).
    Accurate to ~1e-9 for 0.0001 < p < 0.9999.
    """
    p = max(1e-10, min(1.0 - 1e-10, p))

    if p < 0.5:
        return -_bsm_core(p)
    else:
        return _bsm_core(1.0 - p)


def _bsm_core(p: float) -> float:
    """Core of Beasley-Springer-Moro inverse normal approximation."""
    # Rational approximation coefficients
    a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ]
    b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ]
    c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ]
    d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ]

    p_low = 0.02425

    if p < p_low:
        # Rational approximation for lower region
        q = _math.sqrt(-2.0 * _math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    else:
        # Rational approximation for central region
        q = p - 0.5
        r = q * q
        return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )


class GaussianCopula:
    """
    Gaussian copula for correlated PA outcomes within an inning.

    Maintains the previous PA's latent normal variable and generates
    correlated draws for the next PA.
    """

    __slots__ = ("prev_z", "rho")

    def __init__(self, rho: float = COPULA_RHO_BASE):
        self.prev_z = 0.0  # start at mean (no prior information)
        self.rho = rho

    def set_correlation(self, runners_on: bool):
        """Adjust correlation based on game state."""
        self.rho = COPULA_RHO_RUNNERS if runners_on else COPULA_RHO_EMPTY

    def draw(self) -> float:
        """
        Draw a correlated uniform random variable using the copula.

        Returns u' ∈ (0, 1) that is correlated with the previous draw.
        This replaces random.random() in the at-bat outcome selection.
        """
        # Fresh standard normal draw
        z_new = _norm_inv_cdf(random.random())

        # Apply AR(1) correlation structure
        z_correlated = self.rho * self.prev_z + _math.sqrt(1.0 - self.rho**2) * z_new

        # Store for next draw
        self.prev_z = z_correlated

        # Transform back to uniform via CDF
        u = _norm_cdf(z_correlated)
        return max(1e-10, min(1.0 - 1e-10, u))

    def reset(self):
        """Reset at the start of each inning."""
        self.prev_z = 0.0


def simulate_at_bat_copula(cum_weights: list, copula: GaussianCopula) -> str:
    """
    Draw one at-bat outcome using a correlated copula draw instead of
    independent random.random(). This creates realistic rally clustering.
    """
    r = copula.draw()
    i = bisect(cum_weights, r)
    i = min(i, len(OUTCOME_ORDER) - 1)
    return OUTCOME_ORDER[i]


# ── 18. EMPIRICAL BAYES / JAMES-STEIN SHRINKAGE ────────────────────────────
# Raw batting stats are noisy, especially for players with few PA. A .350
# hitter through 80 PA is more likely a .280 true-talent hitter who got
# lucky than a genuine .350 hitter.
#
# The James-Stein estimator shrinks individual estimates toward the grand
# mean, with shrinkage proportional to 1/n (more PA = less shrinkage).
#
# For a stat X with observed rate x_i from n_i trials:
#   x_shrunk = grand_mean + (1 - B) * (x_i - grand_mean)
#   where B = σ²_prior / (σ²_prior + σ²_sampling/n_i)
#
# σ²_prior is estimated from the cross-sectional variance of all players
# σ²_sampling is the binomial variance p(1-p)/n
#
# This is the mathematical foundation of PECOTA, Marcel, and ZiPS.

# League-wide prior parameters for key rates (2022-2024 MLB)
JAMES_STEIN_PRIORS = {
    "k_rate": {"mean": 0.224, "var": 0.0025},  # K% population variance
    "bb_rate": {"mean": 0.082, "var": 0.0008},  # BB%
    "hr_rate": {"mean": 0.032, "var": 0.0003},  # HR/PA
    "babip": {"mean": 0.298, "var": 0.0006},  # BABIP
    "iso": {"mean": 0.155, "var": 0.0012},  # Isolated power
}


def james_stein_shrink(observed: float, n_trials: int, stat_key: str) -> float:
    """
    Apply James-Stein shrinkage to an observed rate.

    Args:
        observed: raw observed rate (e.g., K/PA = 0.280)
        n_trials: number of trials (plate appearances)
        stat_key: which stat to shrink ("k_rate", "bb_rate", etc.)

    Returns:
        Shrunk estimate of the true talent rate.
    """
    prior = JAMES_STEIN_PRIORS.get(stat_key)
    if prior is None:
        return observed

    grand_mean = prior["mean"]
    prior_var = prior["var"]

    # Sampling variance for a binomial proportion
    sampling_var = observed * (1.0 - observed) / max(n_trials, 1)

    # Shrinkage factor B: how much to pull toward the mean
    # B = 1 means fully shrink to mean (no data); B = 0 means trust observed
    total_var = prior_var + sampling_var
    if total_var <= 0:
        return observed
    shrinkage_b = sampling_var / total_var

    return grand_mean + (1.0 - shrinkage_b) * (observed - grand_mean)


def build_shrunk_probs(stats: dict) -> list:
    """
    Build batter outcome probabilities using James-Stein shrinkage on
    each component rate. This produces better estimates for small-sample
    batters and reduces overfitting to noisy stats.

    OUTCOME_ORDER = [WALK, K, 1B, 2B, 3B, HR, OUT]
    """
    pa = max(int(stats.get("plateAppearances", 0) or 0), 1)
    ab = max(int(stats.get("atBats", 0) or 0), 1)
    bb = int(stats.get("baseOnBalls", 0) or 0)
    k = int(stats.get("strikeOuts", 0) or 0)
    hr = int(stats.get("homeRuns", 0) or 0)
    h = int(stats.get("hits", 0) or 0)
    doubles = int(stats.get("doubles", 0) or 0)
    triples = int(stats.get("triples", 0) or 0)
    singles = h - doubles - triples - hr

    # Shrink each rate toward its population prior
    k_rate = james_stein_shrink(k / pa, pa, "k_rate")
    bb_rate = james_stein_shrink(bb / pa, pa, "bb_rate")
    hr_rate = james_stein_shrink(hr / pa, pa, "hr_rate")

    # BABIP for hit distribution
    bip = ab - k - hr
    hits_bip = h - hr
    raw_babip = hits_bip / max(bip, 1)
    shrunk_babip = james_stein_shrink(raw_babip, bip, "babip")

    # ISO for XBH distribution
    raw_iso = (doubles + 2 * triples + 3 * hr) / max(ab, 1)
    shrunk_iso = james_stein_shrink(raw_iso, ab, "iso")

    # Reconstruct probabilities from shrunk rates
    out_bip_rate = 1.0 - k_rate - bb_rate - hr_rate
    hit_rate = out_bip_rate * shrunk_babip

    # Split hits into 1B/2B/3B using shrunk ISO as guide
    if h > 0:
        single_frac = max(0.1, singles / max(h, 1))
        double_frac = max(0.01, doubles / max(h, 1))
        triple_frac = max(0.001, triples / max(h, 1))
    else:
        single_frac = 0.65
        double_frac = 0.20
        triple_frac = 0.03

    total_frac = single_frac + double_frac + triple_frac
    single_rate = hit_rate * single_frac / total_frac
    double_rate = hit_rate * double_frac / total_frac
    triple_rate = hit_rate * triple_frac / total_frac

    out_rate = 1.0 - bb_rate - k_rate - single_rate - double_rate - triple_rate - hr_rate
    out_rate = max(0.20, out_rate)

    probs = [bb_rate, k_rate, single_rate, double_rate, triple_rate, hr_rate, out_rate]
    total = sum(probs)
    return [p / total for p in probs]


# ── 19. GENERALIZED PARETO DISTRIBUTION FOR EXTREME SCORING ─────────────────
# Normal Monte Carlo simulations underestimate the probability of extreme
# scoring events (8+ run innings, 15+ run games). This happens because the
# standard outcome draw doesn't capture the heavy tail of MLB run scoring.
#
# Extreme Value Theory (EVT) models the tail of a distribution using the
# Generalized Pareto Distribution (GPD):
#   P(X > x | X > u) = (1 + ξ(x-u)/σ)^(-1/ξ)
#
# where:
#   u = threshold (we use 4 runs/inning as the "extreme" boundary)
#   ξ (xi) = shape parameter (>0 means heavy tail)
#   σ (sigma) = scale parameter
#
# Parameters fitted from 2019-2024 MLB inning-level scoring data:
#   ξ ≈ 0.15 (moderate heavy tail — extreme innings do happen)
#   σ ≈ 1.8
#
# We use this to adjust the probability of continuing to score within an
# inning once the score reaches the extreme threshold.
GPD_THRESHOLD = 4  # runs in an inning before EVT kicks in
GPD_XI = 0.15  # shape parameter (>0 = heavy tail)
GPD_SIGMA = 1.8  # scale parameter


def gpd_survival(x: float, xi: float = GPD_XI, sigma: float = GPD_SIGMA) -> float:
    """
    Generalized Pareto survival function: P(X > x).

    For xi > 0 (heavy tail): S(x) = (1 + xi*x/sigma)^(-1/xi)
    For xi = 0 (exponential): S(x) = exp(-x/sigma)
    """
    if x <= 0:
        return 1.0
    if abs(xi) < 1e-10:
        return _math.exp(-x / sigma)
    inner = 1.0 + xi * x / sigma
    if inner <= 0:
        return 0.0
    return inner ** (-1.0 / xi)


def gpd_extreme_inning_modifier(runs_this_inning: int) -> float:
    """
    When an inning has already scored 4+ runs, compute the probability
    of continuing to score based on the GPD tail model.

    Returns a multiplier on hit probabilities (>1.0 means the extreme
    tail is heavier than what our base model produces).
    """
    if runs_this_inning < GPD_THRESHOLD:
        return 1.0

    excess = runs_this_inning - GPD_THRESHOLD

    # GPD says: given we're already in the tail, what's the conditional
    # probability of scoring MORE runs?
    # P(X > excess+1 | X > excess) = S(excess+1) / S(excess)
    conditional = gpd_survival(excess + 1) / max(gpd_survival(excess), 1e-10)

    # Our base model's implicit conditional (from sim statistics) is roughly
    # 0.35 per additional run once you're at 4+ (it's hard to keep scoring).
    base_conditional = 0.35

    # The ratio tells us how to adjust: if GPD says it's more likely than
    # our base model, boost hit probs; if less, suppress them.
    ratio = conditional / base_conditional
    return max(0.80, min(1.30, ratio))


def apply_extreme_scoring_mod(probs: list, runs_this_inning: int) -> list:
    """Apply GPD extreme scoring modifier when an inning reaches 4+ runs."""
    mod = gpd_extreme_inning_modifier(runs_this_inning)
    if 0.99 <= mod <= 1.01:
        return probs
    adjusted = list(probs)
    adjusted[2] *= mod  # single
    adjusted[3] *= mod  # double
    adjusted[4] *= mod  # triple
    adjusted[5] *= mod  # HR
    # Compensate outs inversely
    adjusted[6] *= 1.0 / max(mod, 0.5)
    total = sum(adjusted)
    return [p / total for p in adjusted]


# ── 20. GAUSSIAN MIXTURE MODEL FOR GAME SCRIPTS ─────────────────────────────
# Not all games follow the same scoring dynamics. A pitchers' duel between
# two aces plays out fundamentally differently from a Coors Field slugfest.
# Rather than using one set of probabilities, we model games as being drawn
# from a mixture of K latent "game script" archetypes.
#
# Each archetype has:
#   - A probability of occurring (mixing weight π_k)
#   - Mean total runs (μ_k) and standard deviation (σ_k)
#   - Probability adjustments to all batters (K rate, HR rate, etc.)
#
# The mixing weights are determined by the matchup:
#   - Two aces → higher weight on "pitchers_duel"
#   - Two bad pitchers at Coors → higher weight on "slugfest"
#   - Big mismatch → higher weight on "blowout"
#
# At the START of each simulation, we probabilistically select a game
# script, then run the entire game under that script's modifiers.
# This introduces realistic game-level variance that per-PA models miss.

GAME_SCRIPTS = {
    "pitchers_duel": {
        "total_runs_mean": 5.5,
        "total_runs_std": 2.0,
        "k_mult": 1.06,
        "bb_mult": 0.94,
        "hr_mult": 0.90,
        "hit_mult": 0.95,
    },
    "low_scoring": {
        "total_runs_mean": 7.0,
        "total_runs_std": 2.5,
        "k_mult": 1.03,
        "bb_mult": 0.97,
        "hr_mult": 0.95,
        "hit_mult": 0.98,
    },
    "normal": {
        "total_runs_mean": 9.0,
        "total_runs_std": 3.5,
        "k_mult": 1.00,
        "bb_mult": 1.00,
        "hr_mult": 1.00,
        "hit_mult": 1.00,
    },
    "high_scoring": {
        "total_runs_mean": 11.5,
        "total_runs_std": 3.8,
        "k_mult": 0.97,
        "bb_mult": 1.03,
        "hr_mult": 1.06,
        "hit_mult": 1.03,
    },
    "slugfest": {
        "total_runs_mean": 14.0,
        "total_runs_std": 4.5,
        "k_mult": 0.94,
        "bb_mult": 1.06,
        "hr_mult": 1.12,
        "hit_mult": 1.06,
    },
}


def compute_script_weights(away_pitcher: dict, home_pitcher: dict, venue: str = None) -> dict:
    """
    Compute mixing weights for each game script archetype based on
    the pitching matchup and venue.

    Uses pitcher ERA/FIP to estimate matchup quality, then assigns
    weights to each script via a softmax-like function.
    """
    # Estimate combined pitching quality
    try:
        away_era = float(away_pitcher.get("era", "4.00") or "4.00")
        away_fip = float(away_pitcher.get("fip", away_era) or away_era)
        home_era = float(home_pitcher.get("era", "4.00") or "4.00")
        home_fip = float(home_pitcher.get("fip", home_era) or home_era)
    except (ValueError, TypeError):
        away_era = away_fip = home_era = home_fip = 4.00

    avg_quality = (away_era * 0.4 + away_fip * 0.6 + home_era * 0.4 + home_fip * 0.6) / 2.0

    # Park factor boost for known hitter-friendly parks
    park_boost = 0.0
    if venue:
        hitter_parks = {
            "Coors Field": 1.5,
            "Great American Ball Park": 0.5,
            "Fenway Park": 0.4,
            "Globe Life Field": 0.3,
            "Citizens Bank Park": 0.3,
        }
        pitcher_parks = {
            "Oracle Park": -0.5,
            "Petco Park": -0.4,
            "Dodger Stadium": -0.3,
            "Tropicana Field": -0.3,
            "Oakland Coliseum": -0.4,
        }
        park_boost = hitter_parks.get(venue, pitcher_parks.get(venue, 0.0))

    # Compute script scores (unnormalized log-probabilities)
    # Lower avg_quality (better pitching) → more weight on pitchers_duel
    # Higher avg_quality (worse pitching) → more weight on slugfest
    quality_centered = avg_quality - 4.00 + park_boost * 0.5  # 0 = average

    scores = {
        "pitchers_duel": max(0.01, 0.15 - quality_centered * 0.12),
        "low_scoring": max(0.01, 0.22 - quality_centered * 0.06),
        "normal": 0.35,
        "high_scoring": max(0.01, 0.18 + quality_centered * 0.06),
        "slugfest": max(0.01, 0.10 + quality_centered * 0.10),
    }

    total = sum(scores.values())
    return {k: v / total for k, v in scores.items()}


def select_game_script(weights: dict) -> str:
    """Probabilistically select a game script from the mixture weights."""
    r = random.random()
    cumulative = 0.0
    for script, weight in weights.items():
        cumulative += weight
        if r <= cumulative:
            return script
    return "normal"


def apply_game_script(precomp: list, script: str) -> list:
    """
    Apply a game script's probability modifiers to all precomputed batters.
    Blends cumulative weight arrays based on the script's K/BB/HR/hit multipliers.
    """
    s = GAME_SCRIPTS.get(script)
    if s is None or script == "normal":
        return precomp

    k_m = s["k_mult"]
    bb_m = s["bb_mult"]
    hr_m = s["hr_mult"]
    hit_m = s["hit_mult"]

    # If all multipliers are ~1.0, skip
    if all(0.99 <= m <= 1.01 for m in [k_m, bb_m, hr_m, hit_m]):
        return precomp

    result = []
    for batter_cum in precomp:
        # Convert cumulative → raw probs
        raw = [batter_cum[0]] + [
            batter_cum[j] - batter_cum[j - 1] for j in range(1, len(batter_cum))
        ]
        # Apply multipliers: [BB, K, 1B, 2B, 3B, HR, OUT]
        raw[0] *= bb_m
        raw[1] *= k_m
        raw[2] *= hit_m
        raw[3] *= hit_m
        raw[4] *= hit_m
        raw[5] *= hr_m
        total = sum(raw)
        raw = [p / total for p in raw]
        # Convert back to cumulative
        cum = list(accumulate(raw))
        result.append(cum)
    return result


# ── Run expectancy (RE24) base-out state table ────────────────────────────────
# Expected runs from each base-out state (2020-2024 MLB average).
# Used to weight strategic decisions like sac fly and stolen base thresholds.
# Format: (runners_tuple, outs) → expected_runs
# runners_tuple: (b1, b2, b3) booleans
RE24 = {
    # 0 outs
    (False, False, False, 0): 0.50,
    (True, False, False, 0): 0.87,
    (False, True, False, 0): 1.10,
    (True, True, False, 0): 1.45,
    (False, False, True, 0): 1.35,
    (True, False, True, 0): 1.78,
    (False, True, True, 0): 1.97,
    (True, True, True, 0): 2.28,
    # 1 out
    (False, False, False, 1): 0.27,
    (True, False, False, 1): 0.53,
    (False, True, False, 1): 0.67,
    (True, True, False, 1): 0.92,
    (False, False, True, 1): 0.92,
    (True, False, True, 1): 1.12,
    (False, True, True, 1): 1.38,
    (True, True, True, 1): 1.55,
    # 2 outs
    (False, False, False, 2): 0.10,
    (True, False, False, 2): 0.22,
    (False, True, False, 2): 0.32,
    (True, True, False, 2): 0.44,
    (False, False, True, 2): 0.36,
    (True, False, True, 2): 0.49,
    (False, True, True, 2): 0.57,
    (True, True, True, 2): 0.75,
}


def get_run_expectancy(b1: bool, b2: bool, b3: bool, outs: int) -> float:
    """Return expected runs for this base-out state."""
    return RE24.get((b1, b2, b3, outs), 0.50)


# ── Pitch-count fatigue within a game ────────────────────────────────────────
# As a starter's pitch count climbs, velocity drops and command fades.
# Observed MLB thresholds (from pitch-by-pitch Statcast data):
#   < 60 pitches:  fresh — no penalty
#   60-79:  mild fatigue — slight contact quality increase
#   80-99:  meaningful fade — K% drops, walks tick up
#   100+:   clearly laboring — managers usually pull here
#
# We model pitch count based on estimated pitches per plate appearance (~3.8)
# and apply a cumulative multiplier to the starter's probs each half-inning.
PITCHES_PER_PA = 3.8  # MLB average pitches per plate appearance

# pitch_count → (k_mod, bb_mod, hit_mod)
# hit_mod boosts all hit outcomes; k_mod reduces Ks; bb_mod increases BBs
PITCH_COUNT_FATIGUE = [
    (0, 1.00, 1.00, 1.00),  # 0-59: fresh
    (60, 0.97, 1.03, 1.02),  # 60-79: mild fade
    (80, 0.93, 1.07, 1.05),  # 80-99: real fade
    (100, 0.88, 1.12, 1.09),  # 100+: clearly laboring
]


def apply_pitch_count_fatigue(probs: list, estimated_pitches: float) -> list:
    """
    Degrade starter quality as estimated pitch count rises.
    Returns adjusted probs with more contact/walks, fewer Ks.

    OUTCOME_ORDER = [WALK, K, 1B, 2B, 3B, HR, OUT]
                       0   1   2   3   4   5   6
    """
    k_mod = bb_mod = hit_mod = 1.0
    for threshold, km, bm, hm in reversed(PITCH_COUNT_FATIGUE):
        if estimated_pitches >= threshold:
            k_mod, bb_mod, hit_mod = km, bm, hm
            break
    if k_mod == 1.0 and bb_mod == 1.0:
        return probs
    adjusted = list(probs)
    adjusted[0] *= bb_mod  # walk
    adjusted[1] *= k_mod  # strikeout
    adjusted[2] *= hit_mod  # single
    adjusted[3] *= hit_mod  # double
    adjusted[4] *= hit_mod  # triple
    adjusted[5] *= hit_mod  # HR
    total = sum(adjusted)
    return [p / total for p in adjusted]


def simulate_half_inning(
    precomp_lineup: list,
    lineup_pos: int,
    risp_precomp: list = None,
    batter_rates: list = None,
    inning: int = 4,
    run_diff: int = 0,
    copula: "GaussianCopula | None" = None,
    pitcher_hmm: "PitcherHMM | None" = None,
) -> tuple:
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
    inning:         0-indexed inning number (for leverage calculations)
    run_diff:       score difference from batting team's perspective (+ = leading)

    Returns: (runs_scored, new_lineup_pos)
    The lineup_pos return value lets the next inning pick up where this one left off.
    """
    outs = 0
    runs = 0
    b1 = b2 = b3 = False  # nobody on base
    # Track which lineup slot is on each base (for speed grades)
    b1_who = b2_who = b3_who = -1
    pos = lineup_pos

    # Reset copula correlation at inning start
    if copula:
        copula.reset()

    def _speed(slot):
        if batter_rates and 0 <= slot < len(batter_rates):
            return batter_rates[slot].get("speed", "avg")
        return "avg"

    while outs < 3:
        # ── Stolen base attempt (between at-bats, runner on 1st only) ────────
        if batter_rates and b1 and not b2 and outs < 2:
            runner_idx = (pos - 1) % 9
            r = batter_rates[runner_idx]
            if random.random() < r["sb_rate"]:
                re_current = get_run_expectancy(b1, b2, b3, outs)
                re_success = get_run_expectancy(False, True, b3, outs)
                re_caught = get_run_expectancy(False, b2, b3, outs + 1)
                re_gain = re_success - re_current
                re_loss = re_current - re_caught
                denom = re_gain + re_loss
                break_even = (re_loss / denom) if denom > 0 else 0.72
                if r["sb_success"] >= break_even:
                    if random.random() < r["sb_success"]:
                        b2, b1 = True, False
                        b2_who, b1_who = b1_who, -1
                    else:
                        b1 = False
                        b1_who = -1
                        outs += 1
                        if outs >= 3:
                            break

        # ── Select probability array ──────────────────────────────────────────
        if risp_precomp and (b2 or b3):
            cum_weights = risp_precomp[pos % 9]
        else:
            cum_weights = precomp_lineup[pos % 9]

        # ── Situational hitting modifier (dynamic, per-PA) ───────────────────
        sit_key = get_situation_key(outs, b2, b3, inning, run_diff)
        if sit_key:
            cw = cum_weights
            raw = [cw[0]] + [cw[j] - cw[j - 1] for j in range(1, len(cw))]
            raw = apply_situational_mod(raw, sit_key)
            cum_weights = list(accumulate(raw))

        # ── Leverage index modifier (dynamic, per-PA) ────────────────────────
        li = compute_leverage_index(outs, b1, b2, inning, run_diff)
        if li < 0.9 or li > 1.1:
            cw = cum_weights
            raw = [cw[0]] + [cw[j] - cw[j - 1] for j in range(1, len(cw))]
            raw = apply_leverage_modifier(raw, li)
            cum_weights = list(accumulate(raw))

        # ── HMM pitcher state adjustment (dynamic, per-PA) ───────────────────
        if pitcher_hmm:
            hmm_adj = pitcher_hmm.get_state_adjustment()
            if any(abs(v - 1.0) > 0.005 for v in hmm_adj.values()):
                cw = cum_weights
                raw = [cw[0]] + [cw[j] - cw[j - 1] for j in range(1, len(cw))]
                raw = apply_hmm_adjustment(raw, hmm_adj)
                cum_weights = list(accumulate(raw))

        # ── GPD extreme scoring modifier (4+ runs this inning) ───────────────
        if runs >= GPD_THRESHOLD:
            cw = cum_weights
            raw = [cw[0]] + [cw[j] - cw[j - 1] for j in range(1, len(cw))]
            raw = apply_extreme_scoring_mod(raw, runs)
            cum_weights = list(accumulate(raw))

        # ── Copula correlation: set rho based on runners ─────────────────────
        if copula:
            copula.set_correlation(runners_on=(b1 or b2 or b3))
            outcome = simulate_at_bat_copula(cum_weights, copula)
        else:
            outcome = simulate_at_bat(cum_weights)

        # ── Feed outcome to HMM for Bayesian state update ────────────────────
        if pitcher_hmm:
            pitcher_hmm.observe(outcome)

        if outcome in (STRIKEOUT, OUT):
            if outcome == OUT and batter_rates and outs < 2:
                batter_idx = pos % 9

                if b1:
                    if random.random() < batter_rates[batter_idx]["gdp_rate"]:
                        outs += 2
                        b1 = False
                        b1_who = -1
                        pos += 1
                        continue

                if b3:
                    if random.random() < batter_rates[batter_idx]["sf_rate"]:
                        runs += 1
                        b3 = False
                        b3_who = -1
                        outs += 1
                        pos += 1
                        continue

            if outcome == OUT and random.random() < ERROR_RATE:
                sp_b1 = _speed(b1_who) if b1 else "avg"
                sp_b2 = _speed(b2_who) if b2 else "avg"
                sp_b3 = _speed(b3_who) if b3 else "avg"
                b1, b2, b3, new_runs = advance_runners(b1, b2, b3, SINGLE, sp_b1, sp_b2, sp_b3)
                runs += new_runs
                # Batter reached on error → track on 1st
                b1_who = pos % 9
                pos += 1
                continue

            outs += 1

            if (b1 or b2 or b3) and random.random() < WILD_PITCH_RATE:
                if b3:
                    runs += 1
                b3, b3_who = b2, b2_who
                b2, b2_who = b1, b1_who
                b1, b1_who = False, -1

        else:
            if (b1 or b2 or b3) and random.random() < WILD_PITCH_RATE:
                if b3:
                    runs += 1
                b3, b3_who = b2, b2_who
                b2, b2_who = b1, b1_who
                b1, b1_who = False, -1

            sp_b1 = _speed(b1_who) if b1 else "avg"
            sp_b2 = _speed(b2_who) if b2 else "avg"
            sp_b3 = _speed(b3_who) if b3 else "avg"
            b1, b2, b3, new_runs = advance_runners(b1, b2, b3, outcome, sp_b1, sp_b2, sp_b3)
            runs += new_runs

            # Track who's on base after the play
            batter_slot = pos % 9
            if outcome == SINGLE:
                b1_who = batter_slot
            elif outcome == DOUBLE:
                b2_who = batter_slot
            elif outcome == TRIPLE:
                b1_who = b2_who = -1
                b3_who = batter_slot
            elif outcome == HR:
                b1_who = b2_who = b3_who = -1
            elif outcome == WALK:
                if not b2 and not b3:
                    b1_who = batter_slot
                elif b1 and not b3:
                    b1_who = batter_slot

        pos += 1

    return runs, pos % 9


# ── STEP 5: Simulate one full game ──────────────────────────────


def simulate_game(
    away_precomp_early: list,  # innings 1-2: fresh starter
    home_precomp_early: list,
    away_precomp_mid: list = None,  # innings 3-5: starter showing fatigue
    home_precomp_mid: list = None,
    away_precomp_late: list = None,  # innings 6-7: middle relievers
    home_precomp_late: list = None,
    away_precomp_risp: list = None,  # RISP probs for away batters (b2 or b3 occupied)
    home_precomp_risp: list = None,  # RISP probs for home batters
    away_batter_rates: list = None,  # per-batter GIDP/SF/SB rates for away batters
    home_batter_rates: list = None,  # per-batter GIDP/SF/SB rates for home batters
    away_precomp_late2: list = None,  # innings 8-9: closer/setup tier
    home_precomp_late2: list = None,
    innings: int = 9,
    bullpen_start: int = 5,  # 0-indexed: inning 6 in human terms
    closer_start: int = 7,  # 0-indexed: inning 8 in human terms
    mid_start: int = 2,  # 0-indexed: inning 3 in human terms
    resolve_ties: bool = True,  # False = return tied score as-is (for F3/F5/F7 push calc)
    away_pitcher_stats: dict = None,  # away starter stats (for stamina durability)
    home_pitcher_stats: dict = None,  # home starter stats (for stamina durability)
) -> tuple:
    """
    Simulate one complete 9-inning game with 3 pitching phases + RISP clutch stats:

      Phase 1 (innings 1-2):   Starter is fresh. Uses early probs.
      Phase 2 (innings 3-5):   Starter tires. Uses mid probs (ERA/WHIP scaled up).
      Phase 3 (innings 6-9+):  Bullpen takes over. Uses late probs.

      Dynamic starter removal: if the starter gets hammered early, the bullpen
      comes in sooner. If the starter is dominant, he may go deeper. This better
      captures the real distribution of game scripts — bad starts pull the pen
      early; gems suppress it until very late.

      RISP: whenever a runner reaches 2nd or 3rd, the current batter's probs
            switch to their RISP (runners in scoring position) stats. Clutch
            hitters get a boost; weak RISP batters get a penalty.

    Returns: (away_runs, home_runs)
    """
    away_runs = 0
    home_runs = 0
    away_pos = 0
    home_pos = 0

    # Bayesian in-game pitcher quality tracking
    away_pitcher_state = InGamePitcherState(away_pitcher_stats)
    home_pitcher_state = InGamePitcherState(home_pitcher_stats)

    # Hidden Markov Model for pitcher latent states
    away_hmm = PitcherHMM(away_pitcher_stats)
    home_hmm = PitcherHMM(home_pitcher_stats)

    # Gaussian copula for intra-inning PA correlation
    away_copula = GaussianCopula()
    home_copula = GaussianCopula()

    # Momentum tracker (hot-inning correlation)
    momentum = MomentumTracker()

    # Dynamic starter removal tracking.
    # "away" starter = the pitcher throwing for the away team → HOME batters face them
    # "home" starter = the pitcher throwing for the home team → AWAY batters face them
    #
    # away_starter_runs: runs HOME batters have scored off the AWAY starter
    # home_starter_runs: runs AWAY batters have scored off the HOME starter
    away_starter_runs = 0
    home_starter_runs = 0
    away_starter_pulled = False  # away team's SP removed → home_cur switches to late
    home_starter_pulled = False  # home team's SP removed → away_cur switches to late

    # TTO tracking: count batters faced by each starter.
    # away_starter faces HOME batters → home_batters_faced
    # home_starter faces AWAY batters → away_batters_faced
    away_batters_faced = 0  # batters HOME lineup has sent up vs away starter
    home_batters_faced = 0  # batters AWAY lineup has sent up vs home starter

    # Estimated pitch counts for each starter (used for in-game fatigue)
    away_pitch_count = 0.0  # pitches thrown by away starter to home batters
    home_pitch_count = 0.0  # pitches thrown by home starter to away batters

    def _should_pull_starter(runs_allowed: int, inning: int) -> bool:
        """
        Probabilistic starter removal based on runs allowed and game situation.
        Reflects real MLB manager decisions about when to go to the bullpen.

        Thresholds derived from 2022-2024 MLB starter removal patterns:
          - 5+ runs by end of inning 2: almost always pulled (~90%)
          - 4+ runs by end of inning 3: usually pulled (~75%)
          - 5+ runs by end of inning 4: almost certainly pulled (~90%)
          - 3+ runs by end of inning 5: likely pulled (~60%)
          Normal scheduled removal at inning 6 is handled by bullpen_start.
        """
        r = random.random()
        if inning == 1 and runs_allowed >= 5:
            return r < 0.90
        if inning == 2 and runs_allowed >= 4:
            return r < 0.75
        if inning == 3 and runs_allowed >= 5:
            return r < 0.90
        if inning == 4 and runs_allowed >= 3:
            return r < 0.60
        return False

    for inning in range(innings):
        # ── Check if away starter should be pulled (affects HOME batters) ──────
        if not away_starter_pulled:
            if inning >= bullpen_start:
                away_starter_pulled = bool(home_precomp_late)
            elif (
                inning > 1
                and home_precomp_late
                and _should_pull_starter(away_starter_runs, inning - 1)
            ):
                away_starter_pulled = True

        # ── Check if home starter should be pulled (affects AWAY batters) ──────
        if not home_starter_pulled:
            if inning >= bullpen_start:
                home_starter_pulled = bool(away_precomp_late)
            elif (
                inning > 1
                and away_precomp_late
                and _should_pull_starter(home_starter_runs, inning - 1)
            ):
                home_starter_pulled = True

        # ── Pick probs for AWAY batters (they face the HOME pitching staff) ─────
        if home_starter_pulled:
            if inning >= closer_start and away_precomp_late2:
                away_cur = away_precomp_late2  # innings 8-9: closer/setup tier
            elif away_precomp_late:
                away_cur = away_precomp_late  # innings 6-7: middle relievers
            else:
                away_cur = away_precomp_early
        elif inning >= mid_start and away_precomp_mid:
            away_cur = away_precomp_mid  # innings 3-5: tired home starter
        else:
            away_cur = away_precomp_early  # innings 1-2: fresh home starter

        # ── Pick probs for HOME batters (they face the AWAY pitching staff) ─────
        if away_starter_pulled:
            if inning >= closer_start and home_precomp_late2:
                home_cur = home_precomp_late2  # innings 8-9: closer/setup tier
            elif home_precomp_late:
                home_cur = home_precomp_late  # innings 6-7: middle relievers
            else:
                home_cur = home_precomp_early
        elif inning >= mid_start and home_precomp_mid:
            home_cur = home_precomp_mid  # innings 3-5: tired away starter
        else:
            home_cur = home_precomp_early  # innings 1-2: fresh away starter

        # ── Score-state leverage (innings 7+) ────────────────────────────────
        # In real MLB, when a team leads by 5+ runs in late innings, managers
        # deploy mop-up arms (worse ERA) rather than burning setup/closer.
        # The trailing team gets better relief arms to stop the bleeding.
        #
        # We model this by selecting between two precomputed versions of the
        # bullpen probs: the normal late probs (good pen) and a slightly
        # degraded version (mop-up arm = more hittable).
        #
        # Degradation: in a blowout the mop-up arm effectively adds ~0.5 ERA
        # worth of quality reduction. We approximate this by blending the late
        # probs 70% toward league-average offense (more hittable pitcher).
        # The trailing team's bullpen is NOT degraded (they're using their best).
        run_diff = away_runs - home_runs  # positive = away leading

        def _mopup(cur, factor=0.20):
            """Blend cumulative-weight arrays toward league avg (mop-up arm is more hittable)."""
            if not cur:
                return cur
            # Build league-avg cumulative weights once
            from itertools import accumulate as _acc

            lg_cum = list(_acc(LEAGUE_AVG_PROBS))
            result = []
            for batter_cum in cur:
                blended = [c * (1 - factor) + lg * factor for c, lg in zip(batter_cum, lg_cum)]
                result.append(blended)
            return result

        if inning >= 6:  # inning 7+ (0-indexed)
            if run_diff >= 5:
                # Away team has big lead → home batters face away mop-up arm
                home_cur = _mopup(home_cur)
            elif run_diff <= -5:
                # Home team has big lead → away batters face home mop-up arm
                away_cur = _mopup(away_cur)
            elif abs(run_diff) <= 2:
                # High-leverage: close game in late innings → manager deploys
                # best available arm. We model this as the inverse of mop-up:
                # blend probs AWAY from league average (pitcher is more dominant).
                # Factor -0.10 means 10% better than normal bullpen quality.
                away_cur = _mopup(away_cur, factor=-0.10)
                home_cur = _mopup(home_cur, factor=-0.10)

        # Away team bats (facing home pitching staff)
        # Apply TTO + pitch-count fatigue while home starter is still in.
        # home_batters_faced tracks AWAY batters seen by the home starter.
        if not home_starter_pulled:
            away_tto = 1 + home_batters_faced // 9
            tto_away_cur = apply_tto(away_cur, away_tto)
            # Continuous stamina decay based on pitch count + pitcher durability
            if home_pitch_count >= 30:
                decay = compute_stamina_decay(home_pitch_count, home_pitcher_stats)
                from itertools import accumulate as _acc2

                tto_away_cur = [
                    list(
                        _acc2(
                            apply_stamina_decay(
                                [c - p for c, p in zip(bslot, [0] + list(bslot[:-1]))],
                                decay,
                            )
                        )
                    )
                    for bslot in tto_away_cur
                ]
        else:
            tto_away_cur = away_cur  # bullpen resets TTO and stamina decay

        # Apply momentum (hot-inning carry-over)
        away_mom = momentum.get_momentum(is_away=True)
        if away_mom > 0.005:
            tto_away_cur = apply_momentum(tto_away_cur, away_mom)

        # Apply Bayesian in-game pitcher adjustment (home starter quality)
        if not home_starter_pulled:
            p_adj = home_pitcher_state.get_adjustment()
            if abs(p_adj["k_mult"] - 1.0) > 0.005:
                from itertools import accumulate as _acc3

                tto_away_cur = [
                    list(
                        _acc3(
                            apply_ingame_pitcher_adj(
                                [c - p for c, p in zip(bs, [0] + list(bs[:-1]))],
                                p_adj,
                            )
                        )
                    )
                    for bs in tto_away_cur
                ]

        # run_diff from away team's perspective (batting team)
        away_rdiff = away_runs - home_runs
        # Away batters face the home pitcher → use home_hmm for HMM state
        runs, new_away_pos = simulate_half_inning(
            tto_away_cur,
            away_pos,
            away_precomp_risp,
            away_batter_rates,
            inning=inning,
            run_diff=away_rdiff,
            copula=away_copula,
            pitcher_hmm=home_hmm if not home_starter_pulled else None,
        )
        batters_this_half = (new_away_pos - away_pos) % len(away_cur) or len(away_cur)
        if not home_starter_pulled:
            home_starter_runs += runs
            home_batters_faced += batters_this_half
            home_pitch_count += batters_this_half * PITCHES_PER_PA
            # Bayesian update: each batter faced is an observation
            for _ in range(batters_this_half):
                home_pitcher_state.update(outcome_good=(runs == 0))
            if runs > 0:
                for _ in range(runs):
                    home_pitcher_state.record_run()
        away_pos = new_away_pos
        away_runs += runs

        # Update momentum after away half-inning
        momentum.update_after_half(is_away=True, runs_scored=runs)

        # Walk-off rule: home team wins in 9th without finishing if already ahead
        if inning == innings - 1 and home_runs > away_runs:
            break

        # Home team bats (facing away pitching staff)
        # Apply TTO + pitch-count fatigue while away starter is still in.
        # away_batters_faced tracks HOME batters seen by the away starter.
        if not away_starter_pulled:
            home_tto = 1 + away_batters_faced // 9
            tto_home_cur = apply_tto(home_cur, home_tto)
            if away_pitch_count >= 30:
                decay = compute_stamina_decay(away_pitch_count, away_pitcher_stats)
                from itertools import accumulate as _acc2

                tto_home_cur = [
                    list(
                        _acc2(
                            apply_stamina_decay(
                                [c - p for c, p in zip(bslot, [0] + list(bslot[:-1]))],
                                decay,
                            )
                        )
                    )
                    for bslot in tto_home_cur
                ]
        else:
            tto_home_cur = home_cur  # bullpen resets TTO and stamina decay

        # Apply momentum (hot-inning carry-over)
        home_mom = momentum.get_momentum(is_away=False)
        if home_mom > 0.005:
            tto_home_cur = apply_momentum(tto_home_cur, home_mom)

        # Apply Bayesian in-game pitcher adjustment (away starter quality)
        if not away_starter_pulled:
            p_adj = away_pitcher_state.get_adjustment()
            if abs(p_adj["k_mult"] - 1.0) > 0.005:
                from itertools import accumulate as _acc3

                tto_home_cur = [
                    list(
                        _acc3(
                            apply_ingame_pitcher_adj(
                                [c - p for c, p in zip(bs, [0] + list(bs[:-1]))],
                                p_adj,
                            )
                        )
                    )
                    for bs in tto_home_cur
                ]

        # run_diff from home team's perspective (batting team)
        home_rdiff = home_runs - away_runs
        # Home batters face the away pitcher → use away_hmm for HMM state
        runs, new_home_pos = simulate_half_inning(
            tto_home_cur,
            home_pos,
            home_precomp_risp,
            home_batter_rates,
            inning=inning,
            run_diff=home_rdiff,
            copula=home_copula,
            pitcher_hmm=away_hmm if not away_starter_pulled else None,
        )
        batters_this_half = (new_home_pos - home_pos) % len(home_cur) or len(home_cur)
        if not away_starter_pulled:
            away_starter_runs += runs
            away_batters_faced += batters_this_half
            away_pitch_count += batters_this_half * PITCHES_PER_PA
            for _ in range(batters_this_half):
                away_pitcher_state.update(outcome_good=(runs == 0))
            if runs > 0:
                for _ in range(runs):
                    away_pitcher_state.record_run()
        home_pos = new_home_pos
        home_runs += runs

        # Update momentum after home half-inning
        momentum.update_after_half(is_away=False, runs_scored=runs)

    # Extra innings if tied (up to 3 extra) — use bullpen probs
    # Skip if resolve_ties=False (segment bets: tied score = push)
    if resolve_ties:
        extra_away = away_precomp_late or away_precomp_mid or away_precomp_early
        extra_home = home_precomp_late or home_precomp_mid or home_precomp_early
        extra = 0
        while away_runs == home_runs and extra < 3:
            runs, away_pos = simulate_half_inning(
                extra_away, away_pos, away_precomp_risp, away_batter_rates
            )
            away_runs += runs
            runs, home_pos = simulate_half_inning(
                extra_home, home_pos, home_precomp_risp, home_batter_rates
            )
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
    away_team: str,
    home_team: str,
    away_lineup: list,  # batter stat dicts (L/R split) for away team
    home_lineup: list,  # batter stat dicts (L/R split) for home team
    away_pitcher: dict,  # away starting pitcher's season stats
    home_pitcher: dict,  # home starting pitcher's season stats
    weather: dict = None,  # ballpark weather
    away_recent: list = None,  # last-20-game stats for away batters
    home_recent: list = None,  # last-20-game stats for home batters
    away_rp_stats: list = None,  # away batters' stats vs relief pitchers (bullpen)
    home_rp_stats: list = None,  # home batters' stats vs relief pitchers (bullpen)
    away_bullpen: dict = None,  # away team's actual bullpen ERA/WHIP (for innings 6-9)
    home_bullpen: dict = None,  # home team's actual bullpen ERA/WHIP (for innings 6-9)
    venue: str = None,  # ballpark name for park factor adjustment
    away_pitcher_log: list = None,  # away starter's last 10 game logs (ip, er per start)
    home_pitcher_log: list = None,  # home starter's last 10 game logs
    away_rest_days: int = None,  # days since away starter's last start
    home_rest_days: int = None,  # days since home starter's last start
    away_risp_stats: list = None,  # away batters' RISP stats (runners on 2nd/3rd)
    home_risp_stats: list = None,  # home batters' RISP stats
    away_matchup_stats: list = None,  # away batters' career stats vs home starter
    home_matchup_stats: list = None,  # home batters' career stats vs away starter
    away_daynight_stats: list = None,  # away batters' day or night game splits
    home_daynight_stats: list = None,  # home batters' day or night game splits
    away_homeaway_stats: list = None,  # away batters' road (away) splits
    home_homeaway_stats: list = None,  # home batters' home splits
    away_batter_rest: int = None,
    home_batter_rest: int = None,
    away_savant: list = None,  # Statcast data per away batter
    home_savant: list = None,  # Statcast data per home batter
    umpire_name: str = None,  # home plate umpire for K/BB modifier
    away_bp_fatigue: dict = None,  # bullpen fatigue {era_modifier, whip_modifier, fatigue}
    home_bp_fatigue: dict = None,  # bullpen fatigue for home team
    series_game_number: int = 1,  # game number in current series (1=opener, 2/3/4=familiarity boost)
    away_catcher_cs: float = 0.28,  # opposing (home) catcher caught-stealing rate
    home_catcher_cs: float = 0.28,  # opposing (away) catcher caught-stealing rate
    away_bp_depth: dict = None,  # bullpen depth score for away team
    home_bp_depth: dict = None,  # bullpen depth score for home team
    n: int = 100_000,
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

    # ── Series familiarity: batters improve vs a pitcher they've seen ────────
    # In game 2+ of a series, batters have already seen the starter's stuff.
    # Research shows ~5% wOBA boost in game 2, ~9% in game 3+.
    # We model this by blending pitcher stats slightly toward league average
    # (the pitcher appears "easier" because batters have scouted him live).
    #
    # series_game_number: 1 = series opener, 2 = second game, 3+ = third+
    # We degrade the pitcher ERA/FIP to simulate batter familiarity advantage.
    _SERIES_ERA_BUMP = {1: 0.0, 2: 0.15, 3: 0.28}  # ERA added to starter
    _era_bump = _SERIES_ERA_BUMP.get(series_game_number, 0.28)

    def _apply_series_familiarity(pitcher):
        if _era_bump == 0 or not pitcher:
            return pitcher
        p = dict(pitcher)
        try:
            p["era"] = str(round(float(p.get("era", 4.20)) + _era_bump, 2))
            p["fip"] = str(round(float(p.get("fip", 4.00)) + _era_bump * 0.8, 2))
        except (ValueError, TypeError):
            pass
        return p

    away_pitcher = _apply_series_familiarity(away_pitcher)
    home_pitcher = _apply_series_familiarity(home_pitcher)

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
    # away batters face home pitcher → opp catcher is home team's catcher (home_catcher_cs)
    # home batters face away pitcher → opp catcher is away team's catcher (away_catcher_cs)
    away_precomp_early = precompute_lineup(
        away_lineup,
        home_pitcher,
        weather,
        away_recent,
        venue,
        away_matchup_stats,
        away_daynight_stats,
        away_batter_rest,
        away_savant,
        umpire_name,
        away_homeaway_stats,
        opp_catcher_cs=home_catcher_cs,
    )
    home_precomp_early = precompute_lineup(
        home_lineup,
        away_pitcher,
        weather,
        home_recent,
        venue,
        home_matchup_stats,
        home_daynight_stats,
        home_batter_rest,
        home_savant,
        umpire_name,
        home_homeaway_stats,
        opp_catcher_cs=away_catcher_cs,
    )

    # ── Phase 2: innings 3-5 (starter showing fatigue) ───────────────
    home_pitcher_mid = build_fatigued_pitcher_stats(home_pitcher, home_fatigue["phase2"])
    away_pitcher_mid = build_fatigued_pitcher_stats(away_pitcher, away_fatigue["phase2"])
    away_precomp_mid = precompute_lineup(
        away_lineup,
        home_pitcher_mid,
        weather,
        away_recent,
        venue,
        away_matchup_stats,
        away_daynight_stats,
        away_batter_rest,
        away_savant,
        umpire_name,
        away_homeaway_stats,
        opp_catcher_cs=home_catcher_cs,
    )
    home_precomp_mid = precompute_lineup(
        home_lineup,
        away_pitcher_mid,
        weather,
        home_recent,
        venue,
        home_matchup_stats,
        home_daynight_stats,
        home_batter_rest,
        home_savant,
        umpire_name,
        home_homeaway_stats,
        opp_catcher_cs=away_catcher_cs,
    )

    # ── RISP: clutch stats when runner on 2nd or 3rd ─────────────────
    away_precomp_risp = None
    home_precomp_risp = None

    if away_risp_stats and any(
        s for s in away_risp_stats if s and s.get("plateAppearances", 0) >= 20
    ):
        away_precomp_risp = precompute_lineup(
            away_risp_stats,
            home_pitcher,
            weather,
            away_recent,
            venue,
            away_matchup_stats,
            away_daynight_stats,
            away_batter_rest,
            away_savant,
            umpire_name,
            away_homeaway_stats,
            opp_catcher_cs=home_catcher_cs,
        )
    if home_risp_stats and any(
        s for s in home_risp_stats if s and s.get("plateAppearances", 0) >= 20
    ):
        home_precomp_risp = precompute_lineup(
            home_risp_stats,
            away_pitcher,
            weather,
            home_recent,
            venue,
            home_matchup_stats,
            home_daynight_stats,
            home_batter_rest,
            home_savant,
            umpire_name,
            home_homeaway_stats,
            opp_catcher_cs=away_catcher_cs,
        )

    # ── Late game: batters vs the bullpen (innings 6+) ───────────────
    # Use real team bullpen ERA/WHIP if available, otherwise league average.
    # A team with a 3.00 bullpen ERA gets a real edge in innings 6-9.
    def _apply_bp_fatigue(bp_stats, fatigue):
        """Scale bullpen ERA/WHIP up when the pen is tired."""
        if not fatigue or not bp_stats:
            return bp_stats
        era_mod = fatigue.get("era_modifier", 1.0)
        whip_mod = fatigue.get("whip_modifier", 1.0)
        result = dict(bp_stats)
        try:
            result["era"] = str(round(float(result.get("era", 4.20)) * era_mod, 2))
            result["whip"] = str(round(float(result.get("whip", 1.30)) * whip_mod, 3))
        except (ValueError, TypeError):
            pass
        return result

    away_bp_pitcher_raw = away_bullpen if away_bullpen else LEAGUE_AVG_PITCHER
    home_bp_pitcher_raw = home_bullpen if home_bullpen else LEAGUE_AVG_PITCHER
    away_bp_pitcher = _apply_bp_fatigue(away_bp_pitcher_raw, away_bp_fatigue)
    home_bp_pitcher = _apply_bp_fatigue(home_bp_pitcher_raw, home_bp_fatigue)

    # ── Bullpen depth adjustment ───────────────────────────────────────────
    # Teams with more elite arms (ERA < 3.00) can better navigate high-leverage
    # situations. We adjust the bullpen ERA used for the 8th-9th innings based
    # on how many elite arms the team has available.
    #
    # elite_arms: 0 → no adjustment, 1 → -0.15 ERA, 2 → -0.28 ERA, 3+ → -0.38 ERA
    # This stacks on top of the closer/setup tier adjustment from bullpen tiers.
    def _apply_depth_adjustment(bp_stats, depth_dict):
        if not depth_dict or not bp_stats:
            return bp_stats
        elite = depth_dict.get("elite_arms", 0)
        if elite == 0:
            return bp_stats
        era_reduction = min(0.38, elite * 0.15 - (elite - 1) * 0.02)
        result = dict(bp_stats)
        try:
            result["era"] = str(round(max(1.50, float(result.get("era", 4.20)) - era_reduction), 2))
        except (ValueError, TypeError):
            pass
        return result

    away_bp_pitcher = _apply_depth_adjustment(away_bp_pitcher, away_bp_depth)
    home_bp_pitcher = _apply_depth_adjustment(home_bp_pitcher, home_bp_depth)

    # ── Bullpen quality tiers ──────────────────────────────────────────────
    # A team's bullpen ERA is the average across all relievers. But in reality:
    #   Innings 6-7: middle relievers (worse than average — ~0.40 ERA higher)
    #   Innings 8-9: setup man + closer (better than average — ~0.35 ERA lower)
    #
    # We model this by building two pitcher profiles from the same bullpen ERA:
    #   early_pen = bullpen ERA + 0.40 (less skilled arms in the 6th/7th)
    #   late_pen  = bullpen ERA - 0.35 (closer/setup quality in the 8th/9th)
    #
    # simulate_game uses bullpen_start=5 (0-indexed inning 6). We split at
    # inning 8 (0-indexed 7) to switch from early to late pen.
    def _tier_bullpen(bp_stats, era_delta):
        """Return a copy of bp_stats with ERA adjusted by era_delta."""
        if not bp_stats:
            return bp_stats
        result = dict(bp_stats)
        try:
            base_era = float(result.get("era", 4.20))
            result["era"] = str(round(max(1.50, base_era + era_delta), 2))
        except (ValueError, TypeError):
            pass
        return result

    away_bp_early = _tier_bullpen(away_bp_pitcher, +0.40)  # middle relievers
    away_bp_late = _tier_bullpen(away_bp_pitcher, -0.35)  # closer/setup
    home_bp_early = _tier_bullpen(home_bp_pitcher, +0.40)
    home_bp_late = _tier_bullpen(home_bp_pitcher, -0.35)

    away_precomp_late = None
    away_precomp_late2 = None  # innings 8-9: closer/setup tier
    home_precomp_late = None
    home_precomp_late2 = None

    if away_rp_stats and any(s for s in away_rp_stats if s):
        away_precomp_late = precompute_lineup(
            away_rp_stats, home_bp_early, weather, away_recent, venue
        )
        away_precomp_late2 = precompute_lineup(
            away_rp_stats, home_bp_late, weather, away_recent, venue
        )
    if home_rp_stats and any(s for s in home_rp_stats if s):
        home_precomp_late = precompute_lineup(
            home_rp_stats, away_bp_early, weather, home_recent, venue
        )
        home_precomp_late2 = precompute_lineup(
            home_rp_stats, away_bp_late, weather, home_recent, venue
        )

    # ── Compute per-batter situational rates (GIDP, sac fly, stolen base) ──────
    # Derived from season stats already in memory — no extra API calls needed.
    # away_batter_rates describes away BATTERS (how they run, hit with runners on)
    # home_batter_rates describes home BATTERS
    away_batter_rates = compute_batter_rates(away_lineup)
    home_batter_rates = compute_batter_rates(home_lineup)

    # ── Run the simulation loop ───────────────────────────────────────
    away_wins = 0
    home_wins = 0
    away_total = 0
    home_total = 0
    away_cover = 0  # wins by 2+ (covers -1.5 run line)
    home_cover = 0
    away_cover_p15 = 0  # wins by 2+ (alt: +1.5, away must win outright OR lose by 1)
    home_cover_p15 = 0
    away_cover_m25 = 0  # wins by 3+ (alt: -2.5)
    home_cover_m25 = 0
    total_runs_hist = Counter()  # {total_runs: count} for O/U model
    score_pairs = Counter()  # {(away_runs, home_runs): count} for SGP correlated probs

    # Gaussian Mixture Model: compute game script mixing weights
    script_weights = compute_script_weights(away_pitcher, home_pitcher, venue)

    def _sim(**kw):
        # Select game script archetype for this simulation
        script = select_game_script(script_weights)

        # Apply game script modifiers to all precomputed lineups
        a_early = apply_game_script(away_precomp_early, script)
        h_early = apply_game_script(home_precomp_early, script)
        a_mid = apply_game_script(away_precomp_mid, script) if away_precomp_mid else None
        h_mid = apply_game_script(home_precomp_mid, script) if home_precomp_mid else None
        a_late = apply_game_script(away_precomp_late, script) if away_precomp_late else None
        h_late = apply_game_script(home_precomp_late, script) if home_precomp_late else None
        a_risp = apply_game_script(away_precomp_risp, script) if away_precomp_risp else None
        h_risp = apply_game_script(home_precomp_risp, script) if home_precomp_risp else None
        a_late2 = apply_game_script(away_precomp_late2, script) if away_precomp_late2 else None
        h_late2 = apply_game_script(home_precomp_late2, script) if home_precomp_late2 else None

        return simulate_game(
            a_early,
            h_early,
            a_mid,
            h_mid,
            a_late,
            h_late,
            a_risp,
            h_risp,
            away_batter_rates,
            home_batter_rates,
            a_late2,
            h_late2,
            **kw,
        )

    for _ in range(n):
        a, h = _sim(
            away_pitcher_stats=away_pitcher,
            home_pitcher_stats=home_pitcher,
        )
        away_total += a
        home_total += h
        total_runs_hist[a + h] += 1
        score_pairs[(a, h)] += 1
        if a > h:
            away_wins += 1
            if a - h >= 2:
                away_cover += 1
            if a - h >= 3:
                away_cover_m25 += 1
        elif a < h:
            home_wins += 1
            if h - a >= 2:
                home_cover += 1
            if h - a >= 3:
                home_cover_m25 += 1
        # +1.5: team covers if they win OR lose by exactly 1
        if a >= h - 1:
            away_cover_p15 += 1
        if h >= a - 1:
            home_cover_p15 += 1

    # ── F3 / F5 / F7 inning-segment simulations ──────────────────────
    # Simulate a smaller batch to get win probs through each inning cutoff.
    # Ties count as a push (neither team wins the bet), matching how F5
    # betting markets work.
    n_seg = max(n // 4, 10_000)
    seg_results = {}
    for label, inn in (("f3", 3), ("f5", 5), ("f7", 7)):
        aw = hw = at = ht = ties = 0
        for _ in range(n_seg):
            a, h = _sim(
                innings=inn,
                resolve_ties=False,
                away_pitcher_stats=away_pitcher,
                home_pitcher_stats=home_pitcher,
            )
            at += a
            ht += h
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
            "tie_pct": round(ties / n_seg * 100, 1),
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
    HOME_FIELD_BOOST = 2.0  # percentage points
    adj_home = raw_home_pct + HOME_FIELD_BOOST
    adj_away = raw_away_pct - HOME_FIELD_BOOST
    # Clamp so neither goes below 1% or above 99%
    adj_home = min(max(adj_home, 1.0), 99.0)
    adj_away = min(max(adj_away, 1.0), 99.0)
    # Re-normalize to exactly 100
    total = adj_away + adj_home
    adj_away = round(adj_away / total * 100, 1)
    adj_home = round(adj_home / total * 100, 1)

    # ── Summarize fatigue profile for display in the pitcher card ────────
    def _fatigue_label(log, fatigue):
        if not log:
            return "Typical", None
        avg_ip = sum(g["ip"] for g in log) / len(log)
        p2 = fatigue["phase2"]
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

    # ── Confidence score ─────────────────────────────────────────────
    # Distance of top win% from 50% → maps to 0-100 confidence scale
    top_pct = max(adj_away, adj_home)
    confidence_score = int(min(100, (top_pct - 50) / 30 * 100))  # 50% = 0, 80% = 100

    # Run total std dev from histogram
    hist = total_runs_hist
    avg_r = (away_total + home_total) / n
    if hist:
        variance = sum((k - avg_r) ** 2 * v for k, v in hist.items()) / n
        run_std = round(variance**0.5, 2)
    else:
        run_std = 3.0

    return {
        "away_team": away_team,
        "home_team": home_team,
        "simulations": n,
        "confidence_score": confidence_score,
        "run_std": run_std,
        "away_win_pct": adj_away,
        "home_win_pct": adj_home,
        "away_win_pct_raw": round(raw_away_pct, 1),  # pre-HFA, for display
        "home_win_pct_raw": round(raw_home_pct, 1),
        "away_avg_runs": round(away_total / n, 2),
        "home_avg_runs": round(home_total / n, 2),
        "uses_bullpen_data": away_precomp_late is not None,
        # Pitcher fatigue profile
        "away_fatigue_label": away_fatigue_label,
        "away_avg_ip": away_avg_ip,
        "away_avg_pitches": away_fatigue.get("avg_pitches"),
        "away_last_pitches": away_fatigue.get("last_pitches"),
        "away_pitch_carryover": away_fatigue.get("pitch_carryover", False),
        "home_fatigue_label": home_fatigue_label,
        "home_avg_ip": home_avg_ip,
        "home_avg_pitches": home_fatigue.get("avg_pitches"),
        "home_last_pitches": home_fatigue.get("last_pitches"),
        "home_pitch_carryover": home_fatigue.get("pitch_carryover", False),
        # Pitcher rest days
        "away_rest_days": away_rest_label,
        "away_rest_type": away_rest_type,
        "home_rest_days": home_rest_label,
        "home_rest_type": home_rest_type,
        # Run line coverage
        "away_cover_pct": round(away_cover / n * 100, 1),
        "home_cover_pct": round(home_cover / n * 100, 1),
        "away_cover_p15_pct": round(away_cover_p15 / n * 100, 1),
        "home_cover_p15_pct": round(home_cover_p15 / n * 100, 1),
        "away_cover_m25_pct": round(away_cover_m25 / n * 100, 1),
        "home_cover_m25_pct": round(home_cover_m25 / n * 100, 1),
        # Over/Under: full distribution so web layer can calc P(total > any line)
        "total_runs_hist": dict(total_runs_hist),
        "score_pairs": {str(k): v for k, v in score_pairs.items()},
        "run_distribution": compute_run_distribution(
            [k for k, cnt in score_pairs.items() for _ in range(cnt)]
        ),
        "avg_total_runs": round((away_total + home_total) / n, 2),
        # Inning-segment results (F3/F5/F7)
        "f3": seg_results["f3"],
        "f5": seg_results["f5"],
        "f7": seg_results["f7"],
    }


# ── Pitcher strikeout prop model ─────────────────────────────────────────────


def detect_pitcher_form(pitcher_log: list, pitcher_stats: dict) -> dict:
    """
    Compare a starter's last 2 starts K/BB/ERA vs their season average.
    Returns {"form": "declining"|"hot"|"stable", "note": str, "modifier": float}
    modifier < 1.0 = lower projection, > 1.0 = boost
    """
    if not pitcher_log or len(pitcher_log) < 2:
        return {
            "form": "stable",
            "note": "",
            "modifier": 1.0,
            "recent_whip": None,
            "k9_recent": None,
            "bb9_recent": None,
        }

    # Season averages
    ip_season = float(pitcher_stats.get("inningsPitched", 1) or 1)
    k_season = int(pitcher_stats.get("strikeOuts", 0) or 0)
    bb_season = int(pitcher_stats.get("baseOnBalls", 0) or 0)
    er_season = int(pitcher_stats.get("earnedRuns", 0) or 0)
    k9_season = k_season / ip_season * 9 if ip_season > 0 else 7.0
    bb9_season = bb_season / ip_season * 9 if ip_season > 0 else 3.0
    era_season = er_season / ip_season * 9 if ip_season > 0 else 4.20

    # Last 2 starts
    recent = pitcher_log[-2:]
    recent_ip = sum(float(g.get("inningsPitched", 0) or 0) for g in recent)
    recent_k = sum(int(g.get("strikeOuts", 0) or 0) for g in recent)
    recent_bb = sum(int(g.get("baseOnBalls", 0) or 0) for g in recent)
    recent_er = sum(int(g.get("earnedRuns", 0) or 0) for g in recent)

    if recent_ip < 3:
        return {"form": "stable", "note": "", "modifier": 1.0}

    k9_recent = recent_k / recent_ip * 9
    bb9_recent = recent_bb / recent_ip * 9
    era_recent = recent_er / recent_ip * 9

    # Score change
    k_drop = k9_season - k9_recent  # positive = losing Ks
    bb_rise = bb9_recent - bb9_season  # positive = walking more
    era_rise = era_recent - era_season  # positive = ERA spiking

    if k_drop > 2.0 or bb_rise > 1.5 or era_rise > 2.0:
        note = []
        if k_drop > 2.0:
            note.append(f"K/9 down {k_drop:.1f}")
        if bb_rise > 1.5:
            note.append(f"BB/9 up {bb_rise:.1f}")
        if era_rise > 2.0:
            note.append(f"ERA up {era_rise:.1f}")
        modifier = max(0.85, 1.0 - (k_drop * 0.02 + bb_rise * 0.02 + era_rise * 0.01))
        recent_whip = round((recent_bb + recent_er * 0.6) / recent_ip, 2) if recent_ip > 0 else None
        return {
            "form": "declining",
            "note": ", ".join(note),
            "modifier": round(modifier, 3),
            "recent_whip": recent_whip,
            "k9_recent": round(k9_recent, 1),
            "bb9_recent": round(bb9_recent, 1),
        }
    elif k9_recent > k9_season + 1.5 and era_recent < era_season - 1.0:
        recent_whip = round((recent_bb + recent_er * 0.6) / recent_ip, 2) if recent_ip > 0 else None
        return {
            "form": "hot",
            "note": f"K/9 up {k9_recent - k9_season:.1f}, ERA down {era_season - era_recent:.1f}",
            "modifier": min(1.12, 1.0 + (k9_recent - k9_season) * 0.015),
            "recent_whip": recent_whip,
            "k9_recent": round(k9_recent, 1),
            "bb9_recent": round(bb9_recent, 1),
        }
    return {"form": "stable", "note": "", "modifier": 1.0}


def predict_batter_props(
    lineup_stats: list,
    pitcher_stats: dict,
    weather: dict = None,
    venue: str = None,
    batting_slots: int = 9,
    recent_stats_list: list = None,
    umpire_name: str = None,
    opp_catcher_cs: float = None,
) -> list:
    """
    Project per-batter prop lines (hits, HR, RBI, total bases) for one team.

    Uses the same outcome probabilities as the simulation engine so predictions
    are consistent with the win/run totals.

    Returns a list of dicts (one per batter slot, in batting order):
      {
        "slot":          int,
        "avg_hits":      float,  # expected hits per game
        "avg_hr":        float,  # expected HR per game
        "avg_rbi":       float,  # expected RBI per game
        "avg_tb":        float,  # expected total bases per game
        "avg_walks":     float,  # expected walks per game
        "hit_pct":       float,  # P(at least 1 hit)
        "hr_pct":        float,  # P(at least 1 HR)
        "multi_hit_pct": float,  # P(2+ hits)
        "tb_pct_2plus":  float,  # P(2+ total bases)
      }
    """
    import math

    # Expected PAs per slot — leadoff gets most, 9-hole gets fewest.
    # TTO-adjusted: early order batters face the starter more (worse ERA);
    # later batters may catch the bullpen. We apply a TTO boost to slots 1-3
    # since they're most likely to face the starter a 2nd+ time.
    PA_BY_SLOT = [4.8, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.6]
    # Slots 1-3 will see TTO 2 by inning 5-6 — apply small PA-weighted boost
    TTO_PA_BOOST = [0.06, 0.05, 0.04, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0]

    park = BALLPARK_FACTORS.get(venue, {}) if venue else {}
    hr_park = park.get("hr", 1.0)
    hit_park = park.get("hit", 1.0)

    # RBI model using RE24-derived runner-on-base rates by slot.
    # Slots behind good OBP hitters see more runners on base.
    # Approximated from MLB lineup simulation data:
    #   Slot 1 (leadoff):  fewest runners on (hits leadoff), ~20% ROB
    #   Slot 3-5 (heart):  most runners on, ~35-38% ROB
    #   Slot 8-9:          fewer runners, ~25% ROB
    ROB_BY_SLOT = [0.20, 0.28, 0.35, 0.38, 0.37, 0.33, 0.30, 0.27, 0.24]

    results = []
    for i, stats in enumerate(lineup_stats[:9]):
        slot = i + 1
        pa_base = PA_BY_SLOT[i] if i < len(PA_BY_SLOT) else 3.8
        tto_boost = TTO_PA_BOOST[i] if i < len(TTO_PA_BOOST) else 0.0
        pa_expected = pa_base + tto_boost
        rob_rate = ROB_BY_SLOT[i] if i < len(ROB_BY_SLOT) else 0.30

        # Build base outcome probs with same pipeline as the sim engine
        probs = build_batter_probs(stats)

        # Blend recent form if available
        if recent_stats_list and i < len(recent_stats_list):
            recent = recent_stats_list[i]
            if recent and recent.get("plateAppearances", 0) >= 10:
                recent_probs = build_batter_probs(recent)
                pa_scale = min(recent.get("plateAppearances", 0) / 20.0, 1.0)
                weight = adaptive_recent_weight(probs, recent_probs) * pa_scale
                probs = blend_probs(probs, recent_probs, recent_weight=weight)

        # Apply pitcher modifier
        probs = apply_pitcher_modifier(probs, pitcher_stats)

        # Umpire zone shift
        if umpire_name:
            probs = apply_umpire_modifier(probs, umpire_name)

        # Catcher framing
        if opp_catcher_cs is not None:
            probs = apply_catcher_framing(probs, opp_catcher_cs)

        # Apply weather
        if weather:
            probs = apply_weather_modifier(probs, weather)

        # Apply park factor
        adj = list(probs)
        adj[5] *= hr_park
        adj[2] *= hit_park
        adj[3] *= hit_park
        total = sum(adj)
        probs = [p / total for p in adj]

        # wOBA clamp — keep within realistic range
        probs = clamp_woba(probs)

        # OUTCOME_ORDER = [WALK, K, 1B, 2B, 3B, HR, OUT]
        p_walk = probs[0]
        p_1b = probs[2]
        p_2b = probs[3]
        p_3b = probs[4]
        p_hr = probs[5]
        p_hit = p_1b + p_2b + p_3b + p_hr

        # Expected per PA
        exp_hits_pa = p_hit
        exp_hr_pa = p_hr
        exp_tb_pa = p_1b * 1 + p_2b * 2 + p_3b * 3 + p_hr * 4
        exp_walk_pa = p_walk

        # Scale to expected PAs
        avg_hits = exp_hits_pa * pa_expected
        avg_hr = exp_hr_pa * pa_expected
        avg_tb = exp_tb_pa * pa_expected
        avg_walks = exp_walk_pa * pa_expected

        # RBI model using slot-specific runner-on-base rate.
        # Expected RBI = HR * (1 + runners_on_at_HR) + non-HR hits * P(runner scores)
        # runners_on_at_HR ≈ rob_rate * 2.3 (avg runners on per base-occupied)
        # non-HR RBI: hit with runner on → runner scores ~62% on single, ~87% on double
        avg_runners_on = rob_rate * 2.1  # avg runners on base (weighted by occupancy)
        rbi_from_hr = avg_hr * (1.0 + avg_runners_on * 0.90)  # 90% of runners score on HR
        rbi_from_1b = (p_1b * pa_expected) * rob_rate * 0.35  # single scores ~35% of ROB
        rbi_from_2b = (p_2b * pa_expected) * rob_rate * 0.65  # double scores ~65% of ROB
        rbi_from_3b = (p_3b * pa_expected) * rob_rate * 0.90  # triple scores ~90% of ROB
        avg_rbi = rbi_from_hr + rbi_from_1b + rbi_from_2b + rbi_from_3b

        # P(at least 1 hit)
        hit_pct = round((1 - (1 - p_hit) ** pa_expected) * 100, 1)

        # P(at least 1 HR)
        hr_pct = round((1 - (1 - p_hr) ** pa_expected) * 100, 1)

        # P(2+ hits) using Poisson approximation
        lam = avg_hits
        p_0 = math.exp(-lam)
        p_1 = lam * math.exp(-lam)
        multi_hit_pct = round(max(0.0, 1 - p_0 - p_1) * 100, 1)

        # P(2+ total bases) — useful for TB props
        lam_tb = avg_tb
        p_tb0 = math.exp(-lam_tb)
        p_tb1 = lam_tb * math.exp(-lam_tb)
        tb_pct_2plus = round(max(0.0, 1 - p_tb0 - p_tb1) * 100, 1)

        # Runs scored model — how often does this batter cross home plate?
        # A batter scores by: getting on base AND being driven home by someone else.
        # Simplified: P(score | reach base) × P(reach base per PA) × PAs
        # P(reach base) = p_hit + p_walk
        # P(score | reach base) depends on slot — leadoff has most subsequent batters
        # and more innings, cleanup has fewer but more dangerous lineup behind them.
        SCORE_RATE_BY_SLOT = [0.42, 0.40, 0.38, 0.36, 0.34, 0.30, 0.27, 0.24, 0.22]
        score_rate = SCORE_RATE_BY_SLOT[i] if i < len(SCORE_RATE_BY_SLOT) else 0.30
        p_reach = p_hit + p_walk
        avg_runs = p_reach * pa_expected * score_rate

        results.append(
            {
                "slot": slot,
                "avg_hits": round(avg_hits, 2),
                "avg_hr": round(avg_hr, 3),
                "avg_rbi": round(avg_rbi, 2),
                "avg_tb": round(avg_tb, 2),
                "avg_walks": round(avg_walks, 2),
                "avg_runs": round(avg_runs, 2),
                "hit_pct": hit_pct,
                "hr_pct": hr_pct,
                "multi_hit_pct": multi_hit_pct,
                "tb_pct_2plus": tb_pct_2plus,
            }
        )

    return results


def predict_pitcher_ks(
    pitcher_stats: dict,
    lineup_stats: list,
    umpire_name: str = None,
    pitcher_log: list = None,
    fatigue: dict = None,
) -> dict:
    """
    Predict how many strikeouts a starting pitcher will record.

    Formula:
      1. Pitcher K/9 from season stats
      2. Opposing lineup K rate (how often each batter strikes out)
      3. Umpire K factor (wide-zone umps add Ks)
      4. Expected innings (from recent game log avg IP, capped at 6.0)
      5. Fatigue: short-rest pitchers tend to go fewer innings

    Returns:
      {
        "model_k":        float,   # predicted strikeout total
        "model_k_low":    float,   # 16th percentile (roughly -1 std)
        "model_k_high":   float,   # 84th percentile (roughly +1 std)
        "expected_ip":    float,
      }
    """
    # ── Pitcher K/9 ────────────────────────────────────────────────
    ip_season = float(pitcher_stats.get("inningsPitched", 0) or 0)
    k_season = int(pitcher_stats.get("strikeOuts", 0) or 0)
    if ip_season >= 10:
        k_per_9 = k_season / ip_season * 9
    else:
        k_per_9 = 8.5  # league average starter K/9

    # ── Opposing lineup K rate ────────────────────────────────────
    opp_k_rates = []
    for s in lineup_stats:
        pa = s.get("plateAppearances", 0) or 0
        ks = s.get("strikeOuts", 0) or 0
        if pa >= 20:
            opp_k_rates.append(ks / pa)
    opp_k_rate = sum(opp_k_rates) / len(opp_k_rates) if opp_k_rates else 0.224  # lg avg

    # Blend: pitcher K rate adjusted for opponent K tendency
    # League avg batter K rate ~22.4% — if opp is 25%, pitcher gets +1 K/9 boost
    LG_K_RATE = 0.224
    opp_factor = opp_k_rate / LG_K_RATE  # >1 = strikeout-prone lineup
    blended_k9 = k_per_9 * (0.70 + 0.30 * opp_factor)

    # ── Umpire adjustment ─────────────────────────────────────────
    ump_k_factor = 1.0
    if umpire_name:
        tend = UMP_TENDENCIES.get(umpire_name, {})
        ump_k_factor = tend.get("k", 1.0)
    blended_k9 *= ump_k_factor

    # ── Expected innings ──────────────────────────────────────────
    if pitcher_log:
        recent_ip = [g.get("ip", 0) for g in pitcher_log[-5:] if g.get("ip")]
        avg_ip = sum(recent_ip) / len(recent_ip) if recent_ip else 5.0
    else:
        avg_ip = 5.0

    # Fatigue: if pitcher is on short rest, expect fewer innings
    if fatigue:
        phase2 = fatigue.get("phase2", 1.0)
        if phase2 >= 1.14:  # short rest / gasses quickly
            avg_ip = min(avg_ip, 4.5)
        elif phase2 >= 1.08:
            avg_ip = min(avg_ip, 5.5)

    expected_ip = min(avg_ip, 6.0)  # rarely goes beyond 6 in modern game

    # ── Final K prediction ─────────────────────────────────────────
    model_k = blended_k9 / 9 * expected_ip

    # Rough variance: K totals follow Poisson-ish distribution
    # std dev ~ sqrt(mean) but empirically ~1.5 for starters
    std = max(1.5, model_k**0.5)
    return {
        "model_k": round(model_k, 1),
        "model_k_low": round(max(0, model_k - std), 1),
        "model_k_high": round(model_k + std, 1),
        "expected_ip": round(expected_ip, 1),
    }


def optimize_batting_order(
    lineup_stats: list,
    pitcher_stats: dict,
    weather: dict = None,
    venue: str = None,
) -> dict:
    """
    Suggest an optimized batting order using wOBA-based scoring.

    Classic lineup construction rules:
      - Slot 1: best OBP (gets the most PAs, needs to be on base)
      - Slot 2: second-best OBP or best all-around hitter
      - Slot 3: best overall hitter (highest wOBA)
      - Slot 4: best power (highest SLG/HR rate)
      - Slot 5: second-best power
      - Slots 6-9: descending wOBA

    Returns:
      {
        "current":   [{"slot": int, "name": str, "obp": float, "slg": float, "woba": float}, ...],
        "optimized": [...same structure but reordered...],
        "changes":   int,   # how many slots differ
        "run_gain":  float, # estimated extra runs per game
      }
    """
    import itertools

    if not lineup_stats or len(lineup_stats) < 2:
        return {}

    park = BALLPARK_FACTORS.get(venue, {}) if venue else {}
    hr_park = park.get("hr", 1.0)
    hit_park = park.get("hit", 1.0)

    # Score each batter
    scored = []
    for i, stats in enumerate(lineup_stats[:9]):
        probs = build_batter_probs(stats)
        probs = apply_pitcher_modifier(probs, pitcher_stats)
        if weather:
            probs = apply_weather_modifier(probs, weather)

        adj = list(probs)
        adj[5] *= hr_park
        adj[2] *= hit_park
        adj[3] *= hit_park
        total = sum(adj)
        probs = [p / total for p in adj]

        p_walk = probs[0]
        p_1b = probs[2]
        p_2b = probs[3]
        p_3b = probs[4]
        p_hr = probs[5]
        p_hit = p_1b + p_2b + p_3b + p_hr
        p_obp = p_hit + p_walk

        # wOBA weights (approximate)
        woba = p_walk * 0.69 + p_1b * 0.888 + p_2b * 1.271 + p_3b * 1.616 + p_hr * 2.101
        slg = p_1b * 1 + p_2b * 2 + p_3b * 3 + p_hr * 4

        scored.append(
            {
                "orig_slot": i + 1,
                "name": stats.get("name", f"Batter {i + 1}"),
                "obp": round(p_obp, 3),
                "slg": round(slg, 3),
                "woba": round(woba, 3),
                "p_hr": round(p_hr, 4),
            }
        )

    # Build current order info
    current = [
        {
            "slot": b["orig_slot"],
            "name": b["name"],
            "obp": b["obp"],
            "slg": b["slg"],
            "woba": b["woba"],
        }
        for b in scored
    ]

    # Sort by wOBA descending to rank players
    ranked = sorted(scored, key=lambda x: x["woba"], reverse=True)

    # Apply lineup construction rules:
    # Slot 1 → highest OBP, Slot 2 → 2nd highest OBP, 3 → top wOBA,
    # 4 → top HR, 5 → 2nd HR, 6-9 → remaining by wOBA desc
    by_obp = sorted(scored, key=lambda x: x["obp"], reverse=True)
    by_woba = sorted(scored, key=lambda x: x["woba"], reverse=True)
    by_hr = sorted(scored, key=lambda x: x["p_hr"], reverse=True)

    assigned = []
    used = set()

    def pick(pool, used):
        for p in pool:
            if p["name"] not in used:
                used.add(p["name"])
                return p

    slot1 = pick(by_obp, used)
    slot2 = pick(by_obp, used)
    slot3 = pick(by_woba, used)
    slot4 = pick(by_hr, used)
    slot5 = pick(by_hr, used)
    rest = [b for b in by_woba if b["name"] not in used]

    optimized_order = [slot1, slot2, slot3, slot4, slot5] + rest

    optimized = [
        {
            "slot": i + 1,
            "name": b["name"],
            "obp": b["obp"],
            "slg": b["slg"],
            "woba": b["woba"],
            "orig_slot": b["orig_slot"],
        }
        for i, b in enumerate(optimized_order)
        if b
    ]

    # Count changes
    changes = sum(1 for a, b in zip(current, optimized) if a["name"] != b["name"])

    # Estimate run gain: top-of-order wOBA improvement × ~0.3 runs per wOBA point per slot
    current_top3_woba = sum(c["woba"] for c in current[:3]) / 3
    optimized_top3_woba = sum(o["woba"] for o in optimized[:3]) / 3
    run_gain = round((optimized_top3_woba - current_top3_woba) * 3.0, 2)

    return {
        "current": current,
        "optimized": optimized,
        "changes": changes,
        "run_gain": run_gain,
    }
