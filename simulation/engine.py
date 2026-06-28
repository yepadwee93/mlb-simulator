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

    total_hr_factor = hr_temp_factor * hr_wind_factor
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

    # ── Apply all adjustments ──────────────────────────────────────────────
    adjusted = list(probs)
    adjusted[0] *= bb_scale  # walk — driven by BB%
    adjusted[1] *= k_scale  # strikeout — driven by K%
    adjusted[2] *= hit_scale  # single
    adjusted[3] *= hit_scale  # double
    adjusted[4] *= hit_scale  # triple
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

        # Lineup protection: pitchers issue more walks to good hitters when
        # the next batter is significantly better (don't want to face the
        # cleanup guy with a runner on). Conversely, they attack the zone
        # more aggressively when a weak hitter follows.
        # OUTCOME_ORDER index 0 = WALK
        try:
            n_batters = len(lineup_stats)
            curr_ops = float(stats.get("ops", "0") or "0")
            next_stats = lineup_stats[(i + 1) % n_batters]
            next_ops = float(next_stats.get("ops", "0") or "0")
            ops_diff = next_ops - curr_ops
            if ops_diff >= 0.120:
                probs[0] *= 1.18  # next batter is a monster -- lots of nibbling
            elif ops_diff >= 0.060:
                probs[0] *= 1.10  # next batter is clearly better
            elif ops_diff >= 0.030:
                probs[0] *= 1.05  # next batter is somewhat better
            elif ops_diff <= -0.060:
                probs[0] *= 0.94  # next batter is weaker -- pitcher attacks zone
            total = sum(probs)
            probs = [p / total for p in probs]
        except (ValueError, TypeError, ZeroDivisionError):
            pass  # missing OPS data -- skip protection modifier

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

        # wOBA sanity clamp: if compounding factors pushed the implied wOBA
        # outside the realistic MLB range [0.200, 0.450], scale it back.
        # This prevents extreme edge cases (e.g. hot streak + hitter's park +
        # weak pitcher + strong home split all stacking) from producing
        # absurd per-PA run values that skew O/U and win% predictions.
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

        if ab >= 100:
            # GIDP rate: GDP / (non-K outs) — these are the outs where a DP is even possible
            non_k_outs = max(ab - hits - k, 1)
            gdp_rate = gdp / non_k_outs

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


def advance_runners(b1: bool, b2: bool, b3: bool, outcome: str) -> tuple:
    """
    Given who's on base and what just happened, return:
      (new_b1, new_b2, new_b3, runs_scored)

    Uses MLB-research baserunning probabilities instead of deterministic rules.
    The old model assumed singles always scored runners from 2nd and doubles
    always scored everyone — this inflated run totals by ~0.4 runs/game and
    caused the O/U model to skew over.

    Empirical rates (2022-2024 MLB average):
      Single:
        - Runner on 3rd → scores 87% of the time (held 13% on aggressive D)
        - Runner on 2nd → scores 62% (held at 3rd 38%)
        - Runner on 1st → reaches 3rd 27%, stays at 2nd 73%
      Double:
        - Runner on 3rd → scores 95%
        - Runner on 2nd → scores 87%
        - Runner on 1st → scores 52%, stays at 3rd 48%
      Triple: all runners score (essentially 100%)
      HR: all runners + batter score (100%)
      Walk: pure force advance (no randomness needed)
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
        new_b1 = True  # batter always reaches 1st
        new_b2 = False
        new_b3 = False

        if b3:
            if r() < 0.87:
                runs += 1  # scores
            else:
                new_b3 = True  # held at 3rd (rare)

        if b2:
            if r() < 0.62:
                runs += 1  # scores from 2nd
            else:
                new_b3 = True  # held at 3rd

        if b1:
            if r() < 0.27:
                new_b3 = True  # first-to-third (speed play)
            else:
                new_b2 = True  # normal: 1st to 2nd

        return new_b1, new_b2, new_b3, runs

    if outcome == DOUBLE:
        runs = 0
        new_b1 = False
        new_b2 = True  # batter always at 2nd
        new_b3 = False

        if b3:
            if r() < 0.95:
                runs += 1
            else:
                new_b3 = True

        if b2:
            if r() < 0.87:
                runs += 1
            else:
                new_b3 = True

        if b1:
            if r() < 0.52:
                runs += 1  # scores from 1st on a double
            else:
                new_b3 = True  # stops at 3rd

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


def simulate_half_inning(
    precomp_lineup: list, lineup_pos: int, risp_precomp: list = None, batter_rates: list = None
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
            runner_idx = (pos - 1) % 9  # runner on 1st was the prior batter
            r = batter_rates[runner_idx]
            if random.random() < r["sb_rate"]:
                if random.random() < r["sb_success"]:
                    b2, b1 = True, False  # successful steal of 2nd
                else:
                    b1 = False  # caught stealing — runner is out
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
                        outs += 2  # batter AND runner are both out
                        b1 = False  # runner on 1st is erased
                        pos += 1
                        continue

                # ── Sac fly: runner on 3rd, <2 outs, non-K out ───────────────
                # Ball is caught in the outfield — runner tags and scores.
                # (Only checked if GIDP didn't fire above)
                if b3:
                    if random.random() < batter_rates[batter_idx]["sf_rate"]:
                        runs += 1  # run scores on the sac fly
                        b3 = False  # runner on 3rd scores
                        outs += 1
                        pos += 1
                        continue

            # -- Error: on a non-K out, small chance batter reaches base
            if outcome == OUT and random.random() < ERROR_RATE:
                # Fielding error -- batter reaches 1st, runners advance
                b1, b2, b3, new_runs = advance_runners(b1, b2, b3, SINGLE)
                runs += new_runs
                pos += 1
                continue  # no out recorded

            outs += 1

            # -- Wild pitch / passed ball: advance all runners one base
            # Only matters when there are runners on base.
            if (b1 or b2 or b3) and random.random() < WILD_PITCH_RATE:
                if b3:
                    runs += 1
                b3 = b2
                b2 = b1
                b1 = False

        else:
            # -- Wild pitch before the hit: runners advance on passed ball
            if (b1 or b2 or b3) and random.random() < WILD_PITCH_RATE:
                if b3:
                    runs += 1
                b3 = b2
                b2 = b1
                b1 = False

            b1, b2, b3, new_runs = advance_runners(b1, b2, b3, outcome)
            runs += new_runs

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

        # Away team bats (facing home pitching staff)
        runs, away_pos = simulate_half_inning(
            away_cur, away_pos, away_precomp_risp, away_batter_rates
        )
        if not home_starter_pulled:
            home_starter_runs += runs  # track only while home starter is still in
        away_runs += runs

        # Walk-off rule: home team wins in 9th without finishing if already ahead
        if inning == innings - 1 and home_runs > away_runs:
            break

        # Home team bats (facing away pitching staff)
        runs, home_pos = simulate_half_inning(
            home_cur, home_pos, home_precomp_risp, home_batter_rates
        )
        if not away_starter_pulled:
            away_starter_runs += runs  # track only while away starter is still in
        home_runs += runs

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

    def _sim(**kw):
        return simulate_game(
            away_precomp_early,
            home_precomp_early,
            away_precomp_mid,
            home_precomp_mid,
            away_precomp_late,
            home_precomp_late,
            away_precomp_risp,
            home_precomp_risp,
            away_batter_rates,
            home_batter_rates,
            away_precomp_late2,
            home_precomp_late2,
            **kw,
        )

    for _ in range(n):
        a, h = _sim()
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
            a, h = _sim(innings=inn, resolve_ties=False)
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
) -> list:
    """
    Project per-batter prop lines (hits, HR, RBI, total bases) for one team.

    Uses the same outcome probabilities as the simulation engine so predictions
    are consistent with the win/run totals.

    Returns a list of dicts (one per batter slot, in batting order):
      {
        "slot":        int,    # 1-9
        "avg_hits":    float,  # expected hits per game
        "avg_hr":      float,  # expected HR per game
        "avg_rbi":     float,  # expected RBI per game (approximated)
        "avg_tb":      float,  # expected total bases per game
        "hit_pct":     float,  # P(at least 1 hit)
        "hr_pct":      float,  # P(at least 1 HR)
        "multi_hit_pct": float,  # P(2+ hits)
      }
    """
    # Expected plate appearances per slot (leadoff sees more PAs)
    # Rough MLB average: team gets ~38 PAs/game, distributed across 9 slots
    PA_BY_SLOT = [4.8, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.6]

    park = BALLPARK_FACTORS.get(venue, {}) if venue else {}
    hr_park = park.get("hr", 1.0)
    hit_park = park.get("hit", 1.0)

    results = []
    for i, stats in enumerate(lineup_stats[:9]):
        slot = i + 1
        pa_expected = PA_BY_SLOT[i] if i < len(PA_BY_SLOT) else 3.8

        # Build base outcome probs from batter stats
        probs = build_batter_probs(stats)

        # Apply pitcher modifier
        probs = apply_pitcher_modifier(probs, pitcher_stats)

        # Apply weather
        if weather:
            probs = apply_weather_modifier(probs, weather)

        # Apply park factor to HR and hits
        adj = list(probs)
        adj[5] *= hr_park  # HR
        adj[2] *= hit_park  # single
        adj[3] *= hit_park  # double
        total = sum(adj)
        probs = [p / total for p in adj]

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

        # Scale to expected PAs
        avg_hits = exp_hits_pa * pa_expected
        avg_hr = exp_hr_pa * pa_expected
        avg_tb = exp_tb_pa * pa_expected

        # RBI approximation: ~30% of hits drive in a run (runners on base ~30% of time)
        # HRs always score at least 1 run
        avg_rbi = avg_hr * 1.3 + (avg_hits - avg_hr) * 0.28

        # P(at least 1 hit) = 1 - P(no hits in all PAs)
        p_no_hit_pa = 1 - p_hit
        hit_pct = round((1 - p_no_hit_pa**pa_expected) * 100, 1)

        # P(at least 1 HR)
        p_no_hr_pa = 1 - p_hr
        hr_pct = round((1 - p_no_hr_pa**pa_expected) * 100, 1)

        # P(2+ hits) using Poisson approximation
        import math

        lam = avg_hits
        p_0 = math.exp(-lam)
        p_1 = lam * math.exp(-lam)
        multi_hit_pct = round((1 - p_0 - p_1) * 100, 1)

        results.append(
            {
                "slot": slot,
                "avg_hits": round(avg_hits, 2),
                "avg_hr": round(avg_hr, 3),
                "avg_rbi": round(avg_rbi, 2),
                "avg_tb": round(avg_tb, 2),
                "hit_pct": hit_pct,
                "hr_pct": hr_pct,
                "multi_hit_pct": multi_hit_pct,
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
