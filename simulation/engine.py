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


def precompute_lineup(lineup_stats: list, pitcher_stats: dict) -> list:
    """
    Pre-calculate cumulative probability arrays for every batter in a lineup.
    Doing this ONCE before the simulation loop (instead of inside it) is the
    key performance optimization — we avoid repeating the same math 100,000 times.

    Returns a list of (outcomes, cumulative_weights) tuples, one per batter.
    """
    result = []
    for stats in lineup_stats:
        probs     = build_batter_probs(stats)
        adj       = apply_pitcher_modifier(probs, pitcher_stats)
        cum_weights = list(accumulate(adj))   # e.g. [0.08, 0.30, 0.45, ...]
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
    away_precomp: list,
    home_precomp: list,
    innings: int = 9,
) -> tuple:
    """
    Simulate one complete 9-inning game.
    Returns: (away_runs, home_runs)
    """
    away_runs = 0
    home_runs = 0
    away_pos  = 0
    home_pos  = 0

    for inning in range(innings):
        # Away team bats
        runs, away_pos = simulate_half_inning(away_precomp, away_pos)
        away_runs += runs

        # Walk-off rule: home team wins in 9th without finishing if already ahead
        if inning == innings - 1 and home_runs > away_runs:
            break

        # Home team bats
        runs, home_pos = simulate_half_inning(home_precomp, home_pos)
        home_runs += runs

    # Extra innings if tied (up to 3 extra)
    extra = 0
    while away_runs == home_runs and extra < 3:
        runs, away_pos = simulate_half_inning(away_precomp, away_pos)
        away_runs += runs
        runs, home_pos = simulate_half_inning(home_precomp, home_pos)
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
    away_team:    str,
    home_team:    str,
    away_lineup:  list,   # list of stat dicts for the 9 away batters
    home_lineup:  list,   # list of stat dicts for the 9 home batters
    away_pitcher: dict,   # away starting pitcher's season stats
    home_pitcher: dict,   # home starting pitcher's season stats
    n:            int = 100_000,
) -> dict:
    """
    Run N Monte Carlo simulations and return aggregated results.

    The key optimization: pre-compute probability arrays ONCE, then
    reuse them across all N games. This is what makes 100,000 games fast.
    """

    # Pre-compute batter probabilities adjusted for each opposing pitcher
    # away batters face the home pitcher, and vice versa
    away_precomp = precompute_lineup(away_lineup, home_pitcher)
    home_precomp = precompute_lineup(home_lineup, away_pitcher)

    # Run the simulation loop
    away_wins  = 0
    home_wins  = 0
    away_total = 0
    home_total = 0

    for _ in range(n):
        a, h = simulate_game(away_precomp, home_precomp)
        away_total += a
        home_total += h
        if a > h:
            away_wins += 1
        else:
            home_wins += 1

    return {
        "away_team":     away_team,
        "home_team":     home_team,
        "simulations":   n,
        "away_win_pct":  round(away_wins  / n * 100, 1),
        "home_win_pct":  round(home_wins  / n * 100, 1),
        "away_avg_runs": round(away_total / n, 2),
        "home_avg_runs": round(home_total / n, 2),
    }
