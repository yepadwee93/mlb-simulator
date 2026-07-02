"""
nfl_engine.py
-------------
NFL simulation engine — drive-chain Monte Carlo with multiple models.

Models:
  1. Monte Carlo (drive-chain): Simulates each drive as a sequence of plays,
     modeling scoring probability based on offensive/defensive efficiency.
  2. Elo: Rating-based win probability using historical performance.
  3. DVOA-style: Defense-adjusted Value Over Average using per-play efficiency.
  4. Turnover model: Weighs turnover margin and red zone efficiency.
  5. Log5: Bradley-Terry model combining team win rates.

Each model produces independent win probabilities, then a weighted consensus
determines the final prediction.
"""

import math
import random
from collections import Counter

# ── League averages (2025 NFL season approximations) ────────────
LEAGUE_AVG_PPG = 22.5
LEAGUE_AVG_YPG = 340.0
LEAGUE_AVG_PASS_YPG = 220.0
LEAGUE_AVG_RUSH_YPG = 120.0
LEAGUE_AVG_TOP = 30.0  # time of possession in minutes
LEAGUE_AVG_TURNOVERS = 1.3  # per game
LEAGUE_AVG_SACKS = 2.5
LEAGUE_AVG_THIRD_PCT = 0.39  # 3rd down conversion rate
LEAGUE_AVG_RZ_PCT = 0.58  # red zone TD rate
LEAGUE_AVG_FG_PCT = 0.85
LEAGUE_AVG_PENALTY_YPG = 50.0

# ── Drive outcome probabilities (NFL averages) ──────────────────
# Based on NFL seasonal averages for drive outcomes
DRIVE_OUTCOMES = {
    "touchdown": 0.22,
    "field_goal": 0.12,
    "punt": 0.38,
    "turnover": 0.12,
    "turnover_on_downs": 0.06,
    "end_of_half": 0.05,
    "safety": 0.005,
    "missed_fg": 0.035,
}

# ── Home field advantage ────────────────────────────────────────
HOME_ADVANTAGE_POINTS = 2.5  # NFL home teams average ~2.5 pts better


# ── Elo System ──────────────────────────────────────────────────
DEFAULT_ELO = 1500
ELO_K = 20  # K-factor for NFL (lower than MLB — fewer games)
HFA_ELO = 48  # Home field advantage in Elo points


def elo_expected(elo_a, elo_b):
    """Expected win probability for team A vs team B."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def elo_update(elo, expected, actual, k=ELO_K, mov=1):
    """Update Elo with margin-of-victory multiplier."""
    mov_mult = math.log(max(abs(mov), 1) + 1) * (2.2 / (2.2 + 0.001 * abs(elo - 1500)))
    return elo + k * mov_mult * (actual - expected)


# ── Team power ratings (pre-season 2026 estimates) ──────────────
# These get updated as games are played. Starting points based on
# end-of-2025 performance + offseason moves.
PRESEASON_ELO = {
    "KC": 1620,
    "DET": 1610,
    "PHI": 1600,
    "BUF": 1595,
    "BAL": 1590,
    "SF": 1580,
    "DAL": 1560,
    "CIN": 1555,
    "MIA": 1550,
    "HOU": 1545,
    "GB": 1540,
    "LAR": 1535,
    "PIT": 1530,
    "MIN": 1530,
    "SEA": 1520,
    "JAX": 1515,
    "CLE": 1510,
    "NYJ": 1510,
    "TB": 1505,
    "NO": 1505,
    "ATL": 1500,
    "DEN": 1500,
    "IND": 1495,
    "LAC": 1495,
    "CHI": 1490,
    "LV": 1480,
    "TEN": 1475,
    "WAS": 1475,
    "ARI": 1470,
    "NYG": 1465,
    "NE": 1460,
    "CAR": 1450,
}


def _get_elo(team_abbr):
    return PRESEASON_ELO.get(team_abbr, DEFAULT_ELO)


# ── Drive simulation ───────────────────────────────────────────


def _adjust_drive_probs(base_probs, off_efficiency, def_efficiency):
    """
    Adjust drive outcome probabilities based on team strengths.
    off_efficiency: multiplier (>1 = better offense)
    def_efficiency: multiplier (>1 = better defense against opponent)
    """
    probs = dict(base_probs)

    # Better offense → more TDs and FGs, fewer punts
    td_boost = (off_efficiency - 1.0) * 0.15
    fg_boost = (off_efficiency - 1.0) * 0.05
    probs["touchdown"] = max(0.05, min(0.45, probs["touchdown"] + td_boost))
    probs["field_goal"] = max(0.03, min(0.25, probs["field_goal"] + fg_boost))

    # Better defense → fewer TDs/FGs, more punts and turnovers
    def_penalty = (def_efficiency - 1.0) * 0.12
    probs["touchdown"] = max(0.05, min(0.45, probs["touchdown"] - def_penalty))
    probs["field_goal"] = max(0.03, min(0.25, probs["field_goal"] - def_penalty * 0.5))
    probs["punt"] += def_penalty * 0.8
    probs["turnover"] += def_penalty * 0.3

    # Normalize
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def _simulate_drive(off_probs):
    """Simulate a single drive, returns points scored."""
    r = random.random()
    cumulative = 0.0
    for outcome, prob in off_probs.items():
        cumulative += prob
        if r <= cumulative:
            if outcome == "touchdown":
                # PAT: 94% make rate in NFL
                return 7 if random.random() < 0.94 else 6
            elif outcome == "field_goal":
                return 3
            elif outcome == "safety":
                return -2  # Safety scores for defense
            else:
                return 0
    return 0


def _compute_efficiency(team_stats):
    """
    Compute offensive and defensive efficiency multipliers from team stats.
    Returns (off_eff, def_eff) where 1.0 = league average.
    """
    if not team_stats:
        return 1.0, 1.0

    # Offensive efficiency: points per game relative to league average
    ppg = team_stats.get("general_pointsPerGame", LEAGUE_AVG_PPG)
    off_eff = ppg / LEAGUE_AVG_PPG if LEAGUE_AVG_PPG else 1.0

    # Defensive efficiency: opponent points per game (inverse — lower is better)
    opp_ppg = team_stats.get("general_pointsAgainstPerGame", LEAGUE_AVG_PPG)
    def_eff = LEAGUE_AVG_PPG / opp_ppg if opp_ppg else 1.0

    return off_eff, def_eff


# ── Monte Carlo Drive-Chain Simulation ──────────────────────────


def simulate_game(
    home_stats,
    away_stats,
    home_abbr="HOME",
    away_abbr="AWAY",
    n_sims=10000,
    home_elo=None,
    away_elo=None,
):
    """
    Simulate an NFL game n_sims times using drive-chain model.

    Returns dict with:
      - home_win_pct, away_win_pct
      - home_avg_pts, away_avg_pts
      - score_distribution
      - model results from each individual model
    """
    home_off_eff, home_def_eff = _compute_efficiency(home_stats)
    away_off_eff, away_def_eff = _compute_efficiency(away_stats)

    # Home field advantage
    home_off_eff *= 1.0 + (HOME_ADVANTAGE_POINTS / LEAGUE_AVG_PPG) * 0.5
    away_off_eff *= 1.0 - (HOME_ADVANTAGE_POINTS / LEAGUE_AVG_PPG) * 0.3

    # Compute drive probabilities
    # Home offense vs away defense
    home_drive_probs = _adjust_drive_probs(DRIVE_OUTCOMES, home_off_eff, away_def_eff)
    # Away offense vs home defense
    away_drive_probs = _adjust_drive_probs(DRIVE_OUTCOMES, away_off_eff, home_def_eff)

    home_wins = 0
    away_wins = 0
    ties = 0
    home_scores = []
    away_scores = []
    score_pairs = Counter()

    drives_per_half = 6  # ~12 drives per team per game (NFL average)

    for _ in range(n_sims):
        home_pts = 0
        away_pts = 0

        # Simulate drives (alternating possession)
        for half in range(2):
            for drive_num in range(drives_per_half):
                # Away team drives
                pts = _simulate_drive(away_drive_probs)
                if pts == -2:
                    home_pts += 2  # Safety scores for home
                else:
                    away_pts += pts

                # Home team drives
                pts = _simulate_drive(home_drive_probs)
                if pts == -2:
                    away_pts += 2
                else:
                    home_pts += pts

        # Overtime if tied (simplified)
        if home_pts == away_pts:
            ot_r = random.random()
            if ot_r < 0.52:  # Home team slight OT advantage
                home_pts += 3 + (4 if random.random() < 0.35 else 0)
            elif ot_r < 0.97:
                away_pts += 3 + (4 if random.random() < 0.35 else 0)
            # ~3% chance remains a tie (rare in NFL)

        if home_pts > away_pts:
            home_wins += 1
        elif away_pts > home_pts:
            away_wins += 1
        else:
            ties += 1

        home_scores.append(home_pts)
        away_scores.append(away_pts)
        score_pairs[(away_pts, home_pts)] += 1

    home_win_pct = round(home_wins / n_sims * 100, 1)
    away_win_pct = round(away_wins / n_sims * 100, 1)
    home_avg_pts = round(sum(home_scores) / n_sims, 1)
    away_avg_pts = round(sum(away_scores) / n_sims, 1)

    # ── Elo model ───────────────────────────────────────────────
    h_elo = home_elo or _get_elo(home_abbr)
    a_elo = away_elo or _get_elo(away_abbr)
    elo_home_pct = round(elo_expected(h_elo + HFA_ELO, a_elo) * 100, 1)
    elo_away_pct = round(100 - elo_home_pct, 1)

    # ── DVOA-style model ────────────────────────────────────────
    dvoa_home = _dvoa_win_prob(home_stats, away_stats, home=True)
    dvoa_away = round(100 - dvoa_home, 1)

    # ── Turnover model ──────────────────────────────────────────
    to_home = _turnover_model(home_stats, away_stats, home=True)
    to_away = round(100 - to_home, 1)

    # ── Log5 model ──────────────────────────────────────────────
    home_wp = home_stats.get("general_winPercent", 0.5) if home_stats else 0.5
    away_wp = away_stats.get("general_winPercent", 0.5) if away_stats else 0.5
    log5_home = _log5(home_wp, away_wp, home=True)
    log5_away = round(100 - log5_home, 1)

    # ── Weighted consensus ──────────────────────────────────────
    weights = {
        "monte_carlo": 0.35,
        "elo": 0.20,
        "dvoa": 0.20,
        "turnover": 0.10,
        "log5": 0.15,
    }
    consensus_home = round(
        home_win_pct * weights["monte_carlo"]
        + elo_home_pct * weights["elo"]
        + dvoa_home * weights["dvoa"]
        + to_home * weights["turnover"]
        + log5_home * weights["log5"],
        1,
    )
    consensus_away = round(100 - consensus_home, 1)

    models = {
        "monte_carlo": {
            "home_pct": home_win_pct,
            "away_pct": away_win_pct,
            "weight": weights["monte_carlo"],
        },
        "elo": {
            "home_pct": elo_home_pct,
            "away_pct": elo_away_pct,
            "home_elo": h_elo,
            "away_elo": a_elo,
            "weight": weights["elo"],
        },
        "dvoa": {
            "home_pct": dvoa_home,
            "away_pct": dvoa_away,
            "weight": weights["dvoa"],
        },
        "turnover": {
            "home_pct": to_home,
            "away_pct": to_away,
            "weight": weights["turnover"],
        },
        "log5": {
            "home_pct": log5_home,
            "away_pct": log5_away,
            "weight": weights["log5"],
        },
    }

    predicted_winner = home_abbr if consensus_home >= consensus_away else away_abbr
    predicted_spread = round(home_avg_pts - away_avg_pts, 1)

    return {
        "home_team": home_abbr,
        "away_team": away_abbr,
        "home_win_pct": consensus_home,
        "away_win_pct": consensus_away,
        "home_avg_pts": home_avg_pts,
        "away_avg_pts": away_avg_pts,
        "predicted_winner": predicted_winner,
        "predicted_spread": predicted_spread,
        "predicted_total": round(home_avg_pts + away_avg_pts, 1),
        "models": models,
        "consensus_home_pct": consensus_home,
        "consensus_away_pct": consensus_away,
        "n_sims": n_sims,
        "score_pairs": dict(score_pairs),
        "models_agree": sum(
            1 for m in models.values() if (m["home_pct"] > 50) == (consensus_home > 50)
        ),
    }


# ── DVOA-style model ────────────────────────────────────────────


def _dvoa_win_prob(home_stats, away_stats, home=True):
    """
    DVOA-inspired model: uses per-play efficiency metrics.
    Compares offensive efficiency vs opposing defensive efficiency.
    """
    if not home_stats or not away_stats:
        return 55.0 if home else 45.0

    # Offensive yards per play
    home_oypp = home_stats.get("passing_yardsPerPassAttempt", 6.5)
    away_oypp = away_stats.get("passing_yardsPerPassAttempt", 6.5)

    # Defensive yards allowed per play
    home_dypp = home_stats.get("defensive_yardsPerPassAttemptAllowed", 6.5)
    away_dypp = away_stats.get("defensive_yardsPerPassAttemptAllowed", 6.5)

    # Rushing efficiency
    home_rypc = home_stats.get("rushing_yardsPerRushAttempt", 4.3)
    away_rypc = away_stats.get("rushing_yardsPerRushAttempt", 4.3)

    # Combined efficiency score
    home_off_score = home_oypp * 0.6 + home_rypc * 0.4
    away_off_score = away_oypp * 0.6 + away_rypc * 0.4
    home_def_score = 1.0 / max(home_dypp * 0.6 + 4.3 * 0.4, 1.0)
    away_def_score = 1.0 / max(away_dypp * 0.6 + 4.3 * 0.4, 1.0)

    home_power = home_off_score * home_def_score
    away_power = away_off_score * away_def_score

    # Home advantage
    if home:
        home_power *= 1.06

    total = home_power + away_power
    if total == 0:
        return 50.0

    return round(home_power / total * 100, 1)


# ── Turnover model ──────────────────────────────────────────────


def _turnover_model(home_stats, away_stats, home=True):
    """
    Turnover-focused model: teams that protect the ball and force turnovers
    have a significant edge.
    """
    if not home_stats or not away_stats:
        return 55.0 if home else 45.0

    # Turnover differential
    home_to_diff = home_stats.get("general_turnoverDifferential", 0)
    away_to_diff = away_stats.get("general_turnoverDifferential", 0)

    # Red zone efficiency
    home_rz = home_stats.get("scoring_redZonePct", LEAGUE_AVG_RZ_PCT)
    away_rz = away_stats.get("scoring_redZonePct", LEAGUE_AVG_RZ_PCT)

    # Third down conversion
    home_3rd = home_stats.get("general_thirdDownPct", LEAGUE_AVG_THIRD_PCT)
    away_3rd = away_stats.get("general_thirdDownPct", LEAGUE_AVG_THIRD_PCT)

    # Composite score
    home_score = home_to_diff * 3 + home_rz * 20 + home_3rd * 15
    away_score = away_to_diff * 3 + away_rz * 20 + away_3rd * 15

    if home:
        home_score += 2  # Small HFA

    total = abs(home_score) + abs(away_score)
    if total == 0:
        return 50.0

    # Convert to probability using logistic function
    diff = home_score - away_score
    prob = 1.0 / (1.0 + math.exp(-diff * 0.08))
    return round(prob * 100, 1)


# ── Log5 model ──────────────────────────────────────────────────


def _log5(home_wp, away_wp, home=True):
    """
    Log5 / Bradley-Terry model.
    Given each team's win percentage, compute head-to-head probability.
    """
    home_wp = max(0.01, min(0.99, home_wp))
    away_wp = max(0.01, min(0.99, away_wp))

    p = (home_wp * (1 - away_wp)) / (home_wp * (1 - away_wp) + away_wp * (1 - home_wp))

    # Home field adjustment
    if home:
        p = p * 1.06 / (p * 1.06 + (1 - p))

    return round(p * 100, 1)
