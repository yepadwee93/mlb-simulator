"""
web.py
------
The Flask web app. Run this to open the simulator in your browser.

Usage:
    python web.py
Then open:  http://127.0.0.1:5000
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date


def _today_est():
    """Return today's date in US/Eastern time (avoids UTC midnight drift on Vercel)."""
    import datetime as _dt

    try:
        from zoneinfo import ZoneInfo

        return _dt.datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return (_dt.datetime.utcnow() - _dt.timedelta(hours=5)).date()


from dotenv import load_dotenv

load_dotenv()  # loads ODDS_API_KEY from .env file

import os

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_wtf.csrf import CSRFProtect, generate_csrf

from data.auth import check_password, create_user, get_user_by_id
from data.bet_tracker import get_all_bets, get_bet_stats, log_bet, settle_bets
from data.mlb_api import (
    compute_injury_impact,
    get_ballpark_weather,
    get_batter_risp_stats,
    get_batter_sitcode_stats,
    get_batter_split_stats,
    get_batter_vs_pitcher,
    get_bullpen_depth_score,
    get_catcher_cs_rate,
    get_game_lineup,
    get_game_umpire,
    get_lineup_status,
    get_live_scores,
    get_pitcher_arsenal,
    get_pitcher_game_log,
    get_pitcher_hand,
    get_player_recent_stats,
    get_player_season_stats,
    get_recent_transactions,
    get_savant_stats_all,
    get_series_game_number,
    get_team_bullpen_stats,
    get_team_bullpen_usage,
    get_team_il_players,
    get_team_rest_days,
    get_team_streak,
    get_today_schedule,
)
from data.my_picks import add_pick, get_all_picks, get_pick_stats, update_pick_results
from data.odds_api import (
    calc_edge,
    calc_ev,
    calc_kelly,
    format_odds,
    get_line_movement,
    get_mlb_events,
    get_mlb_odds,
    get_mlb_runline,
    get_mlb_totals,
    get_player_props,
    get_public_betting_pcts,
    get_requests_remaining,
)
from data.tracker import (
    delete_game_note,
    get_accuracy_stats,
    get_all_predictions,
    get_game_notes,
    get_odds_history,
    log_odds,
    log_prediction,
    save_game_note,
    settle_odds_history,
    update_results,
)
from simulation.engine import (
    detect_pitcher_form,
    optimize_batting_order,
    predict_batter_props,
    predict_pitcher_ks,
    run_simulation,
)

app = Flask(__name__, template_folder="app/templates")
_secret = os.environ.get("SECRET_KEY")
if not _secret:
    raise RuntimeError("SECRET_KEY environment variable must be set")
app.secret_key = _secret
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    WTF_CSRF_TIME_LIMIT=None,
    WTF_CSRF_HEADERS=["X-CSRFToken"],  # accept token from JS header
)

csrf = CSRFProtect(app)
limiter = Limiter(get_remote_address, app=app, default_limits=[], storage_uri="memory://")


# Inject CSRF token into every response so JS can read it from a meta tag
@app.after_request
def set_csrf_cookie(response):
    response.set_cookie("csrf_token", generate_csrf(), samesite="Lax", httponly=False)
    return response


# ── Flask-Login setup ─────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view = "login_page"
login_manager.login_message = "Please sign in to access the simulator."


class User(UserMixin):
    def __init__(self, data):
        self.id = str(data["id"])
        self.username = data["username"]

    def get_id(self):
        return self.id


@login_manager.user_loader
def load_user(user_id):
    data = get_user_by_id(int(user_id))
    return User(data) if data else None


def _uid():
    """Return the current user's ID for Supabase queries."""
    return int(current_user.id) if current_user.is_authenticated else None


_score_pairs_cache = {}  # game_pk -> score_pairs dict for SGP correlated prob (max 30 entries)
_batter_props_cache = {}  # game_pk -> {away: [...], home: [...], away_team, home_team, ...}
_SCORE_PAIRS_MAXSIZE = 30

N_SIMS_SINGLE = 50_000  # simulations for one-game view (50k ≈ same accuracy as 100k, 2x faster)
N_SIMS_ALL = 25_000  # simulations per game in simulate-all (faster)
PARLAY_THRESHOLD = 65.0  # min win % to include in best parlay
BET_THRESHOLD = 62.0  # min win % to show in best bets section


# ── Shared helpers ────────────────────────────────────────────────


def fetch_batter_stats_with_splits(batters, pitcher_hand):
    """
    For each batter, fetch their split stats vs the starting pitcher's hand
    (L or R) and blend them with season totals weighted by sample size.

    OLD approach: use split stats if 20+ PA, otherwise ignore entirely.
    NEW approach: always blend split stats into season totals, with the
    split weight scaling up as the sample size grows:
      <10 PA  → 15% split weight (tiny sample, mostly noise)
      20 PA   → 35% split weight (meaningful platoon signal)
      50+ PA  → 60% split weight (strong platoon data, trust it heavily)

    This means even a small platoon sample adds useful signal rather than
    being thrown away, while large samples drive the prediction more.
    """
    from simulation.engine import blend_probs, build_batter_probs

    stats_list = []
    for b in batters:
        if not b.get("id"):
            stats_list.append({})
            continue
        try:
            season = get_player_season_stats(b["id"], group="hitting")
            split = get_batter_split_stats(b["id"], pitcher_hand)
            split_pa = split.get("plateAppearances", 0) if split else 0

            if not split or split_pa < 5:
                # No useful split data — use season totals only
                stats_list.append(season)
                continue

            # Scale weight: 5 PA → ~0.12, 20 PA → 0.35, 50+ PA → 0.60
            split_weight = min(0.12 + (split_pa / 50.0) * 0.48, 0.60)

            # Blend at probability level for accuracy
            season_pa = season.get("plateAppearances", 1) or 1
            if season_pa >= 10:
                s_probs = build_batter_probs(season)
                r_probs = build_batter_probs(split)
                blended = blend_probs(s_probs, r_probs, recent_weight=split_weight)
                # Return split stats dict with blended PA-weighted rates baked in
                # We do this by returning the higher-PA season stats as base
                # and marking it as pre-blended so engine doesn't double-count
                merged = dict(season)
                merged["_split_blended"] = True
                merged["_blended_probs"] = blended
                stats_list.append(merged)
            else:
                stats_list.append(split if split_pa >= 20 else season)

        except Exception:
            try:
                stats_list.append(get_player_season_stats(b["id"], group="hitting"))
            except Exception:
                stats_list.append({})
    return stats_list


def _get_day_night(game_time_str):
    """
    Returns 'd' (day) or 'n' (night) based on the game's UTC start time.
    Day games typically start before 5:30pm ET (21:30 UTC).
    Night games start 6pm ET or later (22:00+ UTC).
    """
    if not game_time_str:
        return "n"
    try:
        from datetime import datetime as dt

        utc_hour = dt.fromisoformat(game_time_str.replace("Z", "+00:00")).hour
        # 16-21 UTC = roughly 12pm-5pm ET = day game
        return "d" if 16 <= utc_hour < 22 else "n"
    except Exception:
        return "n"


def build_game_result(game, n_sims, use_splits=True):
    """
    Full pipeline for one game:
      fetch lineup → get pitcher hands → fetch batter stats →
      get weather → run simulation → return result dict.

    use_splits: if True, fetch L/R split stats per batter (more accurate but slower).
                Set to False for simulate-all mode to keep things fast.

    Returns None if the lineup isn't available yet.
    """
    lineup = get_game_lineup(game["gamePk"])
    if not lineup["away_batters"] or not lineup["home_batters"]:
        return None

    away_pitcher = lineup.get("away_pitcher") or {}
    home_pitcher = lineup.get("home_pitcher") or {}

    # Get which hand each pitcher throws with
    away_hand = get_pitcher_hand(away_pitcher["id"]) if away_pitcher.get("id") else "R"
    home_hand = get_pitcher_hand(home_pitcher["id"]) if home_pitcher.get("id") else "R"

    if use_splits:
        # Detailed mode: use L/R split stats for each batter (more accurate)
        away_batter_stats = fetch_batter_stats_with_splits(lineup["away_batters"], home_hand)
        home_batter_stats = fetch_batter_stats_with_splits(lineup["home_batters"], away_hand)
    else:
        # Fast mode: just use season totals (good enough for parlay overview)
        away_batter_stats = [
            get_player_season_stats(b["id"], group="hitting") if b.get("id") else {}
            for b in lineup["away_batters"]
        ]
        home_batter_stats = [
            get_player_season_stats(b["id"], group="hitting") if b.get("id") else {}
            for b in lineup["home_batters"]
        ]

    # Platoon matchup score
    def _platoon_score(batter_stats_list):
        scores = []
        for s in batter_stats_list:
            if not s:
                continue
            ops = float(s.get("obp", 0) or 0) + float(s.get("slg", 0) or 0)
            if ops > 0:
                scores.append(ops)
        if not scores:
            return 50
        avg_ops = sum(scores) / len(scores)
        return int(min(100, max(0, (avg_ops / 0.720) * 50)))

    away_platoon_score = _platoon_score(away_batter_stats)
    home_platoon_score = _platoon_score(home_batter_stats)
    platoon_edge = away_platoon_score - home_platoon_score

    # Fetch pitcher season stats + team bullpen stats + last 10 starts in parallel
    def _fetch_pitcher_and_bullpen(pitcher_id, team_id):
        sp = get_player_season_stats(pitcher_id, group="pitching") if pitcher_id else {}
        bp = get_team_bullpen_stats(team_id) if team_id else {"era": "4.20", "whip": "1.30"}
        log = get_pitcher_game_log(pitcher_id, num_games=10) if pitcher_id else []
        return sp, bp, log

    with ThreadPoolExecutor(max_workers=2) as ex:
        away_p_future = ex.submit(
            _fetch_pitcher_and_bullpen, away_pitcher.get("id"), game.get("away_id")
        )
        home_p_future = ex.submit(
            _fetch_pitcher_and_bullpen, home_pitcher.get("id"), game.get("home_id")
        )
        away_pitcher_stats, away_bullpen_stats, away_pitcher_log = away_p_future.result()
        home_pitcher_stats, home_bullpen_stats, home_pitcher_log = home_p_future.result()

    # Calculate days since each starter's last start
    def _rest_days(game_log):
        """Return days since the pitcher's most recent start, or None if unknown."""
        if not game_log:
            return None
        last_date_str = game_log[-1].get("date", "")
        if not last_date_str:
            return None
        try:
            from datetime import datetime as dt

            last_date = dt.strptime(last_date_str[:10], "%Y-%m-%d").date()
            return (_today_est() - last_date).days
        except Exception:
            return None

    away_rest_days = _rest_days(away_pitcher_log)
    home_rest_days = _rest_days(home_pitcher_log)

    # Fetch recent form + display splits for each batter — single-game mode only
    # In parallel we fetch: recent 20 games, vs LHP, vs RHP, vs SP, vs RP
    away_recent = []
    home_recent = []
    away_display_extra = []  # list of dicts with vs_lhp/vs_rhp/vs_sp/vs_rp stats
    home_display_extra = []

    if use_splits:
        day_night = _get_day_night(game.get("game_time", ""))

        def _fetch_all_batter_data(batter, opp_pitcher_id=None, is_home=False):
            """Fetch recent form + all split types + matchup history + day/night + home/away for one batter.
            Returns (recent, lhp, rhp, sp, rp, risp, vs_pitcher, day_night_stat, home_away_stat)."""
            pid = batter.get("id")
            if not pid:
                return {}, {}, {}, {}, {}, {}, {}, {}, {}
            try:
                recent = get_player_recent_stats(pid, num_games=20)
            except Exception:
                recent = {}
            try:
                vs_lhp = get_batter_sitcode_stats(pid, "vl")
            except Exception:
                vs_lhp = {}
            try:
                vs_rhp = get_batter_sitcode_stats(pid, "vr")
            except Exception:
                vs_rhp = {}
            try:
                vs_sp = get_batter_sitcode_stats(pid, "vsp")
            except Exception:
                vs_sp = {}
            try:
                vs_rp = get_batter_sitcode_stats(pid, "vrp")
            except Exception:
                vs_rp = {}
            try:
                risp = get_batter_risp_stats(pid)
            except Exception:
                risp = {}
            try:
                vs_pitcher = get_batter_vs_pitcher(pid, opp_pitcher_id) if opp_pitcher_id else {}
            except Exception:
                vs_pitcher = {}
            try:
                dn_stat = get_batter_sitcode_stats(pid, day_night)
            except Exception:
                dn_stat = {}
            try:
                ha_code = "h" if is_home else "a"
                home_away_stat = get_batter_sitcode_stats(pid, ha_code)
            except Exception:
                home_away_stat = {}
            return recent, vs_lhp, vs_rhp, vs_sp, vs_rp, risp, vs_pitcher, dn_stat, home_away_stat

        # Away batters face the HOME pitcher; home batters face the AWAY pitcher
        n_away = len(lineup["away_batters"])
        home_pid = home_pitcher.get("id")
        away_pid = away_pitcher.get("id")

        with ThreadPoolExecutor(max_workers=30) as ex:
            away_futures = [
                ex.submit(_fetch_all_batter_data, b, home_pid, False)
                for b in lineup["away_batters"]
            ]
            home_futures = [
                ex.submit(_fetch_all_batter_data, b, away_pid, True) for b in lineup["home_batters"]
            ]
            all_results_data = [f.result() for f in away_futures + home_futures]

        # Split results back into away and home
        away_data = all_results_data[:n_away]
        home_data = all_results_data[n_away:]

        away_recent = [d[0] for d in away_data]
        home_recent = [d[0] for d in home_data]
        away_display_extra = [
            {
                "vs_lhp": d[1],
                "vs_rhp": d[2],
                "vs_sp": d[3],
                "vs_rp": d[4],
                "risp": d[5],
                "vs_pitcher": d[6],
                "day_night": d[7],
                "home_away": d[8],
            }
            for d in away_data
        ]
        home_display_extra = [
            {
                "vs_lhp": d[1],
                "vs_rhp": d[2],
                "vs_sp": d[3],
                "vs_rp": d[4],
                "risp": d[5],
                "vs_pitcher": d[6],
                "day_night": d[7],
                "home_away": d[8],
            }
            for d in home_data
        ]

    # Fetch Statcast barrel rate + hard-hit % from Baseball Savant (cached per session)
    try:
        savant_all = get_savant_stats_all()
        away_savant = [savant_all.get(b.get("id"), {}) for b in lineup["away_batters"]]
        home_savant = [savant_all.get(b.get("id"), {}) for b in lineup["home_batters"]]
    except Exception:
        away_savant = None
        home_savant = None

    # How many days rest do the batters on each team have?
    try:
        away_batter_rest = get_team_rest_days(game["away_team"])
        home_batter_rest = get_team_rest_days(game["home_team"])
    except Exception:
        away_batter_rest = None
        home_batter_rest = None

    try:
        away_bp_fatigue = get_team_bullpen_usage(game["away_id"])
        home_bp_fatigue = get_team_bullpen_usage(game["home_id"])
    except Exception:
        away_bp_fatigue = None
        home_bp_fatigue = None

    # ── Injury / IL alerts ────────────────────────────────────────────
    try:
        away_il = get_team_il_players(game["away_id"])
        home_il = get_team_il_players(game["home_id"])
        away_transactions = get_recent_transactions(game["away_id"], days_back=3)
        home_transactions = get_recent_transactions(game["home_id"], days_back=3)
    except Exception:
        away_il = home_il = away_transactions = home_transactions = []

    # ── Injury impact scores ─────────────────────────────────────────
    try:
        away_injury_impact = compute_injury_impact(away_il)
        home_injury_impact = compute_injury_impact(home_il)
    except Exception:
        away_injury_impact = home_injury_impact = {
            "score": 0,
            "grade": "None",
            "color": "#555870",
            "key_players": [],
        }

    # ── Series context ────────────────────────────────────────────────
    try:
        series_game_num = get_series_game_number(game["gamePk"], game["away_id"], game["home_id"])
    except Exception:
        series_game_num = 1

    # ── Lineup confirmation status ───────────────────────────────────
    try:
        lineup_status = get_lineup_status(game["gamePk"])
    except Exception:
        lineup_status = "unknown"

    # ── Team streaks ─────────────────────────────────────────────────
    try:
        away_streak = get_team_streak(game["away_id"])
        home_streak = get_team_streak(game["home_id"])
    except Exception:
        away_streak = home_streak = {
            "streak": 0,
            "type": "",
            "label": "",
            "hot": False,
            "cold": False,
        }

    # ── Catcher arm ratings ───────────────────────────────────────────
    try:
        away_catcher_cs = get_catcher_cs_rate(game["away_id"])
        home_catcher_cs = get_catcher_cs_rate(game["home_id"])
    except Exception:
        away_catcher_cs = home_catcher_cs = 0.28

    # ── Bullpen depth scores ──────────────────────────────────────────
    try:
        away_bp_depth = get_bullpen_depth_score(game["away_id"])
        home_bp_depth = get_bullpen_depth_score(game["home_id"])
    except Exception:
        away_bp_depth = home_bp_depth = {
            "score": 50,
            "grade": "Average",
            "reliable_arms": 0,
            "elite_arms": 0,
            "avg_era": 4.20,
            "avg_whip": 1.30,
        }

    # Fetch umpire for K/BB tendency modifier
    try:
        umpire_name = get_game_umpire(game["gamePk"])
    except Exception:
        umpire_name = None

    # Fetch weather for this ballpark at game time
    weather = get_ballpark_weather(game["venue"], game_time_utc=game.get("game_time", ""))

    # Extract vs-RP stats for SP→bullpen transition in the simulation
    # These are the batter's historical stats against relief pitchers specifically.
    # The sim uses them for innings 6-9 instead of the starting pitcher probs.
    away_rp_stats = (
        [ex.get("vs_rp", {}) for ex in away_display_extra] if away_display_extra else None
    )
    home_rp_stats = (
        [ex.get("vs_rp", {}) for ex in home_display_extra] if home_display_extra else None
    )

    # Extract RISP stats — used in the sim whenever a runner is on 2nd or 3rd
    away_risp_stats = (
        [ex.get("risp", {}) for ex in away_display_extra] if away_display_extra else None
    )
    home_risp_stats = (
        [ex.get("risp", {}) for ex in home_display_extra] if home_display_extra else None
    )

    # Extract matchup history stats — career stats for each batter vs the opposing starter
    away_matchup_stats = (
        [ex.get("vs_pitcher", {}) for ex in away_display_extra] if away_display_extra else None
    )
    home_matchup_stats = (
        [ex.get("vs_pitcher", {}) for ex in home_display_extra] if home_display_extra else None
    )

    # Extract day/night splits
    away_daynight_stats = (
        [ex.get("day_night", {}) for ex in away_display_extra] if away_display_extra else None
    )
    home_daynight_stats = (
        [ex.get("day_night", {}) for ex in home_display_extra] if home_display_extra else None
    )

    # Extract home/away splits (away batters playing on the road, home batters at their park)
    away_homeaway_stats = (
        [ex.get("home_away", {}) for ex in away_display_extra] if away_display_extra else None
    )
    home_homeaway_stats = (
        [ex.get("home_away", {}) for ex in home_display_extra] if home_display_extra else None
    )

    # Run the simulation — all factors now wired in:
    #   - L/R pitcher splits (base stats)
    #   - Recent form blend (last 20 games)
    #   - Ballpark factors (Coors, Oracle Park, Yankee Stadium, etc.)
    #   - Weather effects on HR (wind, temperature on top of park baseline)
    #   - SP vs RP: innings 1-5 starter, innings 6-9 bullpen
    result = run_simulation(
        away_team=game["away_team"],
        home_team=game["home_team"],
        away_lineup=away_batter_stats,
        home_lineup=home_batter_stats,
        away_pitcher=away_pitcher_stats,
        home_pitcher=home_pitcher_stats,
        weather=weather,
        away_recent=away_recent if use_splits else None,
        home_recent=home_recent if use_splits else None,
        away_rp_stats=away_rp_stats if use_splits else None,
        home_rp_stats=home_rp_stats if use_splits else None,
        away_bullpen=away_bullpen_stats,
        home_bullpen=home_bullpen_stats,
        venue=game["venue"],
        away_pitcher_log=away_pitcher_log,
        home_pitcher_log=home_pitcher_log,
        away_rest_days=away_rest_days,
        home_rest_days=home_rest_days,
        away_risp_stats=away_risp_stats if use_splits else None,
        home_risp_stats=home_risp_stats if use_splits else None,
        away_matchup_stats=away_matchup_stats if use_splits else None,
        home_matchup_stats=home_matchup_stats if use_splits else None,
        away_daynight_stats=away_daynight_stats if use_splits else None,
        home_daynight_stats=home_daynight_stats if use_splits else None,
        away_homeaway_stats=away_homeaway_stats if use_splits else None,
        home_homeaway_stats=home_homeaway_stats if use_splits else None,
        away_batter_rest=away_batter_rest,
        home_batter_rest=home_batter_rest,
        away_savant=away_savant,
        home_savant=home_savant,
        umpire_name=umpire_name,
        away_bp_fatigue=away_bp_fatigue,
        home_bp_fatigue=home_bp_fatigue,
        series_game_number=series_game_num,
        away_catcher_cs=away_catcher_cs,
        home_catcher_cs=home_catcher_cs,
        away_bp_depth=away_bp_depth,
        home_bp_depth=home_bp_depth,
        n=n_sims,
    )

    # ── Pitcher K prop predictions ──────────────────────────────────────
    try:
        away_form = detect_pitcher_form(away_pitcher_log, away_pitcher_stats)
        home_form = detect_pitcher_form(home_pitcher_log, home_pitcher_stats)
        away_k_pred = predict_pitcher_ks(
            away_pitcher_stats,
            home_batter_stats,
            umpire_name=umpire_name,
            pitcher_log=away_pitcher_log,
            fatigue=away_fatigue if "away_fatigue" in dir() else None,
        )
        home_k_pred = predict_pitcher_ks(
            home_pitcher_stats,
            away_batter_stats,
            umpire_name=umpire_name,
            pitcher_log=home_pitcher_log,
            fatigue=home_fatigue if "home_fatigue" in dir() else None,
        )
    except Exception:
        away_k_pred = {}
        away_form = {"form": "stable", "note": "", "modifier": 1.0}
        home_form = {"form": "stable", "note": "", "modifier": 1.0}
        home_k_pred = {}
    result["away_k_pred"] = away_k_pred
    result["away_form"] = away_form
    result["home_form"] = home_form
    result["away_platoon"] = away_platoon_score
    result["home_platoon"] = home_platoon_score
    result["platoon_edge"] = platoon_edge
    result["away_hand"] = away_hand
    result["home_hand"] = home_hand
    result["hr_park_factor"] = round(
        __import__("simulation.engine", fromlist=["BALLPARK_FACTORS"])
        .BALLPARK_FACTORS.get(game.get("venue", ""), {})
        .get("hr", 1.0),
        2,
    )
    result["wind_effect"] = weather.get("wind_effect", "neutral") if weather else "neutral"
    result["lineup_status"] = lineup_status
    result["wind_label"] = weather.get("wind_label", "") if weather else ""

    # ── Last start mini-recap ──────────────────────────────────────────
    def _last_start(log):
        if not log:
            return None
        g = log[-1]
        ip = g.get("inningsPitched") or g.get("ip") or ""
        er = g.get("earnedRuns", "")
        ks = g.get("strikeOuts", "")
        opp = g.get("opponent", "") or g.get("opp", "")
        dt = g.get("date", "")
        parts = []
        if ip:
            parts.append(f"{ip} IP")
        if er != "":
            parts.append(f"{er} ER")
        if ks != "":
            parts.append(f"{ks} K")
        if opp:
            parts.append(f"vs {opp}")
        from datetime import date as _d
        from datetime import datetime as _dt

        if dt:
            try:
                d = _dt.strptime(dt[:10], "%Y-%m-%d").date()
                days = (_d.today() - d).days
                parts.append(f"{days}d ago")
            except Exception:
                pass
        return ", ".join(parts) if parts else None

    result["away_last_start"] = _last_start(away_pitcher_log)
    result["home_last_start"] = _last_start(home_pitcher_log)
    result["away_streak"] = away_streak
    result["home_streak"] = home_streak
    result["series_game_num"] = series_game_num
    result["away_catcher_cs"] = away_catcher_cs
    result["home_catcher_cs"] = home_catcher_cs
    result["away_bp_depth"] = away_bp_depth
    result["home_bp_depth"] = home_bp_depth
    result["away_il"] = away_il
    result["home_il"] = home_il
    result["away_transactions"] = away_transactions
    result["home_transactions"] = home_transactions
    result["away_injury_impact"] = away_injury_impact
    result["home_injury_impact"] = home_injury_impact
    result["home_k_pred"] = home_k_pred

    # Pitcher K line probabilities for SGP builder (Poisson-based)
    import math as _math

    def _k_line_probs(k_pred):
        if not k_pred or not k_pred.get("model_k"):
            return {}
        lam = k_pred["model_k"]
        lines = {}
        for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
            threshold = int(line + 0.5)
            p_under = sum(
                _math.exp(-lam) * lam**k / _math.factorial(k)
                for k in range(threshold)
            )
            lines[str(line)] = round((1 - p_under) * 100, 1)
        return lines

    result["away_k_lines"] = _k_line_probs(away_k_pred)
    result["home_k_lines"] = _k_line_probs(home_k_pred)

    # Attach IDs for logo/headshot URLs in templates
    result["away_team_id"] = game.get("away_id")
    result["home_team_id"] = game.get("home_id")
    result["away_bp_fatigue"] = (away_bp_fatigue or {}).get("fatigue", "normal")
    result["home_bp_fatigue"] = (home_bp_fatigue or {}).get("fatigue", "normal")
    result["away_bp_ip"] = (away_bp_fatigue or {}).get("total_bp_ip", None)
    result["home_bp_ip"] = (home_bp_fatigue or {}).get("total_bp_ip", None)
    result["away_pitcher_id"] = away_pitcher.get("id")
    result["home_pitcher_id"] = home_pitcher.get("id")
    result["away_pitcher_name"] = away_pitcher.get("name", "TBD")
    result["home_pitcher_name"] = home_pitcher.get("name", "TBD")
    result["umpire"] = umpire_name

    # Build per-batter stat rows for display (season + recent + L/R + SP/RP)
    def build_batter_display(batters, season_stats_list, recent_stats_list, extra_list):
        rows = []
        for i, b in enumerate(batters):
            s = season_stats_list[i] if i < len(season_stats_list) else {}
            r = recent_stats_list[i] if recent_stats_list and i < len(recent_stats_list) else {}
            ex = extra_list[i] if extra_list and i < len(extra_list) else {}

            vs_lhp = ex.get("vs_lhp", {})
            vs_rhp = ex.get("vs_rhp", {})
            vs_sp = ex.get("vs_sp", {})
            vs_rp = ex.get("vs_rp", {})
            risp = ex.get("risp", {})
            vs_pitcher = ex.get("vs_pitcher", {})
            dn_stat = ex.get("day_night", {})

            # Season: singles vs extra base hits
            hits = s.get("hits", 0)
            doubles = s.get("doubles", 0)
            triples = s.get("triples", 0)
            hr = s.get("homeRuns", 0)
            xbh = doubles + triples + hr
            singles = max(0, hits - xbh)

            # Recent form: batting average over last 20 games
            r_ab = r.get("atBats", 0)
            r_hits = r.get("hits", 0)
            r_avg = round(r_hits / r_ab, 3) if r_ab > 0 else None

            def _fmt_avg(d):
                """Format avg from a split stats dict."""
                v = d.get("avg", "")
                return v if v else "—"

            def _fmt_ops(d):
                v = d.get("ops", "")
                return v if v else "—"

            rows.append(
                {
                    **b,
                    # Season stats
                    "avg": s.get("avg", "—"),
                    "obp": s.get("obp", "—"),
                    "slg": s.get("slg", "—"),
                    "ops": s.get("ops", "—"),
                    "hr": hr,
                    "rbi": s.get("rbi", 0),
                    "so": s.get("strikeOuts", 0),
                    "bb": s.get("baseOnBalls", 0),
                    "pa": s.get("plateAppearances", 0),
                    # Hit type breakdown
                    "singles": singles,
                    "doubles": doubles,
                    "triples": triples,
                    "xbh": xbh,
                    # vs LHP / RHP
                    "vs_lhp_avg": _fmt_avg(vs_lhp),
                    "vs_lhp_ops": _fmt_ops(vs_lhp),
                    "vs_lhp_pa": vs_lhp.get("plateAppearances", 0),
                    "vs_rhp_avg": _fmt_avg(vs_rhp),
                    "vs_rhp_ops": _fmt_ops(vs_rhp),
                    "vs_rhp_pa": vs_rhp.get("plateAppearances", 0),
                    # vs SP / RP (early vs late game)
                    "vs_sp_avg": _fmt_avg(vs_sp),
                    "vs_sp_ops": _fmt_ops(vs_sp),
                    "vs_sp_pa": vs_sp.get("plateAppearances", 0),
                    "vs_rp_avg": _fmt_avg(vs_rp),
                    "vs_rp_ops": _fmt_ops(vs_rp),
                    "vs_rp_pa": vs_rp.get("plateAppearances", 0),
                    # Recent form (last 20 games)
                    "recent_avg": f".{int(r_avg * 1000):03d}" if r_avg is not None else "—",
                    "recent_games": r.get("games_found", 0),
                    "recent_hr": r.get("homeRuns", 0),
                    "recent_rbi": r.get("rbi", 0),
                    # RISP / clutch stats
                    "risp_avg": _fmt_avg(risp),
                    "risp_ops": _fmt_ops(risp),
                    "risp_pa": risp.get("plateAppearances", 0),
                    # Career matchup history vs this game's opposing starter
                    "vs_pitcher_avg": _fmt_avg(vs_pitcher),
                    "vs_pitcher_ops": _fmt_ops(vs_pitcher),
                    "vs_pitcher_pa": vs_pitcher.get("plateAppearances", 0),
                    "vs_pitcher_hr": vs_pitcher.get("homeRuns", 0),
                    # Day/night splits
                    "dn_avg": _fmt_avg(dn_stat),
                    "dn_ops": _fmt_ops(dn_stat),
                    "dn_pa": dn_stat.get("plateAppearances", 0),
                    "dn_label": "Day" if day_night == "d" else "Night",
                }
            )
        return rows

    away_display = build_batter_display(
        lineup["away_batters"],
        away_batter_stats,
        away_recent if use_splits else [],
        away_display_extra if use_splits else [],
    )
    home_display = build_batter_display(
        lineup["home_batters"],
        home_batter_stats,
        home_recent if use_splits else [],
        home_display_extra if use_splits else [],
    )

    # Attach extra context for display
    result.update(
        {
            "gamePk": game["gamePk"],
            "venue": game["venue"],
            "status": game["status"],
            "away_pitcher_name": away_pitcher.get("name", "TBD"),
            "home_pitcher_name": home_pitcher.get("name", "TBD"),
            "away_pitcher_hand": away_hand,
            "home_pitcher_hand": home_hand,
            "away_era": away_pitcher_stats.get("era", "N/A"),
            "away_whip": away_pitcher_stats.get("whip", "N/A"),
            "away_wl": f"{away_pitcher_stats.get('wins', 0)}-{away_pitcher_stats.get('losses', 0)}",
            "away_bp_era": away_bullpen_stats.get("era", "N/A"),
            "away_bp_whip": away_bullpen_stats.get("whip", "N/A"),
            "home_era": home_pitcher_stats.get("era", "N/A"),
            "home_whip": home_pitcher_stats.get("whip", "N/A"),
            "home_wl": f"{home_pitcher_stats.get('wins', 0)}-{home_pitcher_stats.get('losses', 0)}",
            "home_bp_era": home_bullpen_stats.get("era", "N/A"),
            "home_bp_whip": home_bullpen_stats.get("whip", "N/A"),
            # Pitcher fatigue profile (from last 10 starts game log)
            "away_fatigue_label": result.get("away_fatigue_label", "Typical"),
            "away_avg_ip": result.get("away_avg_ip"),
            "home_fatigue_label": result.get("home_fatigue_label", "Typical"),
            "home_avg_ip": result.get("home_avg_ip"),
            # Pitcher rest days
            "away_rest_days": result.get("away_rest_days"),
            "away_rest_type": result.get("away_rest_type"),
            "home_rest_days": result.get("home_rest_days"),
            "home_rest_type": result.get("home_rest_type"),
            "away_batters": away_display,
            "home_batters": home_display,
            "weather": weather,
            "use_recent": use_splits,  # flag for template to show/hide recent form column
            "game_time_utc": game.get("game_time", ""),
        }
    )
    return result


# ── Routes ────────────────────────────────────────────────────────

# ── Auth routes ───────────────────────────────────────────────────


@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")
@csrf.exempt
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user_data = check_password(username, password)
        if user_data:
            login_user(User(user_data), remember=True)
            next_url = request.args.get("next", "")
            from urllib.parse import urlparse

            if next_url and urlparse(next_url).netloc == "":
                return redirect(next_url)
            return redirect(url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
@csrf.exempt
def register_page():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        if password != confirm:
            error = "Passwords don't match."
        else:
            ok, result = create_user(username, password)
            if ok:
                user_data = get_user_by_id(result)
                login_user(User(user_data), remember=True)
                # Save email + alerts preference if provided at signup
                email = request.form.get("email", "").strip()
                alerts = bool(request.form.get("email_alerts"))
                if email:
                    try:
                        from data.email_alerts import update_user_email

                        update_user_email(result, email, alerts)
                    except Exception:
                        pass
                return redirect(url_for("index"))
            else:
                error = result
    return render_template("register.html", error=error)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login_page"))


@app.route("/")
def index():
    # Allow browsing any date via ?date=2026-06-28
    selected = request.args.get("date", _today_est().isoformat())
    try:
        from datetime import datetime

        selected_date = datetime.strptime(selected, "%Y-%m-%d").date()
    except ValueError:
        selected_date = _today_est()

    from datetime import timedelta

    prev_date = (selected_date - timedelta(days=1)).isoformat()
    next_date = (selected_date + timedelta(days=1)).isoformat()
    is_today = selected_date == _today_est()

    games = get_today_schedule(game_date=selected_date.isoformat())
    label = selected_date.strftime("%A, %B %d %Y")

    # Multi-day strip: today + next 6 days (fetched lightweight — just game count)
    from datetime import timedelta as _td

    today = _today_est()
    multi_day = []
    for offset in range(7):
        d = today + _td(days=offset)
        try:
            day_games = get_today_schedule(game_date=d.isoformat())
            count = len(day_games)
        except Exception:
            count = 0
        multi_day.append(
            {
                "date": d.isoformat(),
                "label": d.strftime("%a"),
                "day_num": d.day,
                "count": count,
                "is_today": d == today,
                "selected": d == selected_date,
            }
        )
    # Also include yesterday so you can go back
    yesterday = today - _td(days=1)
    try:
        y_games = get_today_schedule(game_date=yesterday.isoformat())
        y_count = len(y_games)
    except Exception:
        y_count = 0
    multi_day.insert(
        0,
        {
            "date": yesterday.isoformat(),
            "label": yesterday.strftime("%a"),
            "day_num": yesterday.day,
            "count": y_count,
            "is_today": False,
            "selected": yesterday == selected_date,
        },
    )

    # Public betting % — keyed by frozenset so game cards can look up by team names
    try:
        pub_pcts_raw = get_public_betting_pcts()
        pub_pcts = {tuple(sorted(k)): v for k, v in pub_pcts_raw.items()}
    except Exception:
        pub_pcts = {}

    # Line movement — for reverse line movement detection
    try:
        lm_raw = get_line_movement()
        line_movement = {tuple(sorted(k)): v for k, v in lm_raw.items()}
    except Exception:
        line_movement = {}

    # Check if logged-in user has an email set (for the signup nudge banner)
    show_email_nudge = False
    if current_user.is_authenticated:
        try:
            from data.email_alerts import get_user_email_settings

            prefs = get_user_email_settings(current_user.id)
            show_email_nudge = not prefs.get("email")
        except Exception:
            pass

    return render_template(
        "index.html",
        games=games,
        date=label,
        selected_date=selected_date.isoformat(),
        prev_date=prev_date,
        next_date=next_date,
        is_today=is_today,
        pub_pcts=pub_pcts,
        line_movement=line_movement,
        multi_day=multi_day,
        game_notes=get_game_notes(current_user.id) if current_user.is_authenticated else {},
        api_remaining=get_requests_remaining(),
        show_email_nudge=show_email_nudge,
        error=request.args.get("error"),
    )


@app.route("/simulate/<int:game_pk>")
@login_required
def simulate(game_pk):
    """Simulate a single game and show the detailed results page."""
    games = get_today_schedule()
    game = next((g for g in games if g["gamePk"] == game_pk), None)
    if game is None:
        abort(404)

    # Read sim count from URL e.g. /simulate/822961?sims=500000
    allowed = {50_000, 100_000, 500_000, 1_000_000}
    try:
        n_sims = int(request.args.get("sims", N_SIMS_SINGLE))
        if n_sims not in allowed:
            n_sims = N_SIMS_SINGLE
    except ValueError:
        n_sims = N_SIMS_SINGLE

    try:
        result = build_game_result(game, n_sims=n_sims)
    except Exception:
        result = None

    if result is None:
        return render_template(
            "no_lineup.html",
            away_team=game.get("away_team", "Away"),
            home_team=game.get("home_team", "Home"),
            game_time=game.get("game_time", ""),
        )

    # Cache score_pairs for SGP correlated probability calculations
    if "score_pairs" in result:
        if len(_score_pairs_cache) >= _SCORE_PAIRS_MAXSIZE:
            _score_pairs_cache.pop(next(iter(_score_pairs_cache)))
        _score_pairs_cache[game_pk] = result["score_pairs"]

    # Log this prediction for accuracy tracking
    try:
        log_prediction(
            game_pk=game_pk,
            game_date=_today_est().isoformat(),
            away_team=result["away_team"],
            home_team=result["home_team"],
            away_win_pct=result["away_win_pct"],
            home_win_pct=result["home_win_pct"],
            away_avg_runs=result["away_avg_runs"],
            home_avg_runs=result["home_avg_runs"],
            n_sims=n_sims,
            source="single",
        )
    except Exception:
        pass  # never let logging break the simulation

    # Fetch moneyline odds (cached for 30 min — only 1 API call per 30 min window)
    all_odds = get_mlb_odds()
    odds_key = frozenset([game["away_team"], game["home_team"]])
    game_odds = all_odds.get(odds_key, {})

    if game_odds:
        result["away_implied_pct"] = game_odds["away_implied_pct"]
        result["home_implied_pct"] = game_odds["home_implied_pct"]
        result["away_avg_odds"] = format_odds(game_odds["away_avg_odds"])
        result["home_avg_odds"] = format_odds(game_odds["home_avg_odds"])
        result["away_edge"] = calc_edge(result["away_win_pct"], game_odds["away_implied_pct"])
        result["home_edge"] = calc_edge(result["home_win_pct"], game_odds["home_implied_pct"])
        result["away_ev"] = calc_ev(result["away_win_pct"], game_odds["away_avg_odds"])
        result["home_ev"] = calc_ev(result["home_win_pct"], game_odds["home_avg_odds"])
        result["away_kelly"] = calc_kelly(result["away_win_pct"], game_odds["away_avg_odds"])
        result["home_kelly"] = calc_kelly(result["home_win_pct"], game_odds["home_avg_odds"])
        result["books_used"] = game_odds["books_used"]
        result["best_away_odds"] = format_odds(
            game_odds.get("best_away_odds", game_odds["away_avg_odds"])
        )
        result["best_home_odds"] = format_odds(
            game_odds.get("best_home_odds", game_odds["home_avg_odds"])
        )
        result["best_away_book"] = game_odds.get("best_away_book", "")
        result["best_home_book"] = game_odds.get("best_home_book", "")
    else:
        result["away_implied_pct"] = None
        result["home_implied_pct"] = None
        result["away_avg_odds"] = None
        result["home_avg_odds"] = None
        result["away_edge"] = None
        result["home_edge"] = None
        result["away_ev"] = None
        result["home_ev"] = None
        result["away_kelly"] = None
        result["home_kelly"] = None
        result["books_used"] = 0

    # ── Log odds snapshot to history ─────────────────────────────────
    if game_odds:
        try:
            all_totals_snap = get_mlb_totals()
            ou_snap = all_totals_snap.get(odds_key, {})
            log_odds(
                game_pk=game_pk,
                game_date=_today_est().isoformat(),
                away_team=result["away_team"],
                home_team=result["home_team"],
                away_ml=game_odds.get("away_avg_odds"),
                home_ml=game_odds.get("home_avg_odds"),
                away_implied_pct=game_odds.get("away_implied_pct"),
                home_implied_pct=game_odds.get("home_implied_pct"),
                over_under=ou_snap.get("line"),
                model_away_pct=result.get("away_win_pct"),
                model_home_pct=result.get("home_win_pct"),
                model_away_runs=result.get("away_avg_runs"),
                model_home_runs=result.get("home_avg_runs"),
            )
        except Exception:
            pass

    # ── Over/Under + Run line odds ────────────────────────────────────
    try:
        all_totals = get_mlb_totals()
        ou = all_totals.get(odds_key, {})
        if ou:
            hist = result.get("total_runs_hist", {})
            n_sims = result.get("simulations", 1)
            ou_line = ou["line"]
            over_count = sum(cnt for tot, cnt in hist.items() if tot > ou_line)
            model_over = round(over_count / n_sims * 100, 1)
            model_under = round(100 - model_over, 1)
            result["ou_line"] = ou_line
            result["model_over_pct"] = model_over
            result["model_under_pct"] = model_under
            result["ou_over_implied"] = ou["over_implied"]
            result["ou_under_implied"] = ou["under_implied"]
            result["ou_over_odds"] = format_odds(ou["over_odds"])
            result["ou_under_odds"] = format_odds(ou["under_odds"])
            result["ou_over_edge"] = round(model_over - ou["over_implied"], 1)
            result["ou_under_edge"] = round(model_under - ou["under_implied"], 1)
            result["ou_over_ev"] = calc_ev(model_over, ou["over_odds"])
            result["ou_under_ev"] = calc_ev(model_under, ou["under_odds"])
    except Exception:
        pass
    try:
        all_rl = get_mlb_runline()
        rl = all_rl.get(odds_key, {})
        if rl:
            result["away_rl_odds"] = format_odds(rl["away_rl_odds"])
            result["home_rl_odds"] = format_odds(rl["home_rl_odds"])
            result["away_rl_implied"] = rl["away_rl_implied"]
            result["home_rl_implied"] = rl["home_rl_implied"]
            result["away_rl_edge"] = round(
                result.get("away_cover_pct", 0) - rl["away_rl_implied"], 1
            )
            result["home_rl_edge"] = round(
                result.get("home_cover_pct", 0) - rl["home_rl_implied"], 1
            )
            result["away_rl_ev"] = calc_ev(result.get("away_cover_pct", 0), rl["away_rl_odds"])
            result["home_rl_ev"] = calc_ev(result.get("home_cover_pct", 0), rl["home_rl_odds"])
            result["away_rl_kelly"] = calc_kelly(
                result.get("away_cover_pct", 0), rl["away_rl_odds"]
            )
            result["home_rl_kelly"] = calc_kelly(
                result.get("home_cover_pct", 0), rl["home_rl_odds"]
            )
    except Exception:
        pass

    # ── Public betting percentages (Action Network) ──────────────────
    try:
        pub_pcts = get_public_betting_pcts()
        pub = pub_pcts.get(odds_key, {})
        result["away_bet_pct"] = pub.get("away_bet_pct")
        result["home_bet_pct"] = pub.get("home_bet_pct")
        result["away_money_pct"] = pub.get("away_money_pct")
        result["home_money_pct"] = pub.get("home_money_pct")
        result["sharp_indicator"] = pub.get("sharp_indicator")
    except Exception:
        result["away_bet_pct"] = result["home_bet_pct"] = None
        result["away_money_pct"] = result["home_money_pct"] = None
        result["sharp_indicator"] = None

    # Props are NOT auto-fetched here — user clicks "Load Props" button
    # which calls /props/<game_pk> separately (saves 2 API calls per simulation)
    result["props_by_player"] = {}

    # ── Batter prop model predictions ────────────────────────────────────
    try:
        away_batter_props = predict_batter_props(
            away_batter_stats,
            home_pitcher_stats,
            weather=weather,
            venue=game.get("venue"),
            recent_stats_list=away_recent,
            umpire_name=umpire_name,
            opp_catcher_cs=home_catcher_cs,
        )
        home_batter_props = predict_batter_props(
            home_batter_stats,
            away_pitcher_stats,
            weather=weather,
            venue=game.get("venue"),
            recent_stats_list=home_recent,
            umpire_name=umpire_name,
            opp_catcher_cs=away_catcher_cs,
        )
        # Attach batter names to each slot
        for i, prop in enumerate(away_batter_props):
            b = lineup["away_batters"][i] if i < len(lineup["away_batters"]) else {}
            prop["name"] = b.get("name", f"Batter {i + 1}")
        for i, prop in enumerate(home_batter_props):
            b = lineup["home_batters"][i] if i < len(lineup["home_batters"]) else {}
            prop["name"] = b.get("name", f"Batter {i + 1}")
    except Exception:
        away_batter_props = []
        home_batter_props = []

    result["away_batter_props"] = away_batter_props
    result["home_batter_props"] = home_batter_props

    # Cache batter props for SGP correlated probability calculations
    if away_batter_props or home_batter_props:
        if len(_batter_props_cache) >= _SCORE_PAIRS_MAXSIZE:
            _batter_props_cache.pop(next(iter(_batter_props_cache)))
        _batter_props_cache[game_pk] = {
            "away": away_batter_props,
            "home": home_batter_props,
            "away_team": result.get("away_team", ""),
            "home_team": result.get("home_team", ""),
        }

    # ── Batting order optimizer ───────────────────────────────────────
    try:
        result["away_order_opt"] = optimize_batting_order(
            away_batter_stats,
            home_pitcher_stats,
            weather=weather,
            venue=game.get("venue"),
        )
        result["home_order_opt"] = optimize_batting_order(
            home_batter_stats,
            away_pitcher_stats,
            weather=weather,
            venue=game.get("venue"),
        )
    except Exception:
        result["away_order_opt"] = {}
        result["home_order_opt"] = {}

    # ── Pitcher arsenal ───────────────────────────────────────────────
    try:
        result["away_arsenal"] = (
            get_pitcher_arsenal(away_pitcher.get("id")) if away_pitcher.get("id") else {}
        )
        result["home_arsenal"] = (
            get_pitcher_arsenal(home_pitcher.get("id")) if home_pitcher.get("id") else {}
        )
    except Exception:
        result["away_arsenal"] = {}
        result["home_arsenal"] = {}

    # ── Stadium profile ───────────────────────────────────────────────
    result["stadium_profile"] = _build_stadium_profile(
        game.get("venue", ""),
        result.get("away_lineup", []),
        result.get("home_lineup", []),
    )

    return render_template("result.html", game_pk=game_pk, n_sims=n_sims, **result)


def _build_stadium_profile(venue, away_batters, home_batters):
    """Build a stadium split profile for the result page."""
    from simulation.engine import BALLPARK_FACTORS, DEFAULT_PARK_FACTOR

    park = BALLPARK_FACTORS.get(venue, DEFAULT_PARK_FACTOR)
    hr_f = park.get("hr", 1.0)
    hit_f = park.get("hit", 1.0)
    run_f = park.get("run", 1.0)

    if hr_f >= 1.10:
        park_type = "hitter-friendly"
        park_color = "#81c784"
    elif hr_f <= 0.90:
        park_type = "pitcher-friendly"
        park_color = "#ef5350"
    else:
        park_type = "neutral"
        park_color = "#9aa0b8"

    def batter_impact(batters):
        impacts = []
        for s in (batters or [])[:9]:
            name = s.get("name", "")
            if not name:
                continue
            from simulation.engine import build_batter_probs

            try:
                probs = build_batter_probs(s)
            except Exception:
                continue
            p_hr = probs[5]
            p_hit = probs[2] + probs[3] + probs[4] + probs[5]
            # Power hitter = p_hr > 4% per PA
            is_power = p_hr > 0.04
            # Contact hitter = p_hit > 28% per PA
            is_contact = p_hit > 0.28
            adj_hr = round((hr_f - 1.0) * 100, 0)
            adj_hit = round((hit_f - 1.0) * 100, 0)
            if is_power and abs(adj_hr) >= 5:
                note = f"HR factor {'+' if adj_hr > 0 else ''}{int(adj_hr)}%"
                flag = "power"
            elif is_contact and abs(adj_hit) >= 3:
                note = f"Hit factor {'+' if adj_hit > 0 else ''}{int(adj_hit)}%"
                flag = "contact"
            else:
                note = None
                flag = None
            impacts.append(
                {
                    "name": name,
                    "note": note,
                    "flag": flag,
                    "p_hr": round(p_hr * 100, 1),
                    "p_hit": round(p_hit * 100, 1),
                }
            )
        return impacts

    return {
        "venue": venue,
        "hr_factor": round(hr_f, 3),
        "hit_factor": round(hit_f, 3),
        "run_factor": round(run_f, 3),
        "park_type": park_type,
        "park_color": park_color,
        "away_impacts": batter_impact(away_batters),
        "home_impacts": batter_impact(home_batters),
    }


@app.route("/api/notes/<int:game_pk>", methods=["POST"])
@login_required
def api_save_note(game_pk):
    data = request.get_json() or {}
    note = (data.get("note") or "").strip()
    if not note:
        return jsonify({"ok": False, "error": "empty"}), 400
    try:
        save_game_note(
            user_id=current_user.id,
            game_pk=game_pk,
            game_date=data.get("game_date", _today_est().isoformat()),
            away_team=data.get("away_team", ""),
            home_team=data.get("home_team", ""),
            note=note,
        )
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"ok": False, "error": "An error occurred"}), 500


@app.route("/api/notes/<int:game_pk>", methods=["DELETE"])
@login_required
def api_delete_note(game_pk):
    try:
        delete_game_note(current_user.id, game_pk)
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"ok": False, "error": "An error occurred"}), 500


_props_cache = {}  # game_pk → (timestamp, props dict)
_PROPS_TTL = 1800  # 30 min


@app.route("/props/<int:game_pk>")
def load_props(game_pk):
    """
    On-demand player props — only called when user clicks Load Props.
    Cached 30 min per game so repeated clicks don't cost extra credits.
    Returns {} immediately if no event ID found (free — no API call made).
    """
    import time as _time

    now = _time.time()
    cached = _props_cache.get(game_pk)
    if cached and (now - cached[0]) < _PROPS_TTL:
        return jsonify(cached[1])

    games = get_today_schedule()
    game = next((g for g in games if g["gamePk"] == game_pk), None)
    if not game:
        return jsonify({})

    odds_key = frozenset([game["away_team"], game["home_team"]])
    all_events = get_mlb_events()
    event_id = all_events.get(odds_key)
    if not event_id:
        return jsonify({})  # no event found — don't spend credits

    props = get_player_props(event_id)
    _props_cache[game_pk] = (now, props)
    return jsonify(props)


@app.route("/simulate-all")
@login_required
def simulate_all():
    """
    Simulate every active game on today's slate.

    Speed strategy:
      1. Fetch all lineups in parallel
      2. Collect every unique player ID across all games
      3. Fetch ALL player stats in one big parallel batch (no duplicates)
      4. Run simulations (fast — pre-computed probs)

    This way we make the minimum number of API calls and run them
    all at the same time, cutting total time from 3+ minutes to ~20-30s.
    """
    # Use date from query param (passed by index.html button) so UTC clock drift doesn't shift the date
    from datetime import datetime as _dt

    _date_str = request.args.get("date", _today_est().isoformat())
    try:
        _sim_date = _dt.strptime(_date_str, "%Y-%m-%d").date()
    except ValueError:
        _sim_date = _today_est()

    all_games = get_today_schedule(game_date=_sim_date.isoformat())

    # Skip finished games
    games = [
        g
        for g in all_games
        if "final" not in g["status"].lower() and "game over" not in g["status"].lower()
    ]

    if not games:
        from flask import redirect, url_for

        return redirect(
            url_for(
                "index",
                date=_sim_date.isoformat(),
                error="No active or upcoming games to simulate right now.",
            )
        )

    # ── Step 1: Fetch all lineups in parallel ─────────────────────
    lineups = {}

    def fetch_lineup(game):
        try:
            lu = get_game_lineup(game["gamePk"])
            if lu["away_batters"] and lu["home_batters"]:
                return game["gamePk"], lu
        except Exception:
            pass
        return game["gamePk"], None

    with ThreadPoolExecutor(max_workers=9) as ex:
        for gk, lu in ex.map(fetch_lineup, games):
            if lu:
                lineups[gk] = lu

    # ── Step 2: Collect all unique player IDs ─────────────────────
    batter_ids = set()
    pitcher_ids = set()
    for lu in lineups.values():
        for b in lu["away_batters"] + lu["home_batters"]:
            if b.get("id"):
                batter_ids.add(b["id"])
        for side in ("away_pitcher", "home_pitcher"):
            p = lu.get(side)
            if p and p.get("id"):
                pitcher_ids.add(p["id"])

    # ── Step 3: Fetch all stats in one parallel batch ─────────────
    batter_cache = {}
    pitcher_cache = {}

    def _fetch_batter(pid):
        try:
            return pid, get_player_season_stats(pid, group="hitting")
        except Exception:
            return pid, {}

    def _fetch_pitcher(pid):
        try:
            return pid, get_player_season_stats(pid, group="pitching")
        except Exception:
            return pid, {}

    with ThreadPoolExecutor(max_workers=20) as ex:
        for pid, stats in ex.map(_fetch_batter, batter_ids):
            batter_cache[pid] = stats
        for pid, stats in ex.map(_fetch_pitcher, pitcher_ids):
            pitcher_cache[pid] = stats

    # ── Step 4: Fetch weather for each venue ──────────────────────
    venue_weather = {}
    unique_venues = {g["venue"] for g in games}

    def _fetch_weather(venue):
        try:
            return venue, get_ballpark_weather(venue)
        except Exception:
            return venue, {}

    with ThreadPoolExecutor(max_workers=9) as ex:
        for venue, w in ex.map(_fetch_weather, unique_venues):
            venue_weather[venue] = w

    # ── Step 5: Run simulations ───────────────────────────────────
    results = []
    for game in games:
        lu = lineups.get(game["gamePk"])
        if not lu:
            continue

        away_pitcher = lu.get("away_pitcher") or {}
        home_pitcher = lu.get("home_pitcher") or {}
        away_ps = pitcher_cache.get(away_pitcher.get("id"), {})
        home_ps = pitcher_cache.get(home_pitcher.get("id"), {})
        weather = venue_weather.get(game["venue"], {})

        away_stats = [batter_cache.get(b["id"], {}) for b in lu["away_batters"]]
        home_stats = [batter_cache.get(b["id"], {}) for b in lu["home_batters"]]

        result = run_simulation(
            away_team=game["away_team"],
            home_team=game["home_team"],
            away_lineup=away_stats,
            home_lineup=home_stats,
            away_pitcher=away_ps,
            home_pitcher=home_ps,
            weather=weather,
            venue=game["venue"],
            n=N_SIMS_ALL,
        )
        result.update(
            {
                "gamePk": game["gamePk"],
                "venue": game["venue"],
                "status": game["status"],
                "away_pitcher_name": away_pitcher.get("name", "TBD"),
                "home_pitcher_name": home_pitcher.get("name", "TBD"),
                "away_pitcher_hand": "R",
                "home_pitcher_hand": "R",
                "away_era": away_ps.get("era", "N/A"),
                "away_whip": away_ps.get("whip", "N/A"),
                "away_wl": f"{away_ps.get('wins', 0)}-{away_ps.get('losses', 0)}",
                "home_era": home_ps.get("era", "N/A"),
                "home_whip": home_ps.get("whip", "N/A"),
                "home_wl": f"{home_ps.get('wins', 0)}-{home_ps.get('losses', 0)}",
                "away_batters": lu["away_batters"],
                "home_batters": lu["home_batters"],
                "weather": weather,
                "game_time_utc": game.get("game_time", ""),
            }
        )
        results.append(result)

    results.sort(key=lambda r: r["gamePk"])

    if not results:
        from flask import redirect, url_for

        return redirect(
            url_for(
                "index",
                date=_sim_date.isoformat(),
                error="Lineups not posted yet for remaining games — check back closer to game time.",
            )
        )

    # ── Log predictions for accuracy tracking ─────────────────
    try:
        for r in results:
            predicted_winner = (
                r["away_team"] if r["away_win_pct"] >= r["home_win_pct"] else r["home_team"]
            )
            predicted_win_pct = max(r["away_win_pct"], r["home_win_pct"])
            predicted_total = r.get("avg_away_runs", 0) + r.get("avg_home_runs", 0)
            log_prediction(
                game_pk=r["gamePk"],
                game_date=_today_est().isoformat(),
                away_team=r["away_team"],
                home_team=r["home_team"],
                away_win_pct=r["away_win_pct"],
                home_win_pct=r["home_win_pct"],
                away_avg_runs=r.get("avg_away_runs", 0),
                home_avg_runs=r.get("avg_home_runs", 0),
                n_sims=N_SIMS_ALL,
            )
    except Exception:
        pass

    # ── Build the Best Parlay ──────────────────────────────────
    # A parlay is a bet where you pick multiple games and all must win.
    # We pick games where one team has ≥60% win probability.
    parlay_picks = []
    for r in results:
        if r["away_win_pct"] >= PARLAY_THRESHOLD:
            parlay_picks.append(
                {
                    "team": r["away_team"],
                    "opponent": r["home_team"],
                    "win_pct": r["away_win_pct"],
                    "avg_runs": r["away_avg_runs"],
                    "venue": r["venue"],
                }
            )
        elif r["home_win_pct"] >= PARLAY_THRESHOLD:
            parlay_picks.append(
                {
                    "team": r["home_team"],
                    "opponent": r["away_team"],
                    "win_pct": r["home_win_pct"],
                    "avg_runs": r["home_avg_runs"],
                    "venue": r["venue"],
                }
            )

    # Combined parlay probability = multiply all individual win chances
    combined_prob = 1.0
    for pick in parlay_picks:
        combined_prob *= pick["win_pct"] / 100.0
    combined_prob_pct = round(combined_prob * 100, 1)

    # ── Parlay EV calculator ──────────────────────────────────────────
    parlay_ev = None
    parlay_true_odds = None
    parlay_book_odds = None
    if len(parlay_picks) >= 2:
        true_decimal = 1.0
        for pick in parlay_picks:
            true_decimal *= 1.0 / (pick["win_pct"] / 100.0)
        book_decimal = 1.909 ** len(parlay_picks)
        parlay_ev = round(combined_prob * (book_decimal - 1) * 100 - (1 - combined_prob) * 100, 1)
        if true_decimal >= 2.0:
            parlay_true_odds = "+" + str(int((true_decimal - 1) * 100))
        else:
            parlay_true_odds = "-" + str(int(100 / (true_decimal - 1)))
        if book_decimal >= 2.0:
            parlay_book_odds = "+" + str(int((book_decimal - 1) * 100))
        else:
            parlay_book_odds = "-" + str(int(100 / (book_decimal - 1)))

    # ── Confidence-tiered best bets ───────────────────────────────────
    # Only show games where the model has enough conviction to matter.
    # 62-64% = moderate edge, 65-69% = good, 70%+ = strong lean
    best_bets = []
    for r in results:
        fav_team = None
        fav_pct = None
        fav_opp = None

        if r["away_win_pct"] >= r["home_win_pct"] and r["away_win_pct"] >= BET_THRESHOLD:
            fav_team, fav_pct, fav_opp = r["away_team"], r["away_win_pct"], r["home_team"]
        elif r["home_win_pct"] > r["away_win_pct"] and r["home_win_pct"] >= BET_THRESHOLD:
            fav_team, fav_pct, fav_opp = r["home_team"], r["home_win_pct"], r["away_team"]

        if fav_team:
            if fav_pct >= 70:
                tier, tier_label, tier_color = "strong", "🔥 Strong (70%+)", "#ef5350"
            elif fav_pct >= 65:
                tier, tier_label, tier_color = "good", "✅ Good (65-69%)", "#4caf50"
            else:
                tier, tier_label, tier_color = "moderate", "💛 Moderate (62-64%)", "#ffd54f"

            best_bets.append(
                {
                    "team": fav_team,
                    "opponent": fav_opp,
                    "win_pct": fav_pct,
                    "tier": tier,
                    "tier_label": tier_label,
                    "tier_color": tier_color,
                    "venue": r["venue"],
                    "gamePk": r["gamePk"],
                }
            )

    best_bets.sort(key=lambda b: b["win_pct"], reverse=True)

    # ── Fetch Vegas odds and attach to each result ────────────────────────────
    # Uses the same cached call as the single-game view (30-min cache).
    # Matches by frozenset({away_team, home_team}) so order doesn't matter.
    try:
        all_odds = get_mlb_odds()
    except Exception:
        all_odds = {}

    for r in results:
        odds_key = frozenset([r["away_team"], r["home_team"]])
        game_odds = all_odds.get(odds_key, {})
        if game_odds:
            r["away_implied_pct"] = game_odds["away_implied_pct"]
            r["home_implied_pct"] = game_odds["home_implied_pct"]
            r["away_avg_odds"] = format_odds(game_odds["away_avg_odds"])
            r["home_avg_odds"] = format_odds(game_odds["home_avg_odds"])
            r["away_edge"] = calc_edge(r["away_win_pct"], game_odds["away_implied_pct"])
            r["home_edge"] = calc_edge(r["home_win_pct"], game_odds["home_implied_pct"])
            r["away_ev"] = calc_ev(r["away_win_pct"], game_odds["away_avg_odds"])
            r["home_ev"] = calc_ev(r["home_win_pct"], game_odds["home_avg_odds"])
            r["away_kelly"] = calc_kelly(r["away_win_pct"], game_odds["away_avg_odds"])
            r["home_kelly"] = calc_kelly(r["home_win_pct"], game_odds["home_avg_odds"])
        else:
            for k in (
                "away_implied_pct",
                "home_implied_pct",
                "away_avg_odds",
                "home_avg_odds",
                "away_edge",
                "home_edge",
                "away_ev",
                "home_ev",
                "away_kelly",
                "home_kelly",
            ):
                r[k] = None
        # O/U + run line per game card
        try:
            all_totals = get_mlb_totals()
            all_rl = get_mlb_runline()
            ok = frozenset([r["away_team"], r["home_team"]])
            ou = all_totals.get(ok, {})
            if ou:
                hist = r.get("total_runs_hist", {})
                ns = r.get("simulations", 1)
                oul = ou["line"]
                oc = sum(cnt for tot, cnt in hist.items() if tot > oul)
                mo = round(oc / ns * 100, 1)
                r["ou_line"] = oul
                r["model_over_pct"] = mo
                r["model_under_pct"] = round(100 - mo, 1)
                r["ou_over_implied"] = ou["over_implied"]
                r["ou_under_implied"] = ou["under_implied"]
                r["ou_over_odds"] = format_odds(ou["over_odds"])
                r["ou_under_odds"] = format_odds(ou["under_odds"])
                r["ou_over_edge"] = round(mo - ou["over_implied"], 1)
                r["ou_under_edge"] = round((100 - mo) - ou["under_implied"], 1)
                r["ou_over_ev"] = calc_ev(mo, ou["over_odds"])
                r["ou_under_ev"] = calc_ev(100 - mo, ou["under_odds"])
            rl = all_rl.get(ok, {})
            if rl:
                r["away_rl_odds"] = format_odds(rl["away_rl_odds"])
                r["home_rl_odds"] = format_odds(rl["home_rl_odds"])
                r["away_rl_implied"] = rl["away_rl_implied"]
                r["home_rl_implied"] = rl["home_rl_implied"]
                r["away_rl_edge"] = round(r.get("away_cover_pct", 0) - rl["away_rl_implied"], 1)
                r["home_rl_edge"] = round(r.get("home_cover_pct", 0) - rl["home_rl_implied"], 1)
                r["away_rl_ev"] = calc_ev(r.get("away_cover_pct", 0), rl["away_rl_odds"])
                r["home_rl_ev"] = calc_ev(r.get("home_cover_pct", 0), rl["home_rl_odds"])
        except Exception:
            pass

    # ── Line movement / sharp money signals ─────────────────────────
    try:
        movement = get_line_movement()
        for r in results:
            mv = movement.get(frozenset([r["away_team"], r["home_team"]]), {})
            r["sharp_away"] = mv.get("sharp_away", False)
            r["sharp_home"] = mv.get("sharp_home", False)
            r["away_impl_move"] = mv.get("away_impl_move", 0)
            r["home_impl_move"] = mv.get("home_impl_move", 0)
            r["away_open_odds"] = format_odds(mv["away_open"]) if mv.get("away_open") else None
            r["home_open_odds"] = format_odds(mv["home_open"]) if mv.get("home_open") else None
    except Exception:
        pass

    # Also attach odds to best_bets entries
    for b in best_bets:
        r_match = next((r for r in results if r["gamePk"] == b["gamePk"]), {})
        is_away = b["team"] == r_match.get("away_team")
        b["implied_pct"] = r_match.get("away_implied_pct" if is_away else "home_implied_pct")
        b["odds_str"] = r_match.get("away_avg_odds" if is_away else "home_avg_odds")
        b["edge"] = r_match.get("away_edge" if is_away else "home_edge")

    today = _today_est().strftime("%A, %B %d %Y")
    return render_template(
        "all_results.html",
        results=results,
        parlay_picks=parlay_picks,
        combined_prob=combined_prob_pct,
        parlay_ev=parlay_ev,
        parlay_true_odds=parlay_true_odds,
        parlay_book_odds=parlay_book_odds,
        best_bets=best_bets,
        date=today,
        n_sims=N_SIMS_ALL,
        api_remaining=get_requests_remaining(),
        username=current_user.username,
    )


# ── SGP Correlation Factors ──────────────────────────────────────────────────
# Multiplicative adjustments to base prop probabilities conditioned on game legs.
# Derived from MLB seasonal correlations between game outcomes and batter stats.
_SGP_BATTER_CORR = {
    "ml_same":  {"hits": 1.12, "hr": 1.08, "rbi": 1.18, "tb": 1.14, "runs": 1.20},
    "ml_opp":   {"hits": 0.92, "hr": 0.94, "rbi": 0.85, "tb": 0.90, "runs": 0.82},
    "rl_same":  {"hits": 1.20, "hr": 1.15, "rbi": 1.30, "tb": 1.22, "runs": 1.28},
    "rl_opp":   {"hits": 0.85, "hr": 0.88, "rbi": 0.75, "tb": 0.82, "runs": 0.75},
    "over":     {"hits": 1.10, "hr": 1.12, "rbi": 1.15, "tb": 1.12, "runs": 1.15},
    "under":    {"hits": 0.90, "hr": 0.88, "rbi": 0.85, "tb": 0.88, "runs": 0.85},
}
_SGP_PITCHER_K_CORR = {
    "ml_same": 1.05, "ml_opp": 0.97,
    "rl_same": 1.08, "rl_opp": 0.94,
    "over": 1.03, "under": 1.08,
}


def _classify_game_leg(leg_type, player_team):
    """Map a game leg type + player team to a correlation key."""
    if leg_type in ("ml_away", "ml_home"):
        is_same = (leg_type == "ml_away" and player_team == "away") or (
            leg_type == "ml_home" and player_team == "home"
        )
        return "ml_same" if is_same else "ml_opp"
    if leg_type in ("rl_away", "rl_home"):
        is_same = (leg_type == "rl_away" and player_team == "away") or (
            leg_type == "rl_home" and player_team == "home"
        )
        return "rl_same" if is_same else "rl_opp"
    if leg_type == "over":
        return "over"
    if leg_type == "under":
        return "under"
    return None


def _correlated_prop_prob(base_prob, prop_market, player_team, game_legs):
    """Adjust a base prop probability for correlation with game-level legs."""
    adj = base_prob
    for leg in game_legs:
        key = _classify_game_leg(leg.get("type", ""), player_team)
        if not key:
            continue
        if prop_market == "k":
            factor = _SGP_PITCHER_K_CORR.get(key, 1.0)
        else:
            factor = _SGP_BATTER_CORR.get(key, {}).get(prop_market, 1.0)
        adj *= factor
    return max(0.01, min(0.99, adj))


@app.route("/sgp/<int:game_pk>", methods=["POST"])
def sgp_calc(game_pk):
    """Compute correlated SGP probability from simulation score_pairs + player props."""
    data = request.get_json(silent=True) or {}
    legs = data.get("legs", [])
    prop_legs = data.get("prop_legs", [])

    if not legs and not prop_legs:
        return jsonify({"error": "Add at least one leg"}), 400

    raw_pairs = _score_pairs_cache.get(game_pk)
    if not raw_pairs and legs:
        return jsonify({"error": "Run simulation first"}), 400

    # Game-level probability from score_pairs
    game_prob = 1.0
    total = 0
    game_hits = 0
    if legs and raw_pairs:
        pairs = {}
        for k, v in raw_pairs.items():
            try:
                a, h = k.strip("()").split(", ")
                pairs[(int(a), int(h))] = v
            except Exception:
                pass

        total = sum(pairs.values())
        if total == 0:
            return jsonify({"error": "No simulation data"}), 400

        def leg_hits(a, h, leg):
            t = leg.get("type", "")
            line = float(leg.get("line", 0))
            if t == "ml_away":
                return a > h
            if t == "ml_home":
                return h > a
            if t == "over":
                return (a + h) > line
            if t == "under":
                return (a + h) < line
            if t == "rl_away":
                return (a - h) > line
            if t == "rl_home":
                return (h - a) > line
            return False

        game_hits = 0
        for (a, h), count in pairs.items():
            if all(leg_hits(a, h, leg) for leg in legs):
                game_hits += count
        game_prob = game_hits / total

    # Prop legs: apply correlation adjustments
    uncorrelated_prob = game_prob
    combined_prob = game_prob
    for pl in prop_legs:
        base = float(pl.get("base_prob", 0.5))
        uncorrelated_prob *= base
        adj = _correlated_prop_prob(base, pl.get("market", "hits"), pl.get("team", "away"), legs)
        combined_prob *= adj

    if combined_prob <= 0:
        combined_prob = 0.001

    fair_odds = round(
        (-100 / combined_prob) if combined_prob >= 0.5 else (100 * (1 - combined_prob) / combined_prob),
        0,
    )
    fair_odds_str = f"+{int(fair_odds)}" if fair_odds > 0 else str(int(fair_odds))

    return jsonify(
        {
            "prob": round(combined_prob * 100, 2),
            "fair_odds": fair_odds_str,
            "legs": len(legs),
            "prop_legs": len(prop_legs),
            "total_legs": len(legs) + len(prop_legs),
            "hits": game_hits,
            "total": total,
            "uncorrelated_prob": round(uncorrelated_prob * 100, 2),
        }
    )


@app.route("/parlay")
def parlay_builder():
    """Interactive parlay builder — pick legs from today's games."""
    from datetime import date as _date

    today_str = _today_est().strftime("%A, %B %d %Y")

    # Load today's schedule so the user can pick from real matchups
    try:
        games = get_today_schedule()
    except Exception:
        games = []

    # Fetch current ML odds for each game to compute parlay book odds accurately
    try:
        all_odds = get_mlb_odds()
    except Exception:
        all_odds = {}

    # Build a clean game list with odds attached
    game_list = []
    for g in games:
        key = frozenset([g["away_team"], g["home_team"]])
        odds = all_odds.get(key, {})
        game_list.append(
            {
                "gamePk": g["gamePk"],
                "away_team": g["away_team"],
                "home_team": g["home_team"],
                "away_probable": g.get("away_probable", "TBD"),
                "home_probable": g.get("home_probable", "TBD"),
                "venue": g.get("venue", ""),
                "away_odds": format_odds(odds["away_avg_odds"])
                if odds.get("away_avg_odds")
                else None,
                "home_odds": format_odds(odds["home_avg_odds"])
                if odds.get("home_avg_odds")
                else None,
                "away_implied": odds.get("away_implied_pct"),
                "home_implied": odds.get("home_implied_pct"),
            }
        )

    return render_template(
        "parlay.html", games=game_list, date=today_str, api_remaining=get_requests_remaining()
    )


@app.route("/my-picks", methods=["GET", "POST"])
@login_required
def my_picks():
    """Personal picks log."""
    from flask import request as freq

    if freq.method == "POST":
        data = freq.get_json() or {}
        try:
            add_pick(
                game_pk=data.get("game_pk"),
                game_date=data.get("game_date", _today_est().isoformat()),
                away_team=data.get("away_team", ""),
                home_team=data.get("home_team", ""),
                my_pick=data.get("my_pick") or data.get("pick", ""),
                my_notes=data.get("my_notes", ""),
                bet_type=data.get("bet_type", "moneyline"),
                sim_away_pct=data.get("sim_away_pct"),
                sim_home_pct=data.get("sim_home_pct"),
                sim_away_runs=data.get("sim_away_runs"),
                sim_home_runs=data.get("sim_home_runs"),
                user_id=_uid(),
            )
            return jsonify({"status": "ok"})
        except Exception:
            import traceback

            traceback.print_exc()
            return jsonify({"status": "error", "error": "An error occurred"}), 500
    update_pick_results(user_id=_uid())
    games = get_today_schedule()
    stats = get_pick_stats(user_id=_uid())
    return render_template("my_picks.html", games=games, username=current_user.username, **stats)


@app.route("/my-picks/update", methods=["POST"])
@login_required
def update_my_picks():
    uid = _uid()
    updated = update_pick_results(user_id=uid)
    stats = get_pick_stats(user_id=uid)
    return jsonify({"updated": updated, "my_pct": stats["my_pct"], "sim_pct": stats["sim_pct"]})


@app.route("/odds-history")
@login_required
def odds_history_page():
    date_from = request.args.get("from")
    date_to = request.args.get("to")
    settle_odds_history()
    rows = get_odds_history(limit=300, date_from=date_from, date_to=date_to)

    # Compute summary stats
    total = len(rows)
    settled = [r for r in rows if r.get("actual_winner")]
    vegas_correct = sum(
        1
        for r in settled
        if r.get("actual_winner")
        and (
            (r.get("away_ml") and r["away_ml"] < 0 and r["actual_winner"] == r["away_team"])
            or (r.get("home_ml") and r["home_ml"] < 0 and r["actual_winner"] == r["home_team"])
        )
    )
    model_correct = sum(
        1
        for r in settled
        if r.get("actual_winner")
        and r.get("model_away_pct")
        and (
            (r["model_away_pct"] >= r.get("model_home_pct", 50) and r["actual_winner"] == r["away_team"])
            or (r.get("model_home_pct", 50) > r["model_away_pct"] and r["actual_winner"] == r["home_team"])
        )
    )
    stats = {
        "total": total,
        "settled": len(settled),
        "vegas_correct": vegas_correct,
        "vegas_pct": round(vegas_correct / len(settled) * 100, 1) if settled else 0,
        "model_correct": model_correct,
        "model_pct": round(model_correct / len(settled) * 100, 1) if settled else 0,
    }
    return render_template(
        "odds_history.html", rows=rows, stats=stats, date_from=date_from or "", date_to=date_to or ""
    )


@app.route("/accuracy")
@login_required
def accuracy_page():
    update_results()
    stats = get_accuracy_stats()
    return render_template("accuracy.html", **stats, username=current_user.username)


@app.route("/update-results", methods=["POST"])
@login_required
def trigger_update():
    updated = update_results()
    stats = get_accuracy_stats()
    return jsonify(
        {
            "updated": updated,
            "accuracy_pct": stats["accuracy_pct"],
            "correct_picks": stats["correct_picks"],
            "results_available": stats["results_available"],
        }
    )


@app.route("/api/live-scores")
def api_live_scores():
    """Public JSON endpoint — live/final/scheduled scores."""
    from datetime import date as _date

    game_date = request.args.get("date", _today_est().isoformat())
    try:
        scores = get_live_scores(game_date)
    except Exception:
        scores = []
    return jsonify(scores)


@app.route("/api/scores")
def api_scores():
    """JSON: today's games with live scores for the scoreboard bar."""
    from data.mlb_api import get_live_scores

    try:
        games = get_live_scores() or []
    except Exception:
        games = []
    out = []
    for g in games:
        out.append(
            {
                "away": g.get("away_team", ""),
                "home": g.get("home_team", ""),
                "away_score": g.get("away_score", ""),
                "home_score": g.get("home_score", ""),
                "status": g.get("status", ""),
                "inning": g.get("inning", ""),
                "inning_half": (g.get("inning_half", "") or "").lower(),
                "game_time": g.get("game_time_utc", ""),
                "gamePk": g.get("gamePk", ""),
            }
        )
    return jsonify(out)


@app.route("/api/bets")
@login_required
def api_bets():
    """JSON: all bets for the logged-in user (bankroll page)."""
    return jsonify(get_all_bets(user_id=_uid()))


@app.route("/bankroll")
@login_required
def bankroll_page():
    return render_template("bankroll.html", username=current_user.username)


@app.route("/calculator")
@login_required
def calculator_page():
    return render_template("calculator.html", username=current_user.username)


@app.route("/bets")
@login_required
def bets_dashboard():
    try:
        settle_bets(user_id=_uid())
    except Exception:
        pass
    try:
        stats = get_bet_stats(user_id=_uid())
    except Exception:
        stats = {
            "bets": [],
            "total_bets": 0,
            "settled": 0,
            "pending": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_pct": None,
            "total_wagered": 0,
            "total_profit": 0,
            "roi": 0,
            "avg_clv": None,
            "by_type": {},
        }
    stats["today"] = _today_est().isoformat()
    stats["total_pl"] = stats.get("total_profit", 0)
    stats["profit_loss"] = stats.get("total_profit", 0)
    stats["win_rate"] = stats.get("win_pct", None)
    stats["wagered"] = stats.get("total_wagered", 0)
    stats["recent_bets"] = stats.get("bets", [])[:20]
    return render_template("bets.html", **stats, username=current_user.username)


@app.route("/bets/log", methods=["POST"])
@login_required
def log_bet_route():
    data = request.get_json() or request.form
    try:
        log_bet(
            game_pk=data["game_pk"],
            game_date=data.get("game_date", ""),
            away_team=data["away_team"],
            home_team=data["home_team"],
            bet_on=data["bet_on"],
            bet_type=data.get("bet_type", "ML"),
            odds=int(data["odds"]),
            amount=float(data.get("amount", 100)),
            model_edge=data.get("model_edge"),
            ev=data.get("ev"),
            kelly=data.get("kelly"),
            user_id=_uid(),
        )
        return jsonify({"status": "ok"})
    except Exception:
        return jsonify({"status": "error", "msg": "Invalid request"}), 400


@app.route("/bets/settle", methods=["POST"])
@login_required
def settle_bets_route():
    n = settle_bets(user_id=_uid())
    return jsonify({"settled": n})


# ── Account Settings ─────────────────────────────────────────────────────────


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings_page():
    from data.email_alerts import get_user_email_settings, update_user_email

    uid = _uid()
    message = None
    error = None

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        alerts_on = request.form.get("email_alerts") == "on"
        if email and "@" not in email:
            error = "Please enter a valid email address."
        else:
            ok = update_user_email(uid, email, alerts_on)
            message = "Settings saved." if ok else "Could not save — try again."

    prefs = get_user_email_settings(uid)
    return render_template(
        "settings.html",
        username=current_user.username,
        email=prefs.get("email") or "",
        email_alerts=prefs.get("email_alerts", False),
        message=message,
        error=error,
    )


# ── Daily email alerts cron endpoint ────────────────────────────────────────
# Triggered by Vercel Cron at 10:00 AM ET every day.
# Protected by a shared secret so only the cron job can call it.


@app.route("/send-daily-alerts", methods=["POST", "GET"])
@csrf.exempt
def send_daily_alerts_route():
    from data.email_alerts import send_daily_alerts

    # Simple secret check — set CRON_SECRET in Vercel env vars
    secret = os.getenv("CRON_SECRET")
    if not secret:
        return jsonify({"error": "unauthorized"}), 401
    auth = request.headers.get("Authorization", "") or request.args.get("secret", "")
    if auth != f"Bearer {secret}" and auth != secret:
        return jsonify({"error": "unauthorized"}), 401

    # Run fast simulations on today's games to find top plays
    try:
        games = get_today_schedule()
    except Exception:
        return jsonify({"error": "could not fetch schedule"}), 500

    all_odds_data = {}
    try:
        all_odds_data = get_mlb_odds()
    except Exception:
        pass

    top_plays = []
    for game in games[:15]:  # cap at 15 games to keep cron fast
        try:
            lineup = get_game_lineup(game["gamePk"])
            if not lineup:
                continue

            away_hand = get_pitcher_hand(lineup.get("away_pitcher_id"))
            home_hand = get_pitcher_hand(lineup.get("home_pitcher_id"))

            away_stats = [
                get_player_season_stats(b["id"])
                for b in lineup.get("away_batters", [])
                if b.get("id")
            ]
            home_stats = [
                get_player_season_stats(b["id"])
                for b in lineup.get("home_batters", [])
                if b.get("id")
            ]

            if not away_stats or not home_stats:
                continue

            away_pid = lineup.get("away_pitcher_id")
            home_pid = lineup.get("home_pitcher_id")
            away_p = get_player_season_stats(away_pid, group="pitching") if away_pid else {}
            home_p = get_player_season_stats(home_pid, group="pitching") if home_pid else {}

            result = run_simulation(
                away_team=game["away_team"],
                home_team=game["home_team"],
                away_lineup=away_stats,
                home_lineup=home_stats,
                away_pitcher=away_p,
                home_pitcher=home_p,
                n=100_000,
            )

            odds_key = frozenset([game["away_team"], game["home_team"]])
            game_odds = all_odds_data.get(odds_key, {})

            for side in ("away", "home"):
                team = game[f"{side}_team"]
                opp = game["home_team" if side == "away" else "away_team"]
                win_pct = result[f"{side}_win_pct"]
                if win_pct >= 65.0:
                    ml_raw = game_odds.get(f"{side}_avg_odds")
                    ml_odds = format_odds(ml_raw) if ml_raw else "—"
                    ev = calc_ev(win_pct, ml_raw) if ml_raw else None
                    top_plays.append(
                        {
                            "team": team,
                            "opponent": opp,
                            "win_pct": win_pct,
                            "ml_odds": ml_odds,
                            "ev": ev,
                            "venue": game.get("venue", ""),
                        }
                    )
        except Exception:
            continue

    top_plays.sort(key=lambda x: x["win_pct"], reverse=True)

    site_url = os.getenv("SITE_URL", "https://mlb-simulator-vert.vercel.app")
    result = send_daily_alerts(top_plays, site_url=site_url)
    return jsonify({"plays_found": len(top_plays), "email_result": result})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    print("\n  ⚾ MLB Simulator is running!")
    if debug:
        print(f"  Open:  http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
