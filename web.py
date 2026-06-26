"""
web.py
------
The Flask web app. Run this to open the simulator in your browser.

Usage:
    python web.py
Then open:  http://127.0.0.1:5000
"""

from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()   # loads ODDS_API_KEY from .env file

from flask import Flask, render_template, abort, request

from data.odds_api import get_mlb_odds, get_mlb_events, get_player_props, format_odds, calc_edge, get_requests_remaining, get_mlb_totals, get_mlb_runline, calc_ev, calc_kelly, get_line_movement
from data.tracker import log_prediction, update_results, get_accuracy_stats, get_all_predictions
from data.my_picks import add_pick, update_pick_results, get_all_picks, get_pick_stats
from data.bet_tracker import log_bet, settle_bets, get_bet_stats, get_all_bets
from data.mlb_api import (
    get_today_schedule,
    get_game_lineup,
    get_player_season_stats,
    get_player_recent_stats,
    get_batter_split_stats,
    get_batter_sitcode_stats,
    get_batter_risp_stats,
    get_pitcher_hand,
    get_ballpark_weather,
    get_team_bullpen_stats,
    get_pitcher_game_log,
    get_batter_vs_pitcher,
    get_team_rest_days,
    get_savant_stats_all,
    get_game_umpire,
    get_team_bullpen_usage,
)
from simulation.engine import run_simulation, predict_pitcher_ks, detect_pitcher_form

app = Flask(__name__, template_folder="app/templates")

N_SIMS_SINGLE = 100_000   # simulations for one-game view
N_SIMS_ALL    =  25_000   # simulations per game in simulate-all (faster)
PARLAY_THRESHOLD  = 65.0   # min win % to include in best parlay
BET_THRESHOLD     = 62.0   # min win % to show in best bets section


# ── Shared helpers ────────────────────────────────────────────────

def fetch_batter_stats_with_splits(batters, pitcher_hand):
    """
    For each batter, fetch their split stats vs the starting pitcher's hand
    (L or R). Falls back to season totals if split data is unavailable.

    This is more accurate than season totals because it accounts for
    platoon advantage — e.g. a righty batter who crushes lefty pitchers
    but struggles against righties.
    """
    stats_list = []
    for b in batters:
        if not b.get("id"):
            stats_list.append({})
            continue
        # Try split stats first; fall back to season totals on any error
        try:
            split = get_batter_split_stats(b["id"], pitcher_hand)
            if split and split.get("plateAppearances", 0) >= 20:
                stats_list.append(split)
            else:
                stats_list.append(get_player_season_stats(b["id"], group="hitting"))
        except Exception:
            stats_list.append(get_player_season_stats(b["id"], group="hitting"))
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
        away_batter_stats = [get_player_season_stats(b["id"], group="hitting") if b.get("id") else {} for b in lineup["away_batters"]]
        home_batter_stats = [get_player_season_stats(b["id"], group="hitting") if b.get("id") else {} for b in lineup["home_batters"]]

    # Platoon matchup score
    def _platoon_score(batter_stats_list):
        scores = []
        for s in batter_stats_list:
            if not s:
                continue
            ops = (float(s.get("obp", 0) or 0) + float(s.get("slg", 0) or 0))
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
        sp  = get_player_season_stats(pitcher_id, group="pitching") if pitcher_id else {}
        bp  = get_team_bullpen_stats(team_id) if team_id else {"era": "4.20", "whip": "1.30"}
        log = get_pitcher_game_log(pitcher_id, num_games=10) if pitcher_id else []
        return sp, bp, log

    with ThreadPoolExecutor(max_workers=2) as ex:
        away_p_future = ex.submit(_fetch_pitcher_and_bullpen,
                                  away_pitcher.get("id"), game.get("away_id"))
        home_p_future = ex.submit(_fetch_pitcher_and_bullpen,
                                  home_pitcher.get("id"), game.get("home_id"))
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
            return (date.today() - last_date).days
        except Exception:
            return None

    away_rest_days = _rest_days(away_pitcher_log)
    home_rest_days = _rest_days(home_pitcher_log)

    # Fetch recent form + display splits for each batter — single-game mode only
    # In parallel we fetch: recent 20 games, vs LHP, vs RHP, vs SP, vs RP
    away_recent       = []
    home_recent       = []
    away_display_extra = []   # list of dicts with vs_lhp/vs_rhp/vs_sp/vs_rp stats
    home_display_extra = []

    if use_splits:
        day_night = _get_day_night(game.get("game_time", ""))

        def _fetch_all_batter_data(batter, opp_pitcher_id=None):
            """Fetch recent form + all split types + matchup history + day/night for one batter.
            Returns (recent, lhp, rhp, sp, rp, risp, vs_pitcher, day_night_stat)."""
            pid = batter.get("id")
            if not pid:
                return {}, {}, {}, {}, {}, {}, {}, {}
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
            return recent, vs_lhp, vs_rhp, vs_sp, vs_rp, risp, vs_pitcher, dn_stat

        # Away batters face the HOME pitcher; home batters face the AWAY pitcher
        n_away = len(lineup["away_batters"])
        home_pid = home_pitcher.get("id")
        away_pid = away_pitcher.get("id")

        with ThreadPoolExecutor(max_workers=30) as ex:
            away_futures = [ex.submit(_fetch_all_batter_data, b, home_pid)
                            for b in lineup["away_batters"]]
            home_futures = [ex.submit(_fetch_all_batter_data, b, away_pid)
                            for b in lineup["home_batters"]]
            all_results_data = [f.result() for f in away_futures + home_futures]

        # Split results back into away and home
        away_data = all_results_data[:n_away]
        home_data = all_results_data[n_away:]

        away_recent        = [d[0] for d in away_data]
        home_recent        = [d[0] for d in home_data]
        away_display_extra = [{"vs_lhp": d[1], "vs_rhp": d[2], "vs_sp": d[3], "vs_rp": d[4], "risp": d[5], "vs_pitcher": d[6], "day_night": d[7]} for d in away_data]
        home_display_extra = [{"vs_lhp": d[1], "vs_rhp": d[2], "vs_sp": d[3], "vs_rp": d[4], "risp": d[5], "vs_pitcher": d[6], "day_night": d[7]} for d in home_data]

    # Fetch Statcast barrel rate + hard-hit % from Baseball Savant (cached per session)
    try:
        savant_all = get_savant_stats_all()
        away_savant = [savant_all.get(b.get('id'), {}) for b in lineup['away_batters']]
        home_savant = [savant_all.get(b.get('id'), {}) for b in lineup['home_batters']]
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

    # ── Series context ────────────────────────────────────────────────
    try:
        series_game_num = get_series_game_number(
            game["gamePk"], game["away_id"], game["home_id"])
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
        away_streak = home_streak = {"streak": 0, "type": "", "label": "", "hot": False, "cold": False}

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
        away_bp_depth = home_bp_depth = {"score": 50, "grade": "Average",
            "reliable_arms": 0, "elite_arms": 0, "avg_era": 4.20, "avg_whip": 1.30}

    # Fetch umpire for K/BB tendency modifier
    try:
        umpire_name = get_game_umpire(game["gamePk"])
    except Exception:
        umpire_name = None

    # Fetch weather for this ballpark
    weather = get_ballpark_weather(game["venue"])

    # Extract vs-RP stats for SP→bullpen transition in the simulation
    # These are the batter's historical stats against relief pitchers specifically.
    # The sim uses them for innings 6-9 instead of the starting pitcher probs.
    away_rp_stats = [ex.get("vs_rp", {}) for ex in away_display_extra] if away_display_extra else None
    home_rp_stats = [ex.get("vs_rp", {}) for ex in home_display_extra] if home_display_extra else None

    # Extract RISP stats — used in the sim whenever a runner is on 2nd or 3rd
    away_risp_stats = [ex.get("risp", {}) for ex in away_display_extra] if away_display_extra else None
    home_risp_stats = [ex.get("risp", {}) for ex in home_display_extra] if home_display_extra else None

    # Extract matchup history stats — career stats for each batter vs the opposing starter
    away_matchup_stats = [ex.get("vs_pitcher", {}) for ex in away_display_extra] if away_display_extra else None
    home_matchup_stats = [ex.get("vs_pitcher", {}) for ex in home_display_extra] if home_display_extra else None

    # Extract day/night splits
    away_daynight_stats = [ex.get("day_night", {}) for ex in away_display_extra] if away_display_extra else None
    home_daynight_stats = [ex.get("day_night", {}) for ex in home_display_extra] if home_display_extra else None

    # Run the simulation — all factors now wired in:
    #   - L/R pitcher splits (base stats)
    #   - Recent form blend (last 20 games)
    #   - Ballpark factors (Coors, Oracle Park, Yankee Stadium, etc.)
    #   - Weather effects on HR (wind, temperature on top of park baseline)
    #   - SP vs RP: innings 1-5 starter, innings 6-9 bullpen
    result = run_simulation(
        away_team         = game["away_team"],
        home_team         = game["home_team"],
        away_lineup       = away_batter_stats,
        home_lineup       = home_batter_stats,
        away_pitcher      = away_pitcher_stats,
        home_pitcher      = home_pitcher_stats,
        weather           = weather,
        away_recent       = away_recent if use_splits else None,
        home_recent       = home_recent if use_splits else None,
        away_rp_stats     = away_rp_stats if use_splits else None,
        home_rp_stats     = home_rp_stats if use_splits else None,
        away_bullpen      = away_bullpen_stats,
        home_bullpen      = home_bullpen_stats,
        venue             = game["venue"],
        away_pitcher_log  = away_pitcher_log,
        home_pitcher_log  = home_pitcher_log,
        away_rest_days    = away_rest_days,
        home_rest_days    = home_rest_days,
        away_risp_stats    = away_risp_stats   if use_splits else None,
        home_risp_stats    = home_risp_stats   if use_splits else None,
        away_matchup_stats  = away_matchup_stats  if use_splits else None,
        home_matchup_stats  = home_matchup_stats  if use_splits else None,
        away_daynight_stats = away_daynight_stats if use_splits else None,
        home_daynight_stats = home_daynight_stats if use_splits else None,
        away_batter_rest    = away_batter_rest,
        home_batter_rest    = home_batter_rest,
        away_savant         = away_savant,
        home_savant         = home_savant,
        umpire_name         = umpire_name,
        away_bp_fatigue     = away_bp_fatigue,
        home_bp_fatigue     = home_bp_fatigue,
        series_game_number  = series_game_num,
        away_catcher_cs     = away_catcher_cs,
        home_catcher_cs     = home_catcher_cs,
        n                   = n_sims,
    )

    # ── Pitcher K prop predictions ──────────────────────────────────────
    try:
        away_form = detect_pitcher_form(away_pitcher_log, away_pitcher_stats)
        home_form = detect_pitcher_form(home_pitcher_log, home_pitcher_stats)
        away_k_pred = predict_pitcher_ks(
            away_pitcher_stats, home_batter_stats,
            umpire_name=umpire_name,
            pitcher_log=away_pitcher_log,
            fatigue=away_fatigue if "away_fatigue" in dir() else None,
        )
        home_k_pred = predict_pitcher_ks(
            home_pitcher_stats, away_batter_stats,
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
    result["away_form"]      = away_form
    result["home_form"]      = home_form
    result["away_platoon"]   = away_platoon_score
    result["home_platoon"]   = home_platoon_score
    result["platoon_edge"]   = platoon_edge
    result["away_hand"]      = away_hand
    result["home_hand"]      = home_hand
    result["hr_park_factor"] = round(
        __import__("simulation.engine", fromlist=["BALLPARK_FACTORS"]).BALLPARK_FACTORS
        .get(game.get("venue",""), {}).get("hr", 1.0), 2)
    result["wind_effect"]    = weather.get("wind_effect", "neutral") if weather else "neutral"
    result["lineup_status"]  = lineup_status
    result["wind_label"]     = weather.get("wind_label", "") if weather else ""

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
        if ip: parts.append(f"{ip} IP")
        if er != "": parts.append(f"{er} ER")
        if ks != "": parts.append(f"{ks} K")
        if opp: parts.append(f"vs {opp}")
        from datetime import datetime as _dt, date as _d
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
    result["away_streak"]     = away_streak
    result["home_streak"]     = home_streak
    result["series_game_num"]   = series_game_num
    result["away_catcher_cs"]   = away_catcher_cs
    result["home_catcher_cs"]   = home_catcher_cs
    result["away_bp_depth"]     = away_bp_depth
    result["home_bp_depth"]     = home_bp_depth
    result["away_il"]           = away_il
    result["home_il"]           = home_il
    result["away_transactions"] = away_transactions
    result["home_transactions"] = home_transactions
    result["home_k_pred"] = home_k_pred

    # Attach IDs for logo/headshot URLs in templates
    result["away_team_id"]     = game.get("away_id")
    result["home_team_id"]     = game.get("home_id")
    result["away_bp_fatigue"]  = (away_bp_fatigue or {}).get("fatigue", "normal")
    result["home_bp_fatigue"]  = (home_bp_fatigue or {}).get("fatigue", "normal")
    result["away_bp_ip"]       = (away_bp_fatigue or {}).get("total_bp_ip", None)
    result["home_bp_ip"]       = (home_bp_fatigue or {}).get("total_bp_ip", None)
    result["away_pitcher_id"]  = away_pitcher.get("id")
    result["home_pitcher_id"]  = home_pitcher.get("id")
    result["away_pitcher_name"] = away_pitcher.get("name", "TBD")
    result["home_pitcher_name"] = home_pitcher.get("name", "TBD")
    result["umpire"]           = umpire_name

    # Build per-batter stat rows for display (season + recent + L/R + SP/RP)
    def build_batter_display(batters, season_stats_list, recent_stats_list, extra_list):
        rows = []
        for i, b in enumerate(batters):
            s    = season_stats_list[i] if i < len(season_stats_list) else {}
            r    = recent_stats_list[i] if recent_stats_list and i < len(recent_stats_list) else {}
            ex   = extra_list[i] if extra_list and i < len(extra_list) else {}

            vs_lhp      = ex.get("vs_lhp",      {})
            vs_rhp      = ex.get("vs_rhp",      {})
            vs_sp       = ex.get("vs_sp",       {})
            vs_rp       = ex.get("vs_rp",       {})
            risp        = ex.get("risp",        {})
            vs_pitcher  = ex.get("vs_pitcher",  {})
            dn_stat     = ex.get("day_night",   {})

            # Season: singles vs extra base hits
            hits    = s.get("hits", 0)
            doubles = s.get("doubles", 0)
            triples = s.get("triples", 0)
            hr      = s.get("homeRuns", 0)
            xbh     = doubles + triples + hr
            singles = max(0, hits - xbh)

            # Recent form: batting average over last 20 games
            r_ab   = r.get("atBats", 0)
            r_hits = r.get("hits", 0)
            r_avg  = round(r_hits / r_ab, 3) if r_ab > 0 else None

            def _fmt_avg(d):
                """Format avg from a split stats dict."""
                v = d.get("avg", "")
                return v if v else "—"

            def _fmt_ops(d):
                v = d.get("ops", "")
                return v if v else "—"

            rows.append({
                **b,
                # Season stats
                "avg":      s.get("avg", "—"),
                "obp":      s.get("obp", "—"),
                "slg":      s.get("slg", "—"),
                "ops":      s.get("ops", "—"),
                "hr":       hr,
                "rbi":      s.get("rbi", 0),
                "so":       s.get("strikeOuts", 0),
                "bb":       s.get("baseOnBalls", 0),
                "pa":       s.get("plateAppearances", 0),
                # Hit type breakdown
                "singles":  singles,
                "doubles":  doubles,
                "triples":  triples,
                "xbh":      xbh,
                # vs LHP / RHP
                "vs_lhp_avg": _fmt_avg(vs_lhp),
                "vs_lhp_ops": _fmt_ops(vs_lhp),
                "vs_lhp_pa":  vs_lhp.get("plateAppearances", 0),
                "vs_rhp_avg": _fmt_avg(vs_rhp),
                "vs_rhp_ops": _fmt_ops(vs_rhp),
                "vs_rhp_pa":  vs_rhp.get("plateAppearances", 0),
                # vs SP / RP (early vs late game)
                "vs_sp_avg":  _fmt_avg(vs_sp),
                "vs_sp_ops":  _fmt_ops(vs_sp),
                "vs_sp_pa":   vs_sp.get("plateAppearances", 0),
                "vs_rp_avg":  _fmt_avg(vs_rp),
                "vs_rp_ops":  _fmt_ops(vs_rp),
                "vs_rp_pa":   vs_rp.get("plateAppearances", 0),
                # Recent form (last 20 games)
                "recent_avg":   f".{int(r_avg * 1000):03d}" if r_avg is not None else "—",
                "recent_games": r.get("games_found", 0),
                "recent_hr":    r.get("homeRuns", 0),
                "recent_rbi":   r.get("rbi", 0),
                # RISP / clutch stats
                "risp_avg":        _fmt_avg(risp),
                "risp_ops":        _fmt_ops(risp),
                "risp_pa":         risp.get("plateAppearances", 0),
                # Career matchup history vs this game's opposing starter
                "vs_pitcher_avg":  _fmt_avg(vs_pitcher),
                "vs_pitcher_ops":  _fmt_ops(vs_pitcher),
                "vs_pitcher_pa":   vs_pitcher.get("plateAppearances", 0),
                "vs_pitcher_hr":   vs_pitcher.get("homeRuns", 0),
                # Day/night splits
                "dn_avg":          _fmt_avg(dn_stat),
                "dn_ops":          _fmt_ops(dn_stat),
                "dn_pa":           dn_stat.get("plateAppearances", 0),
                "dn_label":        "Day" if day_night == "d" else "Night",
            })
        return rows

    away_display = build_batter_display(
        lineup["away_batters"], away_batter_stats,
        away_recent if use_splits else [],
        away_display_extra if use_splits else []
    )
    home_display = build_batter_display(
        lineup["home_batters"], home_batter_stats,
        home_recent if use_splits else [],
        home_display_extra if use_splits else []
    )

    # Attach extra context for display
    result.update({
        "gamePk":             game["gamePk"],
        "venue":              game["venue"],
        "status":             game["status"],
        "away_pitcher_name":  away_pitcher.get("name", "TBD"),
        "home_pitcher_name":  home_pitcher.get("name", "TBD"),
        "away_pitcher_hand":  away_hand,
        "home_pitcher_hand":  home_hand,
        "away_era":           away_pitcher_stats.get("era",  "N/A"),
        "away_whip":          away_pitcher_stats.get("whip", "N/A"),
        "away_wl":            f"{away_pitcher_stats.get('wins',0)}-{away_pitcher_stats.get('losses',0)}",
        "away_bp_era":        away_bullpen_stats.get("era",  "N/A"),
        "away_bp_whip":       away_bullpen_stats.get("whip", "N/A"),
        "home_era":           home_pitcher_stats.get("era",  "N/A"),
        "home_whip":          home_pitcher_stats.get("whip", "N/A"),
        "home_wl":            f"{home_pitcher_stats.get('wins',0)}-{home_pitcher_stats.get('losses',0)}",
        "home_bp_era":        home_bullpen_stats.get("era",  "N/A"),
        "home_bp_whip":       home_bullpen_stats.get("whip", "N/A"),
        # Pitcher fatigue profile (from last 10 starts game log)
        "away_fatigue_label": result.get("away_fatigue_label", "Typical"),
        "away_avg_ip":        result.get("away_avg_ip"),
        "home_fatigue_label": result.get("home_fatigue_label", "Typical"),
        "home_avg_ip":        result.get("home_avg_ip"),
        # Pitcher rest days
        "away_rest_days":     result.get("away_rest_days"),
        "away_rest_type":     result.get("away_rest_type"),
        "home_rest_days":     result.get("home_rest_days"),
        "home_rest_type":     result.get("home_rest_type"),
        "away_batters":       away_display,
        "home_batters":       home_display,
        "weather":            weather,
        "use_recent":         use_splits,   # flag for template to show/hide recent form column
    })
    return result


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    # Allow browsing any date via ?date=2026-06-28
    selected = request.args.get("date", date.today().isoformat())
    try:
        from datetime import datetime
        selected_date = datetime.strptime(selected, "%Y-%m-%d").date()
    except ValueError:
        selected_date = date.today()

    from datetime import timedelta
    prev_date = (selected_date - timedelta(days=1)).isoformat()
    next_date = (selected_date + timedelta(days=1)).isoformat()
    is_today  = (selected_date == date.today())

    games = get_today_schedule(game_date=selected_date.isoformat())
    label = selected_date.strftime("%A, %B %d %Y")

    return render_template("index.html",
                           games=games,
                           date=label,
                           selected_date=selected_date.isoformat(),
                           prev_date=prev_date,
                           next_date=next_date,
                           is_today=is_today,
                           api_remaining=get_requests_remaining())


@app.route("/simulate/<int:game_pk>")
def simulate(game_pk):
    """Simulate a single game and show the detailed results page."""
    games = get_today_schedule()
    game  = next((g for g in games if g["gamePk"] == game_pk), None)
    if game is None:
        abort(404)

    # Read sim count from URL e.g. /simulate/822961?sims=500000
    allowed = {100_000, 500_000, 1_000_000}
    try:
        n_sims = int(request.args.get("sims", N_SIMS_SINGLE))
        if n_sims not in allowed:
            n_sims = N_SIMS_SINGLE
    except ValueError:
        n_sims = N_SIMS_SINGLE

    result = build_game_result(game, n_sims=n_sims)
    if result is None:
        return render_template("index.html", games=games,
                               date=date.today().strftime("%A, %B %d %Y"),
                               error="Lineup not available yet. Try a game already in progress.",
                               api_remaining=get_requests_remaining())

    # Log this prediction for accuracy tracking
    try:
        log_prediction(
            game_pk       = game_pk,
            game_date     = date.today().isoformat(),
            away_team     = result["away_team"],
            home_team     = result["home_team"],
            away_win_pct  = result["away_win_pct"],
            home_win_pct  = result["home_win_pct"],
            away_avg_runs = result["away_avg_runs"],
            home_avg_runs = result["home_avg_runs"],
            n_sims        = n_sims,
        )
    except Exception:
        pass  # never let logging break the simulation

    # Fetch moneyline odds (cached for 30 min — only 1 API call per 30 min window)
    all_odds  = get_mlb_odds()
    odds_key  = frozenset([game["away_team"], game["home_team"]])
    game_odds = all_odds.get(odds_key, {})

    if game_odds:
        result["away_implied_pct"] = game_odds["away_implied_pct"]
        result["home_implied_pct"] = game_odds["home_implied_pct"]
        result["away_avg_odds"]    = format_odds(game_odds["away_avg_odds"])
        result["home_avg_odds"]    = format_odds(game_odds["home_avg_odds"])
        result["away_edge"]        = calc_edge(result["away_win_pct"], game_odds["away_implied_pct"])
        result["home_edge"]        = calc_edge(result["home_win_pct"], game_odds["home_implied_pct"])
        result["away_ev"]          = calc_ev(result["away_win_pct"], game_odds["away_avg_odds"])
        result["home_ev"]          = calc_ev(result["home_win_pct"], game_odds["home_avg_odds"])
        result["away_kelly"]       = calc_kelly(result["away_win_pct"], game_odds["away_avg_odds"])
        result["home_kelly"]       = calc_kelly(result["home_win_pct"], game_odds["home_avg_odds"])
        result["books_used"]       = game_odds["books_used"]
        result["best_away_odds"]   = format_odds(game_odds.get("best_away_odds", game_odds["away_avg_odds"]))
        result["best_home_odds"]   = format_odds(game_odds.get("best_home_odds", game_odds["home_avg_odds"]))
        result["best_away_book"]   = game_odds.get("best_away_book", "")
        result["best_home_book"]   = game_odds.get("best_home_book", "")
    else:
        result["away_implied_pct"] = None
        result["home_implied_pct"] = None
        result["away_avg_odds"]    = None
        result["home_avg_odds"]    = None
        result["away_edge"]        = None
        result["home_edge"]        = None
        result["away_ev"]          = None
        result["home_ev"]          = None
        result["away_kelly"]       = None
        result["home_kelly"]       = None
        result["books_used"]       = 0

    # ── Over/Under + Run line odds ────────────────────────────────────
    try:
        all_totals = get_mlb_totals()
        ou = all_totals.get(odds_key, {})
        if ou:
            hist    = result.get("total_runs_hist", {})
            n_sims  = result.get("simulations", 1)
            ou_line = ou["line"]
            over_count = sum(cnt for tot, cnt in hist.items() if tot > ou_line)
            model_over  = round(over_count / n_sims * 100, 1)
            model_under = round(100 - model_over, 1)
            result["ou_line"]          = ou_line
            result["model_over_pct"]   = model_over
            result["model_under_pct"]  = model_under
            result["ou_over_implied"]  = ou["over_implied"]
            result["ou_under_implied"] = ou["under_implied"]
            result["ou_over_odds"]     = format_odds(ou["over_odds"])
            result["ou_under_odds"]    = format_odds(ou["under_odds"])
            result["ou_over_edge"]     = round(model_over  - ou["over_implied"],  1)
            result["ou_under_edge"]    = round(model_under - ou["under_implied"], 1)
            result["ou_over_ev"]       = calc_ev(model_over,  ou["over_odds"])
            result["ou_under_ev"]      = calc_ev(model_under, ou["under_odds"])
    except Exception:
        pass
    try:
        all_rl = get_mlb_runline()
        rl = all_rl.get(odds_key, {})
        if rl:
            result["away_rl_odds"]    = format_odds(rl["away_rl_odds"])
            result["home_rl_odds"]    = format_odds(rl["home_rl_odds"])
            result["away_rl_implied"] = rl["away_rl_implied"]
            result["home_rl_implied"] = rl["home_rl_implied"]
            result["away_rl_edge"]    = round(result.get("away_cover_pct", 0) - rl["away_rl_implied"], 1)
            result["home_rl_edge"]    = round(result.get("home_cover_pct", 0) - rl["home_rl_implied"], 1)
            result["away_rl_ev"]      = calc_ev(result.get("away_cover_pct", 0), rl["away_rl_odds"])
            result["home_rl_ev"]      = calc_ev(result.get("home_cover_pct", 0), rl["home_rl_odds"])
            result["away_rl_kelly"]   = calc_kelly(result.get("away_cover_pct", 0), rl["away_rl_odds"])
            result["home_rl_kelly"]   = calc_kelly(result.get("home_cover_pct", 0), rl["home_rl_odds"])
    except Exception:
        pass

    # Props are NOT auto-fetched here — user clicks "Load Props" button
    # which calls /props/<game_pk> separately (saves 2 API calls per simulation)
    result["props_by_player"] = {}

    return render_template("result.html", game_pk=game_pk, **result)


@app.route("/props/<int:game_pk>")
def load_props(game_pk):
    """
    On-demand player props endpoint — only called when user clicks Load Props.
    Costs 2 API requests (events list + props for this game).
    Returns JSON so the page loads props without a full refresh.
    """
    from flask import jsonify

    games = get_today_schedule()
    game  = next((g for g in games if g["gamePk"] == game_pk), None)
    if not game:
        return jsonify({})

    odds_key   = frozenset([game["away_team"], game["home_team"]])
    all_events = get_mlb_events()
    event_id   = all_events.get(odds_key)
    props      = get_player_props(event_id) if event_id else {}
    return jsonify(props)


@app.route("/simulate-all")
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
    all_games = get_today_schedule()

    # Skip finished games
    games = [g for g in all_games
             if "final" not in g["status"].lower()
             and "game over" not in g["status"].lower()]

    if not games:
        return render_template("index.html",
                               games=all_games,
                               date=date.today().strftime("%A, %B %d %Y"),
                               error="No active or upcoming games to simulate right now.")

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
    batter_ids  = set()
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
    batter_cache  = {}
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
            away_team    = game["away_team"],
            home_team    = game["home_team"],
            away_lineup  = away_stats,
            home_lineup  = home_stats,
            away_pitcher = away_ps,
            home_pitcher = home_ps,
            weather      = weather,
            venue        = game["venue"],
            n            = N_SIMS_ALL,
        )
        result.update({
            "gamePk":            game["gamePk"],
            "venue":             game["venue"],
            "status":            game["status"],
            "away_pitcher_name": away_pitcher.get("name", "TBD"),
            "home_pitcher_name": home_pitcher.get("name", "TBD"),
            "away_pitcher_hand": "R",
            "home_pitcher_hand": "R",
            "away_era":  away_ps.get("era",  "N/A"),
            "away_whip": away_ps.get("whip", "N/A"),
            "away_wl":   f"{away_ps.get('wins',0)}-{away_ps.get('losses',0)}",
            "home_era":  home_ps.get("era",  "N/A"),
            "home_whip": home_ps.get("whip", "N/A"),
            "home_wl":   f"{home_ps.get('wins',0)}-{home_ps.get('losses',0)}",
            "away_batters": lu["away_batters"],
            "home_batters": lu["home_batters"],
            "weather":   weather,
        })
        results.append(result)

    results.sort(key=lambda r: r["gamePk"])

    # Sort results by gamePk to keep a consistent order
    results.sort(key=lambda r: r["gamePk"])

    # ── Build the Best Parlay ──────────────────────────────────
    # A parlay is a bet where you pick multiple games and all must win.
    # We pick games where one team has ≥60% win probability.
    parlay_picks = []
    for r in results:
        if r["away_win_pct"] >= PARLAY_THRESHOLD:
            parlay_picks.append({
                "team":        r["away_team"],
                "opponent":    r["home_team"],
                "win_pct":     r["away_win_pct"],
                "avg_runs":    r["away_avg_runs"],
                "venue":       r["venue"],
            })
        elif r["home_win_pct"] >= PARLAY_THRESHOLD:
            parlay_picks.append({
                "team":        r["home_team"],
                "opponent":    r["away_team"],
                "win_pct":     r["home_win_pct"],
                "avg_runs":    r["home_avg_runs"],
                "venue":       r["venue"],
            })

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
        parlay_ev = round(combined_prob * (book_decimal - 1) * 100
                          - (1 - combined_prob) * 100, 1)
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
        fav_pct  = None
        fav_opp  = None

        if r["away_win_pct"] >= r["home_win_pct"] and r["away_win_pct"] >= BET_THRESHOLD:
            fav_team, fav_pct, fav_opp = r["away_team"], r["away_win_pct"], r["home_team"]
        elif r["home_win_pct"] > r["away_win_pct"] and r["home_win_pct"] >= BET_THRESHOLD:
            fav_team, fav_pct, fav_opp = r["home_team"], r["home_win_pct"], r["away_team"]

        if fav_team:
            if fav_pct >= 70:
                tier, tier_label, tier_color = "strong",   "🔥 Strong (70%+)",    "#ef5350"
            elif fav_pct >= 65:
                tier, tier_label, tier_color = "good",     "✅ Good (65-69%)",    "#4caf50"
            else:
                tier, tier_label, tier_color = "moderate", "💛 Moderate (62-64%)", "#ffd54f"

            best_bets.append({
                "team":       fav_team,
                "opponent":   fav_opp,
                "win_pct":    fav_pct,
                "tier":       tier,
                "tier_label": tier_label,
                "tier_color": tier_color,
                "venue":      r["venue"],
                "gamePk":     r["gamePk"],
            })

    best_bets.sort(key=lambda b: b["win_pct"], reverse=True)

    # ── Fetch Vegas odds and attach to each result ────────────────────────────
    # Uses the same cached call as the single-game view (30-min cache).
    # Matches by frozenset({away_team, home_team}) so order doesn't matter.
    try:
        all_odds = get_mlb_odds()
    except Exception:
        all_odds = {}

    for r in results:
        odds_key  = frozenset([r["away_team"], r["home_team"]])
        game_odds = all_odds.get(odds_key, {})
        if game_odds:
            r["away_implied_pct"] = game_odds["away_implied_pct"]
            r["home_implied_pct"] = game_odds["home_implied_pct"]
            r["away_avg_odds"]    = format_odds(game_odds["away_avg_odds"])
            r["home_avg_odds"]    = format_odds(game_odds["home_avg_odds"])
            r["away_edge"]        = calc_edge(r["away_win_pct"], game_odds["away_implied_pct"])
            r["home_edge"]        = calc_edge(r["home_win_pct"], game_odds["home_implied_pct"])
            r["away_ev"]          = calc_ev(r["away_win_pct"], game_odds["away_avg_odds"])
            r["home_ev"]          = calc_ev(r["home_win_pct"], game_odds["home_avg_odds"])
            r["away_kelly"]       = calc_kelly(r["away_win_pct"], game_odds["away_avg_odds"])
            r["home_kelly"]       = calc_kelly(r["home_win_pct"], game_odds["home_avg_odds"])
        else:
            for k in ("away_implied_pct","home_implied_pct","away_avg_odds","home_avg_odds",
                      "away_edge","home_edge","away_ev","home_ev","away_kelly","home_kelly"):
                r[k] = None
        # O/U + run line per game card
        try:
            all_totals = get_mlb_totals()
            all_rl     = get_mlb_runline()
            ok = frozenset([r["away_team"], r["home_team"]])
            ou = all_totals.get(ok, {})
            if ou:
                hist = r.get("total_runs_hist", {})
                ns   = r.get("simulations", 1)
                oul  = ou["line"]
                oc   = sum(cnt for tot, cnt in hist.items() if tot > oul)
                mo   = round(oc / ns * 100, 1)
                r["ou_line"]          = oul
                r["model_over_pct"]   = mo
                r["model_under_pct"]  = round(100 - mo, 1)
                r["ou_over_implied"]  = ou["over_implied"]
                r["ou_under_implied"] = ou["under_implied"]
                r["ou_over_odds"]     = format_odds(ou["over_odds"])
                r["ou_under_odds"]    = format_odds(ou["under_odds"])
                r["ou_over_edge"]     = round(mo - ou["over_implied"], 1)
                r["ou_under_edge"]    = round((100 - mo) - ou["under_implied"], 1)
                r["ou_over_ev"]       = calc_ev(mo, ou["over_odds"])
                r["ou_under_ev"]      = calc_ev(100 - mo, ou["under_odds"])
            rl = all_rl.get(ok, {})
            if rl:
                r["away_rl_odds"]    = format_odds(rl["away_rl_odds"])
                r["home_rl_odds"]    = format_odds(rl["home_rl_odds"])
                r["away_rl_implied"] = rl["away_rl_implied"]
                r["home_rl_implied"] = rl["home_rl_implied"]
                r["away_rl_edge"]    = round(r.get("away_cover_pct", 0) - rl["away_rl_implied"], 1)
                r["home_rl_edge"]    = round(r.get("home_cover_pct", 0) - rl["home_rl_implied"], 1)
                r["away_rl_ev"]      = calc_ev(r.get("away_cover_pct", 0), rl["away_rl_odds"])
                r["home_rl_ev"]      = calc_ev(r.get("home_cover_pct", 0), rl["home_rl_odds"])
        except Exception:
            pass

    # ── Line movement / sharp money signals ─────────────────────────
    try:
        movement = get_line_movement()
        for r in results:
            mv = movement.get(frozenset([r["away_team"], r["home_team"]]), {})
            r["sharp_away"]      = mv.get("sharp_away", False)
            r["sharp_home"]      = mv.get("sharp_home", False)
            r["away_impl_move"]  = mv.get("away_impl_move", 0)
            r["home_impl_move"]  = mv.get("home_impl_move", 0)
            r["away_open_odds"]  = format_odds(mv["away_open"]) if mv.get("away_open") else None
            r["home_open_odds"]  = format_odds(mv["home_open"]) if mv.get("home_open") else None
    except Exception:
        pass

    # Also attach odds to best_bets entries
    for b in best_bets:
        r_match = next((r for r in results if r["gamePk"] == b["gamePk"]), {})
        is_away = (b["team"] == r_match.get("away_team"))
        b["implied_pct"] = r_match.get("away_implied_pct" if is_away else "home_implied_pct")
        b["odds_str"]    = r_match.get("away_avg_odds"    if is_away else "home_avg_odds")
        b["edge"]        = r_match.get("away_edge"        if is_away else "home_edge")

    today = date.today().strftime("%A, %B %d %Y")
    return render_template(
        "all_results.html",
        results       = results,
        parlay_picks      = parlay_picks,
        combined_prob     = combined_prob_pct,
        parlay_ev         = parlay_ev,
        parlay_true_odds  = parlay_true_odds,
        parlay_book_odds  = parlay_book_odds,
        best_bets     = best_bets,
        date          = today,
        n_sims        = N_SIMS_ALL,
    )


@app.route("/my-picks", methods=["GET", "POST"])
def my_picks():
    """Personal picks log — enter your pick, see simulator's take, track results."""
    from flask import request as freq, jsonify

    if freq.method == "POST":
        # Submitted a new pick via the form
        data = freq.get_json() or {}
        add_pick(
            game_pk       = data.get("game_pk"),
            game_date     = data.get("game_date"),
            away_team     = data.get("away_team"),
            home_team     = data.get("home_team"),
            my_pick       = data.get("my_pick"),
            my_notes      = data.get("my_notes", ""),
            sim_away_pct  = data.get("sim_away_pct"),
            sim_home_pct  = data.get("sim_home_pct"),
            sim_away_runs = data.get("sim_away_runs"),
            sim_home_runs = data.get("sim_home_runs"),
        )
        return jsonify({"ok": True})

    # GET — show the page
    stats  = get_pick_stats()
    games  = get_today_schedule()
    return render_template("my_picks.html", games=games, **stats)


@app.route("/my-picks/update", methods=["POST"])
def update_my_picks():
    from flask import jsonify
    updated = update_pick_results()
    stats   = get_pick_stats()
    return jsonify({"updated": updated, "my_pct": stats["my_pct"], "sim_pct": stats["sim_pct"]})


@app.route("/accuracy")
def accuracy():
    """Model accuracy dashboard — how often are we right?"""
    stats = get_accuracy_stats()
    return render_template("accuracy.html", **stats)


@app.route("/update-results", methods=["POST"])
def trigger_update():
    """
    Check the MLB API for final scores on any logged predictions
    that don't have results yet. Called from the accuracy page.
    """
    from flask import jsonify
    updated = update_results()
    stats   = get_accuracy_stats()
    return jsonify({
        "updated": updated,
        "accuracy_pct": stats["accuracy_pct"],
        "correct_picks": stats["correct_picks"],
        "results_available": stats["results_available"],
    })


@app.route("/bets")
def bets_dashboard():
    """ROI tracker — shows all placed bets, win rate, and total P&L."""
    settle_bets()   # auto-settle any finished games
    stats = get_bet_stats()
    from datetime import date as _d
    stats["today"] = _d.today().isoformat()
    return render_template("bets.html", **stats)


@app.route("/bets/log", methods=["POST"])
def log_bet_route():
    """AJAX endpoint — log a new bet from the game card."""
    from flask import jsonify, request as req
    data = req.get_json() or req.form
    try:
        log_bet(
            game_pk   = data["game_pk"],
            game_date = data.get("game_date", ""),
            away_team = data["away_team"],
            home_team = data["home_team"],
            bet_on    = data["bet_on"],
            bet_type  = data.get("bet_type", "ML"),
            odds      = int(data["odds"]),
            amount    = float(data.get("amount", 100)),
            model_edge= data.get("model_edge"),
            ev        = data.get("ev"),
            kelly     = data.get("kelly"),
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 400


@app.route("/bets/settle", methods=["POST"])
def settle_bets_route():
    from flask import jsonify
    n = settle_bets()
    return jsonify({"settled": n})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    print("\n  MLB Simulator is running!")
    if debug:
        print(f"  Open this in your browser:  http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
