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

from data.odds_api import get_mlb_odds, get_mlb_events, get_player_props, format_odds, calc_edge, get_requests_remaining
from data.tracker import log_prediction, update_results, get_accuracy_stats, get_all_predictions
from data.my_picks import add_pick, update_pick_results, get_all_picks, get_pick_stats
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
)
from simulation.engine import run_simulation

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
        n                 = n_sims,
    )

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
        result["books_used"]       = game_odds["books_used"]
    else:
        result["away_implied_pct"] = None
        result["home_implied_pct"] = None
        result["away_avg_odds"]    = None
        result["home_avg_odds"]    = None
        result["away_edge"]        = None
        result["home_edge"]        = None
        result["books_used"]       = 0

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

    today = date.today().strftime("%A, %B %d %Y")
    return render_template(
        "all_results.html",
        results       = results,
        parlay_picks  = parlay_picks,
        combined_prob = combined_prob_pct,
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


if __name__ == "__main__":
    print("\n  MLB Simulator is running!")
    print("  Open this in your browser:  http://127.0.0.1:5000\n")
    app.run(debug=True)
