-- ============================================================
-- MLB Simulator — Supabase Schema
-- Paste this into your Supabase project's SQL Editor and run it.
-- ============================================================

-- Users (we manage passwords ourselves with werkzeug, not Supabase Auth)
CREATE TABLE IF NOT EXISTS users (
    id          BIGSERIAL PRIMARY KEY,
    username    TEXT NOT NULL,
    password    TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS users_username_ci ON users (LOWER(username));

-- Per-user bets
CREATE TABLE IF NOT EXISTS bets (
    id           BIGSERIAL PRIMARY KEY,
    user_id      BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    game_pk      TEXT,
    game_date    TEXT,
    away_team    TEXT,
    home_team    TEXT,
    bet_on       TEXT,
    bet_type     TEXT DEFAULT 'ML',
    odds         INTEGER,
    amount       NUMERIC(10,2),
    result       TEXT DEFAULT 'pending',
    payout       NUMERIC(10,2),
    model_edge   NUMERIC(6,2),
    ev           NUMERIC(6,2),
    kelly        NUMERIC(6,4),
    closing_line INTEGER,
    clv          NUMERIC(6,2),
    logged_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    settled_at   TIMESTAMPTZ
);

-- Global model predictions (shared across all users)
CREATE TABLE IF NOT EXISTS predictions (
    id               BIGSERIAL PRIMARY KEY,
    logged_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    game_date        TEXT,
    game_pk          TEXT UNIQUE,
    away_team        TEXT,
    home_team        TEXT,
    away_win_pct     NUMERIC(5,2),
    home_win_pct     NUMERIC(5,2),
    away_avg_runs    NUMERIC(5,2),
    home_avg_runs    NUMERIC(5,2),
    predicted_winner TEXT,
    n_sims           INTEGER,
    actual_away_runs INTEGER,
    actual_home_runs INTEGER,
    actual_winner    TEXT,
    correct_pick     SMALLINT,
    run_diff_error   NUMERIC(5,2)
);

-- Per-user personal picks
CREATE TABLE IF NOT EXISTS picks (
    id               BIGSERIAL PRIMARY KEY,
    user_id          BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    logged_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    game_date        TEXT,
    game_pk          TEXT,
    away_team        TEXT,
    home_team        TEXT,
    my_pick          TEXT,
    my_notes         TEXT,
    sim_pick         TEXT,
    sim_away_pct     NUMERIC(5,2),
    sim_home_pct     NUMERIC(5,2),
    sim_away_runs    NUMERIC(5,2),
    sim_home_runs    NUMERIC(5,2),
    actual_away_runs INTEGER,
    actual_home_runs INTEGER,
    actual_winner    TEXT,
    my_pick_correct  SMALLINT,
    sim_pick_correct SMALLINT,
    run_diff_error   NUMERIC(5,2)
);

-- Confidence grade + score on predictions (for #71 confidence history chart)
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS confidence_grade TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS confidence_score NUMERIC(5,1);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS confidence_signals INTEGER;

-- Odds snapshots for line movement tracking
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id           BIGSERIAL PRIMARY KEY,
    game_pk      TEXT NOT NULL,
    game_date    TEXT,
    away_team    TEXT,
    home_team    TEXT,
    away_odds    INTEGER,
    home_odds    INTEGER,
    ou_line      NUMERIC(4,1),
    snapshot_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_odds_snap_gpk ON odds_snapshots(game_pk);

-- RLM alerts log
CREATE TABLE IF NOT EXISTS rlm_alerts (
    id           BIGSERIAL PRIMARY KEY,
    game_pk      TEXT NOT NULL,
    game_date    TEXT,
    away_team    TEXT,
    home_team    TEXT,
    direction    TEXT,
    bet_pct      NUMERIC(5,1),
    line_open    INTEGER,
    line_current INTEGER,
    alert_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
