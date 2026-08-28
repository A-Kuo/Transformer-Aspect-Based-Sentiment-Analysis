-- ============================================================
-- ABSA model registry — Neon Postgres schema
-- ============================================================
-- Applied idempotently by src/db.py::ensure_schema(). No migration
-- tool: this schema changes a few times a year at most (a new metric
-- field, maybe a new track), which doesn't justify Alembic.
--
-- IMPORTANT: the trained weights themselves are never stored here —
-- only where to find them on Hugging Face Hub, plus their provenance
-- and eval metrics. The app reads these tables at startup to decide
-- which checkpoint to download; nothing here ever trains a model.

CREATE TABLE IF NOT EXISTS absa_runs (
    model_version        TEXT PRIMARY KEY,
    -- "track" separates model architectures sharing this registry
    -- (the BERT baseline in src/ vs. a future V3 hybrid-SSM run).
    -- Only one exists today, but V2/ and V3/ are already separate
    -- architecture experiments in this repo, so this avoids a
    -- schema change the day a V3 run needs to coexist here too.
    track                 TEXT NOT NULL DEFAULT 'bert-absa',
    run_timestamp          TIMESTAMPTZ NOT NULL,
    -- Distinguishes a real Kaggle GPU training run from a local
    -- smoke-test push used to verify this pipeline end-to-end
    -- without spending Kaggle GPU hours.
    environment             TEXT NOT NULL CHECK (environment IN ('kaggle_gpu', 'local_smoke')),
    -- Pointer to the weights on Hugging Face Hub. Never store the
    -- ~440MB .pt file itself in Postgres — Neon's row/column storage
    -- isn't built for that, and it would make every row fetch slow.
    hf_repo_id              TEXT NOT NULL,
    hf_revision              TEXT NOT NULL,  -- commit SHA, pinned for reproducibility
    hf_filename               TEXT NOT NULL DEFAULT 'checkpoint_best.pt',
    -- Snapshot of config.yaml's data.{dataset_name,dataset_subset,
    -- train_size,val_size,test_size,seed} at train time, so a run's
    -- provenance survives config.yaml changing later.
    data_config               JSONB NOT NULL,
    -- Snapshot of config.yaml's training.* block at train time.
    hyperparameters            JSONB NOT NULL,
    -- The exact dict returned by src/evaluate.py::evaluate_model().
    -- Stored as one JSONB blob rather than normalized into per-metric
    -- columns/tables: the shape is nested (per-class precision/
    -- recall/F1, confusion matrices, per-aspect cross-task breakdown,
    -- latency percentiles) and evolves with evaluate.py. Normalizing
    -- it would mean a schema migration every time evaluate_model()'s
    -- output shape changes; JSONB lets the app just read
    -- metrics->'sentiment'->'per_class'->'positive' etc. without one.
    metrics                     JSONB NOT NULL,
    -- Exactly one active run per track at a time. get_active_run()
    -- relies on this to pick a single checkpoint without needing a
    -- separate "current pointer" table.
    is_active                    BOOLEAN NOT NULL DEFAULT FALSE,
    notes                        TEXT
);

CREATE INDEX IF NOT EXISTS idx_absa_runs_active
    ON absa_runs (track, is_active);

CREATE TABLE IF NOT EXISTS absa_example_predictions (
    id                     BIGSERIAL PRIMARY KEY,
    model_version           TEXT NOT NULL REFERENCES absa_runs (model_version) ON DELETE CASCADE,
    -- Matches a key in app.py's EXAMPLE_REVIEWS dict (e.g.
    -- "[Mixed] Great quality, slow shipping"), so predictions from
    -- different runs on the SAME example text can be compared
    -- side-by-side across model versions for audit/regression checks.
    example_key              TEXT NOT NULL,
    text                      TEXT NOT NULL,
    aspect                    TEXT NOT NULL,
    aspect_confidence          DOUBLE PRECISION NOT NULL,
    sentiment                   TEXT NOT NULL,
    sentiment_confidence         DOUBLE PRECISION NOT NULL,
    UNIQUE (model_version, example_key)
);

CREATE INDEX IF NOT EXISTS idx_absa_example_predictions_model_version
    ON absa_example_predictions (model_version);
