"""
Neon Postgres connection/schema utilities for the ABSA model registry.

Kaggle notebooks train and evaluate this project's model; a loader script
(scripts/load_absa_results.py) pushes the results into the tables this
module manages. The app (app.py) only ever reads from them at startup to
decide which trained checkpoint to download from Hugging Face Hub — nothing
here trains a model or stores its weights. See db/schema.sql for the full
schema and the reasoning behind it.

DATABASE_URL is the single source of truth for the connection string
(populated from a local, gitignored .env by the caller via
python-dotenv's load_dotenv() — this module doesn't call it itself, so it
stays usable from contexts that already loaded the environment their own
way, e.g. a real deployment's env vars).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import tomllib
from sqlalchemy import Engine, text

_REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = _REPO_ROOT / "db" / "schema.sql"
ENV_PATH = _REPO_ROOT / ".env"


def normalize_database_url(url: str) -> str:
    """Point a Postgres URL at the psycopg3 driver this project installs.

    Neon (and most providers) hand out a bare ``postgresql://`` string, but
    SQLAlchemy maps that to psycopg2, which isn't a dependency here — the
    result is a confusing ModuleNotFoundError at connect time rather than a
    clear one. Rewriting the scheme means the connection string can be
    pasted verbatim from the Neon dashboard into DATABASE_URL.

    Also accepts a full ``[connections.postgresql]`` TOML block (the same
    shape Streamlit's secrets.toml uses), since that's the form already
    pasted into this project's .env — being lenient about which shape
    arrives beats a silent, confusing connection failure.
    """
    url = url.strip()
    if url.startswith("["):
        try:
            parsed = tomllib.loads(url).get("connections", {}).get("postgresql", {}).get("url")
        except tomllib.TOMLDecodeError:
            parsed = None
        if parsed:
            url = parsed.strip()
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def _read_env_toml_url() -> Optional[str]:
    """Read .env's raw contents as a [connections.postgresql] TOML block.

    python-dotenv's KEY=VALUE parser doesn't understand TOML table headers,
    so a connection string pasted in that shape (matching the Streamlit
    secrets.toml format this project's Neon credential was already copied
    from) never reaches os.environ at all — this reads the file directly.
    """
    if not ENV_PATH.is_file():
        return None
    try:
        parsed = tomllib.loads(ENV_PATH.read_text())
    except tomllib.TOMLDecodeError:
        return None
    return parsed.get("connections", {}).get("postgresql", {}).get("url")


def resolve_database_url() -> Optional[str]:
    """The connection string, normalized for psycopg3, or None if unset.

    Prefers a plain DATABASE_URL env var (populated from .env by the
    caller's load_dotenv(), or set directly in a real deployment). Falls
    back to a [connections.postgresql] TOML block in .env, since that's
    the shape already used here.
    """
    env_url = os.environ.get("DATABASE_URL") or _read_env_toml_url()
    return normalize_database_url(env_url) if env_url else None


def ensure_schema(engine: Engine) -> None:
    """Apply db/schema.sql. Idempotent (CREATE TABLE/INDEX IF NOT EXISTS),
    so it's safe to call on every loader-script run and every app startup."""
    with engine.begin() as conn:
        conn.exec_driver_sql(SCHEMA_PATH.read_text())


def get_active_run(engine: Engine, track: str = "bert-absa") -> Optional[Dict]:
    """The currently-active run for `track`, or None if none is marked active."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT * FROM absa_runs WHERE is_active = TRUE AND track = :track "
                "ORDER BY run_timestamp DESC LIMIT 1"
            ),
            {"track": track},
        ).mappings().first()
    return dict(row) if row is not None else None


def get_example_predictions(engine: Engine, model_version: str) -> list[Dict]:
    """All example predictions recorded for a given run, for cross-run comparison."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM absa_example_predictions WHERE model_version = :model_version "
                "ORDER BY example_key"
            ),
            {"model_version": model_version},
        ).mappings().all()
    return [dict(r) for r in rows]
