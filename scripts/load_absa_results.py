"""
Loads one ABSA results JSON (see the plan's "Results JSON schema") into the
Postgres absa_runs / absa_example_predictions tables.

This is a one-shot loader: there's exactly one JSON artifact per training
run (produced by notebooks/kaggle_train_and_push.ipynb on Kaggle, or a
hand-written smoke-test JSON to verify this pipeline locally), so the whole
load is one transaction — a partial load (run metadata written but example
predictions failed) is a worse state than "load failed, try again."

Usage:
    python scripts/load_absa_results.py results.json
    python scripts/load_absa_results.py results.json --no-set-active

Reads the connection string via src.db.resolve_database_url() (DATABASE_URL
env var, or a [connections.postgresql] TOML block in .env) after loading
.env with python-dotenv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import db  # noqa: E402  (needs the path insert above)

REQUIRED_KEYS = (
    "model_version",
    "run_timestamp_utc",
    "environment",
    "hf_repo_id",
    "hf_revision",
    "data_config",
    "hyperparameters",
    "metrics",
    "example_predictions",
)

_VALID_ENVIRONMENTS = ("kaggle_gpu", "local_smoke")

_REQUIRED_EXAMPLE_KEYS = (
    "example_key",
    "text",
    "aspect",
    "aspect_confidence",
    "sentiment",
    "sentiment_confidence",
)

_UPSERT_RUN_SQL = text("""
    INSERT INTO absa_runs
        (model_version, track, run_timestamp, environment, hf_repo_id, hf_revision,
         hf_filename, data_config, hyperparameters, metrics, notes)
    VALUES
        (:model_version, :track, :run_timestamp, :environment, :hf_repo_id, :hf_revision,
         :hf_filename, :data_config, :hyperparameters, :metrics, :notes)
    ON CONFLICT (model_version) DO UPDATE SET
        track = EXCLUDED.track, run_timestamp = EXCLUDED.run_timestamp,
        environment = EXCLUDED.environment, hf_repo_id = EXCLUDED.hf_repo_id,
        hf_revision = EXCLUDED.hf_revision, hf_filename = EXCLUDED.hf_filename,
        data_config = EXCLUDED.data_config, hyperparameters = EXCLUDED.hyperparameters,
        metrics = EXCLUDED.metrics, notes = EXCLUDED.notes
""")

_UPSERT_EXAMPLE_SQL = text("""
    INSERT INTO absa_example_predictions
        (model_version, example_key, text, aspect, aspect_confidence, sentiment, sentiment_confidence)
    VALUES
        (:model_version, :example_key, :text, :aspect, :aspect_confidence, :sentiment, :sentiment_confidence)
    ON CONFLICT (model_version, example_key) DO UPDATE SET
        text = EXCLUDED.text, aspect = EXCLUDED.aspect, aspect_confidence = EXCLUDED.aspect_confidence,
        sentiment = EXCLUDED.sentiment, sentiment_confidence = EXCLUDED.sentiment_confidence
""")

_DEACTIVATE_OTHERS_SQL = text(
    "UPDATE absa_runs SET is_active = FALSE WHERE track = :track AND model_version != :model_version"
)
_ACTIVATE_THIS_SQL = text("UPDATE absa_runs SET is_active = TRUE WHERE model_version = :model_version")


def validate_results(results: dict) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in results]
    if missing:
        raise ValueError(f"Results JSON is missing required key(s): {', '.join(missing)}")

    if results["environment"] not in _VALID_ENVIRONMENTS:
        raise ValueError(
            f"environment must be one of {_VALID_ENVIRONMENTS}, got {results['environment']!r}"
        )

    for i, example in enumerate(results["example_predictions"]):
        missing_example_keys = [k for k in _REQUIRED_EXAMPLE_KEYS if k not in example]
        if missing_example_keys:
            raise ValueError(
                f"example_predictions[{i}] is missing key(s): {', '.join(missing_example_keys)}"
            )


def load_results(engine, results: dict, set_active: bool = True) -> None:
    validate_results(results)
    model_version = results["model_version"]
    track = results.get("track", "bert-absa")

    with engine.begin() as conn:
        conn.execute(
            _UPSERT_RUN_SQL,
            {
                "model_version": model_version,
                "track": track,
                "run_timestamp": results["run_timestamp_utc"],
                "environment": results["environment"],
                "hf_repo_id": results["hf_repo_id"],
                "hf_revision": results["hf_revision"],
                "hf_filename": results.get("hf_filename", "checkpoint_best.pt"),
                "data_config": json.dumps(results["data_config"]),
                "hyperparameters": json.dumps(results["hyperparameters"]),
                "metrics": json.dumps(results["metrics"]),
                "notes": results.get("notes"),
            },
        )

        if results["example_predictions"]:
            conn.execute(
                _UPSERT_EXAMPLE_SQL,
                [
                    {
                        "model_version": model_version,
                        "example_key": ex["example_key"],
                        "text": ex["text"],
                        "aspect": ex["aspect"],
                        "aspect_confidence": ex["aspect_confidence"],
                        "sentiment": ex["sentiment"],
                        "sentiment_confidence": ex["sentiment_confidence"],
                    }
                    for ex in results["example_predictions"]
                ],
            )

        if set_active:
            conn.execute(_DEACTIVATE_OTHERS_SQL, {"track": track, "model_version": model_version})
            conn.execute(_ACTIVATE_THIS_SQL, {"model_version": model_version})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path", type=Path, help="Path to a results JSON file")
    parser.add_argument(
        "--no-set-active",
        action="store_true",
        help="Load the run's rows without marking it the active run for its track",
    )
    args = parser.parse_args()

    if not args.results_path.is_file():
        print(f"No such file: {args.results_path}", file=sys.stderr)
        return 2

    try:
        results = json.loads(args.results_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"Malformed JSON in {args.results_path}: {exc}", file=sys.stderr)
        return 2

    try:
        validate_results(results)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    load_dotenv()
    database_url = db.resolve_database_url()
    if not database_url:
        print(
            "No database connection string found — set DATABASE_URL, or put a "
            "[connections.postgresql] TOML block, in .env.",
            file=sys.stderr,
        )
        return 2

    engine = create_engine(database_url)
    db.ensure_schema(engine)
    load_results(engine, results, set_active=not args.no_set_active)

    n_examples = len(results["example_predictions"])
    active_note = "active" if not args.no_set_active else "not set active"
    print(
        f"Loaded {results['model_version']} ({results.get('track', 'bert-absa')}, "
        f"{results['environment']}, {active_note}): {n_examples} example prediction rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
