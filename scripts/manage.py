"""Unified CLI for training, evaluation, and database operations."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import typer
import optuna

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.models import Base  # noqa: E402
from app.db.seed import seed_initial_data  # noqa: E402
from scripts.train_baseline import train_baseline  # noqa: E402
from scripts.train_transformer import (  # noqa: E402
    train_transformer_classifier,
)
from scripts.train_transformer_huffpost import (  # noqa: E402
    train_huffpost_transformer,
)
from scripts.train_forecasting import main as train_forecasting_main  # noqa: E402
from scripts.eval_transformer import evaluate  # noqa: E402
from app.services import create_artifact_publisher  # noqa: E402
from app.services.active_learning import (  # noqa: E402
    collect_active_learning_examples,
)

app = typer.Typer(help="MLOps workflow commands.")


def _session_factory(db_url: str):
    engine = create_engine(db_url, future=True)
    Session = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        future=True,
    )
    return engine, Session


@app.command("seed-db")
def seed_db(
    overwrite: bool = False,
) -> None:
    """Populate the feedback SQLite database with demo data."""

    get_settings.cache_clear()
    settings = get_settings()
    engine, Session = _session_factory(settings.db_url)
    Base.metadata.create_all(engine)
    typer.echo(f"Database: {settings.db_url}")
    with Session() as session:
        result = seed_initial_data(session, overwrite=overwrite)
    engine.dispose()

    inserted = result.as_dict()
    message = (
        "Seeded"
        if overwrite or inserted["feedback"] or inserted["review_items"]
        else "Already populated"
    )
    typer.echo(
        (
            f"{message}: feedback={inserted['feedback']} "
            f"review_items={inserted['review_items']}"
        )
    )


@app.command("train-baseline")
def train_baseline_cmd(
    output_dir: Path = typer.Option(
        Path("models/classifier"),
        help="Directory for TF-IDF + logistic regression artifacts.",
    ),
    limit: Optional[int] = typer.Option(
        None,
        help="Limit number of samples (useful for smoke tests).",
    ),
    dataset: str = typer.Option(
        "huffpost",
        help="Dataset to train on: ag_news or huffpost.",
    ),
) -> None:
    """Train the classic TF-IDF + logistic regression classifier."""

    macro_f1 = train_baseline(str(output_dir), limit, dataset)
    typer.echo(f"Baseline training complete. macro_f1={macro_f1:.4f}")


@app.command("train-transformer")
def train_transformer_cmd(
    model_name: str = typer.Option(
        "distilbert-base-uncased",
        help="HF model name",
    ),
    output_dir: Path = typer.Option(
        Path("models/transformer_classifier"),
        help="Directory for checkpoints and exported model.",
    ),
    limit: Optional[int] = typer.Option(None, help="Subset training records."),
    epochs: int = typer.Option(3, help="Number of fine-tuning epochs."),
    train_batch_size: int = typer.Option(16, help="Training batch size."),
    eval_batch_size: int = typer.Option(32, help="Eval batch size."),
    grad_accum: int = typer.Option(1, help="Gradient accumulation steps."),
) -> None:
    """Fine-tune a transformer classifier on AG News."""

    results = train_transformer_classifier(
        model_name=model_name,
        output_dir=str(output_dir),
        limit=limit,
        epochs=epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        grad_accum=grad_accum,
    )
    summary = ", ".join(f"{k}={v:.4f}" for k, v in results.items())
    typer.echo(f"Transformer training complete. {summary}")


@app.command("eval-transformer")
def eval_transformer_cmd(
    model_dir: Path = typer.Argument(
        ..., help="Path to a trained transformer directory."
    ),
    data_path: Path = typer.Argument(
        ...,
        help="Path to evaluation JSON dataset.",
    ),
    batch_size: int = typer.Option(64, help="Evaluation batch size."),
    max_length: int = typer.Option(256, help="Max token length."),
    use_headline_only: bool = typer.Option(
        False, is_flag=True, help="Use only the headline text when evaluating."
    ),
    test_size: float = typer.Option(
        0.15,
        help="Hold-out fraction for evaluation.",
    ),
    seed: int = typer.Option(42, help="Random seed."),
    limit: Optional[int] = typer.Option(
        None,
        help="Limit evaluation samples.",
    ),
) -> None:
    """Evaluate a trained transformer model on the HuffPost dataset."""

    evaluate(
        model_dir=str(model_dir),
        data_path=str(data_path),
        batch_size=batch_size,
        max_length=max_length,
        use_headline_only=use_headline_only,
        test_size=test_size,
        seed=seed,
        limit=limit,
    )


@app.command("tune")
def tune_cmd(
    n_trials: int = typer.Option(100, help="Number of Optuna trials."),
) -> None:
    """Run hyperparameter tuning with Optuna (example objective)."""

    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return (x - 2) ** 2

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)
    typer.echo(f"Best value: {study.best_value} (params: {study.best_params})")


@app.command("active-finetune")
def active_finetune_cmd(
    base_data_path: Path = typer.Option(
        Path("data/raw/huffpost/News_Category_Dataset_v3.json"),
        help="Path to the baseline HuffPost JSON dataset.",
    ),
    output_dir: Path = typer.Option(
        Path("models/transformer_huffpost"),
        help=("Directory where the fine-tuned model checkpoints will be " "stored."),
    ),
    min_samples: int = typer.Option(
        50,
        help="Minimum samples per category after augmentation.",
    ),
    training_threshold: int = typer.Option(
        20,
        help="Minimum labeled examples required before training proceeds.",
    ),
    epochs: int = typer.Option(
        1,
        help="Number of active-learning fine-tune epochs.",
    ),
    train_batch_size: int = typer.Option(8, help="Training batch size."),
    eval_batch_size: int = typer.Option(16, help="Evaluation batch size."),
    learning_rate: float = typer.Option(
        5e-6,
        help="Learning rate for fine-tuning.",
    ),
    limit: Optional[int] = typer.Option(
        None,
        help="Optional cap on the augmented dataset size for quicker runs.",
    ),
    use_headline_only: bool = typer.Option(
        False,
        is_flag=True,
        help="Use only the headline text when assembling the dataset.",
    ),
    dry_run: bool = typer.Option(
        False,
        is_flag=True,
        help="Skip training and only report labeled example statistics.",
    ),
) -> None:
    """Fine-tune the HuffPost transformer with human-labeled feedback."""

    get_settings.cache_clear()
    settings = get_settings()
    engine, Session = _session_factory(settings.db_url)
    try:
        with Session() as session:
            examples, stats = collect_active_learning_examples(
                session,
                training_threshold=training_threshold,
            )
    finally:
        engine.dispose()

    total = stats["total_examples"]
    typer.echo(
        "Active-learning dataset: "
        f"{stats['review_labeled']} review items + "
        f"{stats['feedback_labeled']} feedback entries = {total} examples"
    )
    if stats.get("distinct_labels"):
        labels_obj = stats.get("distinct_labels") or []
        labels_fmt = ", ".join(str(label) for label in labels_obj)
        typer.echo("Distinct labels: " + labels_fmt)
    latest = stats.get("latest_label_at")
    if isinstance(latest, datetime):
        typer.echo(f"Most recent human label: {latest.isoformat()}")

    if total == 0:
        typer.echo("No labeled examples available yet. Nothing to train.")
        raise typer.Exit(code=0)

    if total < training_threshold:
        typer.echo(
            "Warning: Labeled examples below training threshold "
            f"({total} < {training_threshold})."
        )

    if dry_run:
        typer.echo("Dry run requested; skipping fine-tuning.")
        raise typer.Exit(code=0)

    if not base_data_path.exists():
        typer.echo(f"Base dataset not found: {base_data_path}")
        raise typer.Exit(code=1)

    results = train_huffpost_transformer(
        data_path=str(base_data_path),
        output_dir=str(output_dir),
        min_samples=min_samples,
        limit=limit,
        epochs=epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        use_headline_only=use_headline_only,
        extra_examples=examples,
    )

    summary = ", ".join(f"{k}={v:.4f}" for k, v in results.items())
    typer.echo(f"Active learning fine-tune complete. {summary}")


@app.command("train-forecasting")
def train_forecasting_cmd(
    output_dir: str = typer.Option(
        "models/forecasting", help="Directory to save trained forecasting models"
    ),
    experiment_name: str = typer.Option(
        "news_trends_forecasting", help="MLflow experiment name"
    ),
    max_categories: int = typer.Option(
        10, help="Maximum number of top categories to train models for"
    ),
) -> None:
    """Train time series forecasting models for news category trends.

    Trains hybrid ML/DL models (Prophet, XGBoost, LSTM) for forecasting
    future news article trends by category using historical data.
    """
    # Set up sys.argv for the training script
    import sys

    original_argv = sys.argv[:]
    sys.argv = [
        "train_forecasting.py",
        "--output-dir",
        output_dir,
        "--experiment-name",
        experiment_name,
        "--max-categories",
        str(max_categories),
    ]

    try:
        train_forecasting_main()
    finally:
        sys.argv = original_argv


def _resolve_sources(raw: str, project_root: Path, settings) -> list[Path]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    aliases: dict[str, Path] = {
        "models": project_root / settings.model_dir,
        "config": project_root / ".env",
        "metrics": project_root / "mlruns",
        "processed-data": project_root / "data" / "processed",
    }
    resolved: list[Path] = []
    for token in tokens:
        if token in aliases:
            resolved.append(aliases[token])
            continue
        path = Path(token)
        if not path.is_absolute():
            path = project_root / path
        resolved.append(path)
    return resolved


def _copy_source(src: Path, dest_root: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dest_root / src.name, dirs_exist_ok=True)
    else:
        target = dest_root / src.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


def _describe_sources(sources: list[Path], project_root: Path) -> list[str]:
    described: list[str] = []
    for src in sources:
        try:
            relative = src.relative_to(project_root)
            described.append(relative.as_posix())
        except ValueError:
            described.append(str(src))
    return described


@app.command("bundle-artifacts")
def bundle_artifacts_cmd(
    output_dir: Optional[Path] = typer.Option(
        None, help="Directory to store generated archives."
    ),
    sources: Optional[str] = typer.Option(
        None,
        help=(
            "Comma-separated list of directories/files or aliases to bundle. "
            "Defaults to ARTIFACT_SOURCES in config."
        ),
    ),
    label: Optional[str] = typer.Option(
        None,
        help="Optional label appended to the archive filename.",
    ),
    push: Optional[bool] = typer.Option(
        None,
        "--push/--no-push",
        help=("Upload the created archive to the configured artifact store."),
    ),
    remote_name: Optional[str] = typer.Option(
        None,
        help="Override the object name when uploading to remote storage.",
    ),
) -> None:
    """Create a timestamped zip archive containing model artifacts."""

    get_settings.cache_clear()
    settings = get_settings()
    project_root = ROOT
    source_spec = sources or settings.artifact_default_sources
    resolved_sources = _resolve_sources(source_spec, project_root, settings)

    existing_sources = [src for src in resolved_sources if src.exists()]
    if not existing_sources:
        typer.echo("No artifact sources found to bundle.")
        raise typer.Exit(code=0)

    archive_root = output_dir or Path(settings.artifact_root) / "archives"
    archive_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    parts = ["bundle", timestamp]
    if label:
        sanitized = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in label
        )
        if sanitized:
            parts.append(sanitized)
    archive_base = archive_root / "_".join(parts)

    push_enabled = settings.artifact_push_default if push is None else push
    manifest_data: dict[str, object] = {
        "created_at": timestamp,
        "label": label,
        "source_spec": source_spec,
        "sources": _describe_sources(existing_sources, project_root),
        "remote_name": remote_name,
        "push_requested": push_enabled,
        "settings": {
            "classifier_version": settings.classifier_version,
            "summarizer_version": settings.summarizer_version,
            "artifact_store_type": settings.artifact_store_type,
            "artifact_push_default": settings.artifact_push_default,
        },
    }
    if settings.artifact_store_type == "s3":
        manifest_data["settings"]["artifact_store"] = {
            "bucket": settings.artifact_s3_bucket,
            "prefix": settings.artifact_s3_prefix,
            "region": settings.artifact_s3_region,
            "endpoint": settings.artifact_s3_endpoint,
            "path_style": settings.artifact_s3_use_path_style,
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_root = Path(tmpdir)
        manifest_path = temp_root / "artifact-manifest.json"
        manifest_data["archive_name"] = f"{archive_base.name}.zip"
        manifest_path.write_text(
            json.dumps(manifest_data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for src in existing_sources:
            _copy_source(src, temp_root)
        archive_file = shutil.make_archive(str(archive_base), "zip", temp_root)

    archive_path = Path(archive_file)
    typer.echo(f"Created archive: {archive_path}")

    if not push_enabled:
        return

    try:
        publisher = create_artifact_publisher(settings)
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"Artifact push failed: {exc}")
        raise typer.Exit(code=1) from exc

    destination = remote_name or archive_path.name
    try:
        remote_uri = publisher.upload_file(
            archive_path,
            destination_name=destination,
        )
    except Exception as exc:  # pragma: no cover - defensive catch, rethrow
        typer.echo(f"Artifact upload failed: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"Uploaded archive to: {remote_uri}")


if __name__ == "__main__":
    app()
