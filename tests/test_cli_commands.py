"""Smoke tests for the Typer CLI commands."""

import json
import os
from pathlib import Path
import zipfile

import sqlalchemy as sa
from typer.testing import CliRunner

from app.db.models import Base, Feedback, ReviewItem


runner = CliRunner()


def _seed_active_learning_records(db_url: str) -> None:
    engine = sa.create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    Session = sa.orm.sessionmaker(bind=engine, future=True)
    try:
        with Session() as session:
            session.add_all(
                [
                    ReviewItem(
                        text=(
                            "Economic update awaiting verification "
                            "by analysts."
                        ),
                        predicted_label="BUSINESS",
                        confidence_score=0.21,
                        confidence_margin=0.03,
                        model_version="v-cli",
                        labeled=1,
                        true_label="business insights",
                        top_labels=[
                            {"name": "BUSINESS", "prob": 0.52},
                            {"name": "POLITICS", "prob": 0.21},
                        ],
                    ),
                    Feedback(
                        text=(
                            "Reader feedback confirming technology "
                            "categorization."
                        ),
                        predicted_label="TECH",
                        true_label="technology",
                        model_version="v-cli",
                        confidence_score=0.87,
                    ),
                ]
            )
            session.commit()
    finally:
        engine.dispose()


def test_seed_db_cli_command(tmp_path, monkeypatch):
    db_path = tmp_path / "cli_seed.db"
    db_url = f"sqlite:///{db_path.as_posix()}"

    monkeypatch.setenv("DB_URL", db_url)

    from app.core.config import get_settings

    get_settings.cache_clear()

    from scripts.manage import app

    env = {**os.environ, "DB_URL": db_url}
    result = runner.invoke(app, ["seed-db", "--overwrite"], env=env)
    assert result.exit_code == 0, result.output

    assert db_path.exists()

    engine = sa.create_engine(db_url, future=True)
    try:
        insp = sa.inspect(engine)
        assert set(insp.get_table_names()) >= {"feedback", "review_items"}

        with engine.connect() as conn:
            feedback_count = conn.execute(
                sa.select(sa.func.count()).select_from(Feedback)
            ).scalar_one()
            review_count = conn.execute(
                sa.select(sa.func.count()).select_from(ReviewItem)
            ).scalar_one()

        assert feedback_count > 0
        assert review_count > 0
    finally:
        engine.dispose()

    # Second invocation without overwrite should report already populated
    result_again = runner.invoke(app, ["seed-db"], env=env)
    assert result_again.exit_code == 0, result_again.output
    assert "Already populated" in result_again.output


def test_bundle_artifacts_cli(tmp_path):
    source_dir = tmp_path / "bundle_src"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)
    (nested_dir / "artifact.txt").write_text("hello", encoding="utf-8")

    output_dir = tmp_path / "archives"

    from scripts.manage import app

    result = runner.invoke(
        app,
        [
            "bundle-artifacts",
            "--sources",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--label",
            "cli-test",
        ],
    )
    assert result.exit_code == 0, result.output

    archives = list(output_dir.glob("bundle_*.zip"))
    assert len(archives) == 1
    archive_path = archives[0]

    with zipfile.ZipFile(archive_path) as zf:
        assert "bundle_src/nested/artifact.txt" in zf.namelist()
        manifest = json.loads(zf.read("artifact-manifest.json"))
        assert manifest["label"] == "cli-test"
        assert manifest["push_requested"] is False
        assert any("bundle_src" in entry for entry in manifest["sources"])


def test_bundle_artifacts_push_cli(monkeypatch, tmp_path):
    source_dir = tmp_path / "bundle_src"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)
    (nested_dir / "artifact.txt").write_text("hello", encoding="utf-8")

    output_dir = tmp_path / "archives"

    class StubPublisher:
        def __init__(self) -> None:
            self.calls: list[tuple[Path, str | None]] = []

        def upload_file(
            self,
            file_path,
            *,
            destination_name=None,
        ):
            path_obj = Path(file_path)
            self.calls.append((path_obj, destination_name))
            return f"stub://{destination_name or path_obj.name}"

    stub = StubPublisher()

    monkeypatch.setenv("ARTIFACT_STORE_TYPE", "s3")
    monkeypatch.setenv("ARTIFACT_S3_BUCKET", "stub-bucket")
    monkeypatch.setenv("ARTIFACT_PUSH_DEFAULT", "0")

    from app.core.config import get_settings

    get_settings.cache_clear()

    monkeypatch.setattr(
        "scripts.manage.create_artifact_publisher",
        lambda settings: stub,
    )

    from scripts.manage import app

    result = runner.invoke(
        app,
        [
            "bundle-artifacts",
            "--sources",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--push",
            "--remote-name",
            "custom.zip",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Uploaded archive to: stub://custom.zip" in result.output
    assert len(stub.calls) == 1
    uploaded_path, dest_name = stub.calls[0]
    assert dest_name == "custom.zip"
    assert uploaded_path.exists()

    archives = list(output_dir.glob("bundle_*.zip"))
    assert archives, "archive should exist for manifest validation"
    with zipfile.ZipFile(archives[0]) as zf:
        manifest = json.loads(zf.read("artifact-manifest.json"))
        assert manifest["push_requested"] is True
        assert manifest["remote_name"] == "custom.zip"
        assert manifest["settings"]["artifact_store_type"] == "s3"


def test_active_finetune_cli_dry_run(monkeypatch, tmp_path):
    db_path = tmp_path / "active_cli.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    _seed_active_learning_records(db_url)

    monkeypatch.setenv("DB_URL", db_url)

    from app.core.config import get_settings

    get_settings.cache_clear()

    from scripts.manage import app

    env = {**os.environ, "DB_URL": db_url}
    result = runner.invoke(
        app,
        [
            "active-finetune",
            "--training-threshold",
            "1",
            "--dry-run",
        ],
        env=env,
    )

    assert result.exit_code == 0, result.output
    assert "Dry run requested" in result.output
    assert "review items" in result.output


def test_active_finetune_cli_triggers_training(monkeypatch, tmp_path):
    db_path = tmp_path / "active_cli_training.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    _seed_active_learning_records(db_url)

    dataset_path = tmp_path / "dummy_dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")

    monkeypatch.setenv("DB_URL", db_url)

    from app.core.config import get_settings

    get_settings.cache_clear()

    import scripts.manage as manage

    calls: dict[str, object] = {}

    def _fake_train(**kwargs):
        calls["kwargs"] = kwargs
        return {"f1_macro": 0.42, "accuracy": 0.7}

    monkeypatch.setattr(manage, "train_huffpost_transformer", _fake_train)

    env = {**os.environ, "DB_URL": db_url}
    result = runner.invoke(
        manage.app,
        [
            "active-finetune",
            "--base-data-path",
            str(dataset_path),
            "--training-threshold",
            "1",
            "--min-samples",
            "1",
        ],
        env=env,
    )

    assert result.exit_code == 0, result.output
    assert "Active learning fine-tune complete" in result.output
    assert calls
    kwargs = calls["kwargs"]  # type: ignore[index]
    assert isinstance(kwargs, dict)
    assert kwargs["extra_examples"]
    assert len(kwargs["extra_examples"]) == 2
    assert kwargs["limit"] is None
