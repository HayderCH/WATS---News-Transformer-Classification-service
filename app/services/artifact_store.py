from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol

try:  # pragma: no cover - import guard for optional dependency detection
    import boto3  # type: ignore[import]
    from botocore.config import Config  # type: ignore[import]
except ImportError:  # pragma: no cover - surfaced during runtime use
    boto3 = None  # type: ignore
    Config = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from app.core.config import Settings


class ArtifactPublisher(Protocol):
    """Publisher interface for uploading artifact bundles."""

    def upload_file(
        self,
        file_path: Path,
        *,
        destination_name: Optional[str] = None,
    ) -> str:
        """Upload file and return the remote URI."""


@dataclass
class S3ArtifactPublisher:
    bucket: str
    prefix: str = ""
    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    use_path_style: bool = False

    def __post_init__(self) -> None:
        if boto3 is None:
            raise RuntimeError("boto3 is required for S3 artifact publishing.")
        config = None
        if self.use_path_style and Config is not None:
            config = Config(s3={"addressing_style": "path"})
        self._client = boto3.client(
            "s3",
            region_name=self.region,
            endpoint_url=self.endpoint_url,
            config=config,
        )

    def upload_file(
        self,
        file_path: Path,
        *,
        destination_name: Optional[str] = None,
    ) -> str:
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        key = _build_s3_key(self.prefix, destination_name or file_path.name)
        self._client.upload_file(str(file_path), self.bucket, key)
        return f"s3://{self.bucket}/{key}"


def _build_s3_key(prefix: str, name: str) -> str:
    safe_prefix = prefix.strip("/")
    if safe_prefix:
        return f"{safe_prefix}/{name}"
    return name


def create_artifact_publisher(settings: "Settings") -> ArtifactPublisher:
    """Create an artifact publisher based on configuration settings."""

    store_type = (settings.artifact_store_type or "local").strip().lower()
    if store_type in {"local", "filesystem", "none", ""}:
        raise ValueError(
            "Artifact store type 'local' does not support push operations. "
            "Update ARTIFACT_STORE_TYPE to a remote backend (e.g. 's3')."
        )

    if store_type == "s3":
        bucket = settings.artifact_s3_bucket
        if not bucket:
            raise ValueError(
                "ARTIFACT_S3_BUCKET must be set when using the S3 artifact"
                " store."
            )
        return S3ArtifactPublisher(
            bucket=bucket,
            prefix=settings.artifact_s3_prefix,
            region=settings.artifact_s3_region,
            endpoint_url=settings.artifact_s3_endpoint,
            use_path_style=settings.artifact_s3_use_path_style,
        )

    raise ValueError(
        f"Unsupported ARTIFACT_STORE_TYPE '{settings.artifact_store_type}'."
    )


__all__ = [
    "ArtifactPublisher",
    "S3ArtifactPublisher",
    "create_artifact_publisher",
]
