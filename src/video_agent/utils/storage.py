"""Storage utilities — local filesystem or S3/MinIO with presigned URLs."""
from __future__ import annotations

import os
from pathlib import Path

import structlog

from video_agent.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


async def upload_file(local_path: str, key: str) -> str:
    """
    Upload a file to the configured storage backend.
    Returns the storage key / path.
    """
    if settings.storage_backend == "s3":
        return await _upload_s3(local_path, key)
    return _upload_local(local_path, key)


def presigned_url(key: str, expires_in: int = 3600) -> str:
    """Generate a presigned URL for a storage key."""
    if settings.storage_backend == "s3":
        return _presign_s3(key, expires_in)
    return f"file://{key}"


def _upload_local(local_path: str, key: str) -> str:
    dest = os.path.join(settings.local_storage_path, key)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if local_path != dest and os.path.exists(local_path):
        import shutil
        shutil.copy2(local_path, dest)
    return dest


async def _upload_s3(local_path: str, key: str) -> str:
    import boto3  # type: ignore

    s3 = boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        endpoint_url=settings.s3_endpoint_url or None,
    )
    s3.upload_file(local_path, settings.s3_bucket, key)
    return key


def _presign_s3(key: str, expires_in: int) -> str:
    import boto3  # type: ignore

    s3 = boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        endpoint_url=settings.s3_endpoint_url or None,
    )
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": key},
        ExpiresIn=expires_in,
    )
