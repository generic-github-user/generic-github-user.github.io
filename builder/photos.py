from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from .constants import (
    PHOTO_OUTPUT_DIR,
    PHOTO_PREVIEW_DIR,
    PHOTO_PREVIEW_MAX_DIMENSION,
    PHOTO_PUBLIC_BASE_URL,
    PHOTOS_DIR,
    R2_CREDENTIALS_PATH,
    RAW_PHOTOS_DIR,
    RCLONE_CONFIG_PATH,
)


LOGGER = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".heic", ".heif", ".jpeg", ".jpg", ".png", ".webp"}
TIMESTAMP_PATTERN = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})(?:_(?P<micros>\d{1,6}))?")


@dataclass(slots=True)
class PhotoAsset:
    filename: str
    full_output_path: Path
    preview_output_path: Path
    captured_at: datetime
    width: int
    height: int
    preview_width: int
    preview_height: int
    orientation: str
    full_public_url: str
    preview_public_url: str


@dataclass(slots=True)
class R2Config:
    account_id: str
    bucket_name: str
    access_key_id: str
    secret_access_key: str


def prepare_photo_gallery(
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
    preview_dir: Path | None = None,
    public_base_url: str = PHOTO_PUBLIC_BASE_URL,
) -> list[PhotoAsset]:
    source_dir = Path(raw_dir or RAW_PHOTOS_DIR)
    destination_dir = Path(output_dir or PHOTO_OUTPUT_DIR)
    preview_directory = Path(preview_dir or PHOTO_PREVIEW_DIR)

    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(
            prefix=".photos.tmp-",
            dir=destination_dir.parent,
        )
    )
    temp_full_dir = temp_root / destination_dir.name
    temp_preview_dir = temp_root / preview_directory.name
    temp_full_dir.mkdir(parents=True, exist_ok=True)
    temp_preview_dir.mkdir(parents=True, exist_ok=True)

    magick_binary = shutil.which("magick")
    source_files = _iter_source_images(source_dir)
    if source_files and not magick_binary:
        raise RuntimeError("ImageMagick is required to process photographs; install `magick` and retry")
    source_records = [
        (source_path, _read_capture_timestamp(magick_binary or "magick", source_path))
        for source_path in source_files
    ]
    source_records.sort(key=lambda record: (record[1], record[0].name))

    photo_assets: list[PhotoAsset] = []
    for source_path, captured_at in source_records:
        output_name = f"{source_path.stem}.jpg"
        full_output_path = temp_full_dir / output_name
        preview_output_path = temp_preview_dir / output_name
        _convert_image(magick_binary or "magick", source_path, full_output_path)
        _convert_preview_image(magick_binary or "magick", full_output_path, preview_output_path)
        os.utime(full_output_path, (0, 0))
        os.utime(preview_output_path, (0, 0))
        width, height = _identify_dimensions(magick_binary or "magick", full_output_path)
        preview_width, preview_height = _identify_dimensions(magick_binary or "magick", preview_output_path)
        orientation = "landscape" if width >= height else "portrait"
        photo_assets.append(
            PhotoAsset(
                filename=output_name,
                full_output_path=full_output_path,
                preview_output_path=preview_output_path,
                captured_at=captured_at,
                width=width,
                height=height,
                preview_width=preview_width,
                preview_height=preview_height,
                orientation=orientation,
                full_public_url=f"{public_base_url.rstrip('/')}/full_size/{output_name}",
                preview_public_url=f"{public_base_url.rstrip('/')}/previews/{output_name}",
            )
        )

    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_full_dir.replace(destination_dir)
    if preview_directory.exists():
        shutil.rmtree(preview_directory)
    temp_preview_dir.replace(preview_directory)
    shutil.rmtree(temp_root, ignore_errors=True)
    LOGGER.info("Prepared %d photographs in %s and %s", len(photo_assets), destination_dir, preview_directory)
    return photo_assets


def build_photo_page_context(photos: list[PhotoAsset]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {"landscape": [], "portrait": []}
    for photo in sorted(photos, key=lambda asset: asset.captured_at):
        grouped[photo.orientation].append(
            {
                "filename": photo.filename,
                "url": photo.full_public_url,
                "preview_url": photo.preview_public_url,
                "width": photo.preview_width,
                "height": photo.preview_height,
                "alt": _build_alt_text(photo.captured_at),
            }
        )
    return {
        "photo_count": len(photos),
        "gallery_markup": _build_gallery_markup(grouped),
        "photo_groups": grouped,
    }


def ensure_local_rclone_config(
    credentials_path: Path | None = None,
    config_path: Path | None = None,
) -> R2Config:
    parsed = _parse_r2_credentials(Path(credentials_path or R2_CREDENTIALS_PATH))
    output_path = Path(config_path or RCLONE_CONFIG_PATH)
    output_path.write_text(
        "\n".join(
            [
                "[anna-r2]",
                "type = s3",
                "provider = Cloudflare",
                "env_auth = false",
                "region = auto",
                f"access_key_id = {parsed.access_key_id}",
                f"secret_access_key = {parsed.secret_access_key}",
                f"endpoint = https://{parsed.account_id}.r2.cloudflarestorage.com",
                "",
            ]
        ),
        encoding="utf-8",
    )
    LOGGER.info("Wrote local rclone config to %s", output_path)
    return parsed


def sync_photos_to_r2(
    r2_config: R2Config,
    *,
    photos_dir: Path | None = None,
    config_path: Path | None = None,
) -> None:
    rclone_binary = shutil.which("rclone")
    if not rclone_binary:
        raise RuntimeError("rclone is required to upload photographs; install it and retry")
    source_dir = Path(photos_dir or PHOTOS_DIR)
    resolved_config_path = Path(config_path or RCLONE_CONFIG_PATH)
    destination = f"anna-r2:{r2_config.bucket_name}/photos"
    _run_command(
        [
            rclone_binary,
            "sync",
            str(source_dir),
            destination,
            "--config",
            str(resolved_config_path),
            "--fast-list",
        ],
        error_prefix="rclone photo sync failed",
    )
    LOGGER.info("Synced photographs from %s to %s", source_dir, destination)


def _iter_source_images(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []
    return [
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def _convert_image(magick_binary: str, source_path: Path, output_path: Path) -> None:
    _run_command(
        [
            magick_binary,
            str(source_path),
            "-auto-orient",
            "-colorspace",
            "sRGB",
            "-strip",
            "-interlace",
            "Plane",
            "-quality",
            "92",
            str(output_path),
        ],
        error_prefix=f"failed to convert {source_path.name}",
    )


def _convert_preview_image(magick_binary: str, source_path: Path, output_path: Path) -> None:
    _run_command(
        [
            magick_binary,
            str(source_path),
            "-resize",
            f"{PHOTO_PREVIEW_MAX_DIMENSION}x{PHOTO_PREVIEW_MAX_DIMENSION}>",
            "-strip",
            "-interlace",
            "Plane",
            "-quality",
            "85",
            str(output_path),
        ],
        error_prefix=f"failed to generate preview for {source_path.name}",
    )


def _identify_dimensions(magick_binary: str, image_path: Path) -> tuple[int, int]:
    completed = _run_command(
        [magick_binary, "identify", "-format", "%w %h", str(image_path)],
        error_prefix=f"failed to inspect {image_path.name}",
    )
    parts = completed.stdout.strip().split()
    if len(parts) != 2:
        raise RuntimeError(f"unexpected identify output for {image_path}: {completed.stdout!r}")
    return int(parts[0]), int(parts[1])


def _read_capture_timestamp(magick_binary: str, image_path: Path) -> datetime:
    completed = _run_command(
        [
            magick_binary,
            "identify",
            "-format",
            "%[EXIF:DateTimeOriginal]|%[EXIF:OffsetTimeOriginal]|%[EXIF:SubSecTimeOriginal]",
            str(image_path),
        ],
        error_prefix=f"failed to read capture timestamp for {image_path.name}",
    )
    parts = completed.stdout.strip().split("|")
    if len(parts) == 3 and parts[0]:
        raw_timestamp, raw_offset, raw_subsec = parts
        parsed = _parse_exif_datetime(raw_timestamp, raw_offset, raw_subsec)
        if parsed is not None:
            return parsed
    return _parse_filename_timestamp(image_path)


def _parse_r2_credentials(credentials_path: Path) -> R2Config:
    if not credentials_path.exists():
        raise FileNotFoundError(f"R2 credentials file not found: {credentials_path}")
    raw_values: dict[str, str] = {}
    for raw_line in credentials_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        raw_values[_normalize_credential_key(key)] = value.strip()

    access_key_id = _first_matching_value(raw_values, "r2s3accesskeyid", "accesskeyid", "s3accesskeyid")
    secret_access_key = _first_matching_value(raw_values, "r2s3secretaccesskey", "secretaccesskey", "s3secretaccesskey")
    account_id = _first_matching_value(raw_values, "r2accountid", "accountid", "cloudflareaccountid")
    bucket_name = _first_matching_value(raw_values, "r2bucketname", "bucketname", "bucket")
    missing = []
    if not access_key_id:
        missing.append("r2 s3 access key ID")
    if not secret_access_key:
        missing.append("r2 s3 secret access key")
    if not account_id:
        missing.append("r2 account ID")
    if not bucket_name:
        missing.append("r2 bucket name")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"{credentials_path} is missing required values: {joined}. "
            "Add them as `label: value` lines before running a full photo sync."
        )
    return R2Config(
        account_id=account_id,
        bucket_name=bucket_name,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
    )


def _parse_filename_timestamp(path: Path) -> datetime:
    match = TIMESTAMP_PATTERN.match(path.name)
    if not match:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    micros = (match.group("micros") or "0").ljust(6, "0")[:6]
    return datetime.strptime(
        f"{match.group('date')}{match.group('time')}{micros}",
        "%Y%m%d%H%M%S%f",
    ).replace(tzinfo=timezone.utc)


def _parse_exif_datetime(raw_timestamp: str, raw_offset: str, raw_subsec: str) -> datetime | None:
    timestamp = raw_timestamp.strip()
    if not timestamp:
        return None
    try:
        parsed = datetime.strptime(timestamp, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None

    subsec = raw_subsec.strip()
    if subsec:
        micros = (subsec + "000000")[:6]
        if micros.isdigit():
            parsed = parsed.replace(microsecond=int(micros))

    offset = raw_offset.strip()
    if not offset:
        return parsed
    sign = -1 if offset.startswith("-") else 1
    try:
        hours_str, minutes_str = offset[1:].split(":", 1)
        tz_delta = timedelta(hours=int(hours_str), minutes=int(minutes_str))
    except ValueError:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.replace(tzinfo=timezone(sign * tz_delta))


def _build_alt_text(captured_at: datetime) -> str:
    return f"Photograph from {captured_at.strftime('%Y-%m-%d at %H:%M:%S')}"


def _build_gallery_markup(grouped: dict[str, list[dict[str, Any]]]) -> str:
    clusters: list[str] = []
    for orientation in ("landscape", "portrait"):
        photos = grouped.get(orientation) or []
        if not photos:
            continue
        lines = [f'<div class="photograph-cluster photograph-cluster-{orientation}">']
        for photo in photos:
            url = escape(str(photo["url"]), quote=True)
            preview_url = escape(str(photo["preview_url"]), quote=True)
            alt = escape(str(photo["alt"]), quote=True)
            width = int(photo["width"])
            height = int(photo["height"])
            lines.extend(
                [
                    '<figure class="photograph-frame">',
                    f'<a class="photograph-link" href="{url}">',
                    (
                        '<img'
                        f' class="photograph-image"'
                        f' src="{preview_url}"'
                        f' alt="{alt}"'
                        f' width="{width}"'
                        f' height="{height}"'
                        ' loading="lazy"'
                        ' decoding="async"'
                        ' />'
                    ),
                    "</a>",
                    "</figure>",
                ]
            )
        lines.append("</div>")
        clusters.append("\n".join(lines))
    if not clusters:
        return ""
    return '<div class="photograph-gallery">\n' + "\n".join(clusters) + "\n</div>"


def _normalize_credential_key(raw_key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", raw_key.lower())


def _first_matching_value(values: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = values.get(key)
        if value:
            return value
    return ""


def _run_command(command: list[str], *, error_prefix: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip() or "no output"
        raise RuntimeError(f"{error_prefix}: {stderr}") from exc
