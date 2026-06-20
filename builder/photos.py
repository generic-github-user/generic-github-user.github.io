from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from .constants import (
    PHOTO_BUILD_CACHE_PATH,
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
PHOTO_BUILD_CACHE_VERSION = 1


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
    cache_path = PHOTO_BUILD_CACHE_PATH

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
    LOGGER.info("Preparing %d photographs from %s", len(source_files), source_dir)
    existing_cache = _load_photo_build_cache(cache_path)

    photo_assets: list[PhotoAsset] = []
    cache_entries: dict[str, dict[str, Any]] = {}
    uncached_sources: list[tuple[Path, os.stat_result]] = []
    reused_count = 0
    for source_path in source_files:
        source_stat = source_path.stat()
        output_name = f"{source_path.stem}.jpg"
        cached_entry = existing_cache.get(source_path.name)
        if cached_entry and _cache_entry_matches(cached_entry, source_stat):
            full_source = destination_dir / output_name
            preview_source = preview_directory / output_name
            if full_source.exists() and preview_source.exists():
                full_output_path = temp_full_dir / output_name
                preview_output_path = temp_preview_dir / output_name
                _stage_existing_derivative(full_source, full_output_path)
                _stage_existing_derivative(preview_source, preview_output_path)
                photo_assets.append(
                    _photo_asset_from_cache(
                        cached_entry,
                        output_name=output_name,
                        full_output_path=full_output_path,
                        preview_output_path=preview_output_path,
                        public_base_url=public_base_url,
                    )
                )
                cache_entries[source_path.name] = cached_entry
                reused_count += 1
                continue
        uncached_sources.append((source_path, source_stat))

    if reused_count:
        LOGGER.info("Reused cached derivatives for %d/%d photographs", reused_count, len(source_files))
    if uncached_sources:
        LOGGER.info("Reading embedded capture timestamps for %d changed/new photographs", len(uncached_sources))
    capture_timestamps = _read_capture_timestamps(
        magick_binary or "magick",
        [source_path for source_path, _ in uncached_sources],
    )

    total_uncached = len(uncached_sources)
    for index, (source_path, source_stat) in enumerate(uncached_sources, start=1):
        if _should_log_progress(index, total_uncached):
            LOGGER.info("Processing changed photograph %d/%d: %s", index, total_uncached, source_path.name)
        captured_at = capture_timestamps[source_path]
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
        cache_entries[source_path.name] = _build_cache_entry(
            source_stat=source_stat,
            captured_at=captured_at,
            width=width,
            height=height,
            preview_width=preview_width,
            preview_height=preview_height,
            orientation=orientation,
        )

    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_full_dir.replace(destination_dir)
    if preview_directory.exists():
        shutil.rmtree(preview_directory)
    temp_preview_dir.replace(preview_directory)
    shutil.rmtree(temp_root, ignore_errors=True)
    _write_photo_build_cache(cache_path, cache_entries)
    LOGGER.info("Prepared %d photographs in %s and %s", len(photo_assets), destination_dir, preview_directory)
    return photo_assets


def build_photo_page_context(photos: list[PhotoAsset]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {"landscape": [], "portrait": []}
    for photo in sorted(photos, key=lambda asset: _capture_sort_key(asset.captured_at)):
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
    LOGGER.info("Syncing photographs to %s", destination)
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


def _read_capture_timestamps(magick_binary: str, image_paths: list[Path]) -> dict[Path, datetime]:
    if not image_paths:
        return {}

    results: dict[Path, datetime] = {}
    for batch in _batched(image_paths, 32):
        try:
            completed = _run_command(
                [
                    magick_binary,
                    "identify",
                    "-format",
                    "%[EXIF:DateTimeOriginal]|%[EXIF:OffsetTimeOriginal]|%[EXIF:SubSecTimeOriginal]\n",
                    *[str(path) for path in batch],
                ],
                error_prefix="failed to read batched capture timestamps",
            )
            lines = completed.stdout.splitlines()
            if len(lines) != len(batch):
                raise RuntimeError("unexpected batched identify output length")
            for source_path, line in zip(batch, lines, strict=True):
                parsed = _parse_capture_timestamp_line(line)
                results[source_path] = parsed if parsed is not None else _parse_filename_timestamp(source_path)
        except RuntimeError:
            for source_path in batch:
                results[source_path] = _read_capture_timestamp(magick_binary, source_path)
    return results


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
    parsed = _parse_capture_timestamp_line(completed.stdout.strip())
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


def _parse_capture_timestamp_line(line: str) -> datetime | None:
    parts = line.split("|")
    if len(parts) != 3 or not parts[0]:
        return None
    raw_timestamp, raw_offset, raw_subsec = parts
    return _parse_exif_datetime(raw_timestamp, raw_offset, raw_subsec)


def _build_alt_text(captured_at: datetime) -> str:
    return f"Photograph from {captured_at.strftime('%Y-%m-%d at %H:%M:%S')}"


def _capture_sort_key(captured_at: datetime) -> datetime:
    if captured_at.tzinfo is None:
        return captured_at
    return captured_at.replace(tzinfo=None)


def _should_log_progress(index: int, total: int) -> bool:
    if total <= 0:
        return False
    if index == 1 or index == total:
        return True
    if total <= 10:
        return True
    return index % 5 == 0


def _load_photo_build_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("version") != PHOTO_BUILD_CACHE_VERSION:
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    return {str(key): value for key, value in entries.items() if isinstance(value, dict)}


def _write_photo_build_cache(cache_path: Path, entries: dict[str, dict[str, Any]]) -> None:
    cache_path.write_text(
        json.dumps(
            {
                "version": PHOTO_BUILD_CACHE_VERSION,
                "entries": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _cache_entry_matches(entry: dict[str, Any], source_stat: os.stat_result) -> bool:
    return (
        int(entry.get("source_size", -1)) == source_stat.st_size
        and int(entry.get("source_mtime_ns", -1)) == source_stat.st_mtime_ns
    )


def _build_cache_entry(
    *,
    source_stat: os.stat_result,
    captured_at: datetime,
    width: int,
    height: int,
    preview_width: int,
    preview_height: int,
    orientation: str,
) -> dict[str, Any]:
    return {
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "captured_at": captured_at.isoformat(),
        "width": width,
        "height": height,
        "preview_width": preview_width,
        "preview_height": preview_height,
        "orientation": orientation,
    }


def _photo_asset_from_cache(
    entry: dict[str, Any],
    *,
    output_name: str,
    full_output_path: Path,
    preview_output_path: Path,
    public_base_url: str,
) -> PhotoAsset:
    captured_at = datetime.fromisoformat(str(entry["captured_at"]))
    return PhotoAsset(
        filename=output_name,
        full_output_path=full_output_path,
        preview_output_path=preview_output_path,
        captured_at=captured_at,
        width=int(entry["width"]),
        height=int(entry["height"]),
        preview_width=int(entry["preview_width"]),
        preview_height=int(entry["preview_height"]),
        orientation=str(entry["orientation"]),
        full_public_url=f"{public_base_url.rstrip('/')}/full_size/{output_name}",
        preview_public_url=f"{public_base_url.rstrip('/')}/previews/{output_name}",
    )


def _stage_existing_derivative(source_path: Path, destination_path: Path) -> None:
    try:
        os.link(source_path, destination_path)
    except OSError:
        shutil.copy2(source_path, destination_path)


def _batched(items: list[Path], batch_size: int) -> list[list[Path]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


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
