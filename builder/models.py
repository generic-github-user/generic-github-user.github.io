from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import logging
import yaml

from .constants import (
    HEADER_TEMPLATE_PATH,
    JINJA_ENV,
    METADATA_PATH,
    WORD_PATTERN,
    DISPLAY_TIMEZONE,
)


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CommitInfo:
    hexsha: str
    summary: str
    author: str
    committer: str
    authored_datetime: datetime
    committed_datetime: datetime


@dataclass(slots=True)
class Post:
    name: str
    title: str
    tags: list[str]
    location: str
    content: str
    word_count: int
    created_at: datetime
    updated_at: datetime
    commits: list[CommitInfo]


def render_site_header(
    metadata_path: Path | None = None,
    header_template_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    metadata_file = Path(metadata_path or METADATA_PATH)
    header_path = Path(header_template_path or HEADER_TEMPLATE_PATH)
    metadata_obj = metadata or read_metadata(metadata_file)
    navigation = _build_navigation(metadata_obj.get("pages"))
    template = JINJA_ENV.from_string(header_path.read_text(encoding="utf-8"))
    rendered = template.render(main_navigation=navigation)
    LOGGER.debug("Rendered site header")
    return rendered


def read_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("metadata.yaml must contain a mapping at the top level")
    LOGGER.debug("Loaded metadata from %s", metadata_path)
    return data


def parse_frontmatter(raw: str, *, source: str | None = None) -> tuple[dict[str, Any], str]:
    identifier = source or "<string>"
    if not raw.startswith("---"):
        raise ValueError(f"post {identifier} is missing YAML front matter")
    parts = raw.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"post {identifier} has invalid YAML front matter")
    frontmatter_raw = parts[1]
    markdown_content = parts[2].lstrip("\n")
    frontmatter = yaml.safe_load(frontmatter_raw) or {}
    if not isinstance(frontmatter, dict):
        raise ValueError(f"post {identifier} front matter must be a mapping")
    return frontmatter, markdown_content


def read_post_file(post_path: Path) -> tuple[dict[str, Any], str]:
    if not post_path.exists():
        raise FileNotFoundError(f"post source not found: {post_path}")
    raw = post_path.read_text(encoding="utf-8")
    return parse_frontmatter(raw, source=str(post_path))


def build_post_context(post: Post, *, include_body: bool = False, base_slug: str = "posts") -> dict[str, Any]:
    normalized_slug = base_slug.strip("/") or "posts"
    permalink = f"/{normalized_slug}/{post.name}.html"
    relative = f"./{normalized_slug}/{post.name}.html"
    history_url = f"/{normalized_slug}/{post.name}/history/"
    context: dict[str, Any] = {
        "name": post.name,
        "title": post.title or post.name,
        "location": post.location,
        "tags": post.tags,
        "word_count": post.word_count,
        "start_date": format_datetime(post.created_at),
        "update_date": format_datetime(post.updated_at),
        "url": permalink,
        "permalink": permalink,
        "relative_url": relative,
        "history_url": history_url,
    }
    if include_body:
        context["content"] = post.content
    return context


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(WORD_PATTERN.findall(text))


def format_datetime(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    localized = dt.astimezone(DISPLAY_TIMEZONE)
    return localized.strftime("%Y-%m-%d at %H:%M %Z")


def _build_navigation(pages_meta: Any) -> str:
    links = list(_iter_navigation_items(pages_meta))
    if not links:
        return ""
    return f"*{' | '.join(links)}*"


def _iter_navigation_items(pages_meta: Any) -> list[str]:
    if pages_meta is None:
        return []
    entries: list[Any]
    if isinstance(pages_meta, list):
        entries = pages_meta
    else:
        entries = [pages_meta]
    links: list[str] = []
    for entry in entries:
        label: str | None
        slug: str | None
        direct_href: str | None = None
        if isinstance(entry, dict):
            label = _first_non_empty(entry, "label", "title", "name", "slug", "path")
            slug = _first_non_empty(entry, "slug", "path", "name")
            direct_href = _first_non_empty(entry, "href", "url")
        else:
            value = str(entry).strip()
            label = value
            slug = value
        if not label:
            continue
        if direct_href:
            resolved_href = direct_href
        elif slug:
            resolved_href = "/" + slug.lstrip("/")
        else:
            resolved_href = None
        if not resolved_href:
            continue
        links.append(f"[{label}]({resolved_href})")
    return links


def _first_non_empty(entry: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = entry.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return None
