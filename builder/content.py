from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git import Repo

from .constants import METADATA_PATH, NOTES_DIR, POSTS_DIR, REPO_ROOT
from .git_utils import collect_commits
from .models import Post, count_words, read_metadata, read_post_file

LOGGER = logging.getLogger(__name__)


def load_posts(
    metadata_path: Path | None = None,
    posts_dir: Path | None = None,
    repo_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[Post]:
    return _load_entries(
        "posts",
        metadata_path=metadata_path,
        content_dir=posts_dir or POSTS_DIR,
        repo_path=repo_path or REPO_ROOT,
        metadata=metadata,
    )


def load_notes(
    metadata_path: Path | None = None,
    notes_dir: Path | None = None,
    repo_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[Post]:
    return _load_entries(
        "notes",
        metadata_path=metadata_path,
        content_dir=notes_dir or NOTES_DIR,
        repo_path=repo_path or REPO_ROOT,
        metadata=metadata,
    )


def _load_entries(
    metadata_key: str,
    *,
    metadata_path: Path | None = None,
    content_dir: Path,
    repo_path: Path,
    metadata: dict[str, Any] | None = None,
) -> list[Post]:
    metadata_file = Path(metadata_path or METADATA_PATH)
    content_directory = Path(content_dir)
    repo_dir = Path(repo_path)
    repo = Repo(repo_dir)
    metadata_obj = metadata or read_metadata(metadata_file)
    entry_names = metadata_obj.get(metadata_key) or []
    LOGGER.info("Loading %s entries: %d found", metadata_key, len(entry_names))
    entries: list[Post] = []
    for entry_name in entry_names:
        entry_path = content_directory / f"{entry_name}.md"
        frontmatter, content = read_post_file(entry_path)
        commits = collect_commits(repo, entry_path)
        created_at, updated_at = _resolve_post_timestamps(entry_path, commits)
        post = Post(
            name=entry_name,
            title=str(frontmatter.get("title") or ""),
            tags=list(frontmatter.get("tags") or []),
            location=str(frontmatter.get("location") or ""),
            content=content,
            word_count=count_words(content),
            created_at=created_at,
            updated_at=updated_at,
            commits=commits,
        )
        entries.append(post)
    return entries


def _resolve_post_timestamps(post_path: Path, commits: list[Any]) -> tuple[datetime, datetime]:
    if commits:
        created_at = commits[-1].authored_datetime
        updated_at = commits[0].authored_datetime
    else:
        stat = post_path.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
        updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return created_at, updated_at
