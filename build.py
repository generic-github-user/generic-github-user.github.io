from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from rich import print as rprint

import yaml
from git import Commit, Repo
from jinja2 import Environment


REPO_ROOT = Path(__file__).resolve().parent
POSTS_DIR = REPO_ROOT / "posts"
METADATA_PATH = REPO_ROOT / "metadata.yaml"
HEADER_TEMPLATE_PATH = REPO_ROOT / "pages" / "header.md"
JINJA_ENV = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)


@dataclass(slots=True)
class CommitInfo:
    """Minimal information about a commit touching a post source file."""

    hexsha: str
    summary: str
    author: str
    committer: str
    authored_datetime: datetime
    committed_datetime: datetime


@dataclass(slots=True)
class Post:
    """Structured representation of a blog post."""

    name: str
    title: str
    tags: list[str]
    location: str
    content: str
    created_at: datetime
    updated_at: datetime
    commits: list[CommitInfo]


def load_posts(
    metadata_path: Path | None = None,
    posts_dir: Path | None = None,
    repo_path: Path | None = None,
) -> list[Post]:
    """Return Post objects for every entry under `posts` in metadata.yaml."""

    metadata_file = Path(metadata_path or METADATA_PATH)
    posts_directory = Path(posts_dir or POSTS_DIR)
    repo_dir = Path(repo_path or REPO_ROOT)

    repo = Repo(repo_dir)
    metadata = _read_metadata(metadata_file)
    post_names = metadata.get("posts") or []

    posts: list[Post] = []
    for post_name in post_names:
        post_path = posts_directory / f"{post_name}.md"
        frontmatter, content = _read_post_file(post_path)
        commits = _collect_commits(repo, post_path)
        created_at, updated_at = _resolve_post_timestamps(post_path, commits)

        post = Post(
            name=post_name,
            title=str(frontmatter.get("title") or ""),
            tags=list(frontmatter.get("tags") or []),
            location=str(frontmatter.get("location") or ""),
            content=content,
            created_at=created_at,
            updated_at=updated_at,
            commits=commits,
        )
        posts.append(post)

    return posts


def render_site_header(
    metadata_path: Path | None = None,
    header_template_path: Path | None = None,
) -> str:
    """Render the header template populated with navigation links."""

    metadata_file = Path(metadata_path or METADATA_PATH)
    header_path = Path(header_template_path or HEADER_TEMPLATE_PATH)

    metadata = _read_metadata(metadata_file)
    navigation = _build_navigation(metadata.get("pages"))
    template = JINJA_ENV.from_string(header_path.read_text(encoding="utf-8"))
    return template.render(main_navigation=navigation)


def _read_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("metadata.yaml must contain a mapping at the top level")

    return data


def _read_post_file(post_path: Path) -> tuple[dict[str, Any], str]:
    if not post_path.exists():
        raise FileNotFoundError(f"post source not found: {post_path}")

    raw = post_path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise ValueError(f"post {post_path} is missing YAML front matter")

    parts = raw.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"post {post_path} has invalid YAML front matter")

    frontmatter_raw = parts[1]
    markdown_content = parts[2].lstrip("\n")
    frontmatter = yaml.safe_load(frontmatter_raw) or {}
    if not isinstance(frontmatter, dict):
        raise ValueError(f"post {post_path} front matter must be a mapping")

    return frontmatter, markdown_content


def _collect_commits(repo: Repo, post_path: Path) -> list[CommitInfo]:
    commits: list[CommitInfo] = []
    for commit in repo.iter_commits(paths=str(post_path.relative_to(repo.working_tree_dir))):
        commits.append(_to_commit_info(commit))
    return commits


def _to_commit_info(commit: Commit) -> CommitInfo:
    author = f"{commit.author.name} <{commit.author.email}>"
    committer = f"{commit.committer.name} <{commit.committer.email}>"
    return CommitInfo(
        hexsha=commit.hexsha,
        summary=commit.summary,
        author=author,
        committer=committer,
        authored_datetime=commit.authored_datetime,
        committed_datetime=commit.committed_datetime,
    )


def _resolve_post_timestamps(post_path: Path, commits: list[CommitInfo]) -> tuple[datetime, datetime]:
    if commits:
        created_at = commits[-1].authored_datetime
        updated_at = commits[0].authored_datetime
    else:
        stat = post_path.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
        updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

    return created_at, updated_at


def _build_navigation(pages_meta: Any) -> str:
    links = list(_iter_navigation_items(pages_meta))
    if not links:
        return ""
    return f"*{' | '.join(links)}*"


def _iter_navigation_items(pages_meta: Any) -> Iterable[str]:
    if pages_meta is None:
        return []

    entries: Iterable[Any]
    if isinstance(pages_meta, list):
        entries = pages_meta
    else:
        entries = [pages_meta]

    links: list[str] = []
    for entry in entries:
        label: str | None
        slug: str | None

        if isinstance(entry, dict):
            label = _first_non_empty(entry, "label", "title", "name", "slug", "path")
            slug = _first_non_empty(entry, "slug", "path", "href", "url", "name")
        else:
            value = str(entry).strip()
            label = value
            slug = value

        if not label or not slug:
            continue

        href = "/" + slug.lstrip("/")
        links.append(f"[{label}]({href})")

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


__all__ = ["CommitInfo", "Post", "load_posts", "render_site_header"]
