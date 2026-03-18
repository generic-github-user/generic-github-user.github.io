from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pypandoc
import yaml
from git import Commit, Repo
from jinja2 import Environment


REPO_ROOT = Path(__file__).resolve().parent
POSTS_DIR = REPO_ROOT / "posts"
NOTES_DIR = REPO_ROOT / "notes"
PAGES_DIR = REPO_ROOT / "pages"
FILES_DIR = REPO_ROOT / "files"
METADATA_PATH = REPO_ROOT / "metadata.yaml"
HEADER_TEMPLATE_PATH = PAGES_DIR / "header.md"
POST_TEMPLATE_PATH = PAGES_DIR / "post.md"
NOTE_TEMPLATE_PATH = PAGES_DIR / "note.md"
POST_OUTPUT_DIR = REPO_ROOT / "docs" / "posts"
NOTE_OUTPUT_DIR = REPO_ROOT / "docs" / "notes"
POST_HEADER_INCLUDE_PATH = PAGES_DIR / "post_head.html"
H2_ANCHOR_FILTER_PATH = PAGES_DIR / "h2_anchors.lua"
ARROW_FILTER_PATH = PAGES_DIR / "replace_arrows.lua"
PAGES_OUTPUT_DIR = REPO_ROOT / "docs"
DISPLAY_TIMEZONE = ZoneInfo("America/New_York")
JINJA_ENV = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


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
    word_count: int
    created_at: datetime
    updated_at: datetime
    commits: list[CommitInfo]


def load_posts(
    metadata_path: Path | None = None,
    posts_dir: Path | None = None,
    repo_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[Post]:
    """Return Post objects for every entry under `posts` in metadata.yaml."""

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
    """Return Post objects for every entry under `notes` in metadata.yaml."""

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
    metadata_obj = metadata or _read_metadata(metadata_file)
    entry_names = metadata_obj.get(metadata_key) or []
    LOGGER.info("Loading %s entries: %d found", metadata_key, len(entry_names))

    entries: list[Post] = []
    for entry_name in entry_names:
        entry_path = content_directory / f"{entry_name}.md"
        LOGGER.debug("Reading %s/%s", metadata_key, entry_name)
        frontmatter, content = _read_post_file(entry_path)
        commits = _collect_commits(repo, entry_path)
        created_at, updated_at = _resolve_post_timestamps(entry_path, commits)
        word_count = _count_words(content)

        entry = Post(
            name=entry_name,
            title=str(frontmatter.get("title") or ""),
            tags=list(frontmatter.get("tags") or []),
            location=str(frontmatter.get("location") or ""),
            content=content,
            word_count=word_count,
            created_at=created_at,
            updated_at=updated_at,
            commits=commits,
        )
        entries.append(entry)

    return entries


def render_site_header(
    metadata_path: Path | None = None,
    header_template_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Render the header template populated with navigation links."""

    metadata_file = Path(metadata_path or METADATA_PATH)
    header_path = Path(header_template_path or HEADER_TEMPLATE_PATH)

    metadata_obj = metadata or _read_metadata(metadata_file)
    navigation = _build_navigation(metadata_obj.get("pages"))
    template = JINJA_ENV.from_string(header_path.read_text(encoding="utf-8"))
    rendered = template.render(main_navigation=navigation)
    LOGGER.debug("Rendered site header")
    return rendered


def render_pages_to_html(
    metadata_path: Path | None = None,
    pages_dir: Path | None = None,
    files_dir: Path | None = None,
    posts_dir: Path | None = None,
    notes_dir: Path | None = None,
    repo_path: Path | None = None,
    header_template_path: Path | None = None,
    output_dir: Path | None = None,
    pandoc_extra_args: Iterable[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[Path]:
    """Render top-level pages (index, contact, etc.) to HTML files under src/."""

    metadata_file = Path(metadata_path or METADATA_PATH)
    pages_directory = Path(pages_dir or PAGES_DIR)
    static_files_directory = Path(files_dir or FILES_DIR)
    posts_directory = Path(posts_dir or POSTS_DIR)
    notes_directory = Path(notes_dir or NOTES_DIR)
    repo_dir = Path(repo_path or REPO_ROOT)
    header_path = Path(header_template_path or HEADER_TEMPLATE_PATH)
    output_directory = Path(output_dir or PAGES_OUTPUT_DIR)
    output_directory.mkdir(parents=True, exist_ok=True)

    metadata_obj = metadata or _read_metadata(metadata_file)
    page_entries = metadata_obj.get("pages") or []
    if not isinstance(page_entries, list):
        page_entries = [page_entries]
    LOGGER.info("Rendering %d pages", len(page_entries))

    site_header = render_site_header(
        metadata_path=metadata_file,
        header_template_path=header_path,
        metadata=metadata_obj,
    )
    base_pandoc_args = _build_pandoc_base_args(pandoc_extra_args)

    posts_for_listing: list[dict[str, Any]] = []
    try:
        posts = load_posts(
            metadata_path=metadata_file,
            posts_dir=posts_directory,
            repo_path=repo_dir,
            metadata=metadata_obj,
        )
        posts_for_listing = [
            _post_template_context(post, base_slug="posts")
            for post in sorted(posts, key=lambda p: p.updated_at, reverse=True)
        ]
        LOGGER.info("Loaded posts for listing: %d", len(posts))
    except FileNotFoundError:
        posts_for_listing = []

    notes_for_listing: list[dict[str, Any]] = []
    try:
        notes = load_notes(
            metadata_path=metadata_file,
            notes_dir=notes_directory,
            repo_path=repo_dir,
            metadata=metadata_obj,
        )
        notes_for_listing = [
            _post_template_context(note, base_slug="notes")
            for note in sorted(notes, key=lambda n: n.updated_at, reverse=True)
        ]
    except FileNotFoundError:
        notes_for_listing = []

    rendered_paths: list[Path] = []
    for entry in page_entries:
        slug, title, source_name, context = _normalize_page_entry(entry)
        if not slug or not source_name:
            continue

        page_path = pages_directory / f"{source_name}.md"
        if not page_path.exists():
            raise FileNotFoundError(f"page source not found: {page_path}")

        LOGGER.info("Rendering page %s -> %s.html", slug, slug)
        template = JINJA_ENV.from_string(page_path.read_text(encoding="utf-8"))
        rendered_markdown = template.render(
            site_header=site_header,
            page=context,
            posts=posts_for_listing,
            notes=notes_for_listing,
        )

        html = _markdown_to_html(
            rendered_markdown,
            base_pandoc_args,
            metadata={"title": "", "pagetitle": title or slug},
        )
        output_path = output_directory / f"{slug}.html"
        output_path.write_text(html, encoding="utf-8")
        rendered_paths.append(output_path)

    _copy_static_files(static_files_directory, output_directory)

    return rendered_paths


def render_posts_to_html(
    metadata_path: Path | None = None,
    posts_dir: Path | None = None,
    repo_path: Path | None = None,
    header_template_path: Path | None = None,
    post_template_path: Path | None = None,
    output_dir: Path | None = None,
    pandoc_extra_args: Iterable[str] | None = None,
) -> list[Path]:
    """Render every post listed in metadata.yaml into HTML files under src/posts."""

    metadata_file = Path(metadata_path or METADATA_PATH)
    posts_directory = Path(posts_dir or POSTS_DIR)
    repo_dir = Path(repo_path or REPO_ROOT)
    header_path = Path(header_template_path or HEADER_TEMPLATE_PATH)
    template_path = Path(post_template_path or POST_TEMPLATE_PATH)
    output_directory = Path(output_dir or POST_OUTPUT_DIR)
    output_directory.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Rendering posts to HTML")
    metadata_obj = _read_metadata(metadata_file)
    posts = load_posts(
        metadata_path=metadata_file,
        posts_dir=posts_directory,
        repo_path=repo_dir,
        metadata=metadata_obj,
    )
    site_header = render_site_header(
        metadata_path=metadata_file,
        header_template_path=header_path,
        metadata=metadata_obj,
    )

    base_pandoc_args = _build_pandoc_base_args(
        pandoc_extra_args, include_arrow_filter=True
    )

    template_source = template_path.read_text(encoding="utf-8")
    template = JINJA_ENV.from_string(template_source)

    rendered_paths: list[Path] = []
    LOGGER.info("Found %d posts", len(posts))
    for post in posts:
        LOGGER.info("Rendering post %s", post.name)
        post_context = _post_template_context(post, include_body=True, base_slug="posts")
        rendered_markdown = template.render(site_header=site_header, post=post_context)

        visible_title = post.title or post.name
        html = _markdown_to_html(
            rendered_markdown,
            base_pandoc_args,
            metadata={
                "title": "",
                "pagetitle": visible_title,
            },
        )
        output_path = output_directory / f"{post.name}.html"
        output_path.write_text(html, encoding="utf-8")
        rendered_paths.append(output_path)

    return rendered_paths


def render_notes_to_html(
    metadata_path: Path | None = None,
    notes_dir: Path | None = None,
    repo_path: Path | None = None,
    header_template_path: Path | None = None,
    note_template_path: Path | None = None,
    output_dir: Path | None = None,
    pandoc_extra_args: Iterable[str] | None = None,
) -> list[Path]:
    """Render every note listed in metadata.yaml into HTML files under src/notes."""

    metadata_file = Path(metadata_path or METADATA_PATH)
    notes_directory = Path(notes_dir or NOTES_DIR)
    repo_dir = Path(repo_path or REPO_ROOT)
    header_path = Path(header_template_path or HEADER_TEMPLATE_PATH)
    template_candidate = Path(note_template_path or NOTE_TEMPLATE_PATH)
    if not template_candidate.exists():
        template_candidate = POST_TEMPLATE_PATH
    template_path = template_candidate
    output_directory = Path(output_dir or NOTE_OUTPUT_DIR)
    output_directory.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Rendering notes to HTML")
    metadata_obj = _read_metadata(metadata_file)
    notes = load_notes(
        metadata_path=metadata_file,
        notes_dir=notes_directory,
        repo_path=repo_dir,
        metadata=metadata_obj,
    )
    site_header = render_site_header(
        metadata_path=metadata_file,
        header_template_path=header_path,
        metadata=metadata_obj,
    )

    base_pandoc_args = _build_pandoc_base_args(
        pandoc_extra_args, include_arrow_filter=True
    )

    template_source = template_path.read_text(encoding="utf-8")
    template = JINJA_ENV.from_string(template_source)

    rendered_paths: list[Path] = []
    LOGGER.info("Found %d notes", len(notes))
    for note in notes:
        LOGGER.info("Rendering note %s", note.name)
        note_context = _post_template_context(note, include_body=True, base_slug="notes")
        rendered_markdown = template.render(site_header=site_header, post=note_context)

        visible_title = note.title or note.name
        html = _markdown_to_html(
            rendered_markdown,
            base_pandoc_args,
            metadata={
                "title": "",
                "pagetitle": visible_title,
            },
        )
        output_path = output_directory / f"{note.name}.html"
        output_path.write_text(html, encoding="utf-8")
        rendered_paths.append(output_path)

    return rendered_paths


def _read_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("metadata.yaml must contain a mapping at the top level")

    LOGGER.debug("Loaded metadata from %s", metadata_path)
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
    LOGGER.debug("Collected %d commits for %s", len(commits), post_path)
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

        resolved_href: str | None
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


def _copy_static_files(source_dir: Path, destination_root: Path) -> Path | None:
    if not source_dir.exists():
        return None

    destination = destination_root / source_dir.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_dir, destination)
    LOGGER.info("Copied static files from %s to %s", source_dir, destination)
    return destination


def _normalize_page_entry(entry: Any) -> tuple[str, str, str, dict[str, Any]]:
    if isinstance(entry, dict):
        slug = _first_non_empty(entry, "slug", "path", "name")
        title = _first_non_empty(entry, "title", "label", "name", "slug")
        source_name = _first_non_empty(entry, "source", "template", "file", "slug", "path", "name")
        context = dict(entry)
        if slug and "slug" not in context:
            context["slug"] = slug
        if title and "title" not in context:
            context["title"] = title
        return slug or "", title or (slug or ""), source_name or (slug or ""), context

    value = str(entry).strip()
    context = {"title": value, "slug": value}
    return value, value, value, context


def _first_non_empty(entry: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = entry.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return None


def _post_template_context(post: Post, include_body: bool = False, *, base_slug: str = "posts") -> dict[str, Any]:
    normalized_slug = base_slug.strip("/") or "posts"
    permalink = f"/{normalized_slug}/{post.name}.html"
    relative = f"./{normalized_slug}/{post.name}.html"
    context: dict[str, Any] = {
        "name": post.name,
        "title": post.title or post.name,
        "location": post.location,
        "tags": post.tags,
        "word_count": post.word_count,
        "start_date": _format_datetime(post.created_at),
        "update_date": _format_datetime(post.updated_at),
        "url": permalink,
        "permalink": permalink,
        "relative_url": relative,
    }
    if include_body:
        context["content"] = post.content
    return context


def _count_words(text: str) -> int:
    """Best-effort word counting that treats hyphenated/possessive words as single tokens."""

    if not text:
        return 0
    return len(WORD_PATTERN.findall(text))


def _format_datetime(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    localized = dt.astimezone(DISPLAY_TIMEZONE)
    return localized.strftime("%Y-%m-%d at %H:%M %Z")


def _markdown_to_html(
    markdown: str,
    extra_args: Iterable[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> str:
    args = ["--standalone", "--highlight-style=pygments"]
    if extra_args:
        args.extend(extra_args)
    if metadata:
        for key, value in metadata.items():
            if value:
                args.append("--metadata")
                args.append(f"{key}={value}")

    try:
        return pypandoc.convert_text(markdown, "html", format="md", extra_args=args)
    except OSError as exc:  # pandoc binary missing
        raise RuntimeError(
            "pandoc is required to render posts; install it and retry"
        ) from exc


def _build_pandoc_base_args(
    extra_args: Iterable[str] | None = None,
    *,
    include_arrow_filter: bool = False,
) -> list[str]:
    args: list[str] = []
    if POST_HEADER_INCLUDE_PATH.exists():
        args.append(f"--include-in-header={POST_HEADER_INCLUDE_PATH}")
    if H2_ANCHOR_FILTER_PATH.exists():
        args.append(f"--lua-filter={H2_ANCHOR_FILTER_PATH}")
    if include_arrow_filter and ARROW_FILTER_PATH.exists():
        args.append(f"--lua-filter={ARROW_FILTER_PATH}")
    args.append("--section-divs")
    if extra_args:
        args.extend(extra_args)
    return args


__all__ = [
    "CommitInfo",
    "Post",
    "load_posts",
    "load_notes",
    "render_site_header",
    "render_pages_to_html",
    "render_posts_to_html",
    "render_notes_to_html",
]

def main():
    LOGGER.info("Starting site build")
    render_pages_to_html()
    render_posts_to_html()
    render_notes_to_html()
    LOGGER.info("Site build complete")

main()
