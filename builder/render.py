from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from git import Repo

from .constants import (
    FILES_DIR,
    HEADER_TEMPLATE_PATH,
    JINJA_ENV,
    METADATA_PATH,
    NOTE_OUTPUT_DIR,
    NOTE_TEMPLATE_PATH,
    NOTES_DIR,
    PAGES_DIR,
    PAGES_OUTPUT_DIR,
    POST_OUTPUT_DIR,
    POST_TEMPLATE_PATH,
    POSTS_DIR,
    REPO_ROOT,
)
from .content import load_notes, load_posts
from .history import (
    render_note_history,
    render_page_history,
    render_post_history,
    site_header_with_history,
)
from .models import Post, build_post_context, read_metadata, render_site_header
from .pandoc_utils import build_pandoc_base_args, markdown_to_html

LOGGER = logging.getLogger(__name__)


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
    force_history: bool = False,
    listing_posts: list[dict[str, Any]] | None = None,
    listing_notes: list[dict[str, Any]] | None = None,
) -> list[Path]:
    started = perf_counter()
    metadata_file = Path(metadata_path or METADATA_PATH)
    pages_directory = Path(pages_dir or PAGES_DIR)
    static_files_directory = Path(files_dir or FILES_DIR)
    posts_directory = Path(posts_dir or POSTS_DIR)
    notes_directory = Path(notes_dir or NOTES_DIR)
    repo_dir = Path(repo_path or REPO_ROOT)
    header_path = Path(header_template_path or HEADER_TEMPLATE_PATH)
    output_directory = Path(output_dir or PAGES_OUTPUT_DIR)
    output_directory.mkdir(parents=True, exist_ok=True)

    metadata_obj = metadata or read_metadata(metadata_file)
    page_entries = metadata_obj.get("pages") or []
    if not isinstance(page_entries, list):
        page_entries = [page_entries]
    LOGGER.info("Rendering %d pages", len(page_entries))
    normalized_entries = [_normalize_page_entry(entry) for entry in page_entries]
    site_pages = _build_page_targets(normalized_entries)

    site_header = render_site_header(
        metadata_path=metadata_file,
        header_template_path=header_path,
        metadata=metadata_obj,
    )
    base_pandoc_args = build_pandoc_base_args(pandoc_extra_args)

    posts_for_listing = listing_posts if listing_posts is not None else _collect_listing_posts(metadata_file, posts_directory, repo_dir, metadata_obj)
    notes_for_listing = listing_notes if listing_notes is not None else _collect_listing_notes(metadata_file, notes_directory, repo_dir, metadata_obj)
    random_targets_json = json.dumps(_build_random_targets(site_pages, posts_for_listing, notes_for_listing))

    repo = Repo(repo_dir)
    rendered_paths: list[Path] = []
    for slug, title, source_name, context in normalized_entries:
        if not slug or not source_name:
            continue
        page_path = pages_directory / f"{source_name}.md"
        if not page_path.exists():
            raise FileNotFoundError(f"page source not found: {page_path}")
        LOGGER.info("Rendering page %s -> %s.html", slug, slug)
        template_source = page_path.read_text(encoding="utf-8")
        template = JINJA_ENV.from_string(template_source)
        history_url: str | None = None
        if slug != "index":
            history_url = f"/{slug}/history/"
        context.update({
            "posts": posts_for_listing,
            "notes": notes_for_listing,
            "slug": slug,
            "history_url": history_url,
        })
        rendered_markdown = template.render(
            site_header=site_header_with_history(site_header, history_url),
            page=context,
            posts=posts_for_listing,
            notes=notes_for_listing,
            site_pages=site_pages,
            random_targets_json=random_targets_json,
        )
        html = markdown_to_html(
            rendered_markdown,
            base_pandoc_args,
            metadata={"title": "", "pagetitle": title or slug},
        )
        output_path = output_directory / f"{slug}.html"
        output_path.write_text(html, encoding="utf-8")
        rendered_paths.append(output_path)
        if history_url:
            render_page_history(
                slug=slug,
                title=title,
                source_path=page_path,
                repo=repo,
                render_kwargs={"page": context, "posts": posts_for_listing, "notes": notes_for_listing},
                site_header=site_header,
                base_pandoc_args=base_pandoc_args,
                output_directory=output_directory,
                history_url=history_url,
                current_url=f"/{slug}.html",
                force_history=force_history,
            )
    _copy_static_files(static_files_directory, output_directory)
    elapsed = perf_counter() - started
    LOGGER.info("Rendered %d pages in %.2fs", len(rendered_paths), elapsed)
    return rendered_paths


def render_posts_to_html(
    metadata_path: Path | None = None,
    posts_dir: Path | None = None,
    repo_path: Path | None = None,
    header_template_path: Path | None = None,
    post_template_path: Path | None = None,
    output_dir: Path | None = None,
    pandoc_extra_args: Iterable[str] | None = None,
    force_history: bool = False,
    posts: list[Post] | None = None,
) -> list[Path]:
    started = perf_counter()
    metadata_file = Path(metadata_path or METADATA_PATH)
    posts_directory = Path(posts_dir or POSTS_DIR)
    repo_dir = Path(repo_path or REPO_ROOT)
    header_path = Path(header_template_path or PAGES_DIR / "header.md")
    template_path = Path(post_template_path or POST_TEMPLATE_PATH)
    output_directory = Path(output_dir or POST_OUTPUT_DIR)
    output_directory.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Rendering posts to HTML")
    metadata_obj = read_metadata(metadata_file)
    posts_list = posts
    if posts_list is None:
        posts_list = load_posts(
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
    base_pandoc_args = build_pandoc_base_args(pandoc_extra_args, include_arrow_filter=True)
    template_source = template_path.read_text(encoding="utf-8")
    template = JINJA_ENV.from_string(template_source)
    repo = Repo(repo_dir)
    rendered_paths: list[Path] = []
    LOGGER.info("Found %d posts", len(posts_list))
    for post in posts_list:
        LOGGER.info("Rendering post %s", post.name)
        post_context = build_post_context(post, include_body=True, base_slug="posts")
        rendered_markdown = template.render(site_header=site_header, post=post_context)
        visible_title = post.title or post.name
        html = markdown_to_html(
            rendered_markdown,
            base_pandoc_args,
            metadata={"title": "", "pagetitle": visible_title},
        )
        output_path = output_directory / f"{post.name}.html"
        output_path.write_text(html, encoding="utf-8")
        rendered_paths.append(output_path)
        render_post_history(
            post=post,
            repo=repo,
            template=template,
            site_header=site_header,
            base_pandoc_args=base_pandoc_args,
            output_directory=output_directory,
            posts_dir=posts_directory,
            force_history=force_history,
        )
    elapsed = perf_counter() - started
    LOGGER.info("Rendered posts in %.2fs", elapsed)
    return rendered_paths


def render_notes_to_html(
    metadata_path: Path | None = None,
    notes_dir: Path | None = None,
    repo_path: Path | None = None,
    header_template_path: Path | None = None,
    note_template_path: Path | None = None,
    output_dir: Path | None = None,
    pandoc_extra_args: Iterable[str] | None = None,
    force_history: bool = False,
    notes: list[Post] | None = None,
) -> list[Path]:
    started = perf_counter()
    metadata_file = Path(metadata_path or METADATA_PATH)
    notes_directory = Path(notes_dir or NOTES_DIR)
    repo_dir = Path(repo_path or REPO_ROOT)
    header_path = Path(header_template_path or PAGES_DIR / "header.md")
    template_candidate = Path(note_template_path or NOTE_TEMPLATE_PATH)
    if not template_candidate.exists():
        template_candidate = POST_TEMPLATE_PATH
    template_path = template_candidate
    output_directory = Path(output_dir or NOTE_OUTPUT_DIR)
    output_directory.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Rendering notes to HTML")
    metadata_obj = read_metadata(metadata_file)
    notes_list = notes
    if notes_list is None:
        notes_list = load_notes(
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
    base_pandoc_args = build_pandoc_base_args(pandoc_extra_args, include_arrow_filter=True)
    template_source = template_path.read_text(encoding="utf-8")
    template = JINJA_ENV.from_string(template_source)
    repo = Repo(repo_dir)
    rendered_paths: list[Path] = []
    LOGGER.info("Found %d notes", len(notes_list))
    for note in notes_list:
        LOGGER.info("Rendering note %s", note.name)
        note_context = build_post_context(note, include_body=True, base_slug="notes")
        rendered_markdown = template.render(site_header=site_header, post=note_context)
        visible_title = note.title or note.name
        html = markdown_to_html(
            rendered_markdown,
            base_pandoc_args,
            metadata={"title": "", "pagetitle": visible_title},
        )
        output_path = output_directory / f"{note.name}.html"
        output_path.write_text(html, encoding="utf-8")
        rendered_paths.append(output_path)
        render_note_history(
            note=note,
            repo=repo,
            template=template,
            site_header=site_header,
            base_pandoc_args=base_pandoc_args,
            output_directory=output_directory,
            notes_dir=notes_directory,
            force_history=force_history,
        )
    elapsed = perf_counter() - started
    LOGGER.info("Rendered notes in %.2fs", elapsed)
    return rendered_paths


def _collect_listing_posts(
    metadata_file: Path,
    posts_directory: Path,
    repo_dir: Path,
    metadata: dict[str, Any],
    posts: list[Post] | None = None,
) -> list[dict[str, Any]]:
    try:
        posts_list = posts
        if posts_list is None:
            posts_list = load_posts(
                metadata_path=metadata_file,
                posts_dir=posts_directory,
                repo_path=repo_dir,
                metadata=metadata,
            )
        return [
            build_post_context(post, base_slug="posts")
            for post in sorted(posts_list, key=lambda p: p.updated_at, reverse=True)
        ]
    except FileNotFoundError:
        return []


def _collect_listing_notes(
    metadata_file: Path,
    notes_directory: Path,
    repo_dir: Path,
    metadata: dict[str, Any],
    notes: list[Post] | None = None,
) -> list[dict[str, Any]]:
    try:
        notes_list = notes
        if notes_list is None:
            notes_list = load_notes(
                metadata_path=metadata_file,
                notes_dir=notes_directory,
                repo_path=repo_dir,
                metadata=metadata,
            )
        return [
            build_post_context(note, base_slug="notes")
            for note in sorted(notes_list, key=lambda n: n.updated_at, reverse=True)
        ]
    except FileNotFoundError:
        return []


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


def _copy_static_files(source_dir: Path, destination_root: Path) -> Path | None:
    if not source_dir.exists():
        return None
    destination = destination_root / source_dir.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_dir, destination)
    LOGGER.info("Copied static files from %s to %s", source_dir, destination)
    return destination


def _build_page_targets(entries: list[tuple[str, str, str, dict[str, Any]]]) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for slug, title, source_name, _ in entries:
        normalized_slug = (slug or "").strip("/")
        if not normalized_slug or normalized_slug == "random":
            continue
        if not source_name:
            continue
        targets.append(
            {
                "title": title or normalized_slug,
                "slug": normalized_slug,
                "url": f"/{normalized_slug}",
            }
        )
    return targets


def _build_random_targets(
    pages: list[dict[str, str]],
    posts: list[dict[str, Any]],
    notes: list[dict[str, Any]],
) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []

    def _append_target(title: str | None, url: str | None) -> None:
        if not url:
            return
        resolved_title = (title or url).strip() or url
        targets.append({"title": resolved_title, "url": url})

    for page in pages:
        _append_target(page.get("title"), page.get("url"))
    for entry in posts:
        _append_target(entry.get("title") or entry.get("name"), entry.get("url"))
    for entry in notes:
        _append_target(entry.get("title") or entry.get("name"), entry.get("url"))
    return targets
