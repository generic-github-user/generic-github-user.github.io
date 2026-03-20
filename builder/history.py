from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any

from git import Repo

from .constants import HISTORY_HASH_PREFIX, JINJA_ENV, REPO_ROOT
from .git_utils import (
    collect_commits,
    commit_stats_for_path,
    read_file_from_commit,
    read_post_from_commit,
    repo_relative_path,
)
from .models import Post, build_post_context, count_words, format_datetime
from .pandoc_utils import markdown_to_html

LOGGER = logging.getLogger(__name__)


def site_header_with_history(site_header: str, history_url: str | None) -> str:
    if not history_url:
        return site_header
    snippet = f'<div class="history-link"><a href="{history_url}">View revision history</a></div>'
    return f"{site_header}\n\n{snippet}"


def history_banner_block(commit_info, current_url: str) -> str:
    short_hash = commit_info.hexsha[:HISTORY_HASH_PREFIX]
    timestamp = format_datetime(commit_info.authored_datetime)
    escaped_url = html.escape(current_url, quote=True)
    return (
        f'<div class="history-banner">Viewing revision from {timestamp} '
        f'(<code>{short_hash}</code>). '
        f'<a href="{escaped_url}">Return to current version</a></div>'
    )


def prepend_history_banner(markdown: str, banner_html: str | None) -> str:
    if not banner_html:
        return markdown
    return f"{banner_html}\n\n{markdown}"


def write_history_index(
    *,
    history_dir: Path,
    title: str,
    entries: list[dict[str, Any]],
    site_header: str,
    base_pandoc_args: list[str],
    current_url: str,
) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        site_header,
        "",
        f"## Revision history for \"{title}\"",
        "",
        f"[Return to current version]({current_url})",
        "",
    ]
    if not entries:
        lines.append("_No committed history is available yet for this page._")
    else:
        divider = "\n<hr />\n\n"
        rendered_entries: list[str] = []
        for entry in entries:
            heading = f"### {entry['timestamp']} — [`{entry['hash_short']}`]({entry['snapshot_name']})"
            summary = f"`{entry['hash']}` · {entry['message']}"
            diff_line = (
                f"<span class=\"diff-add\">+{entry['insertions']}</span> "
                f"<span class=\"diff-del\">-{entry['deletions']}</span> "
            )
            block_lines = [heading, "", f"{summary} · {diff_line}"]
            rendered_entries.append("\n".join(block_lines))
        lines.append('\n\n'.join(rendered_entries))

    markdown = "\n".join(lines)
    pagetitle = f"Revision history for \"{title}\""
    html_output = markdown_to_html(
        markdown,
        base_pandoc_args,
        metadata={"title": "", "pagetitle": pagetitle},
    )
    (history_dir / "index.html").write_text(html_output, encoding="utf-8")


def render_post_history(
    *,
    post: Post,
    repo: Repo,
    template,
    site_header: str,
    base_pandoc_args: list[str],
    output_directory: Path,
    posts_dir: Path,
    force_history: bool,
) -> None:
    _render_entry_history(
        entry=post,
        repo=repo,
        template=template,
        site_header=site_header,
        base_pandoc_args=base_pandoc_args,
        output_directory=output_directory,
        source_directory=posts_dir,
        base_slug="posts",
        force_history=force_history,
    )


def render_note_history(
    *,
    note: Post,
    repo: Repo,
    template,
    site_header: str,
    base_pandoc_args: list[str],
    output_directory: Path,
    notes_dir: Path,
    force_history: bool,
) -> None:
    _render_entry_history(
        entry=note,
        repo=repo,
        template=template,
        site_header=site_header,
        base_pandoc_args=base_pandoc_args,
        output_directory=output_directory,
        source_directory=notes_dir,
        base_slug="notes",
        force_history=force_history,
    )


def render_page_history(
    *,
    slug: str,
    title: str,
    source_path: Path,
    repo: Repo,
    render_kwargs: dict[str, Any],
    site_header: str,
    base_pandoc_args: list[str],
    output_directory: Path,
    history_url: str,
    current_url: str,
    force_history: bool,
) -> None:
    history_dir = output_directory / slug / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    relative_source = repo_relative_path(source_path, REPO_ROOT)
    commits = collect_commits(repo, source_path)
    entries_for_index: list[dict[str, Any]] = []
    site_header_with_link = site_header_with_history(site_header, history_url)
    if not commits:
        write_history_index(
            history_dir=history_dir,
            title=title,
            entries=entries_for_index,
            site_header=site_header_with_history(site_header, None),
            base_pandoc_args=base_pandoc_args,
            current_url=current_url,
        )
        return
    for commit in commits:
        snapshot_filename = f"{commit.hexsha}.html"
        snapshot_path = history_dir / snapshot_filename
        skip_commit = False
        if force_history or not snapshot_path.exists():
            template_source = read_file_from_commit(repo, commit.hexsha, relative_source)
            if template_source is None:
                skip_commit = True
            else:
                template_obj = JINJA_ENV.from_string(template_source)
                rendered_markdown = template_obj.render(site_header=site_header_with_link, **render_kwargs)
                banner = history_banner_block(commit, current_url)
                rendered_markdown = prepend_history_banner(rendered_markdown, banner)
                pagetitle = f"{title} (revision {commit.hexsha[:HISTORY_HASH_PREFIX]})"
                html_output = markdown_to_html(
                    rendered_markdown,
                    base_pandoc_args,
                    metadata={"title": "", "pagetitle": pagetitle},
                )
                snapshot_path.write_text(html_output, encoding="utf-8")
        if skip_commit:
            continue
        insertions, deletions, lines = commit_stats_for_path(repo, commit.hexsha, relative_source)
        entries_for_index.append(
            {
                "hash": commit.hexsha,
                "hash_short": commit.hexsha[:HISTORY_HASH_PREFIX],
                "timestamp": format_datetime(commit.authored_datetime),
                "message": commit.summary,
                "insertions": insertions,
                "deletions": deletions,
                "lines": lines,
                "snapshot_name": snapshot_filename,
            }
        )
    write_history_index(
        history_dir=history_dir,
        title=title,
        entries=entries_for_index,
        site_header=site_header_with_history(site_header, None),
        base_pandoc_args=base_pandoc_args,
        current_url=current_url,
    )


def _render_entry_history(
    *,
    entry: Post,
    repo: Repo,
    template,
    site_header: str,
    base_pandoc_args: list[str],
    output_directory: Path,
    source_directory: Path,
    base_slug: str,
    force_history: bool,
) -> None:
    history_dir = output_directory / entry.name / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    relative_source = repo_relative_path(source_directory / f"{entry.name}.md", REPO_ROOT)
    history_url = f"/{base_slug}/{entry.name}/history/"
    current_url = f"/{base_slug}/{entry.name}.html"
    entries_for_index: list[dict[str, Any]] = []
    if not entry.commits:
        write_history_index(
            history_dir=history_dir,
            title=entry.title or entry.name,
            entries=entries_for_index,
            site_header=site_header_with_history(site_header, None),
            base_pandoc_args=base_pandoc_args,
            current_url=current_url,
        )
        return
    site_header_with_link = site_header_with_history(site_header, history_url)
    for commit in entry.commits:
        snapshot_filename = f"{commit.hexsha}.html"
        snapshot_path = history_dir / snapshot_filename
        skip_commit = False
        if force_history or not snapshot_path.exists():
            result = read_post_from_commit(repo, commit.hexsha, relative_source)
            if result is None:
                skip_commit = True
            else:
                frontmatter, markdown_body = result
                historic_post = Post(
                    name=entry.name,
                    title=str(frontmatter.get("title") or entry.title or entry.name),
                    tags=list(frontmatter.get("tags") or entry.tags),
                    location=str(frontmatter.get("location") or entry.location),
                    content=markdown_body,
                    word_count=count_words(markdown_body),
                    created_at=entry.created_at,
                    updated_at=commit.authored_datetime,
                    commits=[],
                )
                context = build_post_context(historic_post, include_body=True, base_slug=base_slug)
                rendered_markdown = template.render(site_header=site_header_with_link, post=context)
                banner = history_banner_block(commit, current_url)
                rendered_markdown = prepend_history_banner(rendered_markdown, banner)
                pagetitle = f"{context['title']} (revision {commit.hexsha[:HISTORY_HASH_PREFIX]})"
                html_output = markdown_to_html(
                    rendered_markdown,
                    base_pandoc_args,
                    metadata={"title": "", "pagetitle": pagetitle},
                )
                snapshot_path.write_text(html_output, encoding="utf-8")
        if skip_commit:
            continue
        insertions, deletions, lines = commit_stats_for_path(repo, commit.hexsha, relative_source)
        entries_for_index.append(
            {
                "hash": commit.hexsha,
                "hash_short": commit.hexsha[:HISTORY_HASH_PREFIX],
                "timestamp": format_datetime(commit.authored_datetime),
                "message": commit.summary,
                "insertions": insertions,
                "deletions": deletions,
                "lines": lines,
                "snapshot_name": snapshot_filename,
            }
        )
    write_history_index(
        history_dir=history_dir,
        title=entry.title or entry.name,
        entries=entries_for_index,
        site_header=site_header_with_history(site_header, None),
        base_pandoc_args=base_pandoc_args,
        current_url=current_url,
    )
