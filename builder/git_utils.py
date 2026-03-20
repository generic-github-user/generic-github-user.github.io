from __future__ import annotations

import logging
from pathlib import Path

from git import Commit, Repo
from git.exc import GitCommandError

from .models import CommitInfo, parse_frontmatter


LOGGER = logging.getLogger(__name__)


def collect_commits(repo: Repo, post_path: Path) -> list[CommitInfo]:
    commits: list[CommitInfo] = []
    rel_path = str(post_path.relative_to(Path(repo.working_tree_dir)))
    try:
        log_output = repo.git.log("--follow", "--format=%H", "--", rel_path)
    except GitCommandError:
        LOGGER.warning("Failed to read git log for %s", rel_path)
        log_output = ""
    for line in log_output.splitlines():
        hexsha = line.strip()
        if not hexsha:
            continue
        commit = repo.commit(hexsha)
        commits.append(_to_commit_info(commit))
    LOGGER.debug("Collected %d commits for %s", len(commits), post_path)
    return commits


def repo_relative_path(path: Path, repo_root: Path) -> str:
    return str(path.relative_to(repo_root)).replace("\\", "/")


def commit_stats_for_path(repo: Repo, commit_hex: str, relative_path: str) -> tuple[int, int, int]:
    commit = repo.commit(commit_hex)
    stats = commit.stats.files.get(relative_path) or {}
    insertions = int(stats.get("insertions", 0))
    deletions = int(stats.get("deletions", 0))
    lines = int(stats.get("lines", insertions + deletions))
    return insertions, deletions, lines


def read_post_from_commit(repo: Repo, commit_hex: str, relative_path: str):
    spec = f"{commit_hex}:{relative_path}"
    try:
        raw = repo.git.show(spec)
    except GitCommandError:
        LOGGER.warning("Commit %s is missing %s; skipping history render", commit_hex, relative_path)
        return None
    try:
        return parse_frontmatter(raw, source=spec)
    except ValueError:
        LOGGER.warning("Skipping render for %s because it lacks front matter", spec)
        return None


def read_file_from_commit(repo: Repo, commit_hex: str, relative_path: str) -> str | None:
    spec = f"{commit_hex}:{relative_path}"
    try:
        return repo.git.show(spec)
    except GitCommandError:
        LOGGER.warning("Commit %s is missing %s; skipping history render", commit_hex, relative_path)
        return None


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
