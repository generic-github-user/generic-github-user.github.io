from __future__ import annotations

import logging
import re
from pathlib import Path
from zoneinfo import ZoneInfo

from jinja2 import Environment


REPO_ROOT = Path(__file__).resolve().parents[1]
POSTS_DIR = REPO_ROOT / "posts"
NOTES_DIR = REPO_ROOT / "notes"
PAGES_DIR = REPO_ROOT / "pages"
FILES_DIR = REPO_ROOT / "files"
RAW_PHOTOS_DIR = REPO_ROOT / "raw_photos"
PHOTOS_DIR = REPO_ROOT / "photos"
PHOTO_OUTPUT_DIR = PHOTOS_DIR / "full_size"
METADATA_PATH = REPO_ROOT / "metadata.yaml"
R2_CREDENTIALS_PATH = REPO_ROOT / "r2.txt"
RCLONE_CONFIG_PATH = REPO_ROOT / "rclone-r2.conf"
HEADER_TEMPLATE_PATH = PAGES_DIR / "header.md"
POST_TEMPLATE_PATH = PAGES_DIR / "post.md"
NOTE_TEMPLATE_PATH = PAGES_DIR / "note.md"
POST_OUTPUT_DIR = REPO_ROOT / "docs" / "posts"
NOTE_OUTPUT_DIR = REPO_ROOT / "docs" / "notes"
POST_HEADER_INCLUDE_PATH = PAGES_DIR / "post_head.html"
H2_ANCHOR_FILTER_PATH = PAGES_DIR / "h2_anchors.lua"
ARROW_FILTER_PATH = PAGES_DIR / "replace_arrows.lua"
SHORTEN_LINKS_FILTER_PATH = PAGES_DIR / "shorten_bare_links.lua"
PAGES_OUTPUT_DIR = REPO_ROOT / "docs"
DISPLAY_TIMEZONE = ZoneInfo("America/New_York")
PHOTO_PUBLIC_BASE_URL = "https://images.anna.engineering/photos"
JINJA_ENV = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
HISTORY_HASH_PREFIX = 12


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
