from __future__ import annotations

from typing import Iterable

import logging
import pypandoc

from .constants import (
    ARROW_FILTER_PATH,
    H2_ANCHOR_FILTER_PATH,
    POST_HEADER_INCLUDE_PATH,
    SHORTEN_LINKS_FILTER_PATH,
)


LOGGER = logging.getLogger(__name__)


def markdown_to_html(
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
    except OSError as exc:  # pandoc missing
        raise RuntimeError("pandoc is required to render posts; install it and retry") from exc


def build_pandoc_base_args(
    extra_args: Iterable[str] | None = None,
    *,
    include_arrow_filter: bool = False,
) -> list[str]:
    args: list[str] = []
    if POST_HEADER_INCLUDE_PATH.exists():
        args.append(f"--include-in-header={POST_HEADER_INCLUDE_PATH}")
    if H2_ANCHOR_FILTER_PATH.exists():
        args.append(f"--lua-filter={H2_ANCHOR_FILTER_PATH}")
    if SHORTEN_LINKS_FILTER_PATH.exists():
        args.append(f"--lua-filter={SHORTEN_LINKS_FILTER_PATH}")
    if include_arrow_filter and ARROW_FILTER_PATH.exists():
        args.append(f"--lua-filter={ARROW_FILTER_PATH}")
    args.append("--section-divs")
    if extra_args:
        args.extend(extra_args)
    return args
