from __future__ import annotations

import argparse

from builder import render_pages_to_html, render_posts_to_html, render_notes_to_html


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the personal website")
    parser.add_argument(
        "--rebuild-history",
        action="store_true",
        help="Regenerate all cached history snapshots",
    )
    args = parser.parse_args(argv)

    render_pages_to_html(force_history=args.rebuild_history)
    render_posts_to_html(force_history=args.rebuild_history)
    render_notes_to_html(force_history=args.rebuild_history)


if __name__ == "__main__":
    main()
