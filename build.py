from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

from builder import render_pages_to_html, render_posts_to_html, render_notes_to_html
from builder.content import load_notes, load_posts
from builder.models import build_post_context


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the personal website")
    parser.add_argument(
        "--rebuild-history",
        action="store_true",
        help="Regenerate all cached history snapshots",
    )
    args = parser.parse_args(argv)

    posts = load_posts()
    notes = load_notes()
    post_listings = [
        build_post_context(post, base_slug="posts")
        for post in sorted(posts, key=lambda p: p.updated_at, reverse=True)
    ]
    note_listings = [
        build_post_context(note, base_slug="notes")
        for note in sorted(notes, key=lambda n: n.updated_at, reverse=True)
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                render_pages_to_html,
                force_history=args.rebuild_history,
                listing_posts=post_listings,
                listing_notes=note_listings,
            ),
            executor.submit(
                render_posts_to_html,
                force_history=args.rebuild_history,
                posts=posts,
            ),
            executor.submit(
                render_notes_to_html,
                force_history=args.rebuild_history,
                notes=notes,
            ),
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
