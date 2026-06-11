from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

from builder import render_pages_to_html, render_posts_to_html, render_notes_to_html
from builder.content import load_notes, load_posts
from builder.models import build_post_context
from builder.photos import (
    build_photo_page_context,
    ensure_local_rclone_config,
    prepare_photo_gallery,
    sync_photos_to_r2,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the personal website")
    parser.add_argument(
        "--rebuild-history",
        action="store_true",
        help="Regenerate all cached history snapshots",
    )
    parser.add_argument(
        "--skip-photo-sync",
        action="store_true",
        help="Render and process photographs without uploading them to R2",
    )
    args = parser.parse_args(argv)

    photos = prepare_photo_gallery()
    photo_page_context = build_photo_page_context(photos)
    r2_config = None
    if not args.skip_photo_sync:
        r2_config = ensure_local_rclone_config()

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
                page_context_overrides={"photographs": photo_page_context},
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

    if r2_config is not None:
        sync_photos_to_r2(r2_config)


if __name__ == "__main__":
    main()
