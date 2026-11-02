"""Collect a bounded reply sample; run with python -m entrypoint.collect_x."""

import argparse
import os
from pathlib import Path

from src.ingestion.x_client import XCollectionError, XReplyCollector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("post_url")
    parser.add_argument("--max-comments", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Choose a new filename.")
    try:
        collection = XReplyCollector(os.getenv("X_BEARER_TOKEN", "")).collect(
            args.post_url, max_comments=args.max_comments, max_pages=args.max_pages
        )
    except (ValueError, XCollectionError) as exc:
        parser.exit(1, f"Collection failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(collection.model_dump_json(indent=2))
    print(f"Saved {len(collection.replies)} replies to {args.output}")
    print("Recent search covers 7 days; this is not a complete reply archive.")


if __name__ == "__main__":
    main()
