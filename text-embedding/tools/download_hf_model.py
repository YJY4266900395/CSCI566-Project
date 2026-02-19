#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot to default HF cache.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--revision", default="main", help="Model revision/branch/tag (default: main)")
    parser.add_argument(
        "--token",
        default="",
        help="Hugging Face token. If omitted, uses HF_TOKEN from environment.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Glob pattern to include (can be passed multiple times).",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Glob pattern to exclude (can be passed multiple times).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files exist locally.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN") or None

    print(f"repo_id: {args.repo_id}")
    print(f"revision: {args.revision}")

    cache_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        token=token,
        allow_patterns=args.allow_pattern or None,
        ignore_patterns=args.ignore_pattern or None,
        force_download=args.force_download,
    )

    print("Download complete.")
    print(f"Cache path: {cache_path}")
    print("Default HF cache root is usually ~/.cache/huggingface/hub (unless HF_HOME/HUGGINGFACE_HUB_CACHE is set).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
