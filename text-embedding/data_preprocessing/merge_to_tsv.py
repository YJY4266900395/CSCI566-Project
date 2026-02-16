#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

WS_RE = re.compile(r"\s+")


def default_output_path(input_dir: Path) -> Path:
    return input_dir.parent / f"{input_dir.name}.tsv"


def normalize_text(text: str) -> str:
    # Keep TSV single-line and clean by replacing tabs/newlines with spaces.
    text = text.replace("\t", " ")
    return WS_RE.sub(" ", text).strip()


def list_txt_files(input_dir: Path, limit: int) -> list[Path]:
    files = sorted(
        (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"),
        key=lambda p: int(p.stem),
    )
    return files if limit <= 0 else files[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-video text files into a single TSV (video_id<TAB>text)."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory containing {video_id}.txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output TSV path. Default: <input_dir_name>.tsv in input_dir's parent.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process first N files (0 means all).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir.expanduser()
    output_path: Path = (args.output.expanduser() if args.output else default_output_path(input_dir))
    limit = int(args.limit)

    if not input_dir.is_dir():
        print(f"ERROR: input dir not found: {input_dir}", file=sys.stderr)
        return 1

    txt_files = list_txt_files(input_dir, limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as out:
        for i, txt_file in enumerate(txt_files, start=1):
            video_id = txt_file.stem
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            out.write(f"{video_id}\t{normalize_text(text)}\n")

            if i % 10000 == 0:
                print(f"processed {i}/{len(txt_files)}", file=sys.stderr)

    print(f"wrote {len(txt_files)} rows to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())