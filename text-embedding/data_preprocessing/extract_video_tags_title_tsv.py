#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

# interaction_filtered.csv columns (fixed order)
# user_id,pid,...,hate,tag_name,title,p_hour,...,fre_city_level
_TAG_COL_INDEX = 18
_TITLE_COL_INDEX = 19
_TAIL_COL_COUNT = 8
_MIN_COLS = 28
_WS_RE = re.compile(r"\s+")


@dataclass
class VideoRecord:
    title: str = ""
    tags: list[str] = field(default_factory=list)
    tag_set: set[str] = field(default_factory=set)


def _normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    text = text.strip().strip('"')
    return _WS_RE.sub(" ", text).strip()


def _id_sort_key(video_id: str) -> tuple[int, object]:
    s = str(video_id)
    if s.isdigit():
        return (0, int(s))
    return (1, s)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video_id(pid), merged tag_name, and title from interaction CSV into TSV (no header)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/interaction_filtered.csv"),
        help="Input CSV path with header (default: data/interaction_filtered.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/video_tag_title.tsv"),
        help="Output TSV path without header (default: data/video_tag_title.tsv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only read first N data rows for quick testing (0 means all).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1_000_000,
        help="Log progress every N input rows (default: 1000000).",
    )
    return parser.parse_args()


def _parse_header(header_line: str) -> None:
    cols = [c.strip() for c in header_line.rstrip("\r\n").split(",")]
    if len(cols) < _MIN_COLS:
        raise ValueError(f"Unexpected header columns: {len(cols)} < {_MIN_COLS}")
    required = {"pid", "tag_name", "title"}
    missing = required - set(cols)
    if missing:
        raise ValueError(f"Missing required columns in header: {sorted(missing)}")


def _parse_line_fast(raw_line: str) -> tuple[str, str, str] | None:
    """
    Parse one physical CSV line without multiline quote semantics.
    This avoids runaway parsing when source has broken quotes.
    """
    parts = raw_line.rstrip("\r\n").split(",")
    if len(parts) < _MIN_COLS:
        return None

    # Use fixed positions from both ends:
    # - pid is always column index 1
    # - last 8 columns are fixed tail fields
    # - everything between tag_name and tail belongs to title (may contain commas)
    pid = _normalize_text(parts[1])
    if not pid or not pid.isdigit():
        return None

    cutoff = len(parts) - _TAIL_COL_COUNT
    if cutoff <= _TAG_COL_INDEX:
        return None

    tag_name = _normalize_text(parts[_TAG_COL_INDEX])
    title_parts = parts[_TITLE_COL_INDEX:cutoff]
    title = _normalize_text(",".join(title_parts))

    return pid, tag_name, title


def main() -> int:
    args = _parse_args()
    input_path = args.input.expanduser()
    output_path = args.output.expanduser()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("extract_video_tags_title_tsv")

    if not input_path.is_file():
        raise SystemExit(f"ERROR: input file not found: {input_path}")

    records: dict[str, VideoRecord] = {}
    input_rows = 0
    skipped_rows = 0

    with input_path.open("r", encoding="utf-8", newline="") as f:
        header = f.readline()
        if not header:
            raise SystemExit(f"ERROR: empty input file: {input_path}")
        _parse_header(header)

        for raw_line in f:
            if args.limit > 0 and input_rows >= args.limit:
                break
            input_rows += 1

            parsed = _parse_line_fast(raw_line)
            if parsed is None:
                skipped_rows += 1
                continue

            video_id, tag_name, title = parsed

            record = records.get(video_id)
            if record is None:
                record = VideoRecord(title=title)
                records[video_id] = record
            elif title and not record.title:
                # Keep first non-empty title seen for this video.
                record.title = title

            if tag_name and tag_name not in record.tag_set:
                record.tag_set.add(tag_name)
                record.tags.append(tag_name)

            if args.log_every > 0 and input_rows % args.log_every == 0:
                log.info(
                    "processed_rows=%d unique_video_ids=%d skipped_rows=%d",
                    input_rows,
                    len(records),
                    skipped_rows,
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_rows = 0

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for video_id in sorted(records, key=_id_sort_key):
            record = records[video_id]
            writer.writerow([video_id, "#".join(record.tags), record.title])
            written_rows += 1

    log.info(
        "done input_rows=%d unique_video_ids=%d written_rows=%d skipped_rows=%d output=%s",
        input_rows,
        len(records),
        written_rows,
        skipped_rows,
        output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
