#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

# interaction_filtered.csv columns (fixed order)
_TAG_COL_INDEX = 18
_TITLE_COL_INDEX = 19
_TAIL_COL_COUNT = 8
_MIN_COLS = 28
_WS_RE = re.compile(r"\s+")


@dataclass
class VideoTextRecord:
    title: str = ""
    tags: list[str] = field(default_factory=list)
    tag_set: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class CategoryNode:
    level: int
    category_name_cn: str
    category_name_en: str


@dataclass(frozen=True)
class VideoCategoryCandidate:
    level: int
    root_id: str
    parent_id: str
    category_id: str


def _normalize(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    text = text.strip().strip('"')
    return _WS_RE.sub(" ", text).strip()


def _id_sort_key(video_id: str) -> tuple[int, object]:
    s = str(video_id)
    if s.isdigit():
        return (0, int(s))
    return (1, s)


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build multifield TSV inputs in one pass over interaction_filtered.csv: "
            "video_tag_title.tsv and video_category_combo_cn.tsv."
        )
    )
    p.add_argument(
        "--interaction-file",
        type=Path,
        default=Path("data/interaction_filtered.csv"),
        help="Input interaction CSV path (default: data/interaction_filtered.csv).",
    )
    p.add_argument(
        "--categories-file",
        type=Path,
        default=Path("data/categories_cn_en.csv"),
        help="Input category dictionary CSV path (default: data/categories_cn_en.csv).",
    )
    p.add_argument(
        "--video-tag-title-output",
        type=Path,
        default=Path("data/video_tag_title.tsv"),
        help="Output TSV path for video/tag/title (no header).",
    )
    p.add_argument(
        "--video-category-output",
        type=Path,
        default=Path("data/video_category_combo_cn.tsv"),
        help="Output TSV path for video/category combo (no header).",
    )
    p.add_argument(
        "--category-name-col",
        default="category_name_cn",
        choices=["category_name_cn", "category_name_en"],
        help="Category name column used in combo text (default: category_name_cn).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only read first N data rows for quick testing (0 means all).",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=1_000_000,
        help="Log progress every N rows (default: 1000000).",
    )
    return p.parse_args()


def _load_category_nodes(path: Path) -> dict[str, CategoryNode]:
    nodes: dict[str, CategoryNode] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"category_level", "category_id", "category_name_cn", "category_name_en"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"categories csv missing required columns: {sorted(missing)}")

        for row in reader:
            category_id = _normalize(row.get("category_id"))
            if not category_id or category_id in nodes:
                continue
            nodes[category_id] = CategoryNode(
                level=_to_int(_normalize(row.get("category_level"))),
                category_name_cn=_normalize(row.get("category_name_cn")),
                category_name_en=_normalize(row.get("category_name_en")),
            )

    return nodes


def _pick_best_level(current: VideoCategoryCandidate | None, incoming: VideoCategoryCandidate) -> VideoCategoryCandidate:
    if current is None:
        return incoming
    if incoming.level > current.level:
        return incoming
    return current


def _build_combo(candidate: VideoCategoryCandidate, *, nodes: dict[str, CategoryNode], name_col: str) -> tuple[str, str]:
    inferred_level = nodes.get(candidate.category_id).level if candidate.category_id in nodes else 0
    level = max(candidate.level, inferred_level)

    if level >= 3:
        ids = [candidate.root_id, candidate.parent_id, candidate.category_id]
    elif level == 2:
        ids = [candidate.root_id, candidate.category_id]
    else:
        ids = [candidate.root_id]

    path_ids: list[str] = []
    for cid in ids:
        if cid and (not path_ids or path_ids[-1] != cid):
            path_ids.append(cid)

    names: list[str] = []
    for cid in path_ids:
        node = nodes.get(cid)
        if node is None:
            names.append(cid)
        elif name_col == "category_name_en":
            names.append(node.category_name_en or cid)
        else:
            names.append(node.category_name_cn or cid)

    return "_".join(path_ids), " > ".join(names)


def _write_video_tag_title(path: Path, records: dict[str, VideoTextRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for video_id in sorted(records, key=_id_sort_key):
            record = records[video_id]
            writer.writerow([video_id, "#".join(record.tags), record.title])


def _write_video_category(
    path: Path,
    *,
    all_video_ids: set[str],
    candidates: dict[str, VideoCategoryCandidate],
    nodes: dict[str, CategoryNode],
    name_col: str,
) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_candidate = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for video_id in sorted(all_video_ids, key=_id_sort_key):
            candidate = candidates.get(video_id)
            if candidate is None:
                writer.writerow([video_id, "", ""])
                missing_candidate += 1
                written += 1
                continue
            combo_id, combo_name = _build_combo(candidate, nodes=nodes, name_col=name_col)
            writer.writerow([video_id, combo_id, combo_name])
            written += 1

    return written, missing_candidate


def main() -> int:
    args = _parse_args()

    interaction_path = args.interaction_file.expanduser()
    categories_path = args.categories_file.expanduser()
    tag_title_output = args.video_tag_title_output.expanduser()
    video_category_output = args.video_category_output.expanduser()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("build_multifield_tsvs")

    if not interaction_path.is_file():
        raise SystemExit(f"ERROR: interaction file not found: {interaction_path}")
    if not categories_path.is_file():
        raise SystemExit(f"ERROR: categories file not found: {categories_path}")

    nodes = _load_category_nodes(categories_path)

    video_text_records: dict[str, VideoTextRecord] = {}
    all_video_ids: set[str] = set()
    best_category_by_video: dict[str, VideoCategoryCandidate] = {}

    input_rows = 0
    skipped_rows = 0

    with interaction_path.open("r", encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\r\n")
        if not header:
            raise SystemExit(f"ERROR: empty input file: {interaction_path}")

        columns = [c.strip() for c in header.split(",")]
        required = ["pid", "category_id", "category_level", "parent_id", "root_id", "tag_name", "title"]
        missing = [c for c in required if c not in columns]
        if missing:
            raise SystemExit(f"ERROR: interaction csv missing required columns: {missing}")

        idx_pid = columns.index("pid")
        idx_category_id = columns.index("category_id")
        idx_category_level = columns.index("category_level")
        idx_parent_id = columns.index("parent_id")
        idx_root_id = columns.index("root_id")
        max_required_idx = max(idx_pid, idx_category_id, idx_category_level, idx_parent_id, idx_root_id)

        for raw_line in f:
            if args.limit > 0 and input_rows >= int(args.limit):
                break
            input_rows += 1

            parts = raw_line.rstrip("\r\n").split(",")
            if len(parts) < _MIN_COLS or len(parts) <= max_required_idx:
                skipped_rows += 1
                continue

            video_id = _normalize(parts[idx_pid])
            if not video_id or not video_id.isdigit():
                skipped_rows += 1
                continue
            all_video_ids.add(video_id)

            # title/tag extraction from robust fixed-position slicing.
            cutoff = len(parts) - _TAIL_COL_COUNT
            if cutoff <= _TAG_COL_INDEX:
                skipped_rows += 1
                continue
            tag_name = _normalize(parts[_TAG_COL_INDEX])
            title = _normalize(",".join(parts[_TITLE_COL_INDEX:cutoff]))

            text_record = video_text_records.get(video_id)
            if text_record is None:
                text_record = VideoTextRecord(title=title)
                video_text_records[video_id] = text_record
            elif title and not text_record.title:
                text_record.title = title

            if tag_name and tag_name not in text_record.tag_set:
                text_record.tag_set.add(tag_name)
                text_record.tags.append(tag_name)

            category_id = _normalize(parts[idx_category_id])
            parent_id = _normalize(parts[idx_parent_id])
            root_id = _normalize(parts[idx_root_id])
            level = _to_int(_normalize(parts[idx_category_level]))
            if category_id and parent_id and root_id:
                incoming = VideoCategoryCandidate(
                    level=level,
                    root_id=root_id,
                    parent_id=parent_id,
                    category_id=category_id,
                )
                best_category_by_video[video_id] = _pick_best_level(best_category_by_video.get(video_id), incoming)

            if args.log_every > 0 and input_rows % int(args.log_every) == 0:
                log.info(
                    "processed_rows=%d unique_video_ids=%d",
                    input_rows,
                    len(all_video_ids),
                )

    _write_video_tag_title(tag_title_output, video_text_records)
    written_category, missing_category = _write_video_category(
        video_category_output,
        all_video_ids=all_video_ids,
        candidates=best_category_by_video,
        nodes=nodes,
        name_col=args.category_name_col,
    )

    log.info(
        "done rows=%d video_ids=%d skipped_rows=%d tag_title_rows=%d category_rows=%d missing_category=%d",
        input_rows,
        len(all_video_ids),
        skipped_rows,
        len(video_text_records),
        written_category,
        missing_category,
    )
    log.info("outputs: tag_title=%s category=%s", tag_title_output, video_category_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
