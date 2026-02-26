#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path


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
    return "" if value is None else str(value).strip()


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-video category combo TSV (no header) from interaction_filtered.csv and categories_cn_en.csv."
    )
    parser.add_argument(
        "--interaction-file",
        type=Path,
        default=Path("data/interaction_filtered.csv"),
        help="Input interaction CSV path (default: data/interaction_filtered.csv).",
    )
    parser.add_argument(
        "--categories-file",
        type=Path,
        default=Path("data/categories_cn_en.csv"),
        help="Input category dictionary CSV path (default: data/categories_cn_en.csv).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/video_category_combo_cn.tsv"),
        help="Output TSV path without header (default: data/video_category_combo_cn.tsv).",
    )
    parser.add_argument(
        "--name-col",
        default="category_name_cn",
        choices=["category_name_cn", "category_name_en"],
        help="Category name column used in combo text (default: category_name_cn).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1_000_000,
        help="Log progress every N interaction rows (default: 1000000).",
    )
    return parser.parse_args()


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


def _load_video_category_candidates(
    path: Path,
    *,
    log_every: int,
) -> tuple[set[str], dict[str, VideoCategoryCandidate], int]:
    all_video_ids: set[str] = set()
    best_by_video: dict[str, VideoCategoryCandidate] = {}
    skipped_rows = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\r\n")
        if not header:
            return all_video_ids, best_by_video, skipped_rows

        columns = [c.strip() for c in header.split(",")]
        required = ["pid", "category_id", "category_level", "parent_id", "root_id"]
        missing = [c for c in required if c not in columns]
        if missing:
            raise ValueError(f"interaction csv missing required columns: {missing}")

        idx_pid = columns.index("pid")
        idx_category_id = columns.index("category_id")
        idx_category_level = columns.index("category_level")
        idx_parent_id = columns.index("parent_id")
        idx_root_id = columns.index("root_id")
        max_idx = max(idx_pid, idx_category_id, idx_category_level, idx_parent_id, idx_root_id)

        for row_no, raw_line in enumerate(f, start=2):
            parts = raw_line.rstrip("\r\n").split(",")
            if len(parts) <= max_idx:
                skipped_rows += 1
                continue

            video_id = _normalize(parts[idx_pid])
            if not video_id or not video_id.isdigit():
                skipped_rows += 1
                continue
            all_video_ids.add(video_id)

            category_id = _normalize(parts[idx_category_id])
            parent_id = _normalize(parts[idx_parent_id])
            root_id = _normalize(parts[idx_root_id])
            level = _to_int(_normalize(parts[idx_category_level]))

            if not category_id or not parent_id or not root_id:
                skipped_rows += 1
                continue

            incoming = VideoCategoryCandidate(
                level=level,
                root_id=root_id,
                parent_id=parent_id,
                category_id=category_id,
            )
            best_by_video[video_id] = _pick_best_level(best_by_video.get(video_id), incoming)

            if log_every > 0 and row_no % log_every == 0:
                logging.getLogger("generate_video_category_combo").info(
                    "processed_rows=%d unique_video_ids=%d",
                    row_no - 1,
                    len(all_video_ids),
                )

    return all_video_ids, best_by_video, skipped_rows


def _select_name(node: CategoryNode | None, *, name_col: str, fallback_id: str) -> str:
    if node is None:
        return fallback_id
    if name_col == "category_name_en":
        return node.category_name_en or fallback_id
    return node.category_name_cn or fallback_id


def _build_combo(candidate: VideoCategoryCandidate, *, nodes: dict[str, CategoryNode], name_col: str) -> tuple[str, str]:
    inferred_level = nodes.get(candidate.category_id).level if candidate.category_id in nodes else 0
    level = max(candidate.level, inferred_level)

    if level >= 3:
        ids = [candidate.root_id, candidate.parent_id, candidate.category_id]
    elif level == 2:
        ids = [candidate.root_id, candidate.category_id]
    else:
        ids = [candidate.root_id]

    # Remove adjacent duplicates while preserving order.
    path_ids: list[str] = []
    for cid in ids:
        if cid and (not path_ids or path_ids[-1] != cid):
            path_ids.append(cid)

    names = [_select_name(nodes.get(cid), name_col=name_col, fallback_id=cid) for cid in path_ids]
    return "_".join(path_ids), " > ".join(names)


def main() -> int:
    args = _parse_args()
    interaction_path = args.interaction_file.expanduser()
    categories_path = args.categories_file.expanduser()
    output_path = args.output_file.expanduser()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("generate_video_category_combo")

    if not interaction_path.is_file():
        raise SystemExit(f"ERROR: interaction file not found: {interaction_path}")
    if not categories_path.is_file():
        raise SystemExit(f"ERROR: categories file not found: {categories_path}")

    nodes = _load_category_nodes(categories_path)
    all_video_ids, best_by_video, skipped_rows = _load_video_category_candidates(
        interaction_path,
        log_every=int(args.log_every),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_candidate = 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")

        for video_id in sorted(all_video_ids, key=int):
            candidate = best_by_video.get(video_id)
            if candidate is None:
                missing_candidate += 1
                writer.writerow([video_id, "", ""])
                written += 1
                continue

            combo_id, combo_name = _build_combo(candidate, nodes=nodes, name_col=args.name_col)
            writer.writerow([video_id, combo_id, combo_name])
            written += 1

    log.info(
        "done unique_video_ids=%d written=%d missing_candidate=%d skipped_rows=%d output=%s",
        len(all_video_ids),
        written,
        missing_candidate,
        skipped_rows,
        output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
