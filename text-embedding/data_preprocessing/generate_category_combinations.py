#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Level3Path:
    root_id: str
    parent_id: str
    category_id: str


def _require_columns(fieldnames: list[str] | None, *, name_col: str) -> None:
    required = {"category_level", "category_id", "parent_id", "root_id", name_col}
    missing = required - set(fieldnames or [])
    if missing:
        raise ValueError(f"categories csv missing required columns: {sorted(missing)}")


def _load_categories(
    input_file: Path,
    *,
    name_col: str,
) -> tuple[dict[str, str], list[Level3Path], int]:
    """
    Read categories once and collect:
      - category_id -> category_name map (for all levels)
      - valid level-3 ID paths (root -> parent -> child)
      - count of invalid level-3 rows
    """
    name_map: dict[str, str] = {}
    level3_paths: list[Level3Path] = []
    skipped = 0

    with input_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        _require_columns(reader.fieldnames, name_col=name_col)

        for row in reader:
            cid = str(row.get("category_id", "")).strip()
            if not cid:
                continue

            name = str(row.get(name_col, "")).strip()
            if cid not in name_map:
                name_map[cid] = name

            try:
                level = int(str(row.get("category_level", "")).strip() or 0)
            except ValueError:
                continue

            if level != 3:
                continue

            root_id = str(row.get("root_id", "")).strip()
            parent_id = str(row.get("parent_id", "")).strip()
            if not root_id or not parent_id:
                skipped += 1
                continue

            level3_paths.append(Level3Path(root_id=root_id, parent_id=parent_id, category_id=cid))

    return name_map, level3_paths, skipped


def generate_category_combination_files(
    categories_csv: str | Path,
    output_file: str | Path,
    *,
    name_col: str = "category_name_en",
) -> dict[str, int]:
    """
    Generate a single combo table file with two columns:
      combo_id<TAB>combo_name

    Only generates Level-3 combination files:
      - combo_id:   {cat1}_{cat2}_{cat3}
      - combo_name: cat1_name > cat2_name > cat3_name

    Where:
      - cat1 = root_id (level 1)
      - cat2 = parent_id (level 2)
      - cat3 = category_id (level 3)
    """
    input_path = Path(categories_csv).expanduser()
    output_path = Path(output_file).expanduser()
    name_map, level3_paths, skipped = _load_categories(input_path, name_col=name_col)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_names = 0
    seen: set[str] = set()

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for path in level3_paths:
            ids = [path.root_id, path.parent_id, path.category_id]
            combo_id = f"{path.root_id}_{path.parent_id}_{path.category_id}"
            if combo_id in seen:
                continue

            names = [name_map.get(cid, "") for cid in ids]
            # Skip incomplete paths; output rows must contain all three names.
            if any(not n for n in names):
                missing_names += 1
                continue

            seen.add(combo_id)
            writer.writerow([combo_id, " > ".join(names)])
            written += 1

    return {
        "written": written,
        "skipped": skipped,
        "missing_names": missing_names,
        "rows": len(level3_paths),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a single category combo TSV from categories_cn_en.csv."
    )
    parser.add_argument(
        "--input_file",
        "--categories_csv",
        dest="input_file",
        type=Path,
        required=True,
        help="Path to categories_cn_en.csv",
    )
    parser.add_argument(
        "--output_file",
        "--output",
        "--output_dir",
        dest="output_file",
        type=Path,
        required=True,
        help="Output file path; writes 2-column TSV: combo_id<TAB>combo_name",
    )
    parser.add_argument(
        "--name_col",
        default="category_name_en",
        choices=["category_name_en", "category_name_cn"],
        help="Which category name column to use for output text.",
    )
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("generate_category_combinations")

    stats = generate_category_combination_files(
        args.input_file,
        args.output_file,
        name_col=args.name_col,
    )
    log.info(
        "Done. rows=%d written=%d skipped=%d missing_names=%d output_file=%s name_col=%s",
        stats["rows"],
        stats["written"],
        stats["skipped"],
        stats["missing_names"],
        args.output_file,
        args.name_col,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
