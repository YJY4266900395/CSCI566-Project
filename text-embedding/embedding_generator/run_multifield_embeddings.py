#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .modeling import encode_texts_with_fallback
from .io_utils import make_writer
from .modeling import build_model, resolve_device
from .io_utils import default_embedding_output_path


WS_RE = re.compile(r"\s+")


def _normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return WS_RE.sub(" ", text).strip()


def _id_sort_key(video_id: str) -> tuple[int, object]:
    s = str(video_id)
    if s.isdigit():
        return (0, int(s))
    return (1, s)


def _resolve_output_path(
    *,
    output: str,
    output_root: str,
    model_name: str,
    embedding_dim: int,
    dataset_name: str,
    extension: str,
    fallback_input_path: str,
) -> str:
    if output:
        return output
    return default_embedding_output_path(
        output_root=output_root,
        model_name=model_name,
        input_path=fallback_input_path,
        embedding_dim=embedding_dim,
        dataset_name=dataset_name,
        extension=extension,
    )


def _iter_tsv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            row = line.rstrip("\r\n")
            if not row:
                continue
            yield row.split("\t")


def _load_asr_texts(path: Path) -> dict[str, str]:
    """Load headerless 2-column TSV: video_id<TAB>asr_text."""
    out: dict[str, str] = {}
    for parts in _iter_tsv_rows(path):
        if len(parts) < 2:
            continue
        video_id = _normalize_text(parts[0])
        text = _normalize_text("\t".join(parts[1:]))
        if not video_id:
            continue
        if video_id.lower() in {"video_id", "pid"}:
            continue
        if video_id not in out:
            out[video_id] = text
    return out


def _load_title_tag_texts(path: Path) -> dict[str, str]:
    """
    Load TSV with expected columns:
    - headerless: video_id<TAB>tag_name<TAB>title

    Output text is composed as: "title [SEP] tag_name" when both exist.
    """
    out: dict[str, str] = {}
    for parts in _iter_tsv_rows(path):
        if len(parts) < 3:
            continue
        video_id = _normalize_text(parts[0])
        if not video_id:
            continue
        if video_id.lower() in {"video_id", "pid"}:
            continue

        tag_name = _normalize_text(parts[1])
        title = _normalize_text("\t".join(parts[2:]))
        if title and tag_name:
            text = f"{title} [SEP] {tag_name}"
        else:
            text = title or tag_name

        if video_id not in out:
            out[video_id] = text
        elif not out[video_id] and text:
            out[video_id] = text
    return out


def _load_category_texts(path: Path) -> dict[str, str]:
    """
    Load category TSV with supported shapes:
    - video_id<TAB>category_text
    - video_id<TAB>category_combo_id<TAB>category_combo_name

    If 3 columns are present, category_combo_name is used as text.
    """
    out: dict[str, str] = {}
    for parts in _iter_tsv_rows(path):
        if len(parts) < 2:
            continue

        video_id = _normalize_text(parts[0])
        if not video_id:
            continue
        if video_id.lower() in {"video_id", "pid"}:
            continue

        if len(parts) >= 3:
            # Prefer human-readable combo name.
            text = _normalize_text("\t".join(parts[2:]))
            if not text:
                text = _normalize_text(parts[1])
        else:
            text = _normalize_text(parts[1])

        if video_id not in out:
            out[video_id] = text
        elif not out[video_id] and text:
            out[video_id] = text
    return out


def _resolve_target_dim(model_default_dim: int, requested_dim: int) -> int:
    if requested_dim <= 0:
        raise ValueError("--dimensions must be > 0")
    return min(int(requested_dim), int(model_default_dim))


def _write_concat_parquet(
    *,
    output_path: str,
    video_ids: list[str],
    concat_embeddings: np.ndarray,
    has_asr: np.ndarray,
    has_title_tag: np.ndarray,
    has_category: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dim = int(concat_embeddings.shape[1])
    flat = pa.array(np.asarray(concat_embeddings, dtype=np.float32).reshape(-1), type=pa.float32())
    emb_col = pa.FixedSizeListArray.from_arrays(flat, dim)

    table = pa.Table.from_pydict(
        {
            "video_id": pa.array(video_ids, type=pa.string()),
            "has_asr": pa.array(has_asr.astype(np.uint8), type=pa.uint8()),
            "has_title_tag": pa.array(has_title_tag.astype(np.uint8), type=pa.uint8()),
            "has_category": pa.array(has_category.astype(np.uint8), type=pa.uint8()),
            "embedding": emb_col,
        }
    )
    pq.write_table(table, output_path)


def _write_field_embeddings(
    *,
    output_path: str,
    video_ids: list[str],
    texts: list[str],
    embeddings: np.ndarray,
) -> None:
    writer = make_writer(output_path, total_rows=len(video_ids))
    try:
        writer.init_if_needed(embedding_dim=int(embeddings.shape[1]), has_ids=True)
        writer.write_batch(video_titles=texts, embeddings=embeddings, video_ids=video_ids)
    finally:
        writer.close()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build per-field 256-dim embeddings (asr/title+tag/category), then concat to 768-dim."
    )
    p.add_argument("--asr-input", default="data/asr_cn.tsv", help="ASR TSV path: video_id<TAB>asr_text")
    p.add_argument(
        "--title-tag-input",
        default="data/video_tag_title.tsv",
        help="Title/tag TSV path: video_id<TAB>tag_name<TAB>title (headerless).",
    )
    p.add_argument(
        "--category-input",
        default="data/video_category_combo_cn.tsv",
        help=(
            "Category TSV path: "
            "video_id<TAB>category_text or video_id<TAB>category_combo_id<TAB>category_combo_name."
        ),
    )
    p.add_argument("--model", required=True, help="Local model repo id in org/repo format.")
    p.add_argument(
        "--dimensions",
        type=int,
        default=256,
        help="Per-field embedding dimensions (default: 256).",
    )
    p.add_argument("--batch-size", type=int, default=128, help="Encoding batch size (default: 128).")
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=0,
        help=(
            "Optional max sequence length for tokenizer/model. "
            "0 keeps model default (for bge-m3 on MPS, 512/1024 is recommended)."
        ),
    )
    p.add_argument(
        "--device",
        default="mps",
        choices=["auto", "mps", "cpu"],
        help="Device: mps/auto/cpu (default: mps).",
    )
    p.add_argument(
        "--output",
        default="",
        help="Output parquet path for concat embeddings. If omitted, auto path is used.",
    )
    p.add_argument(
        "--output-root",
        default="output/models",
        help="Root output directory for auto output path (default: output/models).",
    )
    p.add_argument(
        "--dataset-name",
        default="video_multifield_concat",
        help="Dataset slug used for auto output names.",
    )
    p.add_argument(
        "--save-field-embeddings",
        action="store_true",
        help="Also save 256-dim parquet files for asr/title_tag/category fields.",
    )
    p.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Only process first N video_ids after sorting (0 means all).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)
    t0 = time.time()

    asr_path = Path(args.asr_input).expanduser()
    title_tag_path = Path(args.title_tag_input).expanduser()
    category_path = Path(args.category_input).expanduser()

    if not asr_path.is_file():
        raise SystemExit(f"ERROR: --asr-input not found: {asr_path}")
    if not title_tag_path.is_file():
        raise SystemExit(f"ERROR: --title-tag-input not found: {title_tag_path}")
    if not category_path.is_file():
        raise SystemExit(f"ERROR: --category-input not found: {category_path}")

    asr_map = _load_asr_texts(asr_path)
    title_tag_map = _load_title_tag_texts(title_tag_path)
    category_map = _load_category_texts(category_path)

    video_ids = sorted(set(asr_map) | set(title_tag_map) | set(category_map), key=_id_sort_key)
    if args.max_videos > 0:
        video_ids = video_ids[: int(args.max_videos)]
    if not video_ids:
        raise SystemExit("ERROR: no video ids found across input files.")

    log.info(
        "Loaded rows: asr=%d title_tag=%d category=%d video_union=%d",
        len(asr_map),
        len(title_tag_map),
        len(category_map),
        len(video_ids),
    )

    device = resolve_device(args.device)
    log.info("Device selected: %s (requested=%s)", device, args.device)

    model = build_model(args.model, device=device)

    if int(args.max_seq_length) > 0:
        requested_seq_len = int(args.max_seq_length)
        current_seq_len = int(getattr(model, "max_seq_length", 0) or 0)
        target_seq_len = requested_seq_len if current_seq_len <= 0 else min(current_seq_len, requested_seq_len)
        model.max_seq_length = int(target_seq_len)
        log.info(
            "Model max_seq_length set to %d (model default=%d requested=%d)",
            int(model.max_seq_length),
            current_seq_len,
            requested_seq_len,
        )

    model_default_dim = int(model.get_sentence_embedding_dimension())
    embed_dim = _resolve_target_dim(model_default_dim, int(args.dimensions))
    if embed_dim < int(args.dimensions):
        log.warning(
            "Requested dimensions=%d exceeds model default=%d; using %d.",
            int(args.dimensions),
            model_default_dim,
            embed_dim,
        )
    else:
        log.info("Using per-field dimensions=%d", embed_dim)

    n = len(video_ids)
    concat = np.zeros((n, embed_dim * 3), dtype=np.float32)

    asr_texts = [asr_map.get(vid, "") for vid in video_ids]
    title_tag_texts = [title_tag_map.get(vid, "") for vid in video_ids]
    category_texts = [category_map.get(vid, "") for vid in video_ids]

    has_asr = np.fromiter((1 if t else 0 for t in asr_texts), dtype=np.uint8, count=n)
    has_title_tag = np.fromiter((1 if t else 0 for t in title_tag_texts), dtype=np.uint8, count=n)
    has_category = np.fromiter((1 if t else 0 for t in category_texts), dtype=np.uint8, count=n)

    fields = [
        ("asr", asr_texts, 0),
        ("title_tag", title_tag_texts, 1),
        ("category", category_texts, 2),
    ]

    field_outputs: dict[str, str] = {}
    for field_name, texts, field_idx in fields:
        log.info("Encoding field=%s rows=%d", field_name, n)
        emb, model = encode_texts_with_fallback(
            model,
            texts,
            batch_size=int(args.batch_size),
            embedding_dim=embed_dim,
            show_progress_bar=True,
            progress_desc=f"Encoding[{field_name}]",
        )

        start = field_idx * embed_dim
        end = (field_idx + 1) * embed_dim
        concat[:, start:end] = emb

        if args.save_field_embeddings:
            field_output = _resolve_output_path(
                output="",
                output_root=args.output_root,
                model_name=args.model,
                embedding_dim=embed_dim,
                dataset_name=f"{args.dataset_name}_{field_name}",
                extension="parquet",
                fallback_input_path=str(asr_path),
            )
            _write_field_embeddings(
                output_path=field_output,
                video_ids=video_ids,
                texts=texts,
                embeddings=emb,
            )
            field_outputs[field_name] = field_output

        log.info("Encoded field=%s rows=%d non_empty=%d", field_name, n, int(sum(1 for t in texts if t)))

    output_path = _resolve_output_path(
        output=args.output,
        output_root=args.output_root,
        model_name=args.model,
        embedding_dim=embed_dim * 3,
        dataset_name=args.dataset_name,
        extension="parquet",
        fallback_input_path=str(asr_path),
    )
    _write_concat_parquet(
        output_path=output_path,
        video_ids=video_ids,
        concat_embeddings=concat,
        has_asr=has_asr,
        has_title_tag=has_title_tag,
        has_category=has_category,
    )

    dt = time.time() - t0
    log.info("Done. rows=%d dim=%d output=%s seconds=%.2f", n, embed_dim * 3, output_path, dt)
    if field_outputs:
        for name, path in field_outputs.items():
            log.info("Field output (%s): %s", name, path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
