#!/usr/bin/env python3
from __future__ import annotations

import argparse

from .ann import build_ann_index_from_parquet
from .paths import (
    default_ann_index_output_dir,
    infer_dataset_slug_from_embeddings_path,
    infer_model_slug_from_embeddings_path,
    model_slug,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ANN index (HNSW) from embeddings parquet.")
    parser.add_argument("--input", required=True, help="Path to embeddings parquet")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. If omitted, auto-uses output/models/<model>/ann_index/<dataset>_index.",
    )
    parser.add_argument(
        "--output_root",
        default="output/models",
        help="Root output directory used when --output-dir is omitted (default: output/models).",
    )
    parser.add_argument(
        "--model_name",
        default="",
        help="Optional model name for auto output directory (e.g. BAAI/bge-m3).",
    )
    parser.add_argument(
        "--dataset_name",
        default="",
        help="Optional dataset name for auto output directory.",
    )
    parser.add_argument("--embedding-col", default="embedding")
    parser.add_argument("--space", default="cosine", choices=["cosine", "l2"], help="Distance space")
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=200)
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_name_for_path = args.model_name or infer_model_slug_from_embeddings_path(args.input)
        if model_name_for_path == "unknown-model":
            model_name_for_path = "unknown-model"
        dataset_name = args.dataset_name or infer_dataset_slug_from_embeddings_path(args.input)
        output_dir = default_ann_index_output_dir(
            output_root=args.output_root,
            model_name=model_slug(model_name_for_path),
            embeddings_path=args.input,
            dataset_name=dataset_name,
        )

    build_ann_index_from_parquet(
        input_path=args.input,
        output_dir=output_dir,
        embedding_col=args.embedding_col,
        space=args.space,
        m=args.M,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
