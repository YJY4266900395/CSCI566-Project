#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from .ann import load_manifest, query_ann_index, resolve_query_embedding, write_query_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Query ANN index with a single embedding.")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--embedding-json", default="")
    parser.add_argument("--embedding-file", default="")
    parser.add_argument("--embedding-parquet", default="")
    parser.add_argument("--embedding-index-id", type=int, default=-1)
    parser.add_argument("--embedding-col", default="", help="Override embedding column (default: from manifest).")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--ef-search", type=int, default=200)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    manifest = load_manifest(args.index_dir)
    embedding_col = args.embedding_col or str(manifest.get("embedding_col", "embedding"))

    embedding = resolve_query_embedding(
        embedding_json=args.embedding_json,
        embedding_file=args.embedding_file,
        embedding_parquet=args.embedding_parquet,
        embedding_index_id=args.embedding_index_id,
        embedding_col=embedding_col,
    )

    results = query_ann_index(
        args.index_dir,
        embedding,
        topk=args.topk,
        ef_search=args.ef_search,
        manifest=manifest,
    )

    for r in results:
        print(json.dumps(r, ensure_ascii=True))

    write_query_results(results, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
