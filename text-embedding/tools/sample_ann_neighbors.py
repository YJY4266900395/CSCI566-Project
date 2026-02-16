#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import hnswlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_VIDEO_URL_PREFIX = "https://fi.ee.tsinghua.edu.cn/datasets/short-video-dataset/raw_file"


@dataclass(frozen=True)
class IndexManifest:
    space: str
    dimensions: int
    embedding_col: str
    input_embeddings: str


def _load_manifest(index_dir: str) -> IndexManifest:
    path = os.path.join(index_dir, "manifest.json")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return IndexManifest(
        space=str(d["space"]),
        dimensions=int(d["dimensions"]),
        embedding_col=str(d.get("embedding_col", "embedding")),
        input_embeddings=str(d.get("input", "")),
    )


def _load_meta(index_dir: str) -> Tuple[List[str], Optional[List[str]]]:
    path = os.path.join(index_dir, "metadata.parquet")
    table = pq.read_table(path)

    if "video_id" not in table.column_names:
        raise ValueError(f"metadata.parquet missing 'video_id' column: {path}")
    ids = [table["video_id"][i].as_py() for i in range(table.num_rows)]
    ids = ["" if v is None else str(v) for v in ids]

    titles = None
    if "video_title" in table.column_names:
        titles = [table["video_title"][i].as_py() for i in range(table.num_rows)]
        titles = ["" if v is None else str(v).strip() for v in titles]

    return ids, titles


def _video_url(video_id: str, prefix: str) -> str:
    return f"{prefix}/{video_id}.mp4"


def _parquet_rowgroup_offsets(pf: pq.ParquetFile) -> List[int]:
    offsets = [0]
    s = 0
    for rg in range(pf.num_row_groups):
        s += pf.metadata.row_group(rg).num_rows
        offsets.append(s)
    return offsets


def _rowgroup_for_index(offsets: List[int], idx: int) -> int:
    lo, hi = 0, len(offsets) - 2
    while lo <= hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= idx < offsets[mid + 1]:
            return mid
        if idx < offsets[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    raise IndexError(f"row index out of range: {idx}")


def _load_embeddings_for_indices(
    embeddings_parquet: str,
    indices: Sequence[int],
    *,
    embedding_col: str,
    dim: int,
) -> Dict[int, np.ndarray]:
    if not indices:
        return {}

    pf = pq.ParquetFile(embeddings_parquet)
    offsets = _parquet_rowgroup_offsets(pf)

    by_rg: Dict[int, List[int]] = {}
    for idx in indices:
        rg = _rowgroup_for_index(offsets, idx)
        by_rg.setdefault(rg, []).append(idx)

    out: Dict[int, np.ndarray] = {}
    for rg, abs_indices in by_rg.items():
        rg_start = offsets[rg]
        rel = [i - rg_start for i in abs_indices]

        table = pf.read_row_group(rg, columns=[embedding_col])
        take_idx = pa.array(rel, type=pa.int64())
        sub = table.take(take_idx)
        col = sub[embedding_col]

        for abs_i, arr in zip(abs_indices, col):
            vec = np.asarray(arr.as_py(), dtype=np.float32)
            if vec.shape != (dim,):
                raise ValueError(f"Bad embedding shape at row={abs_i}: {vec.shape} (expected ({dim},))")
            out[abs_i] = vec

    return out


def _fmt_item(video_id: str, title: str) -> str:
    if title:
        return f"{video_id}  ({title})"
    return video_id


def main() -> int:
    p = argparse.ArgumentParser(
        description="Randomly sample n items and print their top-k nearest neighbors from an ANN index."
    )
    p.add_argument("--index-dir", required=True, help="ANN index dir (contains index.bin, metadata.parquet, manifest.json)")
    p.add_argument(
        "--embeddings-parquet",
        default="",
        help="Embeddings parquet used to build the index. If omitted, uses manifest.json 'input'.",
    )
    p.add_argument(
        "--mode",
        default="video",
        choices=["video", "text"],
        help="Output style: 'video' (video_id + URL) or 'text' (video_id + video_title).",
    )
    p.add_argument(
        "--video-url-prefix",
        default=DEFAULT_VIDEO_URL_PREFIX,
        help="URL prefix for --mode video.",
    )
    p.add_argument("--n", type=int, default=5, help="Number of query items to sample (default: 5).")
    p.add_argument("--k", type=int, default=10, help="Top-k neighbors to print per query (default: 10).")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    p.add_argument("--ef-search", type=int, default=200, help="HNSW ef_search (default: 200).")
    p.add_argument("--include-self", action="store_true", help="Include the query item itself in results.")
    args = p.parse_args()

    manifest = _load_manifest(args.index_dir)
    embeddings_parquet = args.embeddings_parquet or manifest.input_embeddings
    if not embeddings_parquet:
        raise ValueError("--embeddings-parquet not provided and manifest.json has no 'input' field.")
    if not os.path.exists(embeddings_parquet):
        raise FileNotFoundError(f"Embeddings parquet not found: {embeddings_parquet}")

    ids, titles = _load_meta(args.index_dir)
    if args.mode == "text" and titles is None:
        raise ValueError("metadata.parquet has no 'video_title' column; --mode text requires video_title.")

    n_items = len(ids)
    if n_items == 0:
        raise ValueError("metadata.parquet is empty.")

    idxs = [i for i, vid in enumerate(ids) if str(vid).strip() != ""]
    if not idxs:
        raise ValueError("No usable video_id rows in metadata.parquet.")

    rng = np.random.default_rng(args.seed)
    n = min(int(args.n), len(idxs))
    picked = rng.choice(np.array(idxs, dtype=np.int64), size=n, replace=False).tolist()

    index = hnswlib.Index(space=manifest.space, dim=manifest.dimensions)
    index.load_index(os.path.join(args.index_dir, "index.bin"))
    index.set_ef(int(args.ef_search))

    vecs = _load_embeddings_for_indices(
        embeddings_parquet,
        picked,
        embedding_col=manifest.embedding_col,
        dim=manifest.dimensions,
    )

    for qi in picked:
        qid = ids[qi]
        qtitle = titles[qi] if titles is not None else ""

        if args.mode == "video":
            qurl = _video_url(qid, args.video_url_prefix)
            header = f"Query: {qid}  {qurl}"
            if qtitle:
                header += f"\n  Title: {qtitle}"
            print(header)
        else:
            print(f"Query: {_fmt_item(qid, qtitle)}")

        qvec = vecs.get(qi)
        if qvec is None:
            print("  ERROR: missing embedding vector for this query row\n")
            continue

        want = int(args.k) + 1
        labels, distances = index.knn_query(qvec, k=min(want, n_items))
        labels = labels[0].tolist()
        distances = distances[0].tolist()

        shown = 0
        for lbl, dist in zip(labels, distances):
            if not args.include_self and int(lbl) == int(qi):
                continue

            vid = ids[int(lbl)]
            title = titles[int(lbl)] if titles is not None else ""
            if manifest.space == "cosine":
                score = 1.0 - float(dist)
            else:
                score = -float(dist)

            if args.mode == "video":
                url = _video_url(vid, args.video_url_prefix)
                line = f"  {shown+1:02d}. {vid}  {url}  score={score:.6f}"
                if title:
                    line += f"\n      {title}"
                print(line)
            else:
                print(f"  {shown+1:02d}. {_fmt_item(vid, title)}  score={score:.6f}")

            shown += 1
            if shown >= int(args.k):
                break
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
