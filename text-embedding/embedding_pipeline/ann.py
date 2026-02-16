from __future__ import annotations

import datetime as dt
import json
import os
from bisect import bisect_right
from typing import Any, Dict, List, Sequence

import hnswlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _infer_dim(col: pa.ChunkedArray) -> int:
    # pyarrow.ChunkedArray.combine_chunks() returns an Array.
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    t = arr.type
    if pa.types.is_fixed_size_list(t):
        return int(t.list_size)

    # Fallback: read the first row as python list.
    first = arr[0].as_py()
    if not isinstance(first, list):
        raise ValueError(f"Unsupported embedding type: {t}")
    return int(len(first))


def _to_numpy_2d(col: pa.ChunkedArray, *, dim: int) -> np.ndarray:
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    t = arr.type

    if pa.types.is_fixed_size_list(t):
        # FixedSizeListArray exposes underlying flat values.
        flat = arr.values.to_numpy(zero_copy_only=False)
        out = flat.reshape(-1, dim)
        return np.asarray(out, dtype=np.float32)

    # Slow fallback for variable-length list arrays.
    py = [arr[i].as_py() for i in range(len(arr))]
    out = np.asarray(py, dtype=np.float32)
    if out.ndim != 2 or out.shape[1] != dim:
        raise ValueError(f"Embeddings must be 2D with dim={dim}, got {out.shape}")
    return out


def build_ann_index_from_parquet(
    input_path: str,
    output_dir: str,
    *,
    embedding_col: str = "embedding",
    space: str = "cosine",
    m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 200,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    pf = pq.ParquetFile(input_path)
    n = int(pf.metadata.num_rows)
    if n <= 0:
        raise ValueError("Embeddings parquet is empty.")

    first = pf.read_row_group(0, columns=[embedding_col])
    if embedding_col not in first.column_names:
        raise ValueError(f"Missing embedding column: {embedding_col}")
    dim = _infer_dim(first[embedding_col])

    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=m)

    meta_writer = None
    meta_path = os.path.join(output_dir, "metadata.parquet")

    offset = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        if embedding_col not in table.column_names:
            raise ValueError(f"Missing embedding column: {embedding_col}")

        emb = _to_numpy_2d(table[embedding_col], dim=dim)
        labels = np.arange(offset, offset + emb.shape[0], dtype=np.int64)
        index.add_items(emb, labels)
        offset += int(emb.shape[0])

        meta = table.drop([embedding_col])
        if meta_writer is None:
            meta_writer = pq.ParquetWriter(meta_path, meta.schema)
        meta_writer.write_table(meta)

    if meta_writer is not None:
        meta_writer.close()

    if offset != n:
        raise RuntimeError(f"Row count mismatch while building index: expected {n}, wrote {offset}")

    index.set_ef(ef_search)
    index_path = os.path.join(output_dir, "index.bin")
    index.save_index(index_path)

    manifest = {
        "input": os.path.abspath(input_path),
        "output_dir": os.path.abspath(output_dir),
        "embedding_col": embedding_col,
        "space": space,
        "dimensions": dim,
        "count": n,
        "M": m,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        "created_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "index_file": "index.bin",
        "metadata_file": "metadata.parquet",
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    return manifest


def load_manifest(index_dir: str) -> Dict[str, Any]:
    path = os.path.join(index_dir, "manifest.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parquet_rowgroup_offsets(pf: pq.ParquetFile) -> List[int]:
    offsets = [0]
    s = 0
    for rg in range(pf.num_row_groups):
        s += int(pf.metadata.row_group(rg).num_rows)
        offsets.append(s)
    return offsets


def _rowgroup_for_index(offsets: Sequence[int], index_id: int) -> int:
    return bisect_right(offsets, index_id) - 1


def _load_metadata_rows(
    index_dir: str,
    index_ids: Sequence[int],
    *,
    columns: Sequence[str],
) -> Dict[int, Dict[str, Any]]:
    if not index_ids:
        return {}

    path = os.path.join(index_dir, "metadata.parquet")
    pf = pq.ParquetFile(path)
    total_rows = int(pf.metadata.num_rows)

    for idx in index_ids:
        if idx < 0 or idx >= total_rows:
            raise ValueError(f"metadata index out of range: {idx}")

    available = set(pf.schema_arrow.names)
    selected_cols = [c for c in columns if c in available]
    if not selected_cols:
        return {int(idx): {} for idx in index_ids}

    offsets = _parquet_rowgroup_offsets(pf)
    by_rg: Dict[int, List[int]] = {}
    for idx in index_ids:
        rg = _rowgroup_for_index(offsets, int(idx))
        by_rg.setdefault(rg, []).append(int(idx))

    out: Dict[int, Dict[str, Any]] = {}
    for rg, abs_indices in by_rg.items():
        rg_start = offsets[rg]
        rel = [i - rg_start for i in abs_indices]

        table = pf.read_row_group(rg, columns=selected_cols)
        sub = table.take(pa.array(rel, type=pa.int64()))

        for pos, abs_idx in enumerate(abs_indices):
            row: Dict[str, Any] = {}
            for col in selected_cols:
                row[col] = sub[col][pos].as_py()
            out[abs_idx] = row

    return out


def load_embedding_from_parquet(path: str, index_id: int, *, embedding_col: str = "embedding") -> List[float]:
    pf = pq.ParquetFile(path)
    total_rows = int(pf.metadata.num_rows)
    if index_id < 0 or index_id >= total_rows:
        raise ValueError("index_id out of range")
    if embedding_col not in pf.schema_arrow.names:
        raise ValueError(f"Missing embedding column: {embedding_col}")

    offsets = _parquet_rowgroup_offsets(pf)
    rg = _rowgroup_for_index(offsets, int(index_id))
    rel = int(index_id - offsets[rg])

    table = pf.read_row_group(rg, columns=[embedding_col])
    vec = table[embedding_col][rel].as_py()
    return [float(x) for x in vec]


def load_embedding_from_file(path: str) -> List[float]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        return arr.astype(float).tolist()
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError("Empty embedding file")
    return json.loads(content)


def resolve_query_embedding(
    *,
    embedding_json: str,
    embedding_file: str,
    embedding_parquet: str,
    embedding_index_id: int,
    embedding_col: str,
) -> List[float]:
    if not embedding_json and not embedding_file and not embedding_parquet:
        raise ValueError("Provide --embedding-json, --embedding-file, or --embedding-parquet")
    if embedding_parquet and embedding_index_id < 0:
        raise ValueError("--embedding-index-id is required when using --embedding-parquet")

    if embedding_json:
        return json.loads(embedding_json)
    if embedding_parquet:
        return load_embedding_from_parquet(
            embedding_parquet,
            embedding_index_id,
            embedding_col=embedding_col,
        )
    return load_embedding_from_file(embedding_file)


def query_ann_index(
    index_dir: str,
    embedding: Sequence[float],
    *,
    topk: int = 5,
    ef_search: int = 200,
    manifest: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    manifest_data = manifest or load_manifest(index_dir)
    dim = int(manifest_data["dimensions"])
    space = str(manifest_data["space"])

    vec = np.array(embedding, dtype=np.float32)
    if vec.ndim != 1 or int(vec.shape[0]) != dim:
        raise ValueError(f"Embedding dimension mismatch: expected {dim}, got {vec.shape}")

    index = hnswlib.Index(space=space, dim=dim)
    index.load_index(os.path.join(index_dir, "index.bin"))
    index.set_ef(ef_search)

    labels, distances = index.knn_query(vec, k=topk)
    labels = labels[0].tolist()
    distances = distances[0].tolist()

    meta_rows = _load_metadata_rows(index_dir, labels, columns=["video_id", "video_title"])

    results: List[Dict[str, Any]] = []
    for rank, (idx, dist) in enumerate(zip(labels, distances), start=1):
        row = meta_rows.get(int(idx), {})
        if space == "cosine":
            score = 1.0 - float(dist)
        else:
            score = -float(dist)
        results.append(
            {
                "rank": rank,
                "index_id": int(idx),
                "score": score,
                "video_id": str(row.get("video_id", "")),
                "video_title": str(row.get("video_title", "")),
            }
        )

    return results


def write_query_results(results: Sequence[Dict[str, Any]], output_path: str) -> None:
    if not output_path:
        return
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
