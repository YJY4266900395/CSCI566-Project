from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np


class ParquetWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.writer = None
        self.schema = None
        self.embedding_dim = 0
        self.has_ids = False

    def init_if_needed(self, *, embedding_dim: int, has_ids: bool) -> None:
        if self.writer is not None:
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        self.embedding_dim = int(embedding_dim)
        self.has_ids = bool(has_ids)

        fields = []
        if self.has_ids:
            fields.append(pa.field("video_id", pa.string()))
        fields.append(pa.field("video_title", pa.string()))
        fields.append(pa.field("embedding", pa.list_(pa.float32(), list_size=self.embedding_dim), nullable=False))
        self.schema = pa.schema(fields)

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        self.writer = pq.ParquetWriter(self.output_path, self.schema)

    def write_batch(
        self,
        *,
        video_titles: list[str],
        embeddings: np.ndarray,
        video_ids: Optional[list[object]] = None,
    ) -> None:
        if self.writer is None or self.schema is None:
            raise RuntimeError("Writer is not initialized")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

        import pyarrow as pa

        emb = np.asarray(embeddings, dtype=np.float32)
        flat = pa.array(emb.reshape(-1), type=pa.float32())
        emb_col = pa.FixedSizeListArray.from_arrays(flat, self.embedding_dim)

        columns = {
            "video_title": pa.array(["" if t is None else str(t) for t in video_titles], type=pa.string()),
            "embedding": emb_col,
        }
        if self.has_ids:
            if video_ids is None:
                raise ValueError("video_ids is required")
            columns = {
                "video_id": pa.array(["" if v is None else str(v) for v in video_ids], type=pa.string()),
                **columns,
            }

        self.writer.write_table(pa.Table.from_pydict(columns, schema=self.schema))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class NpyWriter:
    def __init__(self, output_path: str, total_rows: int):
        self.output_path = output_path
        self.total_rows = int(total_rows)
        self.embedding_dim = 0
        self.mm = None
        self.offset = 0

    def init_if_needed(self, *, embedding_dim: int, has_ids: bool) -> None:
        _ = has_ids
        if self.mm is not None:
            return

        self.embedding_dim = int(embedding_dim)
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        self.mm = np.lib.format.open_memmap(
            self.output_path,
            mode="w+",
            dtype=np.float32,
            shape=(self.total_rows, self.embedding_dim),
        )

    def write_batch(
        self,
        *,
        video_titles: list[str],
        embeddings: np.ndarray,
        video_ids: Optional[list[object]] = None,
    ) -> None:
        _ = video_titles, video_ids
        if self.mm is None:
            raise RuntimeError("Writer is not initialized")

        n = int(embeddings.shape[0])
        end = self.offset + n
        if end > self.mm.shape[0]:
            raise ValueError("Write exceeds allocated npy size")
        self.mm[self.offset : end] = embeddings.astype(np.float32, copy=False)
        self.offset = end

    def close(self) -> None:
        if self.mm is not None:
            self.mm.flush()
            self.mm = None


def make_writer(output_path: str, *, total_rows: int):
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".parquet":
        return ParquetWriter(output_path)
    if ext == ".npy":
        return NpyWriter(output_path, total_rows=total_rows)
    raise ValueError("Unsupported output extension. Use .parquet or .npy")


def _slug(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    s = s.strip("-._")
    return s or "unknown"


def _dimension_slug(embedding_dim: int) -> str:
    dim = int(embedding_dim)
    if dim <= 0:
        raise ValueError(f"embedding_dim must be > 0, got: {embedding_dim}")
    return str(dim)


def model_slug(model_name: str) -> str:
    raw = str(model_name or "").strip().rstrip("/\\")
    if not raw:
        return "unknown-model"
    tail = re.split(r"[\\/]", raw)[-1]
    return _slug(tail) or "unknown-model"


def dataset_slug(input_path: str) -> str:
    p = Path(str(input_path or "")).expanduser()
    if p.is_dir():
        return _slug(p.name or "dataset")
    stem = p.stem or p.name
    return _slug(stem or "dataset")


def default_embedding_output_path(
    *,
    output_root: str,
    model_name: str,
    input_path: str,
    embedding_dim: int,
    extension: str = "parquet",
    dataset_name: str = "",
) -> str:
    ext = str(extension or "parquet").lstrip(".")
    ds = _slug(dataset_name) if dataset_name else dataset_slug(input_path)
    dim = _dimension_slug(embedding_dim)
    return os.path.join(str(output_root), model_slug(model_name), "embeddings", dim, f"{ds}.{ext}")


def infer_model_slug_from_embeddings_path(embeddings_path: str) -> str:
    p = Path(str(embeddings_path or ""))
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part == "models" and i + 1 < len(parts):
            return _slug(parts[i + 1])
    return "unknown-model"


def infer_dataset_slug_from_embeddings_path(embeddings_path: str) -> str:
    p = Path(str(embeddings_path or ""))
    return _slug(p.stem or "dataset")


def default_ann_index_output_dir(
    *,
    output_root: str,
    model_name: str,
    embeddings_path: str,
    dataset_name: str = "",
) -> str:
    ds = _slug(dataset_name) if dataset_name else infer_dataset_slug_from_embeddings_path(embeddings_path)
    return os.path.join(str(output_root), model_slug(model_name), "ann_index", f"{ds}_index")
