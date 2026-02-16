from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Protocol

import numpy as np

log = logging.getLogger(__name__)


class EmbeddingWriter(Protocol):
    def init_if_needed(self, *, embedding_dim: int, has_ids: bool) -> None: ...

    def write_batch(
        self,
        *,
        video_titles: List[str],
        embeddings: np.ndarray,
        video_ids: Optional[List[object]] = None,
    ) -> None: ...

    def close(self) -> None: ...


@dataclass
class ParquetWriterConfig:
    output_path: str


class ParquetEmbeddingWriter:
    def __init__(self, cfg: ParquetWriterConfig):
        self._cfg = cfg
        self._writer = None
        self._schema = None
        self._dim: Optional[int] = None
        self._has_ids: Optional[bool] = None

    def init_if_needed(self, *, embedding_dim: int, has_ids: bool) -> None:
        if self._writer is not None:
            return

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Parquet output requires 'pyarrow'. Install it or use --output ...npy instead."
            ) from e

        self._dim = int(embedding_dim)
        self._has_ids = bool(has_ids)

        fields = []
        if has_ids:
            fields.append(pa.field("video_id", pa.string()))
        fields.append(pa.field("video_title", pa.string()))
        # Use a fixed-size list for consistent vector length.
        fields.append(pa.field("embedding", pa.list_(pa.float32(), list_size=self._dim), nullable=False))
        self._schema = pa.schema(fields)

        os.makedirs(os.path.dirname(self._cfg.output_path) or ".", exist_ok=True)
        self._writer = pq.ParquetWriter(self._cfg.output_path, self._schema)

    def write_batch(
        self,
        *,
        video_titles: List[str],
        embeddings: np.ndarray,
        video_ids: Optional[List[object]] = None,
    ) -> None:
        if self._writer is None or self._schema is None or self._dim is None or self._has_ids is None:
            raise RuntimeError("Writer not initialized. Call init_if_needed() first.")

        if embeddings.ndim != 2 or embeddings.shape[1] != self._dim:
            raise ValueError(f"Unexpected embeddings shape {embeddings.shape}, expected (*, {self._dim})")

        # Parquet has no native fixed-size vector type; store as list[float32].
        # We build the list column without converting per-row to Python lists (keeps overhead reasonable).
        import pyarrow as pa

        emb = np.asarray(embeddings, dtype=np.float32)
        flat = pa.array(emb.reshape(-1), type=pa.float32())
        emb_list = pa.FixedSizeListArray.from_arrays(flat, self._dim)

        cols = {}
        if self._has_ids:
            # Keep stable mapping; store ids as strings to avoid mixed int/str parquet types.
            if video_ids is None:
                raise ValueError("Expected video_ids but got None")
            cols["video_id"] = pa.array([None if v is None else str(v) for v in video_ids], type=pa.string())
        cols["video_title"] = pa.array([None if v is None else str(v) for v in video_titles], type=pa.string())
        cols["embedding"] = emb_list

        table = pa.Table.from_pydict(cols, schema=self._schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


@dataclass
class NpyWriterConfig:
    output_path: str
    total_rows: int


class NpyEmbeddingWriter:
    def __init__(self, cfg: NpyWriterConfig):
        self._cfg = cfg
        self._mm = None
        self._offset = 0
        self._dim: Optional[int] = None

    def init_if_needed(self, *, embedding_dim: int, has_ids: bool) -> None:
        if has_ids:
            log.warning("Output is .npy; video_id/video_title are not stored (row order matches input).")
        if self._mm is not None:
            return

        self._dim = int(embedding_dim)
        os.makedirs(os.path.dirname(self._cfg.output_path) or ".", exist_ok=True)

        # Pre-allocate to allow streaming writes.
        self._mm = np.lib.format.open_memmap(
            self._cfg.output_path,
            mode="w+",
            dtype=np.float32,
            shape=(int(self._cfg.total_rows), self._dim),
        )

    def write_batch(
        self,
        *,
        video_titles: List[str],
        embeddings: np.ndarray,
        video_ids: Optional[List[object]] = None,
    ) -> None:
        if self._mm is None or self._dim is None:
            raise RuntimeError("Writer not initialized. Call init_if_needed() first.")

        n = int(embeddings.shape[0])
        end = self._offset + n
        if end > self._mm.shape[0]:
            raise ValueError("Attempted to write past allocated .npy size (row count mismatch).")

        self._mm[self._offset : end] = embeddings.astype(np.float32, copy=False)
        self._offset = end

    def close(self) -> None:
        if self._mm is not None:
            # Flush memmap to disk.
            self._mm.flush()
            self._mm = None


def make_writer(output_path: str, *, total_rows: int) -> EmbeddingWriter:
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".parquet":
        return ParquetEmbeddingWriter(ParquetWriterConfig(output_path=output_path))
    if ext == ".npy":
        return NpyEmbeddingWriter(NpyWriterConfig(output_path=output_path, total_rows=total_rows))
    raise ValueError("Unsupported output extension. Use .parquet or .npy.")


def save_embeddings(
    output_path: str,
    batches: Iterator[Dict[str, object]],
    *,
    total_rows: int,
    embedding_dim: int,
) -> None:
    """
    Convenience wrapper that writes an iterator of batches to disk.

    Each batch must be a dict containing:
      - video_title: list[str]
      - embedding: np.ndarray
      - video_id: optional list[object]
    """
    writer = make_writer(output_path, total_rows=total_rows)
    writer_initialized = False
    try:
        for batch in batches:
            titles = batch["video_title"]
            embeddings = batch["embedding"]
            ids = batch.get("video_id")

            if not writer_initialized:
                writer.init_if_needed(embedding_dim=embedding_dim, has_ids=ids is not None)
                writer_initialized = True
            writer.write_batch(video_titles=titles, embeddings=embeddings, video_ids=ids)  # type: ignore[arg-type]
        if not writer_initialized:
            writer.init_if_needed(embedding_dim=embedding_dim, has_ids=False)
    finally:
        writer.close()
