from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("hnswlib")

from embedding_pipeline.ann import (
    build_ann_index_from_parquet,
    load_embedding_from_parquet,
    query_ann_index,
)


def _write_embeddings_parquet(
    path: Path,
    embeddings: np.ndarray,
    *,
    video_titles: list[str],
    video_ids: list[str] | None = None,
) -> None:
    dim = int(embeddings.shape[1])
    flat = pa.array(np.asarray(embeddings, dtype=np.float32).reshape(-1), type=pa.float32())
    emb_col = pa.FixedSizeListArray.from_arrays(flat, dim)

    cols: dict[str, pa.Array] = {
        "video_title": pa.array(video_titles, type=pa.string()),
        "embedding": emb_col,
    }
    if video_ids is not None:
        cols["video_id"] = pa.array(video_ids, type=pa.string())

    pq.write_table(pa.Table.from_pydict(cols), str(path))


def test_load_embedding_from_parquet_reads_single_row(tmp_path: Path) -> None:
    emb_path = tmp_path / "emb.parquet"
    _write_embeddings_parquet(
        emb_path,
        np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32),
        video_titles=["t1", "t2", "t3"],
        video_ids=["v1", "v2", "v3"],
    )

    row = load_embedding_from_parquet(str(emb_path), 1)
    assert row == [0.0, 1.0]

    with pytest.raises(ValueError):
        load_embedding_from_parquet(str(emb_path), 99)


def test_query_ann_index_returns_expected_neighbors(tmp_path: Path) -> None:
    emb_path = tmp_path / "emb.parquet"
    _write_embeddings_parquet(
        emb_path,
        np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32),
        video_titles=["alpha", "beta", "gamma"],
        video_ids=["v1", "v2", "v3"],
    )

    index_dir = tmp_path / "ann"
    build_ann_index_from_parquet(str(emb_path), str(index_dir), ef_search=64)

    results = query_ann_index(str(index_dir), [1.0, 0.0], topk=2, ef_search=64)

    assert len(results) == 2
    assert results[0]["rank"] == 1
    assert results[0]["video_id"] == "v1"
    assert results[0]["video_title"] == "alpha"
    assert results[0]["score"] > 0.99

    assert results[1]["video_id"] == "v3"
    assert results[1]["video_title"] == "gamma"


def test_query_ann_index_handles_missing_video_id_column(tmp_path: Path) -> None:
    emb_path = tmp_path / "emb_no_id.parquet"
    _write_embeddings_parquet(
        emb_path,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        video_titles=["only-title-1", "only-title-2"],
        video_ids=None,
    )

    index_dir = tmp_path / "ann_no_id"
    build_ann_index_from_parquet(str(emb_path), str(index_dir), ef_search=32)

    results = query_ann_index(str(index_dir), [1.0, 0.0], topk=1, ef_search=32)

    assert len(results) == 1
    assert results[0]["video_id"] == ""
    assert results[0]["video_title"] == "only-title-1"
