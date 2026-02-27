from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embedding_generator.modeling import encode_texts
from embedding_generator.io_utils import make_writer


@dataclass
class DummyModel:
    dim: int = 8
    device: str = "cpu"

    def get_sentence_embedding_dimension(self) -> int:
        return int(self.dim)

    def encode(
        self,
        texts: List[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        # Deterministic embeddings so this test is stable.
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(len(texts), self.dim)).astype(np.float32)
        if normalize_embeddings and len(texts) > 0:
            denom = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / denom
        return emb


def _assert_close(a: float, b: float, *, eps: float = 1e-4) -> None:
    if abs(a - b) > eps:
        raise AssertionError(f"Expected {a} ~= {b} (eps={eps})")


def main() -> int:
    # 1) Encoder behavior: empty titles -> zero vectors; non-empty -> ~unit norm.
    model = DummyModel(dim=8)
    texts = ["你好", "", "   ", None, "世界"]
    emb = encode_texts(model, texts, batch_size=4, embedding_dim=model.get_sentence_embedding_dimension())
    assert emb.shape == (len(texts), model.dim)
    assert np.allclose(emb[1], 0.0)
    assert np.allclose(emb[2], 0.0)
    assert np.allclose(emb[3], 0.0)
    _assert_close(float(np.linalg.norm(emb[0])), 1.0)
    _assert_close(float(np.linalg.norm(emb[4])), 1.0)

    # 2) I/O smoke test for writers using in-memory samples (keeps repo clean).
    with tempfile.TemporaryDirectory(prefix="bge_m3_smoke_") as td:
        samples = [
            ("1", "冬天穿搭推荐"),
            ("2", "#搞笑 今日份快乐"),
            ("3", "重庆火锅真的太香了！"),
            ("4", ""),
            ("5", "  "),
            ("6", "萌宠日常 🐶"),
        ]
        ids = [sid for sid, _ in samples]
        titles = [txt for _, txt in samples]
        total_rows = len(samples)

        out_parquet = os.path.join(td, "smoke.parquet")
        out_npy = os.path.join(td, "smoke.npy")

        for out_path in [out_parquet, out_npy]:
            writer = make_writer(out_path, total_rows=total_rows)
            emb = encode_texts(model, titles, batch_size=4, embedding_dim=model.get_sentence_embedding_dimension())
            writer.init_if_needed(embedding_dim=model.get_sentence_embedding_dimension(), has_ids=True)
            writer.write_batch(video_titles=titles, embeddings=emb, video_ids=ids)
            writer.close()
            assert os.path.exists(out_path)

        # 3) Validate Parquet output shape and schema lightly.
        import pyarrow.parquet as pq

        table = pq.read_table(out_parquet)
        assert table.num_rows == total_rows
        assert "video_title" in table.column_names
        assert "embedding" in table.column_names
        first = table["embedding"][0].as_py()
        assert isinstance(first, list) and len(first) == model.dim

        # 4) Validate NPY output shape.
        mm = np.load(out_npy, mmap_mode="r")
        assert mm.shape == (total_rows, model.dim)

    print("smoke_test_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
