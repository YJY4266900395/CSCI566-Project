from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embedding_pipeline.openai_backend import OpenAIEmbedder


@dataclass
class _FakeDatum:
    embedding: List[float]


@dataclass
class _FakeResp:
    data: List[_FakeDatum]


class _FakeEmbeddingsAPI:
    def __init__(self, dim: int):
        self._dim = dim

    def create(self, *, model: str, input: List[str], dimensions: int):
        assert dimensions == self._dim
        # Deterministic "embedding": [len(text), len(text)+1, ...]
        out = []
        for t in input:
            base = float(len(t))
            out.append(_FakeDatum([base + i for i in range(self._dim)]))
        return _FakeResp(out)


class _FakeClient:
    def __init__(self, dim: int):
        self.embeddings = _FakeEmbeddingsAPI(dim)


def test_openai_backend_shapes_and_empty_handling() -> None:
    dim = 4
    backend = OpenAIEmbedder(
        model="text-embedding-3-small",
        dimensions=dim,
        max_retries=0,
        client=_FakeClient(dim),
        normalize=False,
    )

    texts = ["hi", "", "  ", "world"]
    emb = backend.encode(texts, batch_size=2)
    assert emb.shape == (4, dim)
    assert np.allclose(emb[1], 0.0)
    assert np.allclose(emb[2], 0.0)
    assert not np.allclose(emb[0], 0.0)
    assert not np.allclose(emb[3], 0.0)


def test_openai_backend_normalization() -> None:
    dim = 4
    backend = OpenAIEmbedder(
        model="text-embedding-3-small",
        dimensions=dim,
        max_retries=0,
        client=_FakeClient(dim),
        normalize=True,
    )
    emb = backend.encode(["abc"], batch_size=1)
    n = float(np.linalg.norm(emb[0]))
    assert abs(n - 1.0) < 1e-5

