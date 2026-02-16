from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    # Stable L2 normalization; keep all-zero rows as all-zero.
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return x / denom


class OpenAIEmbedder:
    """
    OpenAI embeddings backend (e.g. text-embedding-3-small).

    Network calls happen here; keep this backend optional.
    """

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 1024,
        request_batch_size: int = 128,
        max_retries: int = 5,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: object = None,
        normalize: bool = True,
    ) -> None:
        if client is not None:
            # Used for testing or advanced customization.
            self._client = client
        else:
            try:
                from openai import OpenAI
            except Exception as e:  # pragma: no cover
                raise RuntimeError("OpenAI backend requires `openai` package. Install it to use --backend openai.") from e

            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
        self.model = str(model)
        self.dimensions = int(dimensions)
        self.request_batch_size = int(request_batch_size)
        self.max_retries = int(max_retries)
        self.normalize = bool(normalize)

    def get_sentence_embedding_dimension(self) -> int:
        return int(self.dimensions)

    def encode(self, texts: List[str], *, batch_size: int) -> np.ndarray:
        """
        Encode a batch of texts into a float32 matrix of shape (N, dimensions).
        Empty/whitespace texts return all-zero vectors.

        `batch_size` here controls request sub-batching (per API call).
        """
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        # Do not modify content; only treat pure-whitespace as empty for safety.
        mask = [t is not None and str(t).strip() != "" for t in texts]
        non_empty = [t for t, keep in zip(texts, mask) if keep]

        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        if not non_empty:
            return out

        emb_non_empty = self._embed_non_empty(non_empty, request_batch_size=batch_size)
        if emb_non_empty.shape != (len(non_empty), self.dimensions):
            raise RuntimeError(f"OpenAI returned unexpected shape {emb_non_empty.shape}")

        j = 0
        for i, keep in enumerate(mask):
            if keep:
                out[i] = emb_non_empty[j]
                j += 1

        if self.normalize:
            out = _l2_normalize(out)
        return out

    def _embed_non_empty(self, texts: List[str], *, request_batch_size: int) -> np.ndarray:
        # Imports inside to avoid requiring openai at module import time.
        from openai import APIError, APITimeoutError, RateLimitError

        all_embs: List[List[float]] = []
        for i in range(0, len(texts), request_batch_size):
            batch = texts[i : i + request_batch_size]

            retries = 0
            while True:
                try:
                    resp = self._client.embeddings.create(
                        model=self.model,
                        input=batch,
                        dimensions=self.dimensions,
                    )
                    all_embs.extend([d.embedding for d in resp.data])
                    break
                except (RateLimitError, APIError, APITimeoutError) as exc:
                    retries += 1
                    if retries > self.max_retries:
                        raise
                    sleep_s = min(2**retries, 60)
                    log.warning("OpenAI embed retry %d/%d after error=%r; sleep=%ss", retries, self.max_retries, exc, sleep_s)
                    time.sleep(sleep_s)

        arr = np.asarray(all_embs, dtype=np.float32)
        if arr.ndim != 2:
            raise RuntimeError(f"OpenAI returned invalid embeddings array ndim={arr.ndim}")
        return arr
