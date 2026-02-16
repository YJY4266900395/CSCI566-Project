from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .model import move_model_to_cpu

log = logging.getLogger(__name__)


def _non_empty_mask(texts: List[str]) -> List[bool]:
    mask: List[bool] = []
    for t in texts:
        if t is None:
            mask.append(False)
            continue
        # Do not modify content; only treat pure-whitespace as empty for safety.
        mask.append(str(t).strip() != "")
    return mask


def encode_titles(
    model: SentenceTransformer,
    texts: List[str],
    *,
    batch_size: int,
    embedding_dim: Optional[int] = None,
) -> np.ndarray:
    """
    Encode titles to dense embeddings (float32), L2-normalized.

    Empty/null titles produce all-zero vectors (not normalized) to avoid NaNs.
    """
    dim = int(embedding_dim) if embedding_dim is not None else int(model.get_sentence_embedding_dimension())
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)

    mask = _non_empty_mask(texts)
    non_empty_texts = [t for t, keep in zip(texts, mask) if keep]

    if not non_empty_texts:
        return np.zeros((len(texts), dim), dtype=np.float32)

    emb_non_empty = model.encode(
        non_empty_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    out = np.zeros((len(texts), dim), dtype=np.float32)

    j = 0
    for i, keep in enumerate(mask):
        if keep:
            out[i] = emb_non_empty[j]
            j += 1

    return out


def encode_titles_with_fallback(
    model: SentenceTransformer,
    texts: List[str],
    *,
    batch_size: int,
    embedding_dim: Optional[int] = None,
) -> Tuple[np.ndarray, SentenceTransformer]:
    """
    Encode with a safety fallback: if running on MPS and we hit a runtime error,
    move the model to CPU and retry once.
    """
    try:
        return (
            encode_titles(model, texts, batch_size=batch_size, embedding_dim=embedding_dim),
            model,
        )
    except RuntimeError as e:
        device_str = str(getattr(model, "device", ""))
        if "mps" not in device_str.lower():
            raise

        log.warning("MPS runtime error; falling back to CPU for retry. error=%r", e)
        model = move_model_to_cpu(model)
        return (
            encode_titles(model, texts, batch_size=batch_size, embedding_dim=embedding_dim),
            model,
        )
