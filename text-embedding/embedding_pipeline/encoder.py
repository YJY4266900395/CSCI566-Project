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


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return x / denom


def _encode_non_empty(
    model: SentenceTransformer,
    texts: List[str],
    *,
    batch_size: int,
    embedding_dim: int,
) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    if emb.ndim == 1:
        emb = emb.reshape(1, -1)

    source_dim = int(emb.shape[1])
    if source_dim < embedding_dim:
        raise RuntimeError(
            f"Model returned dimension={source_dim}, smaller than requested embedding_dim={embedding_dim}."
        )

    # If a smaller target dim is requested, truncate and re-normalize.
    if source_dim > embedding_dim:
        emb = emb[:, :embedding_dim]
        emb = _l2_normalize_rows(emb)

    return emb


def encode_titles(
    model: SentenceTransformer,
    texts: List[str],
    *,
    batch_size: int,
    embedding_dim: Optional[int] = None,
    show_progress_bar: bool = False,
    progress_desc: Optional[str] = None,
) -> np.ndarray:
    """
    Encode texts to dense embeddings (float32), L2-normalized.

    Empty/null texts produce all-zero vectors (not normalized) to avoid NaNs.
    When show_progress_bar=True, display rows progress (rows/s + ETA).
    """
    dim = int(embedding_dim) if embedding_dim is not None else int(model.get_sentence_embedding_dimension())
    total_rows = len(texts)
    if total_rows == 0:
        return np.zeros((0, dim), dtype=np.float32)

    mask = _non_empty_mask(texts)
    non_empty_texts = [t for t, keep in zip(texts, mask) if keep]
    if not non_empty_texts:
        return np.zeros((total_rows, dim), dtype=np.float32)

    if show_progress_bar:
        from tqdm import tqdm

        chunk_rows = max(int(batch_size) * 4, 512)
        emb_non_empty = np.zeros((len(non_empty_texts), dim), dtype=np.float32)

        with tqdm(
            total=len(non_empty_texts),
            desc=progress_desc or "Encoding",
            unit="rows",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            dynamic_ncols=True,
        ) as pbar:
            cursor = 0
            for start in range(0, len(non_empty_texts), chunk_rows):
                chunk_texts = non_empty_texts[start : start + chunk_rows]
                emb_chunk = _encode_non_empty(
                    model,
                    chunk_texts,
                    batch_size=batch_size,
                    embedding_dim=dim,
                )
                n_chunk = int(emb_chunk.shape[0])
                emb_non_empty[cursor : cursor + n_chunk] = emb_chunk
                cursor += n_chunk
                pbar.update(n_chunk)
    else:
        # Fast path: one encode call avoids heavy per-call overhead on large models.
        emb_non_empty = _encode_non_empty(
            model,
            non_empty_texts,
            batch_size=batch_size,
            embedding_dim=dim,
        )

    out = np.zeros((total_rows, dim), dtype=np.float32)
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
    show_progress_bar: bool = False,
    progress_desc: Optional[str] = None,
) -> Tuple[np.ndarray, SentenceTransformer]:
    """
    Encode with a safety fallback: if running on MPS and we hit a runtime error,
    move the model to CPU and retry once.
    """
    try:
        return (
            encode_titles(
                model,
                texts,
                batch_size=batch_size,
                embedding_dim=embedding_dim,
                show_progress_bar=show_progress_bar,
                progress_desc=progress_desc,
            ),
            model,
        )
    except RuntimeError as e:
        device_str = str(getattr(model, "device", ""))
        if "mps" not in device_str.lower():
            raise

        log.warning("MPS runtime error; falling back to CPU for retry. error=%r", e)
        model = move_model_to_cpu(model)
        return (
            encode_titles(
                model,
                texts,
                batch_size=batch_size,
                embedding_dim=embedding_dim,
                show_progress_bar=show_progress_bar,
                progress_desc=progress_desc,
            ),
            model,
        )
