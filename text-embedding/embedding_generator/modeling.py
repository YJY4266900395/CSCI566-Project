from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)
_REPO_ID_RE = re.compile(r"^[^/]+/[^/]+$")


def resolve_device(device_arg: str) -> str:
    """Resolve user device choice to torch device string."""
    device_arg = (device_arg or "mps").lower()

    if device_arg == "cpu":
        return "cpu"

    if device_arg in {"mps", "auto"}:
        if torch.backends.mps.is_available():
            return "mps"
        if device_arg == "mps":
            log.warning("MPS requested but unavailable; falling back to CPU.")
        return "cpu"

    raise ValueError(f"Unsupported device '{device_arg}'. Use one of: mps, cpu, auto.")


@contextmanager
def _hf_offline_mode(enabled: bool):
    """Force HF/Transformers offline mode during model loading."""
    if not enabled:
        yield
        return

    keys = ["TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"]
    old = {k: os.environ.get(k) for k in keys}
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _hf_cache_root() -> Path:
    """Resolve HF cache root from env with explicit precedence."""
    hub_cache = str(os.environ.get("HUGGINGFACE_HUB_CACHE", "")).strip()
    if hub_cache:
        return Path(hub_cache).expanduser()

    hf_home = str(os.environ.get("HF_HOME", "")).strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def _download_hint(repo_id: str) -> str:
    return f"Please download it first with: uv run python tools/download_hf_model.py --repo-id {repo_id}"


def _resolve_latest_snapshot_dir(repo_id: str) -> Path:
    org, repo = repo_id.split("/", 1)
    repo_cache = _hf_cache_root() / f"models--{org}--{repo}"
    snapshots_dir = repo_cache / "snapshots"

    if not snapshots_dir.is_dir():
        raise RuntimeError(
            f"Model '{repo_id}' not found in local HF cache: '{repo_cache}'. {_download_hint(repo_id)}"
        )

    refs_main = repo_cache / "refs" / "main"
    if refs_main.is_file():
        snapshot_id = refs_main.read_text(encoding="utf-8").strip()
        if snapshot_id:
            candidate = snapshots_dir / snapshot_id
            if candidate.is_dir():
                return candidate.resolve()

    snapshot_dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshot_dirs:
        raise RuntimeError(
            f"No local snapshots found for '{repo_id}' under '{snapshots_dir}'. {_download_hint(repo_id)}"
        )

    snapshot_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshot_dirs[0].resolve()


def _resolve_local_model_path(model_name: str) -> str:
    """Resolve --model (org/repo) to local HF snapshot directory path."""
    model_name = str(model_name or "").strip()
    if not model_name:
        raise RuntimeError("--model is required. Pass a model repo id in org/repo format.")

    if _REPO_ID_RE.match(model_name):
        return str(_resolve_latest_snapshot_dir(model_name))

    raise RuntimeError(
        f"Invalid --model value: '{model_name}'. Use repo id in 'org/repo' format, e.g. BAAI/bge-m3."
    )


def build_model(model_name: str, *, device: str) -> SentenceTransformer:
    """Load SentenceTransformer in strict local-only mode."""
    resolved_model_path = _resolve_local_model_path(model_name)
    log.info("Loading model: %s -> %s (device=%s local-only=true)", model_name, resolved_model_path, device)

    try:
        with _hf_offline_mode(True):
            try:
                return SentenceTransformer(
                    resolved_model_path,
                    device=device,
                    model_kwargs={"local_files_only": True},
                    tokenizer_kwargs={"local_files_only": True},
                )
            except TypeError:
                try:
                    return SentenceTransformer(resolved_model_path, device=device, local_files_only=True)
                except TypeError:
                    log.warning("local_files_only kwargs not supported; relying on offline mode env.")
                    return SentenceTransformer(resolved_model_path, device=device)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load local model '{resolved_model_path}'. Ensure the model exists locally and is complete."
        ) from e


def move_model_to_cpu(model: SentenceTransformer) -> SentenceTransformer:
    if getattr(model, "device", None) is None or str(model.device) != "cpu":
        model.to("cpu")
    return model


def _non_empty_mask(texts: list[str]) -> list[bool]:
    mask: list[bool] = []
    for t in texts:
        if t is None:
            mask.append(False)
            continue
        mask.append(str(t).strip() != "")
    return mask


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return x / denom


def _encode_non_empty(
    model: SentenceTransformer,
    texts: list[str],
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

    if source_dim > embedding_dim:
        emb = emb[:, :embedding_dim]
        emb = _l2_normalize_rows(emb)

    return emb


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
    embedding_dim: Optional[int] = None,
    show_progress_bar: bool = False,
    progress_desc: Optional[str] = None,
) -> np.ndarray:
    """Encode texts to float32 L2-normalized embeddings."""
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


def encode_texts_with_fallback(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
    embedding_dim: Optional[int] = None,
    show_progress_bar: bool = False,
    progress_desc: Optional[str] = None,
) -> tuple[np.ndarray, SentenceTransformer]:
    """Encode with safety fallback: on MPS runtime error, retry once on CPU."""
    try:
        return (
            encode_texts(
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
            encode_texts(
                model,
                texts,
                batch_size=batch_size,
                embedding_dim=embedding_dim,
                show_progress_bar=show_progress_bar,
                progress_desc=progress_desc,
            ),
            model,
        )
