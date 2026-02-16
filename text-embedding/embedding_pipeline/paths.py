from __future__ import annotations

import os
import re
from pathlib import Path


def _slug(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    s = s.strip("-._")
    return s or "unknown"


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
    extension: str = "parquet",
    dataset_name: str = "",
) -> str:
    ext = str(extension or "parquet").lstrip(".")
    ds = _slug(dataset_name) if dataset_name else dataset_slug(input_path)
    return os.path.join(str(output_root), model_slug(model_name), "embeddings", f"{ds}.{ext}")


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
