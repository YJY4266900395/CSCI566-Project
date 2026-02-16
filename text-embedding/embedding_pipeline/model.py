from __future__ import annotations

import logging
import os
from contextlib import contextmanager

import torch
from huggingface_hub.constants import HF_HUB_CACHE
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


def resolve_device(device_arg: str) -> str:
    """
    Resolve user device choice to a torch device string.

    Default behavior prefers MPS and falls back to CPU when unavailable.
    """
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
    """
    Force HF/Transformers offline mode to prevent any network calls.
    Restores environment variables afterward.
    """
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


def _assert_local_model_available(model_name: str) -> None:
    """
    Ensure model exists locally. Never download in pipeline runtime.
    """
    if os.path.exists(model_name):
        return

    if "/" in model_name:
        org, repo = model_name.split("/", 1)
        repo_cache = os.path.join(HF_HUB_CACHE, f"models--{org}--{repo}")
        snapshots = os.path.join(repo_cache, "snapshots")
        has_snapshot = os.path.isdir(snapshots) and any(True for _ in os.scandir(snapshots))
        if has_snapshot:
            return
        raise RuntimeError(
            f"Model '{model_name}' is not available locally. Expected cache under: {repo_cache}. "
            "Please download the model first, then rerun."
        )

    raise RuntimeError(
        f"Model path '{model_name}' does not exist locally. "
        "Please download the model first, then rerun."
    )


def build_model(model_name: str, *, device: str) -> SentenceTransformer:
    """
    Load SentenceTransformer model onto the requested device in strict local-only mode.
    """
    log.info("Loading model: %s (device=%s local-only=true)", model_name, device)
    _assert_local_model_available(model_name)

    try:
        with _hf_offline_mode(True):
            # sentence-transformers constructor API differs across versions.
            try:
                return SentenceTransformer(
                    model_name,
                    device=device,
                    model_kwargs={"local_files_only": True},
                    tokenizer_kwargs={"local_files_only": True},
                )
            except TypeError:
                try:
                    return SentenceTransformer(model_name, device=device, local_files_only=True)
                except TypeError:
                    log.warning("local_files_only kwargs not supported; relying on offline mode env.")
                    return SentenceTransformer(model_name, device=device)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load local model '{model_name}'. "
            "Ensure the model exists locally and is complete."
        ) from e


def move_model_to_cpu(model: SentenceTransformer) -> SentenceTransformer:
    """
    Move model to CPU (used for MPS -> CPU fallback).
    """
    if getattr(model, "device", None) is None or str(model.device) != "cpu":
        model.to("cpu")
    return model
