from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)
_REPO_ID_RE = re.compile(r"^[^/]+/[^/]+$")


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


def _hf_cache_root() -> Path:
    """
    Resolve HF cache root from env with explicit precedence.

    1) HUGGINGFACE_HUB_CACHE
    2) HF_HOME/hub
    3) ~/.cache/huggingface/hub
    """
    hub_cache = str(os.environ.get("HUGGINGFACE_HUB_CACHE", "")).strip()
    if hub_cache:
        return Path(hub_cache).expanduser()

    hf_home = str(os.environ.get("HF_HOME", "")).strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def _download_hint(repo_id: str) -> str:
    return (
        "Please download it first with: "
        f"uv run python tools/download_hf_model.py --repo-id {repo_id}"
    )


def _resolve_latest_snapshot_dir(repo_id: str) -> Path:
    org, repo = repo_id.split("/", 1)
    repo_cache = _hf_cache_root() / f"models--{org}--{repo}"
    snapshots_dir = repo_cache / "snapshots"

    if not snapshots_dir.is_dir():
        raise RuntimeError(
            f"Model '{repo_id}' not found in local HF cache: '{repo_cache}'. "
            f"{_download_hint(repo_id)}"
        )

    # Prefer refs/main when available; fallback to newest snapshot by mtime.
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
            f"No local snapshots found for '{repo_id}' under '{snapshots_dir}'. "
            f"{_download_hint(repo_id)}"
        )

    snapshot_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshot_dirs[0].resolve()


def _resolve_local_model_path(model_name: str) -> str:
    """
    Resolve --model (org/repo) to a local snapshot directory path.

    Supported form:
    - org/repo (resolved to latest local HF snapshot)
    """
    model_name = str(model_name or "").strip()
    if not model_name:
        raise RuntimeError(
            "--model is required for local backend. "
            "Pass a model repo id in org/repo format."
        )

    if _REPO_ID_RE.match(model_name):
        return str(_resolve_latest_snapshot_dir(model_name))

    raise RuntimeError(
        f"Invalid --model value: '{model_name}'. "
        "Use repo id in 'org/repo' format, for example: BAAI/bge-m3."
    )


def build_model(model_name: str, *, device: str) -> SentenceTransformer:
    """
    Load SentenceTransformer model onto the requested device in strict local-only mode.
    """
    resolved_model_path = _resolve_local_model_path(model_name)
    log.info(
        "Loading model: %s -> %s (device=%s local-only=true)",
        model_name,
        resolved_model_path,
        device,
    )

    try:
        with _hf_offline_mode(True):
            # sentence-transformers constructor API differs across versions.
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
            f"Failed to load local model '{resolved_model_path}'. "
            "Ensure the model exists locally and is complete."
        ) from e


def move_model_to_cpu(model: SentenceTransformer) -> SentenceTransformer:
    """
    Move model to CPU (used for MPS -> CPU fallback).
    """
    if getattr(model, "device", None) is None or str(model.device) != "cpu":
        model.to("cpu")
    return model
