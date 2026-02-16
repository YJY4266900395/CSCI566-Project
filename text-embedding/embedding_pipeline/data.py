from __future__ import annotations

import os
import re
from typing import Dict, Iterator, List

import pandas as pd
from pandas.errors import EmptyDataError


_TEXT_COL = "video_title"
_ID_COL = "video_id"
_TEXT_FILE_EXT = ".txt"
_TSV_LIKE_EXTS = {".tsv", ".txt"}


def _natural_key(s: str) -> List[object]:
    # Sort like: 1.txt, 2.txt, 10.txt (instead of 1,10,2).
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _resolve_tabular_read_kwargs(input_path: str) -> Dict[str, object]:
    """
    Resolve pandas.read_csv kwargs for file inputs.

    Supported file formats:
      - Headered CSV (default behavior)
      - Headered TSV/TXT with columns video_id/video_title
      - Headerless 2-column TSV/TXT: id<TAB>text
    """
    ext = os.path.splitext(input_path)[1].lower()

    if ext in _TSV_LIKE_EXTS:
        try:
            probe = pd.read_csv(input_path, sep="\t", nrows=1, keep_default_na=False)
            if _TEXT_COL in probe.columns:
                return {"sep": "\t", "header": 0}
        except EmptyDataError:
            # Empty file: keep TSV parsing settings; row iteration will return 0 rows.
            pass

        # Fallback: headerless 2-column TSV/TXT (id, text).
        return {
            "sep": "\t",
            "header": None,
            "names": [_ID_COL, _TEXT_COL],
            "usecols": [0, 1],
        }

    # CSV (comma-separated) with header.
    return {}


def _iter_tabular_chunks(input_path: str, *, chunksize: int) -> Iterator[pd.DataFrame]:
    kwargs = _resolve_tabular_read_kwargs(input_path)

    try:
        reader = pd.read_csv(
            input_path,
            chunksize=chunksize,
            keep_default_na=False,
            **kwargs,
        )
        for chunk in reader:
            yield chunk
    except EmptyDataError:
        return
    except ValueError as e:
        # Improve error message for malformed headerless TSV/TXT inputs.
        if kwargs.get("header") is None and kwargs.get("sep") == "\t":
            raise ValueError(
                f"Failed to parse '{input_path}' as headerless TSV/TXT id-text format. "
                "Expected 2 columns: id<TAB>text."
            ) from e
        raise


def count_rows_csv(input_path: str, *, chunksize: int = 100_000) -> int:
    """
    Count data rows using streaming tabular reads.
    """
    n = 0
    for chunk in _iter_tabular_chunks(input_path, chunksize=chunksize):
        n += len(chunk)
    return int(n)


def count_rows_dir(input_dir: str) -> int:
    n = 0
    for name in os.listdir(input_dir):
        if name.lower().endswith(_TEXT_FILE_EXT):
            n += 1
    return int(n)


def load_texts_from_dir(
    input_dir: str,
    *,
    chunksize: int = 10_000,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> Iterator[Dict[str, object]]:
    """
    Stream texts from a directory of .txt files.

    Yields dicts:
      - video_title: list[str]   (file contents)
      - video_id: list[str]      (filename stem)
    """
    names = [n for n in os.listdir(input_dir) if n.lower().endswith(_TEXT_FILE_EXT)]
    names.sort(key=_natural_key)

    batch_ids: List[str] = []
    batch_texts: List[str] = []
    for name in names:
        path = os.path.join(input_dir, name)
        with open(path, "r", encoding=encoding, errors=errors) as f:
            # Common for `.txt` corpora to end files with a newline; strip trailing newline(s) only.
            txt = f.read().rstrip("\r\n")

        batch_ids.append(os.path.splitext(os.path.basename(name))[0])
        batch_texts.append(txt)

        if len(batch_texts) >= chunksize:
            yield {"video_id": batch_ids, "video_title": batch_texts}
            batch_ids, batch_texts = [], []

    if batch_texts:
        yield {"video_id": batch_ids, "video_title": batch_texts}


def load_titles(input_path: str, *, chunksize: int = 10_000) -> Iterator[Dict[str, object]]:
    """
    Stream input tabular file in chunks. Yields dicts:
      - video_title: list[str]
      - video_id: Optional[list[object]] (only if present)

    Does not modify text content beyond converting nulls to empty strings.
    """
    first = True
    for chunk in _iter_tabular_chunks(input_path, chunksize=chunksize):
        if first:
            if _TEXT_COL not in chunk.columns:
                raise ValueError(f"Missing required column '{_TEXT_COL}' in {input_path}")
            first = False

        raw_titles = chunk[_TEXT_COL].tolist()
        # Preserve text as-is; only coerce null-ish values to empty strings for safety.
        titles = ["" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v) for v in raw_titles]

        out: Dict[str, object] = {"video_title": titles}
        if _ID_COL in chunk.columns:
            # Keep original dtype as much as possible; only normalize missing values.
            ids = chunk[_ID_COL].tolist()
            out["video_id"] = ids

        yield out


def count_rows(input_path: str, *, chunksize: int = 100_000) -> int:
    if os.path.isdir(input_path):
        return count_rows_dir(input_path)
    return count_rows_csv(input_path, chunksize=chunksize)


def load_texts(input_path: str, *, chunksize: int = 10_000) -> Iterator[Dict[str, object]]:
    """
    Unified loader:
      - if input_path is a directory -> read *.txt files
      - else -> treat as tabular file (CSV/TSV)
    """
    if os.path.isdir(input_path):
        return load_texts_from_dir(input_path, chunksize=chunksize)
    return load_titles(input_path, chunksize=chunksize)
