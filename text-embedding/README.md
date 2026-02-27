# Shortvideo-Text-Embedding

Text embedding pipeline for short-video metadata/text.

## 1) Environment Setup

```bash
# Install uv first if needed (macOS)
brew install uv

# Project is pinned to Python 3.12 (.python-version)
uv python install 3.12

# Create/update .venv from pyproject.toml + uv.lock
uv sync --locked
```

Run all commands with `uv run ...`.

## 2) Project Structure

```text
.
├── embedding_generator/          # embedding generation (multifield only)
├── ann_index/                   # ANN build/query
├── data_preprocessing/
├── data/
├── tests/
└── tools/
```

Notes:
- No shared `main.py` entrypoint.
- `embedding_generator` and `ann_index` run independently.
- `embedding_generator` is intentionally flat: `run_multifield_embeddings.py` + `modeling.py` + `io_utils.py`.
- `ann_index` is intentionally flat: `build_index.py` + `query_index.py` + `core.py`.

## 3) Data Preprocessing

### 3.1 One-pass preprocessing (recommended)

```bash
uv run python data_preprocessing/build_multifield_tsvs.py \
  --interaction-file data/interaction_filtered.csv \
  --categories-file data/categories_cn_en.csv \
  --video-tag-title-output data/video_tag_title.tsv \
  --video-category-output data/video_category_combo_cn.tsv \
  --category-name-col category_name_cn
```

Outputs (headerless TSV):
- `data/video_tag_title.tsv`: `video_id<TAB>tag_name<TAB>title`
- `data/video_category_combo_cn.tsv`: `video_id<TAB>category_combo_id<TAB>category_combo_name`

### 3.2 Individual scripts

```bash
uv run python data_preprocessing/extract_video_tags_title_tsv.py \
  --input data/interaction_filtered.csv \
  --output data/video_tag_title.tsv

uv run python data_preprocessing/generate_video_category_combo.py \
  --interaction-file data/interaction_filtered.csv \
  --categories-file data/categories_cn_en.csv \
  --output-file data/video_category_combo_cn.tsv \
  --name-col category_name_cn
```

### 3.3 Merge `{video_id}.txt` to one TSV

```bash
uv run python data_preprocessing/merge_to_tsv.py --input_dir /path/to/title_en_txt
```

## 4) Download Models

Models are downloaded to Hugging Face cache:
- default: `~/.cache/huggingface/hub`
- if `HF_HOME` is set: `$HF_HOME/hub`

```bash
uv run python tools/download_hf_model.py --repo-id BAAI/bge-m3
uv run python tools/download_hf_model.py --repo-id Qwen/Qwen3-Embedding-4B
```

Local model rule:
- pass `--model org/repo`
- pipeline resolves local snapshot with `refs/main` first, then newest `snapshots/*`

## 5) Embedding Generation

### 5.1 Multifield concat embedding (asr + title/tag + category)

```bash
uv run python -m embedding_generator.run_multifield_embeddings \
  --asr-input data/asr_cn.tsv \
  --title-tag-input data/video_tag_title.tsv \
  --category-input data/video_category_combo_cn.tsv \
  --model Qwen/Qwen3-Embedding-4B \
  --dimensions 256 \
  --batch-size 128 \
  --max-seq-length 512 \
  --device mps
```

- each field uses `--dimensions` (default `256`)
- final concat dim is `3 * dimensions` (default `768`)
- progress is shown by default (`rows/s` + ETA)

Output columns:
- `video_id`
- `has_asr`, `has_title_tag`, `has_category` (0/1)
- `embedding` (fixed-size float list)

For `BAAI/bge-m3` on MPS, use `--max-seq-length 512` or `1024`.

## 6) ANN Index

### 6.1 Build index

```bash
uv run python -m ann_index.build_index \
  --input output/models/<model>/embeddings/<dimension>/<dataset>.parquet
```

### 6.2 Query index

```bash
uv run python -m ann_index.query_index \
  --index-dir output/models/<model>/ann_index/<dataset>_index \
  --embedding-parquet output/models/<model>/embeddings/<dimension>/<dataset>.parquet \
  --embedding-index-id 0 \
  --topk 5
```

## 7) Tests

```bash
PYTHONPATH=. uv run pytest -q tests/test_paths.py tests/test_ann.py
PYTHONPATH=. uv run python tests/smoke_test.py
```
