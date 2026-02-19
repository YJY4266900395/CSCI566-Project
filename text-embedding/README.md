# Shortvideo-Text-Embedding

Text embedding pipeline for short-video metadata/text, with:
- local embeddings via any Hugging Face repo id (`org/repo`, sentence-transformers compatible)
- optional OpenAI embeddings (for example `text-embedding-3-small`)
- optional ANN index build/query via HNSW (`hnswlib`)

## Project Structure

```text
.
├── main.py                          # CLI entrypoint for embedding jobs
├── embedding_pipeline/              # Loaders, model backends, writers, ANN tools
├── data_preprocessing/              # Data prep scripts
├── data/                            # Local input/output data (gitignored)
├── tests/                           # Smoke/unit tests
└── tools/                           # Utilities (model download, ANN sampler)
```

## 1) Environment Setup

```bash
# Install uv first if needed (macOS)
brew install uv

# Project is pinned to Python 3.12 (.python-version)
uv python install 3.12

# Create/update .venv from pyproject.toml + uv.lock
uv sync --locked
```

After this, run commands with `uv run ...` (no manual activate needed).

## 2) Data Processing

### 2.1 Merge text files into TSV

Merge `{video_id}.txt` files into one headerless TSV: `video_id<TAB>text`.

```bash
uv run python data_preprocessing/merge_to_tsv.py \
  --input_dir /path/to/title_en_txt
```

Notes:
- filenames are expected to be numeric video IDs (for numeric sort)
- default output path is `<input_dir_name>.tsv` in the input dir's parent
- optional: `--output /path/to/custom.tsv`
- optional: `--limit N` to process only the first `N` files

### 2.2 Generate category-combo TSV

Produces a 2-column TSV:
- column 1: `combo_id` (`cat1_cat2_cat3`)
- column 2: `combo_name` (`cat1 > cat2 > cat3`)
- only Level-3 paths are emitted (`root_id_parent_id_category_id`)
- duplicate `combo_id` rows are deduplicated

```bash
uv run python data_preprocessing/generate_category_combinations.py \
  --input_file /path/to/categories_cn_en.csv \
  --output_file /path/to/category_combo_en.tsv \
  --name_col category_name_en
```

Chinese name output:

```bash
uv run python data_preprocessing/generate_category_combinations.py \
  --input_file /path/to/categories_cn_en.csv \
  --output_file /path/to/category_combo_cn.tsv \
  --name_col category_name_cn
```

### 2.3 Input formats supported by `main.py`

- headered CSV with required `video_title` and optional `video_id`
- headered TSV/TXT with the same column names
- headerless TSV/TXT with exactly 2 columns: `id<TAB>text`
- directory input where each `.txt` file is one row (ID from filename stem)

Recommended for this pipeline: headerless `id<TAB>text` (for example `title_en.tsv`, `category_combo_cn.tsv`).

## 3) Download Model

Use the provided script to download models to Hugging Face default cache.

Default cache root:
- `~/.cache/huggingface/hub`
- if `HF_HOME` is set: `$HF_HOME/hub`

Snapshot layout example:
- `~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<snapshot_id>/`

Download commands:

```bash
uv run python tools/download_hf_model.py --repo-id <org>/<repo>
# example
uv run python tools/download_hf_model.py --repo-id Qwen/Qwen3-Embedding-4B
```

For private models:

```bash
export HF_TOKEN="..."
uv run python tools/download_hf_model.py --repo-id <org>/<private-model>
```

Model parameter rule for local backend:
- `--model` must be repo id format: `org/repo` (for example `Qwen/Qwen3-Embedding-4B`)
- pipeline resolves repo id to local snapshot: prefer `refs/main`, fallback to newest `snapshots/*` by mtime
- `--dimensions` is optional for local backend
- if `--dimensions` is omitted, use model default dimension
- if `--dimensions` is provided, use `min(requested_dim, model_default_dim)`
- if not found locally, run exits with a clear download hint

## 4) Pipeline and Next Steps

### 4.1 Local embedding

```bash
# model default dimension
uv run python main.py \
  --input data/title_en.tsv \
  --backend local \
  --model <org>/<repo> \
  --batch_size 128

# optional: request a smaller output dimension
uv run python main.py \
  --input data/title_en.tsv \
  --backend local \
  --model Qwen/Qwen3-Embedding-4B \
  --dimensions 1024 \
  --batch_size 128
```

Writes to `output/models/<repo>/embeddings/<dimension>/title_en.parquet`.

### 4.2 OpenAI embedding

```bash
export OPENAI_API_KEY="..."
uv run python main.py \
  --input data/title_en.tsv \
  --backend openai \
  --openai_model text-embedding-3-small \
  --dimensions 1024
```

Writes to `output/models/text-embedding-3-small/embeddings/1024/title_en.parquet`.

### 4.3 Build ANN index

```bash
uv run python -m embedding_pipeline.build_ann_index \
  --input output/models/<model>/embeddings/<dimension>/title_en.parquet
```

### 4.4 Query ANN index

```bash
uv run python -m embedding_pipeline.query_ann_index \
  --index-dir output/models/<model>/ann_index/title_en_index \
  --embedding-parquet output/models/<model>/embeddings/<dimension>/title_en.parquet \
  --embedding-index-id 0 \
  --topk 5
```

### 4.5 ANN quality quick check (`tools/`)

`tools/sample_ann_neighbors.py` supports:
- `--mode video`: print `video_id`, score, and dataset URL
- `--mode text`: print `video_id (video_title)` with score

Video mode:

```bash
uv run python tools/sample_ann_neighbors.py \
  --mode video \
  --index-dir output/models/<model>/ann_index/title_en_index \
  --embeddings-parquet output/models/<model>/embeddings/<dimension>/title_en.parquet \
  --n 5 \
  --k 10
```

Text mode:

```bash
uv run python tools/sample_ann_neighbors.py \
  --mode text \
  --index-dir output/models/<model>/ann_index/category_combo_cn_index \
  --embeddings-parquet output/models/<model>/embeddings/<dimension>/category_combo_cn.parquet \
  --n 5 \
  --k 10
```

### 4.6 Output layout

When outputs are omitted, artifacts are auto-organized as:
- `output/models/<model>/embeddings/<dimension>/<dataset>.parquet` (or `.npy`)
- `output/models/<model>/ann_index/<dataset>_index/`

Examples:
- `output/models/Qwen3-Embedding-4B/embeddings/1024/title_en.parquet`
- `output/models/Qwen3-Embedding-4B/ann_index/title_en_index/`
- `output/models/text-embedding-3-small/embeddings/1024/title_en.parquet`
- `output/models/text-embedding-3-small/ann_index/title_en_index/`

Output formats:
- `.parquet` (recommended): includes `video_title`, `embedding`, optional `video_id`
- `.npy`: dense matrix `(N, dim)` only (metadata not stored)

### 4.7 Tests

```bash
uv run python tests/smoke_test.py
uv run pytest -q tests/test_data_loader.py tests/test_paths.py tests/test_openai_backend.py
```
