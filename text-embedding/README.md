# Shortvideo-Text-Embedding

Text embedding pipeline for short-video metadata/text, with:
- local embeddings via any Hugging Face repo id (`org/repo`, sentence-transformers compatible)
- optional OpenAI embeddings (for example `text-embedding-3-small`)
- optional ANN index build/query via HNSW (`hnswlib`)

## Project Structure

```text
.
├── main.py                                  # CLI entrypoint for single-field embedding jobs
├── embedding_pipeline/                      # Loaders, model backends, writers, ANN tools
├── data_preprocessing/                      # Data prep scripts
├── data/                                    # Local input/output data (gitignored)
├── tests/                                   # Smoke/unit tests
└── tools/                                   # Utilities (model download, ANN sampler)
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

## 2) Data Preprocessing

### 2.1 Recommended one-pass preprocessing (optimized)

Use one pass over `interaction_filtered.csv` to generate both multifield inputs:

```bash
uv run python data_preprocessing/build_multifield_tsvs.py \
  --interaction-file data/interaction_filtered.csv \
  --categories-file data/categories_cn_en.csv \
  --video-tag-title-output data/video_tag_title.tsv \
  --video-category-output data/video_category_combo_cn.tsv \
  --category-name-col category_name_cn
```

Outputs (both are **headerless** TSV):
- `data/video_tag_title.tsv`: `video_id<TAB>tag_name<TAB>title`
- `data/video_category_combo_cn.tsv`: `video_id<TAB>category_combo_id<TAB>category_combo_name`

This replaces two separate passes and is faster for large interaction files.

### 2.2 Individual scripts (if needed)

Generate `video_tag_title.tsv` only:

```bash
uv run python data_preprocessing/extract_video_tags_title_tsv.py \
  --input data/interaction_filtered.csv \
  --output data/video_tag_title.tsv
```

Generate `video_category_combo_cn.tsv` only:

```bash
uv run python data_preprocessing/generate_video_category_combo.py \
  --interaction-file data/interaction_filtered.csv \
  --categories-file data/categories_cn_en.csv \
  --output-file data/video_category_combo_cn.tsv \
  --name-col category_name_cn
```

Both outputs are headerless.

### 2.3 Other data prep utilities

Merge `{video_id}.txt` files into one headerless TSV (`video_id<TAB>text`):

```bash
uv run python data_preprocessing/merge_to_tsv.py \
  --input_dir /path/to/title_en_txt
```

Generate category-combo dictionary (level-3 taxonomy):

```bash
uv run python data_preprocessing/generate_category_combinations.py \
  --input_file data/categories_cn_en.csv \
  --output_file data/category_combo_cn.tsv \
  --name_col category_name_cn
```

## 3) Download Model

Use the provided script to download models to Hugging Face default cache.

Default cache root:
- `~/.cache/huggingface/hub`
- if `HF_HOME` is set: `$HF_HOME/hub`

```bash
uv run python tools/download_hf_model.py --repo-id BAAI/bge-m3
uv run python tools/download_hf_model.py --repo-id Qwen/Qwen3-Embedding-4B
```

For private models:

```bash
export HF_TOKEN="..."
uv run python tools/download_hf_model.py --repo-id <org>/<private-model>
```

Local backend model rule:
- `--model` must be `org/repo`
- pipeline resolves local snapshot by: `refs/main` first, else newest `snapshots/*`
- if not found locally, run exits with a download hint

## 4) Embedding Pipelines

### 4.1 Single-field embedding (`main.py`)

```bash
uv run python main.py \
  --input data/asr_cn.tsv \
  --backend local \
  --model BAAI/bge-m3 \
  --dimensions 256 \
  --batch_size 128 \
  --device mps
```

Default output path:
- `output/models/<model>/embeddings/<dimension>/<dataset>.parquet`

### 4.2 Multifield concat embedding (asr + title/tag + category)

`embedding_pipeline.run_multifield_embeddings` uses 3 TSVs:
- `asr`: `video_id<TAB>asr_text`
- `title+tag`: `video_id<TAB>tag_name<TAB>title`
- `category`: `video_id<TAB>category_combo_id<TAB>category_combo_name`

Per-field embedding dim is `--dimensions` (default 256), then concatenated to `3 * dimensions` (default 768).
Progress display is enabled by default and reports `rows/s` plus ETA per field.

```bash
uv run python -m embedding_pipeline.run_multifield_embeddings \
  --asr-input data/asr_cn.tsv \
  --title-tag-input data/video_tag_title.tsv \
  --category-input data/video_category_combo_cn.tsv \
  --model BAAI/bge-m3 \
  --dimensions 256 \
  --batch-size 128 \
  --max-seq-length 512 \
  --device mps
```

Output parquet columns:
- `video_id`
- `has_asr`, `has_title_tag`, `has_category` (0/1)
- `embedding` (fixed size list, default 768)


For `BAAI/bge-m3` on MPS, set `--max-seq-length 512` (or 1024) to avoid very large attention buffers at default seq length 8192.

Optional: save each 256-dim field embedding as parquet too:

```bash
--save-field-embeddings
```

### 4.3 Local bge-m3 smoke test command

```bash
uv run python -m embedding_pipeline.run_multifield_embeddings \
  --asr-input data/asr_cn.tsv \
  --title-tag-input data/video_tag_title.tsv \
  --category-input data/video_category_combo_cn.tsv \
  --model BAAI/bge-m3 \
  --dimensions 256 \
  --batch-size 64 \
  --max-seq-length 512 \
  --device mps \
  --max-videos 64 \
  --output /tmp/video_multifield_concat.bgem3.sample.parquet
```

## 5) ANN (Optional)

Build ANN index from any embedding parquet:

```bash
uv run python -m embedding_pipeline.build_ann_index \
  --input output/models/<model>/embeddings/<dimension>/<dataset>.parquet
```

Query ANN index:

```bash
uv run python -m embedding_pipeline.query_ann_index \
  --index-dir output/models/<model>/ann_index/<dataset>_index \
  --embedding-parquet output/models/<model>/embeddings/<dimension>/<dataset>.parquet \
  --embedding-index-id 0 \
  --topk 5
```

## 6) Tests

```bash
uv run python tests/smoke_test.py
uv run pytest -q tests/test_data_loader.py tests/test_paths.py tests/test_openai_backend.py
```
