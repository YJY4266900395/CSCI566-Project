# Shortvideo-Text-Embedding

Text embedding pipeline for short-video metadata/text, with:
- local embeddings via `BAAI/bge-m3` (sentence-transformers)
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
└── tools/                           # ANN sampling utilities
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1) Prepare `id<TAB>text` input (recommended)

Example: merge text files into one TSV.

```bash
python data_preprocessing/merge_to_tsv.py \
  --input_dir /path/to/title_en_txt
```

By default, output is written to `/path/to/title_en.tsv` (same parent directory as the input folder).

### 2) Local embedding (BGE-M3)

```bash
python main.py \
  --input data/title_en.tsv \
  --backend local \
  --model /path/to/bge-m3 \
  --batch_size 128
```

Writes to `output/models/bge-m3/embeddings/title_en.parquet`.

### 3) OpenAI embedding

```bash
export OPENAI_API_KEY="..."
python main.py \
  --input data/title_en.tsv \
  --backend openai \
  --openai_model text-embedding-3-small \
  --dimensions 1024
```

Writes to `output/models/text-embedding-3-small/embeddings/title_en.parquet`.

### 4) Build and query ANN index

```bash
python -m embedding_pipeline.build_ann_index \
  --input output/models/bge-m3/embeddings/title_en.parquet

python -m embedding_pipeline.query_ann_index \
  --index-dir output/models/bge-m3/ann_index/title_en_index \
  --embedding-parquet output/models/bge-m3/embeddings/title_en.parquet \
  --embedding-index-id 0 \
  --topk 5
```

## Tools (`tools/`)

`tools/` provides a quick sampler to sanity-check ANN quality after index build.

### `sample_ann_neighbors.py`

Use `--mode` to choose output style:
- `video`: print `video_id`, score, and dataset video URL
- `text`: print `video_id (video_title)` with score

Video-style example:

```bash
python tools/sample_ann_neighbors.py \
  --mode video \
  --index-dir output/models/bge-m3/ann_index/title_en_index \
  --embeddings-parquet output/models/bge-m3/embeddings/title_en.parquet \
  --n 5 \
  --k 10
```

Text-style example (for category combos):

```bash
python tools/sample_ann_neighbors.py \
  --mode text \
  --index-dir output/models/bge-m3/ann_index/category_combo_cn_index \
  --embeddings-parquet output/models/bge-m3/embeddings/category_combo_cn.parquet \
  --n 5 \
  --k 10
```

Also supports:
- `--seed` (sampling seed)
- `--ef-search` (HNSW query breadth)
- `--include-self` (include the query row in neighbor results)
- `--video-url-prefix` (custom URL prefix for `--mode video`)

## Output Layout

When `--output` (for `main.py`) or `--output-dir` (for ANN build) is omitted, artifacts are auto-organized as:

- `output/models/<model>/embeddings/<dataset>.parquet` (or `.npy`)
- `output/models/<model>/ann_index/<dataset>_index/`

Examples:
- `output/models/bge-m3/embeddings/title_en.parquet`
- `output/models/bge-m3/ann_index/title_en_index/`
- `output/models/text-embedding-3-small/embeddings/title_en.parquet`
- `output/models/text-embedding-3-small/ann_index/title_en_index/`

## Input Formats

`main.py` supports:
- headered CSV with required `video_title` column and optional `video_id`
- headered TSV/TXT with the same column names
- headerless TSV/TXT with exactly 2 columns: `id<TAB>text`
- directory input where each `.txt` file is one row (ID from filename stem)

For this pipeline, the recommended format is headerless `id<TAB>text` (for example `title_en.tsv`, `category_combo_cn.tsv`).

## Data Preprocessing

### Generate category combo table

Produces a single 2-column TSV file:
- column 1: `combo_id` (`cat1_cat2_cat3`)
- column 2: `combo_name` (`cat1 > cat2 > cat3`)
- only Level-3 category paths are emitted (`root_id_parent_id_category_id`)
- duplicate `combo_id` rows are deduplicated

```bash
python data_preprocessing/generate_category_combinations.py \
  --input_file /path/to/categories_cn_en.csv \
  --output_file /path/to/category_combo_en.tsv \
  --name_col category_name_en
```

Use Chinese category names instead:

```bash
python data_preprocessing/generate_category_combinations.py \
  --input_file /path/to/categories_cn_en.csv \
  --output_file /path/to/category_combo_cn.tsv \
  --name_col category_name_cn
```

### Merge text files to TSV

Merge `{video_id}.txt` files into one headerless TSV: `video_id<TAB>text`.

```bash
python data_preprocessing/merge_to_tsv.py \
  --input_dir /path/to/title_en_txt
```

Notes:
- filenames are expected to be numeric video IDs (for numeric sort)
- default output path is `<input_dir_name>.tsv` in the input dir's parent
- optional: `--output /path/to/custom.tsv`
- optional: `--limit N` to process only the first `N` files

## Output Formats

- `.parquet` (recommended): columns include `video_title`, `embedding`, and optional `video_id`
- `.npy`: dense matrix `(N, dim)` only (row order matches input; metadata not stored)

## Device and Model Loading

- `--device` defaults to `mps`; if MPS is unavailable the pipeline falls back to CPU
- Local backend is strict local-only: if `--model` is missing locally, the run exits with an error
- Download models yourself ahead of time, then pass a local model path (or a cached model name)

Example local model path:

```bash
python main.py --input ... --backend local --model /path/to/bge-m3
```

## Tests

```bash
python tests/smoke_test.py
python -m pytest -q tests/test_data_loader.py tests/test_paths.py tests/test_openai_backend.py
```
