from __future__ import annotations

from embedding_pipeline.paths import (
    default_ann_index_output_dir,
    default_embedding_output_path,
    infer_dataset_slug_from_embeddings_path,
    infer_model_slug_from_embeddings_path,
    model_slug,
)


def test_model_slug_and_default_embedding_path() -> None:
    assert model_slug("BAAI/bge-m3") == "bge-m3"

    out = default_embedding_output_path(
        output_root="output/models",
        model_name="BAAI/bge-m3",
        input_path="/tmp/title_en.tsv",
    )
    assert out == "output/models/bge-m3/embeddings/title_en.parquet"


def test_default_ann_index_output_dir_from_embeddings_path() -> None:
    emb = "output/models/text-embedding-3-small/embeddings/title_en.parquet"
    assert infer_model_slug_from_embeddings_path(emb) == "text-embedding-3-small"
    assert infer_dataset_slug_from_embeddings_path(emb) == "title_en"

    out = default_ann_index_output_dir(
        output_root="output/models",
        model_name="text-embedding-3-small",
        embeddings_path=emb,
    )
    assert out == "output/models/text-embedding-3-small/ann_index/title_en_index"
