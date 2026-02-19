from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Optional


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Text embedding pipeline for video titles.")
    p.add_argument(
        "--input",
        required=True,
        help="Input path: CSV/TSV file (headered, or TSV/TXT id<TAB>text) OR a directory of .txt files (one file per row).",
    )
    p.add_argument(
        "--output",
        default="",
        help=(
            "Output path (.parquet preferred; .npy also supported). "
            "If omitted, auto-uses output/models/<model>/embeddings/<dimension>/"
        ),
    )
    p.add_argument(
        "--output_format",
        default="parquet",
        choices=["parquet", "npy"],
        help="Output extension used when --output is omitted (default: parquet).",
    )
    p.add_argument(
        "--backend",
        default="local",
        choices=["local", "openai"],
        help="Embedding backend: local (sentence-transformers) or openai (API).",
    )
    p.add_argument("--model", default="", help="Local model repo id in org/repo format; required for --backend local.")
    p.add_argument("--openai_model", default="text-embedding-3-small", help="OpenAI embedding model name.")
    p.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help=(
            "Optional embedding dimensions override. "
            "Local backend: default=model dim, values above model dim fall back to model dim. "
            "OpenAI backend: default=1024 when omitted."
        ),
    )
    p.add_argument("--batch_size", type=int, default=128, help="Encoding batch size (default: 128).")
    p.add_argument(
        "--device",
        default="mps",
        choices=["auto", "mps", "cpu"],
        help="Device: mps/auto/cpu (default: mps; falls back to cpu if unavailable).",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=10_000,
        help="Tabular streaming chunksize (rows per chunk, default: 10000).",
    )
    p.add_argument("--max_retries", type=int, default=5, help="OpenAI max retries (default: 5).")
    p.add_argument("--clean", action="store_true", help="Apply optional text cleaning rules before embedding.")
    p.add_argument("--clean_rules", default="config/cleaning.json", help="Path to cleaning rules JSON.")
    p.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize embeddings (default: true).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    t0 = time.time()

    # Lazy imports: keep `python main.py --help` working even if deps aren't installed yet.
    from tqdm import tqdm

    from embedding_pipeline.data import count_rows, load_texts
    from embedding_pipeline.io import make_writer
    from embedding_pipeline.model import build_model, resolve_device
    from embedding_pipeline.paths import default_embedding_output_path

    cleaning_rules = None
    if args.clean:
        from embedding_pipeline.cleaning import clean_text, load_rules

        cleaning_rules = load_rules(args.clean_rules)

    device = "cpu"
    if args.backend == "local":
        device = resolve_device(args.device)
        log.info("Device selected: %s (requested=%s)", device, args.device)
    else:
        log.info("Backend openai selected (device flag ignored).")

    if args.backend == "local" and not str(args.model or "").strip():
        raise SystemExit(
            "ERROR: --model is required for --backend local and must be repo id (org/repo)."
        )

    total_rows = count_rows(args.input, chunksize=args.chunksize)
    log.info("Input rows: %d", total_rows)

    model = None
    openai_backend = None
    embed_dim = None

    if args.backend == "local":
        from embedding_pipeline.encoder import encode_titles_with_fallback

        model = build_model(args.model, device=device)
        model_default_dim = int(model.get_sentence_embedding_dimension())
        if args.dimensions is None:
            embed_dim = model_default_dim
            log.info("Using local default dimensions=%d", embed_dim)
        else:
            requested_dim = int(args.dimensions)
            if requested_dim <= 0:
                raise SystemExit("ERROR: --dimensions must be > 0 when provided.")
            embed_dim = min(requested_dim, model_default_dim)
            if requested_dim > model_default_dim:
                log.warning(
                    "Requested dimensions=%d exceeds model default=%d; using %d.",
                    requested_dim,
                    model_default_dim,
                    embed_dim,
                )
            else:
                log.info("Using local requested dimensions=%d (model default=%d)", embed_dim, model_default_dim)
    else:
        from embedding_pipeline.openai_backend import OpenAIEmbedder

        openai_dimensions = 1024 if args.dimensions is None else int(args.dimensions)
        if openai_dimensions <= 0:
            raise SystemExit("ERROR: --dimensions must be > 0 when provided.")
        openai_backend = OpenAIEmbedder(
            model=args.openai_model,
            dimensions=openai_dimensions,
            request_batch_size=args.batch_size,
            max_retries=args.max_retries,
            normalize=args.normalize,
        )
        embed_dim = int(openai_backend.get_sentence_embedding_dimension())

    model_name_for_path = args.model if args.backend == "local" else args.openai_model
    output_path = args.output or default_embedding_output_path(
        output_root="output/models",
        model_name=model_name_for_path,
        input_path=args.input,
        embedding_dim=embed_dim,
        extension=args.output_format,
    )
    log.info("Output path: %s", output_path)

    writer = make_writer(output_path, total_rows=total_rows)
    writer_initialized = False
    written = 0

    try:
        if total_rows == 0:
            # Still create an empty output file with the correct schema/shape.
            writer.init_if_needed(embedding_dim=embed_dim, has_ids=False)
            log.info("Done. rows=0 dim=%d output=%s seconds=%.2f", embed_dim, output_path, time.time() - t0)
            return 0

        with tqdm(total=total_rows, desc="Encoding", unit="rows") as pbar:
            for batch in load_texts(args.input, chunksize=args.chunksize):
                titles = batch["video_title"]
                ids = batch.get("video_id")

                if cleaning_rules is not None:
                    titles = [clean_text(t, cleaning_rules) for t in titles]

                if args.backend == "local":
                    embeddings, model = encode_titles_with_fallback(
                        model,
                        titles,
                        batch_size=args.batch_size,
                        embedding_dim=embed_dim,
                    )
                else:
                    embeddings = openai_backend.encode(titles, batch_size=args.batch_size)

                if not writer_initialized:
                    writer.init_if_needed(embedding_dim=embed_dim, has_ids=ids is not None)
                    writer_initialized = True

                writer.write_batch(video_titles=titles, embeddings=embeddings, video_ids=ids)

                written += len(titles)
                pbar.update(len(titles))

        dt = time.time() - t0
        log.info(
            "Done. rows=%d dim=%s output=%s seconds=%.2f",
            written,
            embed_dim,
            output_path,
            dt,
        )
        return 0
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
