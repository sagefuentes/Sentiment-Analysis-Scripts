"""Runs RoBERTa-based sentiment analysis on a Twitter dataset.

Reads tweet text from a Parquet file, classifies each tweet as negative,
neutral, or positive using a fine-tuned RoBERTa model from Hugging Face,
and writes the labelled results to a new Parquet file.

Supports crash-safe checkpointing so long runs can be resumed without
reprocessing tweets that were already classified. CPU thermal load is
managed via configurable thread limits and inter-batch sleep intervals.

Typical usage:
    python jan_6_roberta_analysis.py --input data/tweets.parquet
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import pipeline


def run_twitter_sentiment_analysis(
        input_path: Path,
        output_path: Path,
        checkpoint_path: Path,
        test_sample: bool = False,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        batch_size: int = 64,
        checkpoint_every: int = 100,
        inter_batch_sleep: float = 0.1,
        num_threads: int = 4,
) -> None:
    """Classifies tweet sentiment and writes labelled results to Parquet.

    Processes tweets in batches through a HuggingFace sentiment pipeline,
    periodically flushing results to a checkpoint file so that progress is
    preserved across crashes or interruptions. On restart the checkpoint is
    validated before use; a corrupt file (e.g. from a mid-write crash) is
    discarded and the run begins from scratch rather than raising an error.

    Args:
        input_path: Parquet file containing the source tweets. Must include
            a ``sentence_text`` column.
        output_path: Destination Parquet file for the final labelled dataset.
        checkpoint_path: Temporary Parquet file used to store incremental
            results. Deleted automatically on successful completion.
        test_sample: When True, randomly samples 1 000 rows before running
            inference. Useful for quick end-to-end validation. Defaults to
            False.
        model_name: HuggingFace Hub identifier for the sentiment model.
            Defaults to ``cardiffnlp/twitter-roberta-base-sentiment-latest``.
        batch_size: Number of tweets passed to the model in each forward
            pass. Larger values are faster but use more memory. Defaults to
            64.
        checkpoint_every: Number of batches between checkpoint flushes. Lower
            values reduce the work lost on a crash at the cost of more
            frequent disk writes. Defaults to 100.
        inter_batch_sleep: Seconds to sleep after each batch, giving the CPU
            time to dissipate heat. Set to ``0`` to disable. Even 0.1 s
            produces a meaningful reduction in sustained CPU temperature over
            a multi-hour run. Defaults to 0.1.
        num_threads: Maximum number of intra-op threads PyTorch may use for
            CPU operations. This is the most direct lever for controlling CPU
            thermal load — reduce it if temperatures remain too high. Defaults
            to 4.

    Raises:
        FileNotFoundError: If ``input_path`` does not point to an existing
            file.
        ValueError: If the loaded dataframe does not contain a
            ``sentence_text`` column.
    """

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Cap PyTorch's intra-op parallelism. By default PyTorch claims every
    # available core, which is the primary driver of high CPU temperatures
    # during long inference runs. This must be set before the pipeline is
    # initialised.
    torch.set_num_threads(num_threads)
    print(f"--- PyTorch intra-op threads capped at {num_threads} ---")

    print(f"--- Loading data: {input_path.name} ---")
    df = pd.read_parquet(input_path)

    if "sentence_text" not in df.columns:
        raise ValueError("Input dataframe must contain a 'sentence_text' column.")

    if test_sample:
        df = df.sample(n=1000).reset_index(drop=True)

    # Declare the schema explicitly so every row group written to the
    # checkpoint file has a consistent, predictable structure regardless of
    # the pandas dtypes inferred at flush time.
    schema = pa.schema([
        ("roberta_label", pa.string()),
        ("roberta_score", pa.float64()),
    ])

    start_index = 0
    already_processed_rows = []

    if checkpoint_path.exists():
        print(f"--- Found checkpoint at {checkpoint_path}. Validating... ---")
        try:
            # Read the Parquet metadata first as a lightweight integrity check.
            # A crash mid-write leaves a file without a valid footer, which
            # causes read_table to raise ArrowInvalid. Catching that here lets
            # us recover gracefully instead of propagating the error to the
            # caller.
            pq.read_metadata(checkpoint_path)
            existing_df = pd.read_parquet(checkpoint_path)
            start_index = len(existing_df)
            already_processed_rows = existing_df.to_dict("records")
            print(f"--- Checkpoint valid. Resuming from index {start_index} ---")
        except Exception as e:  # pylint: disable=broad-except
            print(f"--- WARNING: Checkpoint is corrupt and will be ignored ({e}) ---")
            print("--- Deleting corrupt checkpoint and restarting from index 0 ---")
            checkpoint_path.unlink()

    device = 0 if torch.cuda.is_available() else -1
    print(f"--- Environment: Using {'GPU' if device == 0 else 'CPU'} ---")

    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
        batch_size=batch_size,
    )

    # Maps the model's internal label tokens to human-readable strings.
    label_map = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
    }

    print(f"--- Processing {len(df) - start_index} remaining tweets ---")
    texts = df["sentence_text"].fillna("").tolist()

    # Accumulates result dicts between checkpoint flushes so we write rows
    # to disk in larger, more efficient chunks rather than one at a time.
    local_batch_accumulator = []
    new_rows_written = 0

    # ParquetWriter does not support true append mode — opening it always
    # creates a fresh file. On resume we therefore re-write the previously
    # checkpointed rows first and then stream the new ones, keeping the file
    # self-contained and correct at all times.
    writer = pq.ParquetWriter(str(checkpoint_path), schema, version="2.6")

    def flush_accumulator() -> None:
        """Writes pending rows in ``local_batch_accumulator`` to the checkpoint file.

        Converts the accumulator to a PyArrow Table, appends it as a new row
        group, increments ``new_rows_written``, and clears the accumulator.
        Does nothing if the accumulator is empty.
        """
        nonlocal new_rows_written
        if not local_batch_accumulator:
            return
        batch_df = pd.DataFrame(local_batch_accumulator)
        table = pa.Table.from_pandas(batch_df, schema=schema, preserve_index=False)
        writer.write_table(table)
        new_rows_written += len(local_batch_accumulator)
        local_batch_accumulator.clear()

    try:
        if already_processed_rows:
            prior_df = pd.DataFrame(already_processed_rows)
            prior_table = pa.Table.from_pandas(
                prior_df, schema=schema, preserve_index=False
            )
            writer.write_table(prior_table)

        gc_counter = 0

        for i in tqdm(range(start_index, len(texts), batch_size)):
            batch_texts = texts[i: i + batch_size]

            with torch.no_grad():
                batch_results = classifier(
                    batch_texts,
                    truncation=True,
                    max_length=512,
                )

            for res in batch_results:
                local_batch_accumulator.append({
                    "roberta_label": label_map.get(res["label"], res["label"]),
                    "roberta_score": res["score"],
                })

            # Count batches relative to the current session's start index so
            # the checkpoint interval fires at the correct frequency regardless
            # of where a resumed run begins.
            batches_this_session = (i - start_index) // batch_size + 1
            if batches_this_session % checkpoint_every == 0:
                flush_accumulator()
                print(f"\n--- Checkpoint saved at batch {batches_this_session} ---")

            # Run the garbage collector every 10 batches rather than every
            # batch. Calling gc.collect() on every iteration produces constant
            # CPU bursts without meaningfully reducing peak memory usage.
            gc_counter += 1
            if gc_counter % 10 == 0:
                if device == 0:
                    torch.cuda.empty_cache()
                gc.collect()

            # A short sleep after each batch allows the CPU to dissipate heat
            # before the next forward pass begins. The cumulative effect over
            # thousands of iterations is a significant reduction in sustained
            # temperature.
            if inter_batch_sleep > 0:
                time.sleep(inter_batch_sleep)

        flush_accumulator()

    except KeyboardInterrupt:
        print("\n--- Interrupted by user. Saving progress and exiting... ---")
        flush_accumulator()
        writer.close()
        print(f"--- Progress saved. {start_index + new_rows_written} rows in checkpoint. ---")
        sys.exit(0)

    finally:
        # Closing the writer flushes the Parquet file footer. Without this
        # call — which must happen even on an unhandled exception — the
        # checkpoint file is left in a corrupt, unreadable state.
        writer.close()

    print("--- Finalizing results ---")

    results_df = pd.read_parquet(checkpoint_path)

    # Align the original dataframe rows with the results using positional
    # indexing so the join is correct even when the source dataframe has a
    # non-default index.
    final_df = pd.concat(
        [df.iloc[: len(results_df)].reset_index(drop=True), results_df],
        axis=1,
    )

    final_df.to_parquet(output_path, index=False)

    checkpoint_path.unlink(missing_ok=True)

    print(f"--- Success! Results saved to {output_path} ---")


def main() -> None:
    """Parses command-line arguments and runs the sentiment analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="RoBERTa sentiment analysis for the January 6th Twitter dataset."
    )
    parser.add_argument(
        "--input",
        default="../data/interim/jan6Dataset.parquet",
        help="Path to the input Parquet file. Must contain a 'sentence_text' column.",
    )
    parser.add_argument(
        "--output",
        default="../data/processed/roberta_results_final.parquet",
        help="Destination path for the final labelled Parquet file.",
    )
    parser.add_argument(
        "--checkpoint",
        default="../data/interim/sentiment_checkpoint.parquet",
        help="Path for the temporary checkpoint file used to resume interrupted runs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="Number of tweets per inference batch. Default: 64.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help=(
            "PyTorch intra-op thread count. Reducing this value lowers CPU "
            "utilisation and temperature. Default: 4."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help=(
            "Seconds to sleep between batches to allow CPU cooling. "
            "Set to 0 to disable. Default: 0.1."
        ),
    )

    args = parser.parse_args()

    run_twitter_sentiment_analysis(
        input_path=Path(args.input),
        output_path=Path(args.output),
        checkpoint_path=Path(args.checkpoint),
        test_sample=False,
        batch_size=args.batch,
        num_threads=args.threads,
        inter_batch_sleep=args.sleep,
    )


if __name__ == "__main__":
    main()