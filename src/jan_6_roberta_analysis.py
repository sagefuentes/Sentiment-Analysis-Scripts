from pathlib import Path
import argparse
import sys

import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm


def run_twitter_sentiment_analysis(
        input_path: Path,
        output_path: Path,
        checkpoint_path: Path,
        test_sample: bool = False,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        batch_size: int = 64,
        checkpoint_every: int = 100
) -> None:
    """
    Analyzes sentiment with checkpointing and memory optimization.

    Args:
        input_path: Path object pointing to the source Parquet file.
        output_path: Path object where the final results will be stored.
        checkpoint_path: Path object for temporary progress storage.
        test_sample: Determine if you would like to run a short sample for testing purposes. Default is False.
        model_name: Hugging Face model identifier. Default is RoBERTa for Twitter.
        batch_size: Number of sequences to process per iteration. Defaults to 64.
        checkpoint_every: Frequency of saving checkpoints (in batches). Defaults to 100.

    Raises:
        FileNotFoundError: If the input_path does not exist.
        ValueError: If the required 'TextClean' column is missing.
        :param test_sample:
    """

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load Data from Parquet
    print(f"--- Loading data: {input_path.name} ---")
    df = pd.read_parquet(input_path)

    if "TextClean" not in df.columns:
        raise ValueError("Input dataframe must contain a 'TextClean' column.")

    # Optional: If you want to sample for testing
    if test_sample:
        df = df.sample(n=1000).reset_index(drop=True)

    # Setup Checkpointing
    results = []
    start_index = 0
    if checkpoint_path.exists():
        print(f"--- Found checkpoint at {checkpoint_path}. Resuming... ---")
        checkpoint_df = pd.read_parquet(checkpoint_path)
        results = checkpoint_df.to_dict('records')
        start_index = len(results)
        print(f"--- Resuming from index {start_index} ---")

    # Initialize Hardware and Pipeline
    device = 0 if torch.cuda.is_available() else -1
    print(f"--- Environment: Using {'GPU' if device == 0 else 'CPU'} ---")

    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
        batch_size=batch_size
    )

    # Map labels
    label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

    # Inference Loop with Checkpoints
    print(f"--- Processing {len(df) - start_index} remaining tweets ---")
    texts = df['TextClean'].fillna("").tolist()

    try:
        # Process in steps to allow for checkpointing
        for i in tqdm(range(start_index, len(texts), batch_size)):
            batch_texts = texts[i: i + batch_size]
            # Truncating to max length due to the hard limit of RoBERTa for 512 tokens
            batch_results = classifier(
                batch_texts,
                truncation=True,
                max_length=512)

            # Map labels immediately
            for res in batch_results:
                res['roberta_label'] = label_map.get(res['label'], res['label'])
                res['roberta_score'] = res.pop('score')
                del res['label']  # Remove old label key
                results.append(res)

            # Save checkpoint
            if i > start_index and (i // batch_size) % checkpoint_every == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_parquet(checkpoint_path)

    except KeyboardInterrupt:
        print("\n--- Interrupted by user. Saving progress... ---")
        pd.DataFrame(results).to_parquet(checkpoint_path)
        sys.exit(0)

    # Final Join and Save
    print("--- Finalizing Data ---")
    results_df = pd.DataFrame(results)
    final_df = pd.concat(
        [df.iloc[:len(results_df)].reset_index(drop=True), results_df], axis=1
    )

    final_df.to_parquet(output_path, index=False)

    # Clean up checkpoint after success
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"--- Success! Saved to {output_path} ---")


def main():
    """Parses arguments for this project and initiates the sentiment analysis pipeline."""
    parser = argparse.ArgumentParser(description="RoBERTa Sentiment Analysis")
    parser.add_argument("--input", default="../Data/jan6dataset.parquet")
    parser.add_argument("--output", default="../Data/roberta_results_final.parquet")
    parser.add_argument("--checkpoint", default="../Data/sentiment_checkpoint.parquet")
    parser.add_argument("--batch", type=int, default=64)

    args = parser.parse_args()

    run_twitter_sentiment_analysis(
        input_path=Path(args.input),
        output_path=Path(args.output),
        checkpoint_path=Path(args.checkpoint),
        test_sample=False,
        batch_size=args.batch)


if __name__ == "__main__":
    main()

