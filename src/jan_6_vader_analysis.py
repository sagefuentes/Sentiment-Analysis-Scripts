"""Runs VADER sentiment analysis on a Twitter dataset.

Reads tweet text from a Parquet file and scores each tweet using VADER
(Valence Aware Dictionary and sEntiment Reasoner), a lexicon and rule-based
sentiment analyser designed specifically for social media text.

For each tweet, four scores are produced: compound, positive, neutral, and
negative. The compound score is a normalised value in [-1.0, 1.0] and is the
primary signal used to derive a human-readable sentiment label. Following
standard VADER convention, the label thresholds are:

    positive  compound >= 0.05
    negative  compound <= -0.05
    neutral   -0.05 < compound < 0.05

Typical usage (local):
    python jan_6_vader_analysis.py --input data/tweets.parquet

Typical usage (Google Colab):
    !python /content/drive/MyDrive/.../jan_6_vader_analysis.py \\
        --input  /content/drive/MyDrive/.../jan6Dataset.parquet \\
        --output /content/drive/MyDrive/.../vader_results_final.parquet
"""

import argparse
from pathlib import Path

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import nltk


# Neutral label boundaries as defined by the original VADER paper.
# Tweets with a compound score strictly between these thresholds are
# labelled neutral; everything else is positive or negative.
_NEUTRAL_LOW: float = -0.05
_NEUTRAL_HIGH: float = 0.05


def _assign_label(compound: float) -> str:
    """Converts a VADER compound score to a human-readable sentiment label.

    Applies the standard VADER thresholds recommended by Hutto & Gilbert
    (2014): compound >= 0.05 is positive, compound <= -0.05 is negative,
    and anything in between is neutral.

    Args:
        compound: Normalised compound sentiment score in [-1.0, 1.0].

    Returns:
        One of ``"positive"``, ``"negative"``, or ``"neutral"``.
    """
    if compound >= _NEUTRAL_HIGH:
        return "positive"
    if compound <= _NEUTRAL_LOW:
        return "negative"
    return "neutral"


def run_vader_sentiment_analysis(
        input_path: Path,
        output_path: Path,
        test_sample: bool = False,
) -> None:
    """Scores tweet sentiment with VADER and writes results to Parquet.

    Loads the source dataset, applies the VADER SentimentIntensityAnalyzer
    to every tweet, and writes a new Parquet file that contains all original
    columns plus five new ones:

    - ``vader_compound``  — normalised score in [-1.0, 1.0]; the primary
      signal for overall sentiment strength and direction.
    - ``vader_positive``  — proportion of text that is positive; in [0.0, 1.0].
    - ``vader_neutral``   — proportion of text that is neutral; in [0.0, 1.0].
    - ``vader_negative``  — proportion of text that is negative; in [0.0, 1.0].
    - ``vader_label``     — human-readable label derived from ``vader_compound``
      using the standard VADER thresholds.

    VADER requires the ``vader_lexicon`` resource from NLTK. This function
    downloads it automatically if it is not already present.

    Args:
        input_path: Parquet file containing the source tweets. Must include
            a ``sentence_text`` column.
        output_path: Destination Parquet file for the final scored dataset.
        test_sample: When True, randomly samples 1 000 rows before scoring.
            Useful for quick end-to-end validation. Defaults to False.

    Raises:
        FileNotFoundError: If ``input_path`` does not point to an existing
            file.
        ValueError: If the loaded dataframe does not contain a
            ``sentence_text`` column.
    """

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Download the VADER lexicon if it is not already cached. The quiet flag
    # suppresses the NLTK download banner so output stays clean on reruns.
    nltk.download("vader_lexicon", quiet=True)

    print(f"--- Loading data: {input_path.name} ---")
    df = pd.read_parquet(input_path)

    if "sentence_text" not in df.columns:
        raise ValueError("Input dataframe must contain a 'sentence_text' column.")

    if test_sample:
        df = df.sample(n=1000).reset_index(drop=True)

    # Replace NaN values with empty strings so the analyser never receives a
    # non-string input, which would raise a TypeError at score time.
    texts = df["sentence_text"].fillna("").tolist()

    analyzer = SentimentIntensityAnalyzer()

    # VADER is fast enough that the entire dataset can be scored in one loop
    # without batching or checkpointing. tqdm provides a progress bar so the
    # run does not appear to hang on very large datasets.
    print(f"--- Scoring {len(texts)} tweets with VADER ---")

    records = []
    for text in tqdm(texts):
        scores = analyzer.polarity_scores(text)
        records.append({
            "vader_compound": scores["compound"],
            "vader_positive": scores["pos"],
            "vader_neutral":  scores["neu"],
            "vader_negative": scores["neg"],
            "vader_label":    _assign_label(scores["compound"]),
        })

    results_df = pd.DataFrame(records)

    # Join results back to the original dataframe column-wise. reset_index
    # ensures alignment is purely positional and is not affected by any
    # non-default index the source dataframe may carry.
    final_df = pd.concat(
        [df.reset_index(drop=True), results_df],
        axis=1,
    )

    # Print a brief distribution summary so the analyst can sanity-check the
    # results before committing the file to disk.
    label_counts = final_df["vader_label"].value_counts()
    total = len(final_df)
    print("\n--- Sentiment distribution ---")
    for label, count in label_counts.items():
        print(f"    {label:<10} {count:>7,}  ({count / total:.1%})")

    print(f"\n--- Compound score summary ---")
    print(final_df["vader_compound"].describe().to_string())

    final_df.to_parquet(output_path, index=False)
    print(f"\n--- Success! Results saved to {output_path} ---")


def main() -> None:
    """Parses command-line arguments and runs the VADER sentiment pipeline."""
    parser = argparse.ArgumentParser(
        description="VADER sentiment analysis for the January 6th Twitter dataset."
    )
    parser.add_argument(
        "--input",
        default="../data/interim/jan6Dataset.parquet",
        help="Path to the input Parquet file. Must contain a 'sentence_text' column.",
    )
    parser.add_argument(
        "--output",
        default="../data/processed/vader_results_final.parquet",
        help="Destination path for the final scored Parquet file.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="When set, scores a random 1 000-row sample instead of the full dataset.",
    )

    args = parser.parse_args()

    run_vader_sentiment_analysis(
        input_path=Path(args.input),
        output_path=Path(args.output),
        test_sample=args.test,
    )


if __name__ == "__main__":
    main()