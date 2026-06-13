# Sentiment-Analysis Scripts

This repo contains scripts I have used for companion project.

## January 6th Twitter Sentiment Analysis — Python Pipeline

Python scripts for sentiment analysis of the January 6th 'Capitol' Twitter dataset,
companion to the [repository for that analysis](https://github.com/sagefuentes/Jan-6-Twitter-Data).
Live version: [here](https://sagefuentes.github.io/Jan-6-Twitter-Data/)

Both scripts read from and write to Parquet, fitting into a broader pipeline
where data cleaning and visualization are handled in R via `arrow`.

---

### Scripts

#### `jan_6_vader_analysis.py`

Scores each tweet using VADER (Valence Aware Dictionary and sEntiment
Reasoner), a lexicon and rule-based analyser designed specifically for social
media text. VADER was run in Python rather than R due to the R package being
removed from CRAN; the archived R version was also substantially slower
(estimated 3+ hours versus minutes in Python).

Each tweet receives four scores:

| Column | Description |
|---|---|
| `vader_compound` | Normalized score in \[-1.0, 1.0\]; primary sentiment signal |
| `vader_positive` | Proportion of text that is positive |
| `vader_neutral` | Proportion of text that is neutral |
| `vader_negative` | Proportion of text that is negative |
| `vader_label` | Human-readable label derived from compound score |

Label thresholds follow the standard VADER convention (Hutto & Gilbert, 2014):
`compound >= 0.05` → positive, `compound <= -0.05` → negative, everything
in between → neutral.

**Usage:**
```bash
python jan_6_vader_analysis.py --input data/interim/jan6Dataset.parquet

# Score a 1,000-row sample for validation
python jan_6_vader_analysis.py --input data/interim/jan6Dataset.parquet --test
```

---

#### `jan_6_roberta_analysis.py`

Classifies each tweet as negative, neutral, or positive using the
`cardiffnlp/twitter-roberta-base-sentiment-latest` checkpoint from Hugging
Face — a RoBERTa model fine-tuned on approximately 124 million tweets.
Unlike VADER, RoBERTa evaluates full sentence context, making it more
sensitive to negation, sarcasm, and syntactic modifiers.

Each tweet receives two outputs:

| Column | Description |
|---|---|
| `roberta_label` | Predicted sentiment class: negative, neutral, or positive |
| `roberta_score` | Model confidence for the assigned label, in \[0.0, 1.0\] |

Supports crash-safe checkpointing so long runs can be resumed without
reprocessing already-classified tweets. CPU thermal load is managed via
configurable thread limits and inter-batch sleep intervals. GPU inference
is used automatically if CUDA is available.

**Usage:**
```bash
python jan_6_roberta_analysis.py --input data/interim/jan6Dataset.parquet

# Validate end-to-end on a 1,000-row sample
python jan_6_roberta_analysis.py --input data/interim/jan6Dataset.parquet --test

# Tune for thermal management on CPU
python jan_6_roberta_analysis.py \
    --input data/interim/jan6Dataset.parquet \
    --threads 2 \
    --sleep 0.2
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--input` | `data/interim/jan6Dataset.parquet` | Input Parquet file |
| `--output` | `data/processed/roberta_results_final.parquet` | Output path |
| `--checkpoint` | `data/interim/sentiment_checkpoint.parquet` | Resume file |
| `--batch` | 64 | Tweets per inference batch |
| `--threads` | 4 | PyTorch intra-op threads; reduce to lower CPU temperature |
| `--sleep` | 0.1 | Seconds between batches for CPU cooling; set to 0 to disable |

---

## Setup

Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
# Install uv if not already installed
pip install uv

# Install dependencies from the lockfile
uv sync
```

To run with GPU support, ensure a CUDA-compatible PyTorch build is installed.
The RoBERTa script will automatically detect and use a GPU if available,
falling back to CPU otherwise.

---

## Data

Input and output data files are not included in this repository. The input
Parquet file is produced by the R cleaning pipeline in the companion
repository. Processed outputs are awaiting to be uploaded.

The input file must contain a `sentence_text` column — the syntactically
preserved tweet text with URLs and mentions removed but casing and punctuation
intact.

---

## References

Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model
for sentiment analysis of social media text. *Proceedings of the Eighth
International AAAI Conference on Weblogs and Social Media (ICWSM-14).*
https://ojs.aaai.org/index.php/ICWSM/article/view/14550

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis,
M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized
BERT pretraining approach. *arXiv.* https://arxiv.org/abs/1907.11692
