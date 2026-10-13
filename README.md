# Kazakh Punctuation Restoration

A transformer-based punctuation restoration pipeline for Kazakh text, originally developed during a hackathon and later refactored into a reusable ML/NLP project.

The task is formulated as word-level sequence labeling with four labels:

- `O` — no punctuation after the word
- `COMMA` — comma-like boundary
- `PERIOD` — sentence-ending boundary
- `QUESTION` — question boundary

## Why this task is challenging

The supplied labeled sample was small and strongly imbalanced. In the original development notebook, the 500-row labeled sample contained only five `QUESTION` labels. The pipeline therefore combines a multilingual transformer, recurrent context modeling, class-aware loss, optional extra data, threshold tuning, and Kazakh-specific post-processing.

## Approach

The main pipeline uses:

- `xlm-roberta-large` as the default transformer backbone;
- optional `nur-dev/roberta-kaz-large` experiments;
- a bidirectional LSTM classification head;
- word-to-subword label alignment;
- overlapping token windows for longer sequences;
- Kazakh question-particle and question-word features;
- focal loss with stronger weighting for rare punctuation classes;
- optional Soft-F1 optimization in later epochs;
- per-class threshold optimization with Nelder-Mead;
- center-weighted merging of overlapping predictions;
- optional ensemble averaging and linguistic post-processing.

## Pipeline

```text
Raw Kazakh text
      │
      ▼
Word / subword alignment
      │
      ▼
Transformer backbone
(XLM-R / KazRoBERTa)
      │
      ▼
Bidirectional LSTM
      │
      ├── Kazakh question features
      ▼
Token classifier
      │
      ▼
Center-weighted window merging
      │
      ▼
Class-specific thresholds
      │
      ▼
Kazakh post-processing rules
      │
      ▼
O / COMMA / PERIOD / QUESTION
```

## Recorded development result

One recorded XLM-RoBERTa experiment in the original hackathon notebooks reached a **Macro F1 of 0.7983** on its development split, with the following per-class values at that checkpoint:

| Metric | Score |
| --- | ---: |
| Macro F1 | 0.7983 |
| COMMA F1 | 0.921 |
| PERIOD F1 | 0.974 |
| QUESTION F1 | 0.500 |

This is a development-split result from the original experimentation workflow, not a claimed leaderboard score. The refactored repository may produce different results depending on the data, split, backbone, and training configuration.

## Repository Structure

```text
kazakh-punctuation-restoration/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── xlm_roberta.yaml
│
├── docs/
│   └── data_format.md
│
├── scripts/
│   ├── build_extra_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
└── src/
    └── kazpunct/
        ├── __init__.py
        ├── constants.py
        ├── data.py
        ├── dataset.py
        ├── model.py
        ├── losses.py
        ├── metrics.py
        ├── training.py
        ├── inference.py
        └── postprocessing.py
```

## Installation

```bash
git clone git@github.com:MikaLinni/kazakh-punctuation-restoration.git
cd kazakh-punctuation-restoration

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Format

Competition data are not included in this repository.

A labeled CSV is expected to contain:

```text
id,input_text,labels
<row id>,<space-separated words>,<space-separated labels>
```

The number of labels must match the number of whitespace-separated words.

For inference, only `id` and `input_text` are required.

See `docs/data_format.md` for details.

## Optional Extra Training Data

The original experiments explored automatically converting public punctuated Kazakh text into additional sequence-labeling examples.

The helper script can build extra training rows from public Hugging Face datasets:

```bash
python scripts/build_extra_data.py \
  --output data/extra_train.csv \
  --max-wiki-articles 3000 \
  --max-literature-texts 2000
```

No downloaded corpus is committed to Git.

## Training

Train a single fold with the default XLM-RoBERTa configuration:

```bash
python scripts/train.py \
  --train-file /path/to/train.csv \
  --output-dir checkpoints/xlm_roberta \
  --folds 5 \
  --fold 0
```

Include extra authorized/public training data:

```bash
python scripts/train.py \
  --train-file /path/to/train.csv \
  --extra-train-file data/extra_train.csv \
  --output-dir checkpoints/xlm_roberta \
  --folds 5 \
  --fold 0
```

Enable the optional Soft-F1 stage from epoch 3:

```bash
python scripts/train.py \
  --train-file /path/to/train.csv \
  --output-dir checkpoints/xlm_roberta \
  --soft-f1-after 3
```

## Evaluation and Threshold Optimization

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/xlm_roberta/best_fold0.pt \
  --data /path/to/validation.csv \
  --optimize-thresholds
```

The script reports Macro F1 over `COMMA`, `PERIOD`, and `QUESTION`, plus per-class F1 values. With `--optimize-thresholds`, it also prints class-specific probability thresholds.

## Prediction

Generate a submission-style CSV:

```bash
python scripts/predict.py \
  --checkpoint checkpoints/xlm_roberta/best_fold0.pt \
  --test-file /path/to/test.csv \
  --output submissions/submission.csv \
  --thresholds '{"COMMA": 0.25, "PERIOD": 0.35, "QUESTION": 0.12}' \
  --use-heuristics
```

Multiple checkpoints can be supplied for logit averaging:

```bash
python scripts/predict.py \
  --checkpoint checkpoints/model_a.pt checkpoints/model_b.pt \
  --test-file /path/to/test.csv \
  --output submissions/ensemble.csv
```

## Data Availability

Hackathon train/test files and generated submissions are intentionally excluded from the repository. The project contains reusable source code, configuration, and documentation only.

## Project History

This repository is a cleaned and modularized version of an iterative hackathon workflow. The original notebooks explored several variants, including XLM-RoBERTa, KazRoBERTa, focal loss, Soft-F1, threshold tuning, temperature scaling, heuristic corrections, and ensemble inference.

## License

A software license has not yet been assigned to this repository.
