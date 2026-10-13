# Data Format

The repository does not distribute the original hackathon train or test files.

## Labeled data

Training and evaluation CSV files must contain at least these columns:

| Column | Description |
| --- | --- |
| `id` | Unique row identifier. Optional for training, recommended for reproducibility. |
| `input_text` | Kazakh text with whitespace-separated words and without the target punctuation labels inserted. |
| `labels` | One whitespace-separated label per word. |

Supported labels:

- `O`
- `COMMA`
- `PERIOD`
- `QUESTION`

The number of labels must equal the number of words in `input_text`.

## Inference data

Inference CSV files require:

| Column | Description |
| --- | --- |
| `id` | Unique row identifier. |
| `input_text` | Kazakh text to label. |

The prediction scripts return a CSV with:

```text
id,labels
```

where `labels` contains one predicted label per input word.
