import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from kazpunct.constants import LABELS


def validate_labeled_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    required = {"input_text", "labels"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = dataframe.copy()
    cleaned = cleaned.dropna(subset=["input_text", "labels"]).reset_index(drop=True)

    bad_rows = []
    allowed = set(LABELS)
    for index, row in cleaned.iterrows():
        words = str(row["input_text"]).split()
        labels = str(row["labels"]).split()
        if len(words) != len(labels) or any(label not in allowed for label in labels):
            bad_rows.append(index)

    if bad_rows:
        preview = bad_rows[:10]
        raise ValueError(
            f"Found {len(bad_rows)} invalid rows. First invalid indices: {preview}"
        )

    return cleaned


def load_labeled_csv(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Labeled CSV not found: {path}")
    return validate_labeled_frame(pd.read_csv(path))


def load_unlabeled_csv(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Inference CSV not found: {path}")

    dataframe = pd.read_csv(path)
    required = {"id", "input_text"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return dataframe.dropna(subset=["id", "input_text"]).reset_index(drop=True)


def label_counts(dataframe: pd.DataFrame) -> Counter:
    counts = Counter()
    for value in dataframe["labels"]:
        counts.update(str(value).split())
    return counts


def extract_sentence_labels(sentence: str) -> List[Tuple[str, str]]:
    """Convert punctuated Kazakh text into word/label pairs."""
    results = []

    for token in sentence.split():
        token = token.strip()
        if not token:
            continue

        token = re.sub(r'^[«»"\'„“”()\[\]{}—–\-]+', "", token)
        label = "O"
        clean = token

        while clean and clean[-1] in '?!.,;:…–—»"\')]}':
            char = clean[-1]
            if char == "?":
                label = "QUESTION"
            elif char in ".!…":
                if label == "O":
                    label = "PERIOD"
            elif char in ",;:":
                if label == "O":
                    label = "COMMA"
            clean = clean[:-1]

        clean = re.sub(r'^[«»"\'„“”()\[\]{}]+', "", clean)
        clean = re.sub(r'[«»"\'„“”()\[\]{}]+$', "", clean)
        clean = clean.lower().strip()

        if not clean or not re.search(r"\w", clean):
            continue

        results.append((clean, label))

    return results


def text_to_training_rows(text: str, max_words_per_row: int = 60) -> List[dict]:
    """Convert raw punctuated Kazakh text into sequence-labeling rows."""
    rows = []
    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
    buffer_words, buffer_labels = [], []

    def flush(words: Iterable[str], labels: Iterable[str]) -> None:
        words = list(words)
        labels = list(labels)
        if len(words) >= 4:
            rows.append({
                "input_text": " ".join(words),
                "labels": " ".join(labels),
            })

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue

        pairs = extract_sentence_labels(sentence)
        if not pairs:
            continue

        words, labels = map(list, zip(*pairs))

        if len(buffer_words) + len(words) > max_words_per_row and buffer_words:
            flush(buffer_words, buffer_labels)
            buffer_words, buffer_labels = [], []

        buffer_words.extend(words)
        buffer_labels.extend(labels)

        if labels[-1] in {"PERIOD", "QUESTION"}:
            flush(buffer_words, buffer_labels)
            buffer_words, buffer_labels = [], []

    if buffer_words:
        flush(buffer_words, buffer_labels)

    return rows
