from typing import Optional

import torch
from torch.utils.data import Dataset

from kazpunct.constants import (
    IGNORE_INDEX,
    LABEL2ID,
    QUESTION_PARTICLES,
    QUESTION_WORDS,
)


class KazPunctDataset(Dataset):
    """Tokenize word-level punctuation labels into fixed overlapping windows."""

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length: int = 256,
        stride: int = 128,
        is_test: bool = False,
    ):
        if max_length < 8:
            raise ValueError("max_length must be at least 8")
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.is_test = is_test
        self.chunks = []

        for row_index, row in dataframe.iterrows():
            words = str(row["input_text"]).split()
            row_id = str(row["id"]) if "id" in dataframe.columns else str(row_index)

            if is_test:
                labels = ["O"] * len(words)
            else:
                labels = str(row["labels"]).split()
                if len(words) != len(labels):
                    raise ValueError(
                        f"Word/label length mismatch for row {row_id}: "
                        f"{len(words)} words vs {len(labels)} labels"
                    )

            token_ids = []
            label_ids = []
            word_ids = []
            first_subtoken = []
            question_features = []

            for word_index, (word, label) in enumerate(zip(words, labels)):
                subtoken_ids = tokenizer.encode(word, add_special_tokens=False)
                if not subtoken_ids:
                    continue

                word_lower = word.lower()
                features = [
                    1.0 if word_lower in QUESTION_PARTICLES else 0.0,
                    1.0 if word_lower in QUESTION_WORDS else 0.0,
                    1.0
                    if any(
                        word_lower.endswith(particle)
                        and len(word_lower) > len(particle)
                        for particle in QUESTION_PARTICLES
                    )
                    else 0.0,
                ]

                token_ids.extend(subtoken_ids)
                label_ids.append(LABEL2ID.get(label, 0))
                label_ids.extend([IGNORE_INDEX] * (len(subtoken_ids) - 1))
                word_ids.extend([word_index] * len(subtoken_ids))
                first_subtoken.extend([True] + [False] * (len(subtoken_ids) - 1))
                question_features.extend([features] * len(subtoken_ids))

            effective_length = max_length - 2
            start = 0

            while start < len(token_ids):
                end = min(start + effective_length, len(token_ids))

                chunk_ids = token_ids[start:end]
                chunk_labels = label_ids[start:end]
                chunk_word_ids = word_ids[start:end]
                chunk_first = first_subtoken[start:end]
                chunk_features = question_features[start:end]

                ids = [tokenizer.cls_token_id] + chunk_ids + [tokenizer.sep_token_id]
                attention_mask = [1] * len(ids)
                labels_with_special = [IGNORE_INDEX] + chunk_labels + [IGNORE_INDEX]
                word_ids_with_special = [-1] + chunk_word_ids + [-1]
                first_with_special = [False] + chunk_first + [False]
                features_with_special = [[0.0, 0.0, 0.0]] + chunk_features + [[0.0, 0.0, 0.0]]

                pad = max_length - len(ids)
                ids += [tokenizer.pad_token_id] * pad
                attention_mask += [0] * pad
                labels_with_special += [IGNORE_INDEX] * pad
                word_ids_with_special += [-1] * pad
                first_with_special += [False] * pad
                features_with_special += [[0.0, 0.0, 0.0]] * pad

                self.chunks.append({
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels_with_special, dtype=torch.long),
                    "question_features": torch.tensor(features_with_special, dtype=torch.float),
                    "row_id": row_id,
                    "word_ids": word_ids_with_special,
                    "first_subtoken": first_with_special,
                    "chunk_token_start": start,
                    "chunk_token_end": end,
                    "n_words": len(words),
                })

                if end == len(token_ids):
                    break
                start += stride

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        chunk = self.chunks[index]
        return {
            "input_ids": chunk["input_ids"],
            "attention_mask": chunk["attention_mask"],
            "labels": chunk["labels"],
            "question_features": chunk["question_features"],
        }

    def get_meta(self, index):
        return self.chunks[index]
