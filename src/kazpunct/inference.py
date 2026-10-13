from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from kazpunct.constants import ID2LABEL
from kazpunct.model import KazPunctModel
from kazpunct.postprocessing import apply_kazakh_heuristics, cascade_predict, softmax


def position_weight(position: int, chunk_length: int) -> float:
    if chunk_length <= 1:
        return 1.0
    center = (chunk_length - 1) / 2.0
    return 1.0 - 0.4 * abs(position - center) / max(center, 1e-8)


@torch.no_grad()
def predict_dataset(model, dataset, device: torch.device, batch_size: int = 16):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    accumulator = defaultdict(lambda: defaultdict(list))
    chunk_index = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        question_features = batch["question_features"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            question_features=question_features,
        ).detach().cpu().float().numpy()

        for batch_index in range(logits.shape[0]):
            metadata = dataset.get_meta(chunk_index + batch_index)
            row_id = metadata["row_id"]
            word_ids = metadata["word_ids"]
            first_flags = metadata["first_subtoken"]
            chunk_length = metadata["chunk_token_end"] - metadata["chunk_token_start"]

            for position, (word_id, is_first) in enumerate(zip(word_ids, first_flags)):
                if word_id < 0 or not is_first:
                    continue
                weight = position_weight(max(position - 1, 0), chunk_length)
                accumulator[row_id][word_id].append((logits[batch_index, position], weight))

        chunk_index += logits.shape[0]

    merged = {}
    for row_id, word_map in accumulator.items():
        merged[row_id] = {}
        for word_id, values in word_map.items():
            vectors = np.asarray([value[0] for value in values])
            weights = np.asarray([value[1] for value in values])
            merged[row_id][word_id] = np.average(vectors, axis=0, weights=weights)
    return merged


def average_prediction_maps(prediction_maps: List[Dict]) -> Dict:
    accumulator = defaultdict(lambda: defaultdict(list))
    for prediction_map in prediction_maps:
        for row_id, word_map in prediction_map.items():
            for word_id, logits in word_map.items():
                accumulator[row_id][word_id].append(logits)

    merged = {}
    for row_id, word_map in accumulator.items():
        merged[row_id] = {
            word_id: np.mean(np.stack(vectors, axis=0), axis=0)
            for word_id, vectors in word_map.items()
        }
    return merged


def predictions_to_dataframe(
    dataframe: pd.DataFrame,
    merged_logits: Dict,
    thresholds: Optional[Dict[str, float]] = None,
    use_heuristics: bool = False,
) -> pd.DataFrame:
    rows = []

    for _, row in dataframe.iterrows():
        row_id = str(row["id"])
        words = str(row["input_text"]).split()
        word_map = merged_logits.get(row_id, {})
        labels = []

        for word_index in range(len(words)):
            logits = word_map.get(word_index)
            if logits is None:
                labels.append("O")
                continue

            if thresholds is None:
                class_id = int(np.argmax(logits))
            else:
                probabilities = softmax(np.asarray(logits)[None, :])
                class_id = int(cascade_predict(probabilities, thresholds)[0])
            labels.append(ID2LABEL[class_id])

        if use_heuristics:
            labels = apply_kazakh_heuristics(words, labels)

        rows.append({
            "id": row_id,
            "labels": " ".join(labels),
        })

    return pd.DataFrame(rows)


def load_checkpoint_model(checkpoint_path: str, device: torch.device, model_name: Optional[str] = None):
    payload = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = payload.get("metadata", {})
        model_name = model_name or metadata.get("model_name")
        model_kwargs = metadata.get("model_kwargs", {})
    else:
        state_dict = payload
        metadata = {}
        model_kwargs = {}

    if not model_name:
        raise ValueError(
            "Checkpoint does not contain model metadata. Pass --model-name explicitly."
        )

    model = KazPunctModel(model_name=model_name, **model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer, metadata
