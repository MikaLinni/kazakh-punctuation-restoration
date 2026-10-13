from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score

from kazpunct.constants import ID2LABEL, QUESTION_PARTICLES, QUESTION_WORDS


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def cascade_predict(probabilities: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    predictions = np.zeros(len(probabilities), dtype=np.int64)
    for class_id, key in ((3, "QUESTION"), (2, "PERIOD"), (1, "COMMA")):
        threshold = float(thresholds[key])
        predictions = np.where(
            (predictions == 0) & (probabilities[:, class_id] > threshold),
            class_id,
            predictions,
        )
    return predictions


def optimize_thresholds(logits: np.ndarray, labels: np.ndarray) -> Tuple[Dict[str, float], float]:
    probabilities = softmax(logits)

    def objective(values):
        values = np.clip(values, 0.01, 0.98)
        thresholds = {
            "COMMA": float(values[0]),
            "PERIOD": float(values[1]),
            "QUESTION": float(values[2]),
        }
        predictions = cascade_predict(probabilities, thresholds)
        scores = [
            f1_score(
                (labels == class_id).astype(int),
                (predictions == class_id).astype(int),
                zero_division=0,
            )
            for class_id in (1, 2, 3)
        ]
        return -float(np.mean(scores))

    initial = np.array([0.25, 0.35, 0.12], dtype=float)
    result = minimize(
        objective,
        initial,
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4},
    )
    best = np.clip(result.x, 0.01, 0.98)
    thresholds = {
        "COMMA": float(best[0]),
        "PERIOD": float(best[1]),
        "QUESTION": float(best[2]),
    }
    return thresholds, -float(result.fun)


def apply_kazakh_heuristics(words: List[str], labels: List[str]) -> List[str]:
    labels = list(labels)
    lowered = [word.lower() for word in words]

    if not words:
        return labels

    if labels[-1] == "O":
        if any(word in QUESTION_PARTICLES or word in QUESTION_WORDS for word in lowered):
            labels[-1] = "QUESTION"
        else:
            labels[-1] = "PERIOD"

    for index, word in enumerate(lowered):
        if index >= 1 and word in QUESTION_PARTICLES and labels[index] in {"O", "PERIOD"}:
            labels[index] = "QUESTION"

    for index in range(1, len(labels)):
        if (
            labels[index] in {"PERIOD", "QUESTION"}
            and labels[index - 1] in {"PERIOD", "QUESTION"}
        ):
            labels[index - 1] = "O"

    return labels
