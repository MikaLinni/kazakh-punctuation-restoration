from typing import Dict

from sklearn.metrics import f1_score

from kazpunct.constants import ID2LABEL


PUNCTUATION_CLASS_IDS = [1, 2, 3]


def compute_f1_metrics(y_true, y_pred) -> Dict[str, float]:
    macro = f1_score(
        y_true,
        y_pred,
        labels=PUNCTUATION_CLASS_IDS,
        average="macro",
        zero_division=0,
    )
    per_class = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        average=None,
        zero_division=0,
    )

    metrics = {"macro_f1": float(macro)}
    for class_id, score in enumerate(per_class):
        metrics[f"{ID2LABEL[class_id].lower()}_f1"] = float(score)
    return metrics
