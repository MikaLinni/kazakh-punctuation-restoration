import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kazpunct.constants import LABEL2ID
from kazpunct.data import load_labeled_csv
from kazpunct.dataset import KazPunctDataset
from kazpunct.inference import load_checkpoint_model, predict_dataset
from kazpunct.metrics import compute_f1_metrics
from kazpunct.postprocessing import cascade_predict, optimize_thresholds, softmax
from kazpunct.training import get_device


def collect_word_logits(dataframe, merged_logits):
    logits, labels = [], []

    for row_index, row in dataframe.iterrows():
        row_id = str(row["id"]) if "id" in dataframe.columns else str(row_index)
        references = str(row["labels"]).split()
        word_map = merged_logits.get(row_id, {})

        for word_index, label in enumerate(references):
            if word_index not in word_map:
                continue
            logits.append(word_map[word_index])
            labels.append(LABEL2ID[label])

    if not logits:
        raise ValueError("No aligned predictions were produced.")

    return np.asarray(logits), np.asarray(labels, dtype=np.int64)


def main(args):
    device = get_device()
    dataframe = load_labeled_csv(args.data)
    model, tokenizer, metadata = load_checkpoint_model(
        args.checkpoint,
        device,
        model_name=args.model_name,
    )

    dataset = KazPunctDataset(
        dataframe,
        tokenizer,
        max_length=int(metadata.get("max_length", args.max_length)),
        stride=int(metadata.get("stride", args.stride)),
    )
    merged = predict_dataset(model, dataset, device, batch_size=args.batch_size)
    logits, labels = collect_word_logits(dataframe, merged)

    baseline_predictions = logits.argmax(axis=-1)
    baseline_metrics = compute_f1_metrics(labels.tolist(), baseline_predictions.tolist())
    print("Baseline:")
    print(json.dumps(baseline_metrics, indent=2))

    if args.optimize_thresholds:
        thresholds, optimized_macro_f1 = optimize_thresholds(logits, labels)
        probabilities = softmax(logits)
        optimized_predictions = cascade_predict(probabilities, thresholds)
        optimized_metrics = compute_f1_metrics(labels.tolist(), optimized_predictions.tolist())
        print("Optimized thresholds:")
        print(json.dumps(thresholds, indent=2))
        print("Optimized metrics:")
        print(json.dumps(optimized_metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Kazakh punctuation checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--optimize-thresholds", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
