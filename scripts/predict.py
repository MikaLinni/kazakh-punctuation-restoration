import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kazpunct.data import load_unlabeled_csv
from kazpunct.dataset import KazPunctDataset
from kazpunct.inference import (
    average_prediction_maps,
    load_checkpoint_model,
    predict_dataset,
    predictions_to_dataframe,
)
from kazpunct.training import get_device


def main(args):
    device = get_device()
    dataframe = load_unlabeled_csv(args.test_file)
    prediction_maps = []

    for checkpoint in args.checkpoint:
        model, tokenizer, metadata = load_checkpoint_model(
            checkpoint,
            device,
            model_name=args.model_name,
        )
        dataset = KazPunctDataset(
            dataframe,
            tokenizer,
            max_length=int(metadata.get("max_length", args.max_length)),
            stride=int(metadata.get("stride", args.stride)),
            is_test=True,
        )
        prediction_maps.append(
            predict_dataset(model, dataset, device, batch_size=args.batch_size)
        )

    merged = average_prediction_maps(prediction_maps)
    thresholds = json.loads(args.thresholds) if args.thresholds else None
    submission = predictions_to_dataframe(
        dataframe,
        merged,
        thresholds=thresholds,
        use_heuristics=args.use_heuristics,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Saved {len(submission)} predictions to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Kazakh punctuation labels.")
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--use-heuristics", action="store_true")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
