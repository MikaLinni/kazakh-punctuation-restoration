import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kazpunct.data import text_to_training_rows


def extract_text(item):
    for key in ("text", "content", "article", "body"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def collect_dataset(dataset_name, split, limit, description):
    dataset = load_dataset(dataset_name, split=split, verification_mode="no_checks")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    rows = []
    for item in tqdm(dataset, desc=description):
        text = extract_text(item)
        if len(text) < 50:
            continue
        rows.extend(text_to_training_rows(text))
    return rows


def main(args):
    rows = []

    if args.max_wiki_articles != 0:
        rows.extend(
            collect_dataset(
                "amandyk/kazakh_wiki_articles",
                "train",
                args.max_wiki_articles,
                "Kazakh Wikipedia",
            )
        )

    if args.max_literature_texts != 0:
        rows.extend(
            collect_dataset(
                "Nothingger/Kazakh-Literature-Collection",
                "train",
                args.max_literature_texts,
                "Kazakh Literature",
            )
        )

    dataframe = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False)
    print(f"Saved {len(dataframe)} rows to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build optional extra Kazakh punctuation data.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-wiki-articles", type=int, default=3000)
    parser.add_argument("--max-literature-texts", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
