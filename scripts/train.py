import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kazpunct.data import load_labeled_csv
from kazpunct.dataset import KazPunctDataset
from kazpunct.losses import FocalLoss, SoftF1Loss
from kazpunct.model import KazPunctModel
from kazpunct.training import (
    build_optimizer,
    build_scheduler,
    evaluate_model,
    get_device,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


def create_splits(dataframe, n_folds, seed):
    has_question = dataframe["labels"].apply(lambda value: int("QUESTION" in str(value).split()))
    min_class = has_question.value_counts().min() if has_question.nunique() > 1 else 0

    if min_class >= n_folds:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(splitter.split(dataframe, has_question))

    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(dataframe))


def train_fold(args, dataframe, train_indices, val_indices, fold_index):
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_df = dataframe.iloc[train_indices].reset_index(drop=True)
    val_df = dataframe.iloc[val_indices].reset_index(drop=True)

    train_dataset = KazPunctDataset(
        train_df,
        tokenizer,
        max_length=args.max_length,
        stride=args.stride,
    )
    val_dataset = KazPunctDataset(
        val_df,
        tokenizer,
        max_length=args.max_length,
        stride=args.stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_kwargs = {
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "dropout": args.dropout,
        "question_feature_dim": 3,
    }
    model = KazPunctModel(model_name=args.model_name, **model_kwargs).to(device)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    criterion = FocalLoss(
        gamma=args.focal_gamma,
        alpha=args.class_weights,
    ).to(device)
    soft_f1 = SoftF1Loss().to(device)

    optimizer = build_optimizer(
        model,
        lr=args.learning_rate,
        head_lr_multiplier=args.head_lr_multiplier,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        loader_length=len(train_loader),
        epochs=args.epochs,
        grad_accum=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
    )

    best_macro_f1 = -1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"best_fold{fold_index}.pt"

    for epoch in range(1, args.epochs + 1):
        use_soft_f1 = args.soft_f1_after > 0 and epoch >= args.soft_f1_after
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            grad_accum=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            soft_f1_criterion=soft_f1,
            use_soft_f1=use_soft_f1,
        )
        metrics = evaluate_model(model, val_loader, criterion, device)

        print(
            f"Fold {fold_index} | epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={metrics['loss']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f} | "
            f"comma={metrics['comma_f1']:.4f} | "
            f"period={metrics['period_f1']:.4f} | "
            f"question={metrics['question_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            metadata = {
                "model_name": args.model_name,
                "model_kwargs": model_kwargs,
                "max_length": args.max_length,
                "stride": args.stride,
                "fold": fold_index,
                "best_macro_f1": best_macro_f1,
            }
            save_checkpoint(str(checkpoint_path), model, metadata)
            print(f"Saved new best checkpoint: {checkpoint_path}")

    return best_macro_f1


def main(args):
    set_seed(args.seed)
    dataframe = load_labeled_csv(args.train_file)

    if args.extra_train_file:
        extra = load_labeled_csv(args.extra_train_file)
        dataframe = pd.concat([dataframe, extra], ignore_index=True)
        dataframe = dataframe.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    splits = create_splits(dataframe, args.folds, args.seed)
    fold_indices = range(args.folds) if args.fold is None else [args.fold]

    scores = {}
    for fold_index in fold_indices:
        train_indices, val_indices = splits[fold_index]
        score = train_fold(args, dataframe, train_indices, val_indices, fold_index)
        scores[str(fold_index)] = score

    print(json.dumps({"fold_macro_f1": scores}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train Kazakh punctuation restoration model.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--extra-train-file", default=None)
    parser.add_argument("--output-dir", default="checkpoints/xlm_roberta")
    parser.add_argument("--model-name", default="xlm-roberta-large")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-lr-multiplier", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--class-weights",
        type=float,
        nargs=4,
        default=[0.05, 1.5, 2.5, 20.0],
        metavar=("O", "COMMA", "PERIOD", "QUESTION"),
    )
    parser.add_argument("--soft-f1-after", type=int, default=0)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
