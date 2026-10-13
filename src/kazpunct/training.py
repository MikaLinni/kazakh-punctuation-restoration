import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from kazpunct.constants import IGNORE_INDEX
from kazpunct.metrics import compute_f1_metrics


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model, lr: float, head_lr_multiplier: float, weight_decay: float):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        is_backbone = name.startswith("backbone.")
        has_no_decay = any(token in name for token in no_decay)

        if is_backbone and has_no_decay:
            backbone_no_decay.append(parameter)
        elif is_backbone:
            backbone_decay.append(parameter)
        elif has_no_decay:
            head_no_decay.append(parameter)
        else:
            head_decay.append(parameter)

    return AdamW([
        {"params": backbone_decay, "lr": lr, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": lr, "weight_decay": 0.0},
        {
            "params": head_decay,
            "lr": lr * head_lr_multiplier,
            "weight_decay": weight_decay,
        },
        {
            "params": head_no_decay,
            "lr": lr * head_lr_multiplier,
            "weight_decay": 0.0,
        },
    ])


def build_scheduler(optimizer, loader_length, epochs, grad_accum, warmup_ratio):
    total_steps = max(1, loader_length * epochs // max(grad_accum, 1))
    warmup_steps = int(total_steps * warmup_ratio)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    criterion,
    device: torch.device,
    grad_accum: int = 1,
    max_grad_norm: float = 1.0,
    soft_f1_criterion=None,
    use_soft_f1: bool = False,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        question_features = batch["question_features"].to(device)

        amp_context = torch.cuda.amp.autocast() if use_amp else nullcontext()
        with amp_context:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                question_features=question_features,
            )
            if use_soft_f1 and soft_f1_criterion is not None:
                loss = soft_f1_criterion(logits, labels)
            else:
                loss = criterion(logits, labels)
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum

    if len(loader) % grad_accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_model(model, loader: DataLoader, criterion, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0

    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        question_features = batch["question_features"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            question_features=question_features,
        )
        loss = criterion(logits, labels)
        total_loss += loss.item()

        predictions = logits.argmax(dim=-1)
        valid = labels != IGNORE_INDEX
        y_true.extend(labels[valid].cpu().tolist())
        y_pred.extend(predictions[valid].cpu().tolist())

    metrics = compute_f1_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def save_checkpoint(path: str, model, metadata: Dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }, output_path)
