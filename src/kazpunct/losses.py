import torch
import torch.nn as nn
import torch.nn.functional as F

from kazpunct.constants import IGNORE_INDEX, NUM_LABELS


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha=None,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if alpha is None:
            alpha = [1.0] * NUM_LABELS
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))

    def forward(self, logits, targets):
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)

        mask = targets != self.ignore_index
        logits = logits[mask]
        targets = targets[mask]

        if targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        probabilities = F.softmax(logits, dim=-1)
        p_t = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = self.alpha[targets]
        loss = -alpha_t * (1.0 - p_t) ** self.gamma * torch.log(p_t + 1e-8)
        return loss.mean()


class SoftF1Loss(nn.Module):
    """Differentiable macro-F1 surrogate over punctuation classes, excluding O."""

    def __init__(self, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)

        mask = targets != self.ignore_index
        logits = logits[mask]
        targets = targets[mask]

        if targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        probabilities = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(targets, NUM_LABELS).float()

        f1_values = []
        for class_id in range(1, NUM_LABELS):
            true_positive = (probabilities[:, class_id] * one_hot[:, class_id]).sum()
            false_positive = (
                probabilities[:, class_id] * (1.0 - one_hot[:, class_id])
            ).sum()
            false_negative = (
                (1.0 - probabilities[:, class_id]) * one_hot[:, class_id]
            ).sum()
            f1 = 2.0 * true_positive / (
                2.0 * true_positive + false_positive + false_negative + 1e-8
            )
            f1_values.append(f1)

        return 1.0 - torch.stack(f1_values).mean()
