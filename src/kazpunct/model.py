import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from kazpunct.constants import NUM_LABELS


class KazPunctModel(nn.Module):
    """Transformer + BiLSTM + question features + token classifier."""

    def __init__(
        self,
        model_name: str = "xlm-roberta-large",
        num_labels: int = NUM_LABELS,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        question_feature_dim: int = 3,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        head_input = lstm_hidden * 2 + question_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(head_input, head_input // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_input // 2, num_labels),
        )

        self.model_name = model_name
        self.question_feature_dim = question_feature_dim

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, question_features=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence = self.dropout(outputs.last_hidden_state)

        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.size(1),
        )

        if question_features is None:
            question_features = torch.zeros(
                *lstm_output.shape[:2],
                self.question_feature_dim,
                dtype=lstm_output.dtype,
                device=lstm_output.device,
            )

        combined = torch.cat([lstm_output, question_features], dim=-1)
        return self.classifier(combined)
