from __future__ import annotations

import torch
from torch import nn

from .classifier import MLPClassifier
from .encoder import MODEL_NAME, MBTIEncoder, build_encoder

class MBTIModel(nn.Module):
    def __init__(self, encoder: MBTIEncoder, hidden_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = MLPClassifier(input_size=hidden_size, hidden_size=hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outs)
        return logits


def build_model() -> MBTIModel:
    encoder = build_encoder(MODEL_NAME)
    hidden = encoder.backbone.config.hidden_size
    return MBTIModel(encoder=encoder, hidden_size=hidden)
