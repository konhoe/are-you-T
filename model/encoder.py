from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "microsoft/deberta-v3-base"
SAFE_TENSORS = True

def load_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name, use_fast=False)

class MBTIEncoder(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            use_safetensors=SAFE_TENSORS,
            trust_remote_code=False,
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        return outputs.last_hidden_state[:, 0]


def build_encoder(model_name: str = MODEL_NAME) -> MBTIEncoder:

    return MBTIEncoder(model_name=model_name)
