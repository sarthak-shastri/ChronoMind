"""Frozen projected CLIP text embeddings with explicit checkpoint loading."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class CLIPTextFeatureExtractor(nn.Module):
    def __init__(self, text_model: nn.Module, tokenizer, *, max_length: int = 77):
        super().__init__()
        if max_length < 2 or max_length > 77:
            raise ValueError("max_length must be between 2 and CLIP's 77-token context")
        self.text_model, self.tokenizer, self.max_length = text_model, tokenizer, max_length
        self.text_model.requires_grad_(False)
        self.text_model.eval()

    @classmethod
    def from_pretrained(cls, model_name="openai/clip-vit-base-patch32", *, revision=None, **kwargs):
        try:
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers to load pretrained CLIP text weights") from exc
        load = {} if revision is None else {"revision": revision}
        tokenizer = CLIPTokenizer.from_pretrained(model_name, **load)
        model = CLIPTextModelWithProjection.from_pretrained(model_name, **load)
        return cls(model, tokenizer, **kwargs)

    def train(self, mode=True):
        super().train(False)
        self.text_model.eval()
        return self

    def forward(self, texts: list[str], *, device=None) -> Tensor:
        if not isinstance(texts, (list, tuple)) or not texts or any(not isinstance(x, str) or not x.strip() for x in texts):
            raise ValueError("texts must be a non-empty batch of non-empty strings")
        encoded = self.tokenizer(list(texts), padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        target = device or next(self.text_model.parameters()).device
        encoded = {k: v.to(target) for k, v in encoded.items()}
        with torch.inference_mode():
            result = self.text_model(**encoded)
            features = result.text_embeds if hasattr(result, "text_embeds") else result[0]
        if features.ndim != 2 or features.shape != (len(texts), 512):
            raise ValueError(f"CLIP text model must return [N,512], got {tuple(features.shape)}")
        return features.float()
