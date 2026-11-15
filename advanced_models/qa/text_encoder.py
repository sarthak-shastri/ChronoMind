"""Frozen projected CLIP text embeddings with explicit checkpoint loading."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import Tensor, nn
from collections import OrderedDict


class CLIPTextFeatureExtractor(nn.Module):
    def __init__(self, text_model: nn.Module, tokenizer, *, max_length: int = 77,
                 cache_max_entries: int = 100_000):
        super().__init__()
        if max_length < 2 or max_length > 77:
            raise ValueError("max_length must be between 2 and CLIP's 77-token context")
        self.text_model, self.tokenizer, self.max_length = text_model, tokenizer, max_length
        if cache_max_entries < 0:
            raise ValueError("cache_max_entries must be non-negative")
        self.cache_max_entries = cache_max_entries
        # Process-local CPU FP32 cache. One vector per distinct input string.
        self._feature_cache: OrderedDict[tuple[str, str, str | None], Tensor] = OrderedDict()
        self._feature_cache_hits = 0
        self._feature_cache_misses = 0
        self.text_model.requires_grad_(False)
        self.text_model.eval()

    @classmethod
    def from_pretrained(cls, model_name="openai/clip-vit-base-patch32", *, revision=None,
                        local_files_only=False, cache_dir=None,
                        output_loading_info=False, tokenizer_kwargs=None, **kwargs):
        try:
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers to load pretrained CLIP text weights") from exc
        is_local_dir = Path(model_name).expanduser().is_dir()
        load = {"local_files_only": local_files_only,
                "output_loading_info": output_loading_info}
        if revision is not None and not is_local_dir:
            load["revision"] = revision
        if cache_dir is not None:
            load["cache_dir"] = cache_dir
        tokenizer_load = dict(load)
        tokenizer_load.pop("output_loading_info", None)
        tokenizer_load.update(tokenizer_kwargs or {})
        try:
            tokenizer = CLIPTokenizer.from_pretrained(model_name, **tokenizer_load)
            loaded = CLIPTextModelWithProjection.from_pretrained(model_name, **load)
        except OSError as exc:
            if local_files_only:
                raise OSError(f"Could not load local CLIP text checkpoint/tokenizer from {model_name!r}; local_files_only=True prevents network fallback: {exc}") from exc
            raise
        model = loaded[0] if output_loading_info else loaded
        extractor = cls(model, tokenizer, **kwargs)
        extractor.source_revision = revision
        extractor.source_model = model_name
        extractor.loading_info = loaded[1] if output_loading_info else None
        return extractor

    def train(self, mode=True):
        super().train(False)
        self.text_model.eval()
        return self

    def forward(self, texts: list[str], *, device=None) -> Tensor:
        if not isinstance(texts, (list, tuple)) or not texts or any(not isinstance(x, str) or not x.strip() for x in texts):
            raise ValueError("texts must be a non-empty batch of non-empty strings")
        target = device or next(self.text_model.parameters()).device
        if self.cache_max_entries == 0:
            return self._encode(texts, target)
        keys = [self._cache_key(text, target) for text in texts]
        missing_keys: list[tuple[str, str, str | None]] = []
        missing_texts: list[str] = []
        seen: set[tuple[str, str, str | None]] = set()
        for text, key in zip(texts, keys):
            if key not in self._feature_cache and key not in seen:
                missing_keys.append(key)
                missing_texts.append(text)
                seen.add(key)
        if missing_texts:
            features = self._encode(missing_texts, target)
            for key, feature in zip(missing_keys, features):
                self._feature_cache[key] = feature.detach().to(device="cpu", dtype=torch.float32).clone()
                self._feature_cache_misses += 1
                while len(self._feature_cache) > self.cache_max_entries:
                    self._feature_cache.popitem(last=False)
        result = []
        for text, key in zip(texts, keys):
            feature = self._feature_cache.get(key)
            if feature is None:
                # The entry may have been evicted while filling a very large batch.
                feature = self._encode([text], target)[0].detach().to(device="cpu", dtype=torch.float32)
                self._feature_cache[key] = feature.clone()
                self._feature_cache_misses += 1
            else:
                self._feature_cache.move_to_end(key)
                self._feature_cache_hits += 1
            result.append(feature.to(target))
        return torch.stack(result)

    @staticmethod
    def _cache_key(text: str, target: torch.device) -> tuple[str, str, str | None]:
        """Keep cached embeddings separate across device/autocast precision modes."""
        enabled = torch.is_autocast_enabled(target.type)
        dtype = str(torch.get_autocast_dtype(target.type)) if enabled else None
        return text, target.type, dtype

    def _encode(self, texts: list[str], target: torch.device) -> Tensor:
        encoded = self.tokenizer(list(texts), padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        encoded = {k: v.to(target) for k, v in encoded.items()}
        with torch.inference_mode():
            result = self.text_model(**encoded)
            features = result.text_embeds if hasattr(result, "text_embeds") else result[0]
        if features.ndim != 2 or features.shape != (len(texts), 512):
            raise ValueError(f"CLIP text model must return [N,512], got {tuple(features.shape)}")
        return features.float()

    def clear_feature_cache(self) -> None:
        """Drop cached features after changing the model, tokenizer, or max length."""
        self._feature_cache.clear()
        self._feature_cache_hits = 0
        self._feature_cache_misses = 0

    def feature_cache_stats(self) -> dict[str, int]:
        return {"entries": len(self._feature_cache), "hits": self._feature_cache_hits,
                "misses": self._feature_cache_misses}
