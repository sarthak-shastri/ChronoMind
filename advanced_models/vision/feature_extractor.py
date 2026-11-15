"""Frozen CLIP ViT-B/32 frame features for temporal video models.

This module accepts frames already resized to 224 x 224 by the data pipeline.
For the current ReVA preprocessing path, ``imagenet`` mode reverses its
ImageNet normalization and applies CLIP normalization. This deliberately
preserves the existing square-resized geometry; it does not reproduce CLIP's
native aspect-preserving resize and center crop.

Importing this module never loads Transformers or downloads model weights.
Call :meth:`CLIPVisionFeatureExtractor.from_pretrained` explicitly when the
checkpoint is available, or inject a compatible vision model for testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import torch
from torch import Tensor, nn


NormalizationMode = Literal["imagenet", "zero_one", "clip"]

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512
IMAGE_SIZE = 224

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPVisionFeatureExtractor(nn.Module):
    """Extract frozen 512D CLIP image embeddings for each video frame.

    Args:
        vision_model: A ``CLIPVisionModelWithProjection``-compatible module.
            Its forward call must return ``image_embeds`` with shape ``[N,512]``
            (or a tensor of that shape, useful for test doubles).
        input_normalization: Explicit input value convention. ``imagenet`` is
            for Sarthak's normalized frame tensors, ``zero_one`` is RGB in
            ``[0,1]``, and ``clip`` is already CLIP-normalized input.
        frame_chunk_size: Maximum number of frames sent through the vision
            model at once. Set to ``None`` to encode all frames together.
        cuda_mixed_precision: Enable CUDA autocast for the vision forward pass.
            It is a no-op on CPU.
        amp_dtype: Autocast dtype when CUDA mixed precision is enabled.
        output_dtype: Feature dtype returned to the caller; defaults to FP32.
        l2_normalize: Normalize each frame embedding to unit length when true.
    """

    def __init__(
        self,
        vision_model: nn.Module,
        *,
        input_normalization: NormalizationMode,
        frame_chunk_size: int | None = 8,
        cuda_mixed_precision: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        output_dtype: torch.dtype = torch.float32,
        l2_normalize: bool = False,
        feature_cache_namespace: str = "",
    ) -> None:
        super().__init__()
        if input_normalization not in ("imagenet", "zero_one", "clip"):
            raise ValueError(
                "input_normalization must be 'imagenet', 'zero_one', or 'clip'"
            )
        if frame_chunk_size is not None and frame_chunk_size < 1:
            raise ValueError("frame_chunk_size must be positive or None")
        if amp_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("amp_dtype must be torch.float16 or torch.bfloat16")
        if not isinstance(output_dtype, torch.dtype):
            raise TypeError("output_dtype must be a torch.dtype")
        if not isinstance(vision_model, nn.Module):
            raise TypeError("vision_model must be a torch.nn.Module")

        self.vision_model = vision_model
        self.input_normalization = input_normalization
        self.frame_chunk_size = frame_chunk_size
        self.cuda_mixed_precision = cuda_mixed_precision
        self.amp_dtype = amp_dtype
        self.output_dtype = output_dtype
        self.l2_normalize = l2_normalize
        # Process-local cache: normalized video key -> CPU FP32 [T,512].
        # The caller supplies keys only when its dataset preprocessing is deterministic.
        self.feature_cache_namespace = str(feature_cache_namespace)
        self._feature_cache: dict[str, Tensor] = {}
        self._feature_cache_hits = 0
        self._feature_cache_misses = 0

        self.vision_model.requires_grad_(False)
        self.vision_model.eval()
        super().train(False)

    @classmethod
    def from_pretrained(
        cls,
        *,
        input_normalization: NormalizationMode,
        model_name: str = CLIP_MODEL_ID,
        revision: str | None = None,
        local_files_only: bool = False,
        cache_dir: str | None = None,
        output_loading_info: bool = False,
        **kwargs,
    ) -> "CLIPVisionFeatureExtractor":
        """Explicitly load the projected CLIP vision tower from Hugging Face.

        Transformers is imported only when this method is called. Checkpoint
        downloading therefore never occurs as an import side effect.
        """
        try:
            from transformers import CLIPVisionModelWithProjection
        except ImportError as exc:
            raise ImportError(
                "Loading pretrained CLIP weights requires the optional "
                "'transformers' package. Install Transformers, then call "
                "from_pretrained() explicitly."
            ) from exc

        load_kwargs = {"local_files_only": local_files_only,
                       "output_loading_info": output_loading_info}
        is_local_dir = Path(model_name).expanduser().is_dir()
        if revision is not None and not is_local_dir:
            load_kwargs["revision"] = revision
        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir
        try:
            loaded = CLIPVisionModelWithProjection.from_pretrained(model_name, **load_kwargs)
        except OSError as exc:
            if local_files_only:
                raise OSError(f"Could not load local CLIP vision checkpoint from {model_name!r}; local_files_only=True prevents network fallback: {exc}") from exc
            raise
        vision_model = loaded[0] if output_loading_info else loaded
        if vision_model.config.projection_dim != EMBEDDING_DIM:
            raise ValueError(
                f"Expected a {EMBEDDING_DIM}D CLIP projection, got "
                f"{vision_model.config.projection_dim}"
            )
        extractor = cls(
            vision_model,
            input_normalization=input_normalization,
            **kwargs,
        )
        extractor.source_revision = revision
        extractor.source_model = model_name
        extractor.loading_info = loaded[1] if output_loading_info else None
        return extractor

    def train(self, mode: bool = True) -> "CLIPVisionFeatureExtractor":
        """Keep both wrapper and frozen backbone in evaluation mode."""
        super().train(False)
        return self

    @staticmethod
    def _channel_constants(
        values: tuple[float, float, float], frames: Tensor
    ) -> Tensor:
        return frames.new_tensor(values).view(1, 3, 1, 1)

    def _to_clip_normalized(self, frames: Tensor) -> Tensor:
        clip_mean = self._channel_constants(_CLIP_MEAN, frames)
        clip_std = self._channel_constants(_CLIP_STD, frames)

        if self.input_normalization == "clip":
            return frames
        if self.input_normalization == "imagenet":
            imagenet_mean = self._channel_constants(_IMAGENET_MEAN, frames)
            imagenet_std = self._channel_constants(_IMAGENET_STD, frames)
            rgb = frames * imagenet_std + imagenet_mean
        else:  # explicit "zero_one" mode
            if torch.any((frames < 0.0) | (frames > 1.0)):
                raise ValueError("zero_one inputs must have values in [0, 1]")
            rgb = frames

        # Clamp only round-off or interpolation excursions at the RGB boundary.
        rgb = rgb.clamp(0.0, 1.0)
        return (rgb - clip_mean) / clip_std

    @staticmethod
    def _image_embeddings(model_output) -> Tensor:
        if isinstance(model_output, Tensor):
            embeddings = model_output
        else:
            embeddings = getattr(model_output, "image_embeds", None)
        if not isinstance(embeddings, Tensor):
            raise TypeError(
                "vision_model must return a tensor or an object with "
                "a tensor 'image_embeds' attribute"
            )
        return embeddings

    def clear_feature_cache(self) -> None:
        """Drop process-local frame embeddings after changing data/model settings."""
        self._feature_cache.clear()
        self._feature_cache_hits = 0
        self._feature_cache_misses = 0

    def feature_cache_stats(self) -> dict[str, int]:
        return {
            "entries": len(self._feature_cache),
            "hits": self._feature_cache_hits,
            "misses": self._feature_cache_misses,
        }

    def forward(self, video_frames: Tensor, *, cache_keys: Sequence[str | None] | None = None) -> Tensor:
        """Map preprocessed frames ``[B,T,3,224,224]`` to features ``[B,T,512]``."""
        if not isinstance(video_frames, Tensor):
            raise TypeError("video_frames must be a torch.Tensor")
        if video_frames.ndim != 5:
            raise ValueError("video_frames must have shape [B,T,3,224,224]")
        if video_frames.shape[0] < 1 or video_frames.shape[1] < 1:
            raise ValueError("batch size and sequence length must be positive")
        if tuple(video_frames.shape[2:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("video_frames must have shape [B,T,3,224,224]")
        if not video_frames.is_floating_point():
            raise TypeError("video_frames must be a floating-point tensor")
        if not torch.isfinite(video_frames).all():
            raise ValueError("video_frames must contain only finite values")
        if cache_keys is not None:
            if len(cache_keys) != video_frames.shape[0]:
                raise ValueError("cache_keys must contain one key per video")
            if any(key is not None and (not isinstance(key, str) or not key) for key in cache_keys):
                raise ValueError("cache keys must be non-empty strings or None")

        try:
            model_device = next(self.vision_model.parameters()).device
        except StopIteration:
            model_device = video_frames.device
        if model_device != video_frames.device:
            raise ValueError(
                f"video_frames are on {video_frames.device}, but the vision "
                f"model is on {model_device}; move both to the same device"
            )

        batch_size, sequence_length = video_frames.shape[:2]
        if cache_keys is not None:
            output: list[Tensor | None] = [None] * batch_size
            missing_indices: list[int] = []
            key_to_indices: dict[str, list[int]] = {}
            for index, key in enumerate(cache_keys):
                if key is None:
                    missing_indices.append(index)
                    continue
                full_key = f"{self.feature_cache_namespace}\0{key}"
                cached = self._feature_cache.get(full_key)
                if cached is not None and cached.shape == (sequence_length, EMBEDDING_DIM):
                    self._feature_cache_hits += 1
                    output[index] = cached.to(device=video_frames.device, dtype=self.output_dtype)
                else:
                    if cached is not None:
                        del self._feature_cache[full_key]
                    indices = key_to_indices.get(full_key)
                    if indices is None:
                        key_to_indices[full_key] = [index]
                        missing_indices.append(index)
                    else:
                        # Duplicate video in the same batch: encode it once.
                        indices.append(index)
            if missing_indices:
                selected = video_frames[missing_indices]
                computed = self._encode_frames(selected)
                for local_index, batch_index in enumerate(missing_indices):
                    feature = computed[local_index].to(dtype=self.output_dtype)
                    key = cache_keys[batch_index]
                    if key is not None:
                        full_key = f"{self.feature_cache_namespace}\0{key}"
                        cpu_feature = feature.detach().to(device="cpu", dtype=torch.float32).clone()
                        self._feature_cache[full_key] = cpu_feature
                        self._feature_cache_misses += 1
                        output[batch_index] = feature
                        for duplicate_index in key_to_indices.get(full_key, [])[1:]:
                            output[duplicate_index] = feature
                    else:
                        output[batch_index] = feature
            if any(feature is None for feature in output):
                raise RuntimeError("internal visual feature cache error: output was not filled")
            return torch.stack([feature for feature in output if feature is not None])

        return self._encode_frames(video_frames)

    def _encode_frames(self, video_frames: Tensor) -> Tensor:
        batch_size, sequence_length = video_frames.shape[:2]
        flat_frames = video_frames.reshape(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
        chunk_size = self.frame_chunk_size or flat_frames.shape[0]
        chunks = []

        with torch.inference_mode():
            for start in range(0, flat_frames.shape[0], chunk_size):
                frame_chunk = self._to_clip_normalized(
                    flat_frames[start : start + chunk_size]
                )
                if self.cuda_mixed_precision and frame_chunk.device.type == "cuda":
                    with torch.autocast(
                        device_type="cuda", dtype=self.amp_dtype
                    ):
                        model_output = self.vision_model(pixel_values=frame_chunk)
                else:
                    model_output = self.vision_model(pixel_values=frame_chunk)

                embeddings = self._image_embeddings(model_output)
                if embeddings.ndim != 2 or embeddings.shape != (
                    frame_chunk.shape[0],
                    EMBEDDING_DIM,
                ):
                    raise ValueError(
                        "vision_model must return image embeddings with shape "
                        f"[{frame_chunk.shape[0]}, {EMBEDDING_DIM}]"
                    )
                embeddings = embeddings.float()
                if self.l2_normalize:
                    embeddings = torch.nn.functional.normalize(
                        embeddings, p=2, dim=-1
                    )
                chunks.append(embeddings.to(dtype=self.output_dtype))

        return torch.cat(chunks, dim=0).reshape(
            batch_size, sequence_length, EMBEDDING_DIM
        )

