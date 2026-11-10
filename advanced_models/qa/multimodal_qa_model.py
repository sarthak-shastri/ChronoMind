"""Trainable temporal multiple-choice head over frozen CLIP embeddings."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from .text_encoder import CLIPTextFeatureExtractor
from ..temporal.temporal_encoder import TransformerTemporalEncoder
from ..vision.feature_extractor import CLIPVisionFeatureExtractor

LETTERS = ("A", "B", "C", "D")

def answer_index(label: str) -> int:
    if not isinstance(label, str) or label not in LETTERS:
        raise ValueError("answer label must be one of A, B, C, D")
    return LETTERS.index(label)

def answer_letter(index: int) -> str:
    if isinstance(index, bool) or index not in range(4):
        raise ValueError("answer index must be 0..3")
    return LETTERS[index]

class MultimodalVideoQAModel(nn.Module):
    def __init__(self, vision: CLIPVisionFeatureExtractor, text: CLIPTextFeatureExtractor,
                 temporal: TransformerTemporalEncoder, dropout: float = 0.1):
        super().__init__()
        if temporal.embedding_dim != 512:
            raise ValueError("temporal embedding_dim must be 512")
        self.vision, self.text, self.temporal = vision, text, temporal
        self.fusion = nn.Sequential(nn.LayerNorm(512 * 4), nn.Linear(2048, 512), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(512))
        self.scorer = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval(); self.text.eval()
        return self

    def forward(self, video_frames: Tensor, questions: list[str], choices: list[dict[str, str]]):
        if not torch.is_tensor(video_frames) or video_frames.ndim != 5:
            raise ValueError("video_frames must be [B,T,3,224,224]")
        b, t = video_frames.shape[:2]
        if b < 1 or t not in (4, 8, 16) or video_frames.shape[2:] != (3,224,224):
            raise ValueError("video_frames must be [B,T,3,224,224] with T in 4,8,16")
        if len(questions) != b or len(choices) != b:
            raise ValueError("questions and choices must match batch size")
        if any(set(row) != set(LETTERS) or any(not isinstance(row[k], str) or not row[k].strip() for k in LETTERS) for row in choices):
            raise ValueError("each choices row must contain non-empty A/B/C/D strings")
        # The frozen extractor uses inference_mode; clone outside it so trainable
        # temporal layers may safely save these features for backward.
        frames = self.vision(video_frames).clone()
        video = self.temporal(frames)
        flat = [text for row in choices for text in (row[k] for k in LETTERS)]
        question = self.text(questions, device=video.device).clone()
        options = self.text(flat, device=video.device).clone().reshape(b, 4, 512)
        v = video[:, None, :].expand(-1, 4, -1)
        q = question[:, None, :].expand(-1, 4, -1)
        fused = self.fusion(torch.cat((v, q, options, v * options), dim=-1))
        logits = self.scorer(fused).squeeze(-1)
        return {"answer_logits": logits, "prediction": [answer_letter(i) for i in logits.argmax(-1).tolist()],
                "frame_features": frames, "video_features": video}
