"""Frozen CLIP text-to-video similarity baseline."""
import torch

def clip_similarity_logits(video_features, question_features, choice_features):
    """Score each option by cosine similarity to video and question (equal weights)."""
    if video_features.ndim != 2 or question_features.shape != video_features.shape or choice_features.shape != (video_features.shape[0],4,video_features.shape[1]):
        raise ValueError("expected video/question [B,D] and choices [B,4,D]")
    v=torch.nn.functional.normalize(video_features.float(),dim=-1)[:,None,:]
    q=torch.nn.functional.normalize(question_features.float(),dim=-1)[:,None,:]
    a=torch.nn.functional.normalize(choice_features.float(),dim=-1)
    return ((a*v).sum(-1)+(a*q).sum(-1))*0.5
