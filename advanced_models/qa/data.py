"""Adapters for verified Sarthak ReVA dataset samples and labels."""
from __future__ import annotations
from torch.utils.data import Dataset
from .multimodal_qa_model import answer_index

class ReVAQADatasetAdapter(Dataset):
    """Wrap an existing ReVADataset, rejecting missing/corrupt videos explicitly."""
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index):
        row = self.dataset[index]
        if row.get("video_frames") is None or row.get("video_error"):
            raise RuntimeError(f"invalid video for sample {row.get('sample_id')}: {row.get('video_error')}")
        choices = row.get("choices")
        if not isinstance(choices, dict) or tuple(choices.keys()) != ("A", "B", "C", "D"):
            # Normalize mapping order explicitly, while requiring all canonical keys.
            if not isinstance(choices, dict) or set(choices) != {"A", "B", "C", "D"}:
                raise ValueError("dataset choices must map exactly A/B/C/D")
            choices = {key: choices[key] for key in ("A", "B", "C", "D")}
        sample = dict(row)
        source_row = getattr(self.dataset, "rows", [])[index]
        # Sarthak's __getitem__ omits video_path; copy it only from its source annotation.
        sample["video_path"] = source_row.get("video_path", source_row.get("file_path"))
        sample["choices"] = choices
        if row.get("answer") not in (None, ""):
            sample["label"] = answer_index(row["answer"])
        return sample

def collate_qa(samples):
    import torch
    if not samples: raise ValueError("cannot collate empty batch")
    counts = {s["video_frames"].shape[0] for s in samples}
    if len(counts) != 1: raise ValueError("all videos in a batch must have the same frame count")
    return {"video_frames": torch.stack([s["video_frames"] for s in samples]),
            "questions": [s["question"] for s in samples],
            "choices": [s["choices"] for s in samples],
            "labels": torch.tensor([s["label"] for s in samples], dtype=torch.long) if all("label" in s for s in samples) else None,
            "answers": [s.get("answer") for s in samples],
            "sample_ids": [str(s["sample_id"]) for s in samples],
            "categories": [s.get("category") for s in samples],
            "video_paths": [s.get("video_path") for s in samples]}
