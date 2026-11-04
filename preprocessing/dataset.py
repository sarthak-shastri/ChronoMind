import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from .config import resolve_video_path, SUPPORTED_FRAME_COUNTS
from .video_loader import get_video_metadata, read_frames, VideoLoadError
from .frame_sampling import uniform_indices, adaptive_indices
from .transformations import transform_frames
from .cache_manager import FrameCache


def read_annotations(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return rows
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        qa = payload.get("QA")
        if isinstance(qa, list):
            return qa
        for key in ("data", "annotations", "questions"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("Unrecognized annotation structure; expected list or object with QA list")


class ReVADataset(Dataset):
    def __init__(self, annotation_file, video_root, num_frames=8, sampling="uniform",
                 split=None, sample_ids=None, image_size=(224, 224), normalize=True,
                 cache_dir=None, missing_video="raise"):
        if num_frames not in SUPPORTED_FRAME_COUNTS:
            raise ValueError(f"num_frames must be one of {SUPPORTED_FRAME_COUNTS}")
        if sampling not in {"uniform", "adaptive"}:
            raise ValueError("sampling must be 'uniform' or 'adaptive'")
        self.video_root = Path(video_root)
        self.num_frames, self.sampling = num_frames, sampling
        self.image_size, self.normalize = image_size, normalize
        self.missing_video = missing_video
        if missing_video not in {"raise", "zeros"}:
            raise ValueError("missing_video must be 'raise' or 'zeros'")
        self.cache = FrameCache(cache_dir) if cache_dir else None
        rows = read_annotations(annotation_file)
        ids = set(sample_ids) if sample_ids is not None else None
        self.rows = [r for r in rows if (split is None or r.get("split") in (None, split))
                     and (ids is None or str(r.get("qa_id", r.get("sample_id", r.get("id")))) in ids)]
        self.errors = []

    def __len__(self):
        return len(self.rows)

    def _video_tensor(self, row):
        rel = row.get("video_path", row.get("file_path"))
        if not rel:
            raise VideoLoadError("Annotation has no video_path/file_path")
        path = resolve_video_path(self.video_root, rel)
        if not path.is_file():
            raise VideoLoadError(f"Video file not found: {path}")
        key = f"{path.resolve()}|{path.stat().st_size}|{self.num_frames}|{self.sampling}|{self.image_size}|{self.normalize}"
        frames = self.cache.get(key) if self.cache else None
        if frames is None:
            meta = get_video_metadata(path)
            count = meta["frame_count"]
            if not count:
                raise VideoLoadError(f"Video has no readable frames: {path}")
            indices = uniform_indices(count, self.num_frames) if self.sampling == "uniform" else adaptive_indices(count, self.num_frames)
            indices = indices.tolist()
            if len(indices) < self.num_frames:
                indices = (indices + [indices[-1]] * self.num_frames)[:self.num_frames]
            frames = read_frames(path, indices)
            if self.cache:
                self.cache.put(key, frames)
        return transform_frames(frames, self.image_size, self.normalize)

    def __getitem__(self, index):
        row = self.rows[index]
        opts = row.get("options", row.get("choices", {})) or {}
        choices = {k: opts.get(k) for k in ("A", "B", "C", "D")}
        answer = row.get("correct_answer", row.get("answer")) or None
        sample = {"video_frames": None, "question": row.get("question"), "choices": choices,
                  "answer": answer, "video_id": row.get("video_id"),
                  "sample_id": row.get("qa_id", row.get("sample_id", row.get("id"))),
                  "category": row.get("category"), "task": row.get("task")}
        try:
            sample["video_frames"] = self._video_tensor(row)
        except Exception as exc:
            self.errors.append({"sample_id": sample["sample_id"], "error": str(exc)})
            if self.missing_video == "raise":
                raise
            sample["video_frames"] = torch.zeros((self.num_frames, 3, *self.image_size), dtype=torch.float32)
            sample["video_error"] = str(exc)
        return sample


class ReVAVideoDataset(ReVADataset):
    """Video-only view yielding frames and identifiers for metadata/preprocessing jobs."""
    def __getitem__(self, index):
        s = super().__getitem__(index)
        return {k: s[k] for k in ("video_frames", "video_id", "sample_id")}


def collate_reva(batch):
    if not batch:
        return {}
    result = {k: [item.get(k) for item in batch] for k in batch[0]}
    try:
        result["video_frames"] = torch.stack([item["video_frames"] for item in batch])
    except (KeyError, TypeError, RuntimeError):
        pass
    return result


def make_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_reva, **kwargs)
