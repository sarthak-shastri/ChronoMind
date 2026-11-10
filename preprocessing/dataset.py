import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .config import resolve_video_path, SUPPORTED_FRAME_COUNTS
from .video_loader import get_video_metadata, motion_scores, read_frames, VideoLoadError
from .frame_sampling import uniform_indices, adaptive_indices
from .transformations import transform_frames
from .cache_manager import FrameCache
from .hf_access import fetch_dataset_file
from .collate import collate_reva, make_dataloader


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
    def __init__(self, annotation_file, video_root=None, num_frames=8, sampling="uniform",
                 split=None, sample_ids=None, image_size=(224, 224), normalize=True,
                 cache_dir=None, missing_video="raise", dataset_repo_id=None,
                 revision=None, hf_cache_dir=None, local_files_only=False):
        if num_frames not in SUPPORTED_FRAME_COUNTS:
            raise ValueError(f"num_frames must be one of {SUPPORTED_FRAME_COUNTS}")
        if sampling not in {"uniform", "adaptive"}:
            raise ValueError("sampling must be 'uniform' or 'adaptive'")
        self.video_root = Path(video_root) if video_root is not None else None
        self.dataset_repo_id = dataset_repo_id
        self.revision = revision
        self.hf_cache_dir = Path(hf_cache_dir).expanduser() if hf_cache_dir else None
        self.local_files_only = local_files_only
        if self.video_root is None and not self.dataset_repo_id:
            raise ValueError("Provide video_root for local data or dataset_repo_id for Hugging Face data")
        self.num_frames, self.sampling = num_frames, sampling
        self.image_size, self.normalize = image_size, normalize
        self.missing_video = missing_video
        if missing_video not in {"raise", "zeros"}:
            raise ValueError("missing_video must be 'raise' or 'zeros'")
        self.cache = FrameCache(cache_dir) if cache_dir else None
        rows = read_annotations(annotation_file)
        ids = set(str(value) for value in sample_ids) if sample_ids is not None else None
        if split is not None:
            wanted = self._normalize_split(split)
            row_splits = [self._normalize_split(row.get("split")) for row in rows if row.get("split")]
            if row_splits:
                rows = [row for row in rows if self._normalize_split(row.get("split")) == wanted]
            else:
                filename_split = self._normalize_split(Path(annotation_file).stem)
                if filename_split is None:
                    raise ValueError("split filtering requires a split field or split-named annotation file")
                if filename_split != wanted:
                    raise ValueError(f"annotation file is for split '{filename_split}', not '{wanted}'")
        self.rows = [r for r in rows if ids is None or str(r.get("qa_id", r.get("sample_id", r.get("id")))) in ids]
        self.errors = []

    @staticmethod
    def _normalize_split(value):
        if value is None:
            return None
        value = str(value).strip().casefold()
        aliases = {"valid": "val", "validation": "val"}
        if value in aliases:
            return aliases[value]
        # Permit descriptive derived annotation filenames such as train_grouped.json.
        prefix = value.split("_", 1)[0]
        return aliases.get(prefix, prefix) if prefix in {"train", "val", "valid", "validation", "test"} else value

    def __len__(self):
        return len(self.rows)

    def _video_tensor(self, row):
        rel = row.get("video_path", row.get("file_path"))
        if not rel:
            raise VideoLoadError("Annotation has no video_path/file_path")
        path = resolve_video_path(self.video_root, rel) if self.video_root is not None else None
        if path is None or not path.is_file():
            if not self.dataset_repo_id:
                raise VideoLoadError(f"Video file not found: {path}")
            try:
                path = fetch_dataset_file(self.dataset_repo_id, str(rel), revision=self.revision,
                                          cache_dir=self.hf_cache_dir,
                                          local_files_only=self.local_files_only)
            except Exception as exc:
                raise VideoLoadError(f"Could not fetch {rel} from {self.dataset_repo_id}: {exc}") from exc
        key = f"{self.revision}|{path.resolve()}|{path.stat().st_size}|{self.num_frames}|{self.sampling}|{self.image_size}|{self.normalize}"
        frames = self.cache.get(key) if self.cache else None
        if frames is None:
            meta = get_video_metadata(path)
            count = meta["frame_count"]
            if not count:
                raise VideoLoadError(f"Video has no readable frames: {path}")
            if self.sampling == "uniform":
                indices = uniform_indices(count, self.num_frames)
            else:
                scores = motion_scores(path)
                indices = adaptive_indices(count, self.num_frames, scores)
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
