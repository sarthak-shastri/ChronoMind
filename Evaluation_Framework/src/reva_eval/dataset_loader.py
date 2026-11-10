"""Dataset loader and normalizer for ReVA VideoQA annotations.

Converts official annotation formats (current QA-list format and legacy videos mapping)
into canonical ground-truth format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from reva_eval.models import CanonicalGroundTruth, GroundTruthSample, VALID_LABELS


def load_raw_annotation_file(file_path: Union[str, Path]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """Read a raw annotation JSON file and extract metadata, QA items, and detected format.

    Returns:
        (metadata_dict, raw_qa_list, format_name)
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Already canonical ground-truth format
    if isinstance(data, dict) and "schema_version" in data and "samples" in data:
        return data.get("metadata", {}), data.get("samples", []), "canonical"

    # 2. Current top-level QA list format {"metadata": ..., "QA": [...]}
    if isinstance(data, dict) and "QA" in data and isinstance(data["QA"], list):
        return data.get("metadata", {}), data["QA"], "qa_list"

    # 3. Direct list format [{...}, {...}]
    if isinstance(data, list):
        return {}, data, "direct_list"

    # 4. Legacy format: {"videos": {video_key: ...}}
    if isinstance(data, dict) and "videos" in data and isinstance(data["videos"], dict):
        qa_list = []
        for v_key, v_val in data["videos"].items():
            video_path = v_val.get("video_path", v_key) if isinstance(v_val, dict) else v_key
            items = v_val.get("qa_list", []) if isinstance(v_val, dict) else (v_val if isinstance(v_val, list) else [])
            for item in items:
                record = dict(item)
                if "video_path" not in record:
                    record["video_path"] = video_path
                qa_list.append(record)
        return data.get("metadata", {}), qa_list, "legacy_videos"

    # 5. Legacy format: mapping of video_id -> list of qa items or dict with qa_list
    if isinstance(data, dict):
        qa_list = []
        for v_key, v_val in data.items():
            if isinstance(v_val, list):
                for item in v_val:
                    if isinstance(item, dict):
                        record = dict(item)
                        if "video_path" not in record:
                            record["video_path"] = v_key
                        qa_list.append(record)
            elif isinstance(v_val, dict) and "qa_list" in v_val:
                for item in v_val["qa_list"]:
                    if isinstance(item, dict):
                        record = dict(item)
                        if "video_path" not in record:
                            record["video_path"] = v_val.get("video_path", v_key)
                        qa_list.append(record)
        if qa_list:
            return {}, qa_list, "legacy_dict_mapping"

    raise ValueError(f"Unrecognized annotation JSON structure in {path}")


def normalize_annotation(
    file_path: Union[str, Path],
    split: str = "validation",
    dataset: str = "ReVA",
) -> CanonicalGroundTruth:
    """Load and normalize raw annotation JSON into a CanonicalGroundTruth instance."""
    path = Path(file_path)
    metadata, raw_items, detected_format = load_raw_annotation_file(path)

    seen_ids = set()
    duplicate_ids = []
    samples: List[GroundTruthSample] = []
    labeled_count = 0

    for idx, item in enumerate(raw_items):
        # Determine sample identifier
        sample_id = item.get("qa_id") or item.get("sample_id")
        if not sample_id:
            raise ValueError(f"Record at index {idx} in {path.name} is missing 'qa_id' / 'sample_id'")

        sample_id = str(sample_id)
        if sample_id in seen_ids:
            duplicate_ids.append(sample_id)
        seen_ids.add(sample_id)

        # Video path
        video_path = item.get("video_path")
        if not video_path:
            # Fall back to video_id if available, but require a video reference
            video_path = item.get("video_id", f"unknown_video_{idx}")

        # Dataset name & subdir
        dataset_name = item.get("dataset_name")
        subdir = item.get("subdir")

        # Question & Options
        question = item.get("question")
        raw_options = item.get("options")
        options = None
        if isinstance(raw_options, dict):
            options = {k: str(v) for k, v in raw_options.items()}
        elif isinstance(raw_options, list):
            # If options is a list, map [0,1,2,3] -> ['A','B','C','D']
            labels = ["A", "B", "C", "D"]
            options = {labels[i]: str(opt) for i, opt in enumerate(raw_options[:4])}

        # Category and Task (preserve actual strings)
        category = item.get("category") or "Unknown"
        task = item.get("task") or "Unknown"

        # Correct answer
        raw_ans = item.get("correct_answer")
        if raw_ans is None:
            raw_ans = item.get("ground_truth")

        ground_truth: Optional[str] = None
        if raw_ans is not None:
            cleaned_ans = str(raw_ans).strip().upper()
            if cleaned_ans in VALID_LABELS:
                ground_truth = cleaned_ans
                labeled_count += 1
            elif cleaned_ans == "":
                ground_truth = None
            else:
                # Keep unrecognized string or flag as None
                ground_truth = None

        sample = GroundTruthSample(
            sample_id=sample_id,
            video_path=str(video_path),
            dataset_name=dataset_name,
            subdir=subdir,
            question=question,
            options=options,
            category=category,
            task=task,
            ground_truth=ground_truth,
        )
        samples.append(sample)

    if duplicate_ids:
        sample_dups = duplicate_ids[:5]
        raise ValueError(
            f"Ground truth validation failed: found {len(duplicate_ids)} duplicate sample_id(s) in {path.name} "
            f"(e.g., {sample_dups})"
        )

    # Determine whether the split is labeled
    is_labeled = labeled_count > 0

    return CanonicalGroundTruth(
        schema_version="1.0",
        dataset=dataset,
        split=split,
        source_annotation_file=path.name,
        labeled=is_labeled,
        samples=samples,
    )


def save_canonical_ground_truth(gt: CanonicalGroundTruth, output_path: Union[str, Path]) -> Path:
    """Save canonical ground truth to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gt.model_dump(), f, indent=2, ensure_ascii=False)
    return path


def load_canonical_ground_truth(file_path: Union[str, Path]) -> CanonicalGroundTruth:
    """Load canonical ground truth from a JSON file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical ground truth file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CanonicalGroundTruth.model_validate(data)

