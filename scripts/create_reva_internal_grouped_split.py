"""Create a deterministic, video-grouped 80:20 split from ReVA train.json only."""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("A", "B", "C", "D")


def normalize_video_path(value: str) -> str:
    """Normalize separators, redundant dot components, and case for grouping."""
    value = re.sub(r"\\+", "/", str(value or "").strip())
    while value.startswith("./"):
        value = value[2:]
    return posixpath.normpath(value).casefold() if value else ""


def read_annotations(path: str | Path) -> tuple[object, list[dict]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rows = payload.get("QA") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Expected a list of QA records or an object with a QA list")
    return payload, rows


def _hash_video(root: Path, relative_path: str) -> Path | None:
    candidate = (root / relative_path.replace("\\", "/")).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"video path escapes video_root: {relative_path}")
    return candidate if candidate.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_strata(row: dict) -> tuple[str, str, str]:
    category = str(row.get("category") or "Unknown")
    task = str(row.get("task") or "Unknown")
    answer = str(row.get("correct_answer") or "Unlabeled")
    if answer not in LABELS:
        answer = "Invalid/Unlabeled"
    return category, task, answer


def _balance_cost(rows: int, strata: Counter, target_rows: float, target_strata: Counter) -> float:
    # Total row count is weighted most heavily; category, task, and answer
    # marginals guide group choice without splitting any video's QA records.
    total_scale = max(target_rows, 1.0)
    cost = 4.0 * abs(rows - target_rows) / total_scale
    groups = ("category", "task", "answer")
    for prefix in groups:
        keys = [key for key in target_strata if key.startswith(prefix + ":")]
        if keys:
            cost += sum(
                abs(strata[key] - target_strata[key]) / max(target_strata[key], 1.0)
                for key in keys
            ) / len(keys)
    return cost


def _choose_validation_groups(
    groups: dict[str, list[int]], rows: list[dict], fraction: float, seed: int
) -> set[str]:
    rng = random.Random(seed)
    all_strata = Counter()
    group_strata: dict[str, Counter] = {}
    for group, indices in groups.items():
        counts = Counter()
        for index in indices:
            category, task, answer = _record_strata(rows[index])
            counts[f"category:{category}"] += 1
            counts[f"task:{task}"] += 1
            counts[f"answer:{answer}"] += 1
        group_strata[group] = counts
        all_strata.update(counts)

    target_rows = len(rows) * fraction
    target_strata = Counter({key: count * fraction for key, count in all_strata.items()})
    tie_break = {group: rng.random() for group in groups}
    remaining = set(groups)
    selected: set[str] = set()
    selected_rows = 0
    selected_strata = Counter()

    while remaining:
        best_group = min(
            remaining,
            key=lambda group: (
                _balance_cost(
                    selected_rows + len(groups[group]),
                    selected_strata + group_strata[group],
                    target_rows,
                    target_strata,
                ),
                tie_break[group],
            ),
        )
        next_rows = selected_rows + len(groups[best_group])
        # Stop at the closest reachable row count; never split a group to hit
        # exactly 20 percent when the video-level group sizes prevent it.
        if selected_rows and abs(selected_rows - target_rows) <= abs(next_rows - target_rows):
            break
        selected.add(best_group)
        remaining.remove(best_group)
        selected_rows = next_rows
        selected_strata.update(group_strata[best_group])
        if selected_rows >= target_rows:
            break

    if len(groups) > 1 and (not selected or len(selected) == len(groups)):
        # Ensure both partitions exist when a valid group-level partition is
        # possible, choosing the single group nearest the requested fraction.
        selected = {min(groups, key=lambda group: (abs(len(groups[group]) - target_rows), tie_break[group]))}
    return selected


def _distribution(rows: list[dict], field: str) -> dict[str, dict[str, float | int]]:
    counts = Counter()
    for row in rows:
        if field == "answer":
            value = row.get("correct_answer") or "Unlabeled"
            if value not in LABELS:
                value = "Invalid/Unlabeled"
        else:
            value = row.get(field) or "Unknown"
        counts[str(value)] += 1
    total = len(rows)
    return {key: {"count": count, "fraction": count / total if total else 0.0}
            for key, count in sorted(counts.items())}


def create_split(
    annotation_file: str | Path,
    output_dir: str | Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    video_root: str | Path | None = None,
    overwrite: bool = False,
) -> dict:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    template, rows = read_annotations(annotation_file)
    if not rows:
        raise ValueError("train.json contains no QA records")

    path_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("correct_answer") not in LABELS:
            raise ValueError(f"QA record {row.get('qa_id', index)} has invalid/missing A/B/C/D correct_answer")
        options = row.get("options", row.get("choices"))
        if not isinstance(options, dict) or set(options) != set(LABELS):
            raise ValueError(f"QA record {row.get('qa_id', index)} must contain exactly A/B/C/D options")
        key = normalize_video_path(row.get("video_path", row.get("file_path")))
        if not key or key == ".":
            raise ValueError(f"QA record {row.get('qa_id', index)} is missing a usable video path")
        path_to_indices[key].append(index)

    root = Path(video_root).expanduser().resolve() if video_root is not None else None
    missing_paths: list[str] = []
    path_hashes: dict[str, str] = {}
    if root:
        for key in sorted(path_to_indices):
            # Resolve any original spelling that normalized to this path; this
            # handles case/slash aliases on case-sensitive local filesystems.
            file_path = None
            for index in path_to_indices[key]:
                relative = rows[index].get("video_path", rows[index].get("file_path"))
                file_path = _hash_video(root, str(relative))
                if file_path is not None:
                    break
            if file_path is None:
                missing_paths.append(key)
                continue
            digest = _sha256(file_path)
            path_hashes[key] = digest

    # Merge paths with identical available video bytes into the same group.
    parent = {key: key for key in path_to_indices}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(first: str, second: str) -> None:
        a, b = find(first), find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for digest in sorted(set(path_hashes.values())):
        same_hash_paths = sorted(key for key, value in path_hashes.items() if value == digest)
        for key in same_hash_paths[1:]:
            union(same_hash_paths[0], key)

    groups: dict[str, list[int]] = defaultdict(list)
    for path, indices in path_to_indices.items():
        groups[find(path)].extend(indices)
    if len(groups) < 2:
        raise ValueError("At least two independent path/hash groups are required for a train/validation split")

    validation_groups = _choose_validation_groups(groups, rows, val_fraction, seed)
    val_indices = {index for group in validation_groups for index in groups[group]}
    train_rows = [row for index, row in enumerate(rows) if index not in val_indices]
    val_rows = [row for index, row in enumerate(rows) if index in val_indices]
    train_paths = {normalize_video_path(row.get("video_path", row.get("file_path"))) for row in train_rows}
    val_paths = {normalize_video_path(row.get("video_path", row.get("file_path"))) for row in val_rows}
    path_overlap = sorted(train_paths & val_paths)
    if path_overlap:
        raise AssertionError(f"internal split shares normalized video paths: {path_overlap[:5]}")

    def output_payload(selected_rows: list[dict]):
        if isinstance(template, dict):
            result = dict(template)
            result["QA"] = selected_rows
            return result
        return selected_rows

    out = Path(output_dir)
    train_file, val_file = out / "train_internal.json", out / "val_internal.json"
    stats_file = out / "split_statistics.json"
    markdown_file = out / "split_statistics.md"
    existing = [path for path in (train_file, val_file, stats_file, markdown_file) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing derived files: {existing}; pass --overwrite explicitly")
    out.mkdir(parents=True, exist_ok=True)
    train_file.write_text(json.dumps(output_payload(train_rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    val_file.write_text(json.dumps(output_payload(val_rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    hash_groups = []
    for digest in sorted(set(path_hashes.values())):
        matching = sorted(key for key, value in path_hashes.items() if value == digest)
        if len(matching) > 1:
            hash_groups.append({"sha256": digest, "normalized_paths": matching})
    split_hashes = {
        "train": {path_hashes[path] for path in train_paths if path in path_hashes},
        "validation": {path_hashes[path] for path in val_paths if path in path_hashes},
    }
    hash_overlap = sorted(split_hashes["train"] & split_hashes["validation"])
    if hash_overlap:
        raise AssertionError(f"internal split shares identical video hashes: {hash_overlap[:3]}")

    source_fields = ("source_sequence_id", "original_video_id", "source_id", "sequence_id")
    source_id_rows = sum(any(row.get(field) not in (None, "") for field in source_fields) for row in rows)
    stats = {
        "source_annotation": str(Path(annotation_file)),
        "split_method": "deterministic greedy group assignment with row/category/task/answer marginal balancing",
        "seed": seed,
        "requested_validation_fraction": val_fraction,
        "actual_validation_fraction": len(val_rows) / len(rows),
        "total_questions": len(rows),
        "train_questions": len(train_rows),
        "validation_questions": len(val_rows),
        "train_unique_video_paths": len(train_paths),
        "validation_unique_video_paths": len(val_paths),
        "normalized_video_path_overlap": path_overlap,
        "normalized_video_path_overlap_count": len(path_overlap),
        "available_video_hashing": root is not None,
        "video_root": str(root) if root else None,
        "hashed_unique_video_paths": len(path_hashes),
        "missing_video_files": missing_paths if root else None,
        "missing_video_file_count": len(missing_paths) if root else None,
        "identical_file_hash_groups": hash_groups,
        "cross_split_identical_hash_overlap": hash_overlap,
        "source_sequence_id_rows": source_id_rows,
        "source_sequence_independence": "not established; annotations without an original source mapping cannot prove source-sequence independence",
        "category_distribution": {"train": _distribution(train_rows, "category"), "validation": _distribution(val_rows, "category")},
        "task_distribution": {"train": _distribution(train_rows, "task"), "validation": _distribution(val_rows, "task")},
        "answer_distribution": {"train": _distribution(train_rows, "answer"), "validation": _distribution(val_rows, "answer")},
    }
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# ReVA internal train-only grouped split", "",
        f"Source: `{Path(annotation_file)}`. Official validation and test annotations were not read.",
        f"Seed: {seed}; requested validation fraction: {val_fraction:.3f}; actual: {stats['actual_validation_fraction']:.4f}.",
        "Groups are normalized video paths, merged when available local files have the same SHA-256.",
        f"Path overlap: {stats['normalized_video_path_overlap_count']}; cross-split hash overlap: {len(hash_overlap)}.",
        f"Source-sequence independence: {stats['source_sequence_independence']}.",
        "", "See `split_statistics.json` for category, task, answer counts, hash groups, and missing-file details.", "",
    ]
    markdown_file.write_text("\n".join(markdown), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Pinned official train.json only")
    parser.add_argument("--output-dir", required=True, help="Separate derived output directory")
    parser.add_argument("--video-root", help="Optional local video snapshot root for existence checks and SHA-256 grouping")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing derived outputs explicitly")
    args = parser.parse_args()
    print(json.dumps(create_split(args.train, args.output_dir, val_fraction=args.val_fraction,
                                  seed=args.seed, video_root=args.video_root, overwrite=args.overwrite),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
