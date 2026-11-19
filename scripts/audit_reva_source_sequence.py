from __future__ import annotations
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")


def norm_path(value: str) -> str:
    return re.sub(r"\\+", "/", str(value).strip()).lstrip("./").casefold()


def load_rows(path: Path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        return data, data.get("QA", [])
    if isinstance(data, list):
        return None, data
    raise ValueError(f"Unsupported annotations: {path}")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Audit ReVA split leakage by video path, source metadata, and local file hash.")
    ap.add_argument("--annotation-dir", type=Path, required=True, help="Directory containing official train.json, val.json, test.json")
    ap.add_argument("--video-root", type=Path, help="Optional root used to verify referenced files and compute hashes")
    ap.add_argument("--report", type=Path, default=Path("reports/reva_source_sequence_audit.md"))
    ap.add_argument("--corrected-dir", type=Path, help="Optional output for non-overlapping copies; originals remain untouched")
    args = ap.parse_args()

    payloads, rows_by_split, path_splits, path_meta, hash_splits = {}, {}, defaultdict(set), {}, defaultdict(set)
    exact_ids = defaultdict(set)
    missing = defaultdict(list)
    for split in SPLITS:
        payload, rows = load_rows(args.annotation_dir / f"{split}.json")
        payloads[split], rows_by_split[split] = payload, rows
        for row in rows:
            vp = row.get("video_path", row.get("file_path"))
            if not vp:
                continue
            key = norm_path(vp)
            path_splits[key].add(split)
            path_meta[key] = (str(row.get("dataset_name", "")), str(row.get("subdir", "")), str(vp))
            for field in ("video_id", "source_id", "source_sequence_id", "sequence_id", "original_video_id"):
                if row.get(field) not in (None, ""):
                    exact_ids[field].add(str(row[field]))
            if args.video_root:
                local = args.video_root.joinpath(*str(vp).replace("\\", "/").split("/"))
                if local.is_file():
                    hash_splits[sha256(local)].add(split)
                else:
                    missing[split].append(str(vp))

    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    overlaps = {f"{a}-{b}": {p for p, ss in path_splits.items() if a in ss and b in ss} for a, b in pairs}
    path_counts = {s: sum(s in ss for ss in path_splits.values()) for s in SPLITS}
    triple = {p for p, ss in path_splits.items() if len(ss) == 3}

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# ReVA dataset and source-sequence audit", "", "## Method and scope", "",
      "Grouping key for this audit is the normalized annotation `video_path`. Annotation records do not define an explicit video/source-sequence identifier (checked: `video_id`, `source_id`, `source_sequence_id`, `sequence_id`, `original_video_id`). `dataset_name` and `subdir` are provenance labels, not sequence IDs.", "",
      "Exact file hashes are computed only for files present under `--video-root`; hash matches across different paths identify byte-identical files, while non-matching hashes do not prove different source sequences.", "", "## Split overlap by exact normalized video path", "",
      "| Split | QA rows | Unique video paths |", "|---|---:|---:|"]
    for s in SPLITS: lines.append(f"| {s} | {len(rows_by_split[s])} | {path_counts[s]} |")
    lines += ["", "| Pair | Shared paths |", "|---|---:|"]
    for k, v in overlaps.items(): lines.append(f"| {k} | {len(v)} |")
    lines.append(f"| all three | {len(triple)} |")
    lines += ["", "## Overlap by dataset, subdirectory, and video path", "", "| Dataset | Subdirectory | Video path | Splits |", "|---|---|---|---|"]
    for key, ss in sorted(path_splits.items()):
        if len(ss) > 1:
            ds, sub, vp = path_meta[key]
            lines.append(f"| {ds} | {sub} | `{vp}` | {', '.join(sorted(ss))} |")
    hash_overlap = sum(len(ss) > 1 for ss in hash_splits.values())
    lines += ["", "## File verification", "", f"Video root supplied: `{args.video_root if args.video_root else 'no'}`.", f"Shared SHA-256 groups among available files: {hash_overlap}.", f"Missing referenced local files: {sum(map(len, missing.values()))}.", ""]
    if exact_ids:
        lines.append("Identifier-like annotation fields found: " + ", ".join(sorted(exact_ids)) + ".")
    else:
        lines.append("No explicit source/video ID fields were present in the annotation records.")
    lines += ["", "## Recommendation", "", "Use the official source-sequence ID as the grouping key only if ReVA provides a mapping from derived clips to original sequences. The current annotations expose no such ID. Until that mapping is obtained, group by normalized `video_path`; additionally group byte-identical files by SHA-256 when the videos are available. Do not group solely by `dataset_name` or `subdir`, which would merge unrelated videos.", "", "For a leakage-safe local benchmark, preserve the official test set as held out, remove from train every video path (and verified identical-file hash group) found in test, then remove from train any paths retained in validation. Keep originals intact and label derived files clearly; disclose that this creates a new benchmark and can change sample balance. If corrected files are requested, script output below implements path-level filtering and leaves test unchanged."]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.corrected_dir:
        test_paths = {norm_path(r.get("video_path", r.get("file_path", ""))) for r in rows_by_split["test"]}
        val_paths = {norm_path(r.get("video_path", r.get("file_path", ""))) for r in rows_by_split["val"]} - test_paths
        filtered = {
            "train": [r for r in rows_by_split["train"] if norm_path(r.get("video_path", r.get("file_path", ""))) not in test_paths | val_paths],
            "val": [r for r in rows_by_split["val"] if norm_path(r.get("video_path", r.get("file_path", ""))) not in test_paths],
            "test": rows_by_split["test"],
        }
        args.corrected_dir.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            original = payloads[split]
            data = dict(original) if isinstance(original, dict) else {"metadata": {}, "QA": []}
            data["QA"] = filtered[split]
            data.setdefault("metadata", {})["derived_from"] = f"{split}.json"
            data["metadata"]["leakage_filter"] = "path-disjoint; test precedence; generated by audit_reva_source_sequence.py"
            (args.corrected_dir / f"{split}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.corrected_dir / "README.md").write_text("Derived local benchmark copies. Official annotation files were not modified. Test paths were prioritized; validation excludes test paths; training excludes both. No source-sequence IDs were available, so path disjointness is guaranteed but source-sequence disjointness is not. Review before use.\n", encoding="utf-8")
    print(f"Wrote {args.report}")
    if args.corrected_dir: print(f"Wrote corrected copies to {args.corrected_dir}")

if __name__ == "__main__": main()
