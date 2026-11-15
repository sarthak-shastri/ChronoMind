"""Preflight ReVA annotation splits and report the limits of their provenance."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping


def preflight_splits(
    train: str | Path,
    validation: str | Path,
    test: str | Path | None = None,
    *,
    video_root: str | Path | None = None,
) -> dict:
    """Return separate path, source-ID, file-hash, and unknown-source findings.

    Exact video/path or byte-identical file matches are content reuse signals;
    they do not by themselves establish that an answer label leaked. Source
    sequence independence is unverifiable where the annotations lack IDs.
    """
    from scripts.check_reva_split_integrity import audit_report

    files: Mapping[str, str | Path] = {"train": train, "val": validation}
    if test is not None:
        files = {**files, "test": test}
    report = audit_report(files, video_root=video_root)
    unknown = report["unverifiable_source_sequence_relationships"]
    source_issues = [issue for issue in report["issues"] if issue.startswith("source-sequence overlap")]
    path_only_issues = [issue for issue in report["issues"] if not issue.startswith("source-sequence overlap")]
    status = "issues_found" if report["issues"] else (
        "passed_with_limitations" if unknown["status"] == "unverifiable" else "passed"
    )
    return {
        "status": status,
        "audited_splits": list(files),
        "issues": report["issues"],
        "path_only_issues": path_only_issues,
        "source_sequence_warnings": source_issues,
        "path_overlaps": report["path_overlaps"],
        "source_sequence_overlaps": report["source_sequence_overlaps"],
        "identical_content_hash_overlaps": report["identical_content_hash_overlaps"],
        "unverifiable_source_sequence_relationships": unknown,
        "video_hashing": report["video_hashing"],
        "interpretation": (
            "Overlap findings identify reused paths or byte-identical files; "
            "they do not alone prove answer leakage. Missing source IDs leave "
            "relationships between distinct clip paths unknown."
        ),
    }
