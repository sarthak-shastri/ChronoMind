"""
BIMBA baseline preflight and inference handover harness.

STATUS: EXPERIMENTAL — NOT YET VALIDATED END-TO-END.

The previous Colab session successfully loaded the BIMBA adapter and base
model, but generation failed inside Qwen attention because query, key, and
value tensors had inconsistent dtypes. No valid prediction was obtained.

This file deliberately does not claim to perform inference. It provides a
reproducible preflight check until the attention dtype issue is resolved and
the official inference prompt is integrated and tested.
"""

import argparse
import json
import sys
from pathlib import Path

BASE_MODEL = "lmms-lab/LLaVA-Video-7B-Qwen2"
ADAPTER_MODEL = "mmiemon/BIMBA-LLaVA-Qwen2-7B"

EXPECTED_PACKAGES = [
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "peft",
    "decord",
    "av",
]

KNOWN_BLOCKER = (
    "Qwen attention received query.dtype=float16 while key/value.dtype=float32. "
    "The previous lm_head input-casting hook did not resolve this attention "
    "mismatch. Do not treat model loading alone as a successful prediction."
)


def preflight(repo_path: Path, video_path: Path | None = None) -> dict:
    result = {
        "status": "preflight_only",
        "repo_path": str(repo_path),
        "repo_exists": repo_path.is_dir(),
        "video_path": str(video_path) if video_path else None,
        "video_exists": video_path.is_file() if video_path else None,
        "base_model": BASE_MODEL,
        "adapter_model": ADAPTER_MODEL,
        "packages": {},
        "cuda_available": False,
        "gpu_name": None,
        "known_inference_blocker": KNOWN_BLOCKER,
        "prediction": None,
    }

    for package in EXPECTED_PACKAGES:
        try:
            module = __import__(package)
            result["packages"][package] = getattr(module, "__version__", "installed")
        except Exception as exc:
            result["packages"][package] = f"MISSING: {exc}"

    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        result["torch_error"] = str(exc)

    return result


def main():
    parser = argparse.ArgumentParser(description="BIMBA baseline preflight check")
    parser.add_argument("--repo", required=True, help="Path to BIMBA-LLaVA-NeXT")
    parser.add_argument("--video", help="Optional path to a ReVA video")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run environment/path checks only (the only supported mode currently)",
    )
    args = parser.parse_args()

    report = preflight(
        Path(args.repo),
        Path(args.video) if args.video else None,
    )
    print(json.dumps(report, indent=2))

    if not report["repo_exists"]:
        sys.exit("BIMBA repository path does not exist.")
    if args.video and not report["video_exists"]:
        sys.exit("Video path does not exist.")

    print("\nNo prediction was generated. See reports/baseline_handover.md.")


if __name__ == "__main__":
    main()
