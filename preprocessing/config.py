from pathlib import Path

SUPPORTED_FRAME_COUNTS = (4, 8, 12, 16, 32)
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def resolve_video_path(video_root, relative_path):
    root = Path(video_root).expanduser().resolve()
    candidate = (root / str(relative_path)).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Video path escapes video_root: {relative_path}")
    return candidate
