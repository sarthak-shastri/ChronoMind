import numpy as np


def uniform_indices(frame_count, num_frames):
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if frame_count == 1:
        return np.zeros(num_frames, dtype=np.int64)
    return np.rint(np.linspace(0, frame_count - 1, num_frames)).astype(np.int64)


def adaptive_indices(frame_count, num_frames, motion_scores=None):
    """Sample motion-weighted temporal quantiles, filling collisions uniformly and deterministically."""
    if num_frames <= 0 or frame_count <= 0:
        raise ValueError("frame_count and num_frames must be positive")
    if frame_count <= num_frames or motion_scores is None or len(motion_scores) == 0:
        return uniform_indices(frame_count, num_frames)
    scores = np.asarray(motion_scores, dtype=float).reshape(-1)
    if len(scores) != frame_count or not np.isfinite(scores).all() or scores.sum() <= 0:
        return uniform_indices(frame_count, num_frames)
    weights = np.maximum(scores, 0.0) + np.finfo(float).eps
    cdf = np.cumsum(weights) / weights.sum()
    targets = (np.arange(num_frames) + 0.5) / num_frames
    chosen = list(np.unique(np.searchsorted(cdf, targets).clip(0, frame_count - 1)))
    used = set(chosen)
    for index in uniform_indices(frame_count, num_frames):
        if len(chosen) >= num_frames:
            break
        if int(index) not in used:
            chosen.append(int(index))
            used.add(int(index))
    if len(chosen) < num_frames:
        for index in uniform_indices(frame_count, num_frames):
            if len(chosen) >= num_frames:
                break
            chosen.append(int(index))
    return np.sort(np.asarray(chosen[:num_frames], dtype=np.int64))
