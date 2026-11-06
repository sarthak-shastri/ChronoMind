import hashlib
import os
import tempfile
from pathlib import Path
import numpy as np


class FrameCache:
    def __init__(self, root, namespace="frames"):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.hits = 0
        self.misses = 0

    def _path(self, key):
        digest = hashlib.sha256((self.namespace + str(key)).encode()).hexdigest()
        return self.root / f"{digest}.npz"

    def get(self, key):
        path = self._path(key)
        try:
            with np.load(path, allow_pickle=False) as data:
                frames = data["frames"]
            if frames.ndim == 4 and frames.shape[-1] == 3:
                self.hits += 1
                return frames
        except (OSError, KeyError, ValueError):
            pass
        self.misses += 1
        return None

    def put(self, key, frames):
        target = self._path(key)
        fd, tmp = tempfile.mkstemp(dir=str(self.root), suffix=".npz")
        os.close(fd)
        try:
            np.savez_compressed(tmp, frames=np.asarray(frames, dtype=np.uint8))
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
