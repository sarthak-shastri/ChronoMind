"""Optional disk cache interface for model-produced NumPy feature arrays."""
from abc import ABC, abstractmethod
import hashlib
import os
from pathlib import Path
import tempfile
import numpy as np

class FeatureCache(ABC):
    """Minimal feature-cache contract; model code may supply another implementation."""
    @abstractmethod
    def get(self,key):
        raise NotImplementedError
    @abstractmethod
    def put(self,key,features):
        raise NotImplementedError

class NumpyFeatureCache(FeatureCache):
    """Compressed, pickle-free disk cache for NumPy-compatible feature arrays."""
    def __init__(self,root,namespace="features"):
        self.root=Path(root).expanduser(); self.root.mkdir(parents=True,exist_ok=True)
        self.namespace=namespace; self.hits=0; self.misses=0
    def _path(self,key):
        digest=hashlib.sha256((self.namespace+str(key)).encode()).hexdigest()
        return self.root/f"{digest}.npz"
    def get(self,key):
        path=self._path(key)
        try:
            with np.load(path,allow_pickle=False) as data: value=data["features"]
            self.hits+=1; return value
        except (OSError,KeyError,ValueError):
            self.misses+=1; return None
    def put(self,key,features):
        target=self._path(key); fd,tmp=tempfile.mkstemp(dir=str(self.root),suffix=".npz"); os.close(fd)
        try:
            np.savez_compressed(tmp,features=np.asarray(features))
            os.replace(tmp,target)
        finally:
            if os.path.exists(tmp): os.remove(tmp)
