"""Small-file and per-video access helpers for the official Hugging Face dataset."""
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download


def list_dataset_files(repo_id="ReVA-Benchmark/ReVA", revision="main"):
    api = HfApi()
    info = api.dataset_info(repo_id, revision=revision)
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=info.sha)
    return {"repo_id": repo_id, "repo_type": "dataset", "revision": info.sha, "files": files}


def fetch_dataset_file(repo_id, filename, revision=None, cache_dir=None, local_files_only=False):
    """Fetch one exact dataset file; Hub cache prevents repeat downloads."""
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename,
                           revision=revision, cache_dir=str(cache_dir) if cache_dir else None,
                           local_files_only=local_files_only)
    return Path(path)


def fetch_annotation_file(repo_id, filename, revision=None, cache_dir=None):
    return fetch_dataset_file(repo_id, filename, revision, cache_dir)
