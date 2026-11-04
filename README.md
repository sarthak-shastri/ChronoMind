# ChronoMind — ReVA data workstream

This repository contains the Member 1 data engineering and video preprocessing pipeline for the ReVA drone video QA challenge. Dataset media is not stored in Git.

## Environment

Use Python 3.10+ in VS Code. From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Discover actual Hugging Face files

The official dataset card has conflicting annotation filename examples. Discover the live file list and pin its revision before choosing exact annotation files:

```powershell
python scripts/download_dataset.py
```

The command writes `manifests/reva_manifest.json` and prints the real filenames. Then selectively fetch an exact annotation file (not all video data):

```powershell
python scripts/download_dataset.py --filename <exact-name-from-manifest>
```

For local dataset folders, run annotation inspection:

```powershell
python scripts/inspect_dataset.py <annotation-file> --video-root <dataset-root> --output reports/local_inspection.json
```

## Dataset interface

```python
from preprocessing import ReVADataset, make_dataloader

dataset = ReVADataset(annotation_file='path/to/train.json', video_root='path/to/ReVA',
                      num_frames=8, sampling='uniform', cache_dir='data/cache/frames')
sample = dataset[0]
```

Each item preserves the source sample ID, ordered A/B/C/D choices, answer if supplied, video ID, category, and task. `video_frames` is `torch.float32 [T,3,H,W]`, RGB, resized to 224x224 by default, scaled to [0,1] and ImageNet normalized. `missing_video='zeros'` returns a zero tensor and records the failure on the dataset instance; default `raise` surfaces errors.

Supported frame counts are 4, 8, 12, 16, and 32. Sampling is deterministic for a given video and configuration. Adaptive mode computes low-resolution per-frame motion scores and uses motion-weighted temporal quantiles.

## Verification and limitations

Run `python -m pytest -q`. Video decoding requires OpenCV (`opencv-python-headless`). The unit suite covers parser, sampling, cache, missing/corrupt video, tensor contract, unlabeled test answers, and collation. The three actual annotation splits were inspected, and end-to-end samples plus an 18-video source-stratified video analysis were run. Full-dataset video properties and corruption rates remain unknown; see `reports/dataset_analysis.md`.

Team flow: work on a member branch and open a PR into `dev`; promote stable `dev` changes to `main` after review.
## Access only what you need from Hugging Face

The videos are not a Python package to install. The full dataset card reports about 29.9 GB, but each member should fetch only annotation files and videos they actually use. `huggingface_hub` keeps downloaded files in a cache and reuses them on later requests; it does not create a shared live mount of every remote video ([Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download), [cache guide](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)). Each video still has to be transferred once before OpenCV can decode it. The loader now supports fetching each referenced MP4 lazily through the Hub and caching it locally.

First list the current files and write a revision-pinned manifest:

```powershell
python scripts/download_dataset.py
```

Copy the exact train annotation filename from the output, then fetch only that file (and repeat with validation if needed):

```powershell
python scripts/download_dataset.py --filename <exact-train-annotation-filename>
```

For direct video fetches, use the annotation file path cached by the Hub and pass `dataset_repo_id` plus the manifest `revision`:

```python
from preprocessing import ReVADataset
from preprocessing.hf_access import fetch_annotation_file

repo = "ReVA-Benchmark/ReVA"
revision = "<revision-from-manifest>"
cache = r"D:\ChronoMind_HF_Cache"
annotation = fetch_annotation_file(repo, "<exact-annotation-filename>", revision, cache)
dataset = ReVADataset(annotation, dataset_repo_id=repo, revision=revision,
                      hf_cache_dir=cache, num_frames=8, sampling="uniform")
sample = dataset[0]  # fetches only that sample's video; later reads use the Hub cache
```

Use a separate cache folder such as `D:\ChronoMind_HF_Cache`, not inside the Git repository. In Colab, mount Drive and set the cache to a folder under `MyDrive` if you want fetched files to survive runtime resets. A Drive-backed cache is persistent but can be slower; use `/content` for faster temporary downloads and persist the smaller sampled-frame cache to Drive if needed. Mounting Drive does not make remote videos stream without transfer.

## Run the preprocessing pilot

After the manifest and the selected 18 source-stratified videos are available in the Hugging Face cache, run:

```powershell
python scripts/run_preprocessing_experiments.py --limit-videos 18
```

The script runs E1-E7, writes metrics under `reports/preprocessing_runs/`, and places frame-cache arrays under ignored `data/cache/`. It does not download the complete video collection. The default sample covers 18 observed dataset/subdirectory strata. Increase `--limit-videos` only when you have enough disk and time.
## Optional Colab/Drive and feature-cache setup

The cache roots can be resolved from configuration. In Colab, mount Drive yourself, then point the persistent cache root at the mounted folder; temporary work remains on the local runtime disk:

```python
from google.colab import drive
drive.mount('/content/drive')
from preprocessing.storage import resolve_storage_paths
storage = resolve_storage_paths({"storage": {
    "drive_root": "/content/drive/MyDrive/ReVA_cache",
    "local_cache_root": "/content/reva_tmp"
}})
```

Pass `storage["hf_cache_dir"]` to `hf_cache_dir` and `storage["frame_cache_dir"]` to `cache_dir` when creating `ReVADataset`. If no Drive root is configured, both persistent caches use the configured local cache root. `storage["local_temp_dir"]` is for temporary extraction work. Drive mounting itself has not been tested in this environment because it requires your Google login.

For model-produced arrays, `NumpyFeatureCache` implements the optional `FeatureCache` `get`/`put` interface; it is separate from the raw-frame cache and does not run any model.

The `preprocessing.collate` module provides the DataLoader collator/factory. `preprocessing.feature_cache.NumpyFeatureCache` is an optional disk cache for NumPy-compatible model features.
