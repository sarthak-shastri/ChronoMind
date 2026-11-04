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

Supported frame counts are 4, 8, 12, 16, and 32. Sampling is deterministic for a given video and configuration. The adaptive helper accepts motion scores supplied by a caller; automatic motion-score calculation remains a future experiment.

## Verification and limitations

Run `python -m pytest -q`. Video decoding requires OpenCV (`opencv-python-headless`). Tests use synthetic annotations and do not substitute for real video validation. The dataset API was unreachable from the current runtime during setup, so current repository revision, true filenames, real sample statistics, and end-to-end real-video validation are still pending; see `reports/dataset_analysis.md`.

Team flow: work on a member branch and open a PR into `dev`; promote stable `dev` changes to `main` after review.
