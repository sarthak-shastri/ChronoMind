# Preprocessing comparison

E1–E7 ran on the same 18 source-stratified videos from the pinned ReVA revision. The videos were already present in the Hugging Face file cache. First-pass timings cover video metadata access, frame decode/sampling and transform; warm-pass timings cover cached sampled-frame reads and transforms. These are measurements for this selected sample, not a full-corpus estimate.

| Experiment | Frames | Strategy | Success / failures | First pass sec/video | Warm pass sec/video | Sample cache storage | Combined cache hit rate | Observed process RSS |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| E1 | 4 | uniform | 18/18 / 0 | 0.456 | 0.086 | 36.24 MiB | 50% (warm pass 100%) | 284.3 MiB |
| E2 | 8 | uniform | 18/18 / 0 | 0.753 | 0.121 | 73.47 MiB | 50% (warm pass 100%) | 333.5 MiB |
| E3 | 12 | uniform | 18/18 / 0 | 1.077 | 0.162 | 110.26 MiB | 50% (warm pass 100%) | 370.3 MiB |
| E4 | 16 | uniform | 18/18 / 0 | 1.118 | 0.150 | 147.37 MiB | 50% (warm pass 100%) | 341.4 MiB |
| E5 | 32 | uniform | 18/18 / 0 | 2.551 | 0.381 | 295.30 MiB | 50% (warm pass 100%) | 385.9 MiB |
| E6 | 8 | adaptive | 18/18 / 0 | 0.967 | 0.143 | 73.98 MiB | 50% (warm pass 100%) | 340.7 MiB |
| E7 | 12 | adaptive | 18/18 / 0 | 1.290 | 0.169 | 111.01 MiB | 50% (warm pass 100%) | 373.5 MiB |

All seven settings processed 18 videos without failures. The cache-hit rate is 50% across both passes because the first pass populated the frame cache and the second pass hit it; warm-pass hit rate was 100%. The observed process RSS is sampled after each video and includes the Python/PyTorch runtime, so it is not incremental frame-only memory. Compressed frame caches were written under ignored `data/cache/`; the measured per-experiment storage is in the table and detailed data is in `reports/preprocessing_runs/20261104T201206Z/results.json`.

Adaptive sampling computes per-frame absolute grayscale differences at 64x36 during a sequential video decode and uses deterministic motion-weighted temporal quantiles. These experiments measure preprocessing behavior only; they do not measure QA accuracy or imply that more frames improve task performance.
