# Preprocessing comparison

E1-E7 were run on the same three small videos (the pilot sample listed in `video_sample_analysis.json`). Their video files were already in the Hugging Face cache. The first pass therefore measures decoding/sampling/transformation from cached files; the second pass measures reads from the extracted-frame cache. These results are a three-video pilot, not dataset-wide resource estimates.

| Experiment | Frames | Strategy | Success | First pass sec/video | Warm pass sec/video | Cached frame storage | Combined cache hit rate | Sampled process RSS |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| E1 | 4 | uniform | 3/3 | 0.343 | 0.069 | 5.24 MiB | 50% (second pass 100%) | 242.5 MiB |
| E2 | 8 | uniform | 3/3 | 0.570 | 0.095 | 10.58 MiB | 50% (second pass 100%) | 269.8 MiB |
| E3 | 12 | uniform | 3/3 | 0.804 | 0.111 | 15.88 MiB | 50% (second pass 100%) | 308.5 MiB |
| E4 | 16 | uniform | 3/3 | 1.009 | 0.129 | 21.18 MiB | 50% (second pass 100%) | 300.6 MiB |
| E5 | 32 | uniform | 3/3 | 1.959 | 0.245 | 42.44 MiB | 50% (second pass 100%) | 363.5 MiB |
| E6 | 8 | adaptive | 3/3 | 0.615 | 0.070 | 10.60 MiB | 50% (second pass 100%) | 342.3 MiB |
| E7 | 12 | adaptive | 3/3 | 0.839 | 0.102 | 15.93 MiB | 50% (second pass 100%) | 332.1 MiB |

All seven settings processed all three videos with no failures. The subset was 640x360, about six seconds per video. Cache-hit rate is 50% over two passes (first-pass misses plus warm-pass hits); the second-pass hit rate was 100%. RSS is the process working set sampled after each video and includes the Python/PyTorch runtime, so it is not incremental frame-only memory. Compressed sampled-frame caches are under ignored `data/cache/`, not Git. See `reports/preprocessing_runs/20261104T195122Z/results.json` for per-pass timings and exact measurements.

The adaptive strategy uses per-frame absolute grayscale differences on 64x36 sequentially decoded frames as motion scores, then deterministically samples weighted temporal quantiles. The three-video experiment does not establish a QA accuracy benefit; no model evaluation was performed.
