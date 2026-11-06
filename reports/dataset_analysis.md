# Dataset analysis (official ReVA Hub revision `06224fa1b42e5e35677a8b7f2145e06307a9a4ee`)

The pinned Hub manifest lists 1,273 files. Actual annotation filenames are `train.json`, `val.json`, and `test.json`. All three annotation files were inspected. For video analysis, 18 MP4s (26.0 MB total), one per observed `dataset_name`/`subdir` stratum, were selectively fetched and decoded. No complete video corpus was downloaded.

## Split summary

| Split | QA records | Unique videos | Questions per video (min–max) | Answer counts | Duplicate IDs | Missing video refs | Paths absent from manifest |
|---|---:|---:|---:|---|---:|---:|---:|
| train | 15,773 | 1,045 | 1–23 | B 3,943; A 3,943; C 3,943; D 3,944 | 0 | 0 | 0 |
| val | 2,000 | 379 | 1–14 | B 500; A 500; C 500; D 500 | 0 | 0 | 0 |
| test | 4,000 | 1,014 | 1–12 | <empty> 4,000 | 0 | 0 | 0 |

The JSON files contain a `metadata` object and `QA` list. Observed QA fields include `qa_id`, `video_path`, `subdir`, `dataset_name`, `category`, `task`, `question`, `options`, and `correct_answer`. The annotations do not provide a separate `video_id` field; the shared interface therefore returns `video_id=None` and uses the preserved `video_path` to resolve/cache the clip. All rows in the three splits have the core fields populated, exactly A/B/C/D choice keys, no duplicate sample IDs, and referenced video paths present in the pinned manifest. Every test `correct_answer` is empty, as expected. Full per-split details are in `train_inspection.json`, `val_inspection.json`, and `test_inspection.json`.

The same video paths appear in multiple splits: train/val share 378 paths, train/test share 1,012, and val/test share 366. This indicates the split is not video-disjoint (inference from exact path overlap); it should be surfaced to the evaluation owner and considered when interpreting validation results. Official split files were not changed.

## Category distribution

| Category | Train | Val | Test |
|---|---:|---:|---:|
| Causal Reasoning | 1,258 | 220 | 440 |
| Factual Perception | 3,763 | 670 | 1,340 |
| Spatial Reasoning | 7,180 | 610 | 1,220 |
| Temporal Understanding | 3,572 | 500 | 1,000 |

## Task distribution

| Task | Train | Val | Test |
|---|---:|---:|---:|
| Causation Reasoning | 401 | 90 | 180 |
| Change Detection | 1,012 | 250 | 500 |
| Consequence Reasoning | 461 | 50 | 100 |
| General Understanding | 1,032 | 90 | 180 |
| Geometric Relation | 2,875 | 200 | 400 |
| Hypothetical Reasoning | 396 | 80 | 160 |
| Object and Land Cover Recognition | 1,719 | 330 | 660 |
| Perspective and Viewpoint | 2,047 | 240 | 480 |
| Structural Layout | 2,258 | 170 | 340 |
| Temporal Grounding | 1,884 | 320 | 640 |
| Trend and Pattern | 1,688 | 180 | 360 |

## Video overlap between splits

| Pair | Shared video paths |
|---|---:|
| train / val | 378 |
| train / test | 1,012 |
| val / test | 366 |

## Video properties sample

The 18-video sample covers all 18 observed dataset/subdirectory strata. All 18 videos decoded and eight RGB frames were extracted from each. These are sample statistics only.

| Property | Sample result |
|---|---|
| Videos | 18 / 18 decoded; 0 failures |
| Downloaded video bytes | 26,023,294 (24.82 MiB) |
| Resolution counts | 640x360: 6, 640x640: 12 |
| FPS counts (rounded to 0.01) | 23.3: 1, 24: 11, 29.97: 4, 30: 2 |
| Duration (min / median / max) | 4.34s / 5.71s / 13.80s |
| Frame count (min / median / max) | 130 / 137.5 / 414 |

The sample does not estimate the full collection’s video-property distributions or corruption rate. See `reports/video_sample_analysis.json` for per-video measurements.
