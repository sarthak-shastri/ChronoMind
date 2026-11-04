# Dataset analysis (official ReVA Hub revision `06224fa1b42e5e35677a8b7f2145e06307a9a4ee`)

The pinned Hub manifest lists 1,273 files. Actual annotation filenames are `train.json`, `val.json`, and `test.json`. All three annotation files were inspected; three MP4s under 1 MB each were fetched for a decode pilot. No complete video corpus was downloaded.

## Split summary

| Split | QA records | Unique videos | Questions per video (min–max) | Answer counts | Duplicate IDs | Missing video refs | Paths absent from manifest |
|---|---:|---:|---:|---|---:|---:|---:|
| train | 15,773 | 1,045 | 1–23 | B 3,943; A 3,943; C 3,943; D 3,944 | 0 | 0 | 0 |
| val | 2,000 | 379 | 1–14 | B 500; A 500; C 500; D 500 | 0 | 0 | 0 |
| test | 4,000 | 1,014 | 1–12 | <empty> 4,000 | 0 | 0 | 0 |

The JSON files contain a `metadata` object and `QA` list. Observed QA fields include `qa_id`, `video_path`, `subdir`, `dataset_name`, `category`, `task`, `question`, `options`, and `correct_answer`. All rows in the three splits have the core fields populated, exactly A/B/C/D choice keys, no duplicate sample IDs, and referenced video paths present in the pinned manifest. Every test `correct_answer` is empty, as expected. Full per-split details are in `train_inspection.json`, `val_inspection.json`, and `test_inspection.json`.

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

## Video sample and remaining scope

Three small MP4s were fetched and decoded (see `video_sample_analysis.json`). They were all 640x360, around 30 FPS, about six seconds long, and each yielded eight RGB frames. This is a pilot only; it does not estimate full-dataset video properties or failure rate. E1–E7 sample metrics are in `preprocessing_comparison.md`.
