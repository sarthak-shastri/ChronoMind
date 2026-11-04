# Dataset analysis (official ReVA Hub revision `06224fa1b42e5e35677a8b7f2145e06307a9a4ee`)

The live Hub manifest lists 1,273 repository files and confirms the annotation filenames `train.json`, `val.json`, and `test.json`. Only `train.json` and `val.json` were downloaded for this inspection; video files were not downloaded. `test.json` annotations and answers have not been opened. See `manifests/reva_manifest.json` for the pinned revision and exact file list.

## Split summary

| Split | QA records | Unique referenced videos | Questions per video (min–max) | Answers |
|---|---:|---:|---:|---|
| Train | 15,773 | 1,045 | 1–23 | A 3,943; B 3,943; C 3,943; D 3,944 |
| Validation (`val`) | 2,000 | 379 | 1–14 | A 500; B 500; C 500; D 500 |

Both annotation files parse as JSON objects with a `QA` list. Record fields observed: `qa_id`, `video_path`, `subdir`, `dataset_name`, `category`, `task`, `question`, `options`, `correct_answer`. There are no duplicate sample IDs in either inspected split, no records with missing video references, and every referenced video path is present in the pinned Hub manifest. All inspected records have exactly the A/B/C/D option keys.

## Category counts

| Category | Train | Validation |
|---|---:|---:|
| Factual Perception | 3,763 | 670 |
| Temporal Understanding | 3,572 | 500 |
| Spatial Reasoning | 7,180 | 610 |
| Causal Reasoning | 1,258 | 220 |

## Task counts

| Task | Train | Validation |
|---|---:|---:|
| General Understanding | 1,032 | 90 |
| Object and Land Cover Recognition | 1,719 | 330 |
| Change Detection | 1,012 | 250 |
| Temporal Grounding | 1,884 | 320 |
| Trend and Pattern | 1,688 | 180 |
| Geometric Relation | 2,875 | 200 |
| Structural Layout | 2,258 | 170 |
| Perspective and Viewpoint | 2,047 | 240 |
| Causation Reasoning | 401 | 90 |
| Consequence Reasoning | 461 | 50 |
| Hypothetical Reasoning | 396 | 80 |

## Remaining data checks

The manifest confirms all annotation-referenced MP4 paths exist in the Hub repository. A deliberately small subset of three videos (under 1 MB each) was fetched on demand and decoded successfully. Each yielded eight RGB frames. Measured properties: `Hawk_UAV/NJ/DJI_0157_d4_01.mp4` — 29.97 FPS, 180 frames, 6.006 s, 640x360; `VisDrone/uav0000009_03358_v_01.mp4` — 30 FPS, 180 frames, 6.000 s, 640x360; `Hawk_UAV/BE/DJI_0092_d4_04.mp4` — 29.97 FPS, 179 frames, 5.973 s, 640x360. Detailed measurements and cache locations are recorded in `reports/video_sample_analysis.json`.

These three files only validate sample decoding; they do not estimate dataset-wide video properties or corrupted-video rates. Full preprocessing timing/storage experiments remain pending. The loader was also exercised end-to-end on the first train and validation rows; both returned `[8,3,224,224]` float32 tensors without errors.
