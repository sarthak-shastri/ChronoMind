# ReVA split integrity and benchmark protocol

## Verified dataset snapshot and schema

Dataset: `ReVA-Benchmark/ReVA`, pinned repository revision `06224fa1b42e5e35677a8b7f2145e06307a9a4ee`. The annotation JSON has top-level `metadata` and `QA`; QA rows use `qa_id`, `video_path`, `subdir`, `dataset_name`, `category`, `task`, `question`, `options` (A/B/C/D), and `correct_answer`. Test answers are blank; train/val labels should be one of A/B/C/D.

## Paper protocol vs released annotation audit

The ReVA paper states that splitting was performed at the original video-sequence level to prevent adjacent clips from the same source sequence crossing splits. The released annotation schema contains no explicit `source_sequence_id` or equivalent field. Exact normalized `video_path` overlap in the pinned annotations is train/val 378, train/test 1,012, val/test 366, and all-three 365. Thus the released files do not satisfy exact video-path disjointness, even though the paper describes source-sequence disjointness. This audit cannot establish whether any distinct paths came from the same original sequence because the mapping is not published in these annotations.

## Evaluation inputs

The public dataset release includes train, validation, and test annotations; test answers are withheld (blank). Codabench generally evaluates either a result bundle or an executable code submission according to the benchmark organizer's configured phase. We could not locate a public ReVA Codabench task page or its task-specific input/reference files and scoring contract from the supplied project context or public search. Therefore the precise competition-phase inputs, prediction schema, and scoring implementation remain unverified; use the active Codabench task's Data/Rules and scoring package as authoritative before packaging a submission. Do not infer the private evaluation inputs from the public HF test JSON.

## Automated checks and recommendation

Run `python scripts/check_reva_split_integrity.py --train ... --val ... --test ... --video-root ...` before training/evaluation. It exits nonzero for exact path overlap, source ID overlap when an ID exists, duplicate QA IDs, missing local videos (when root supplied), invalid train/validation labels, or invalid options. `pytest tests/test_reva_split_integrity.py -q` exercises pass and fail cases.

Until the dataset owners provide original-source mapping, use normalized `video_path` (and source IDs if later provided) to create a deterministic grouped internal validation set from official training rows only. Keep the official test set as a final evaluation set, but report the released overlap issue and do not claim unseen-video generalization from results on the current overlapping split. Match the active Codabench phase's exact submission contract for competition scores.
