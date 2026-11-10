# ReVA leakage-safe internal split

Only official `train.json` QA rows are eligible for these derived splits. Before splitting, all train rows whose source-sequence ID or normalized video path occurs in either official validation or test are excluded. Remaining QA rows are grouped and assigned deterministically using seed 42 (or `--seed`). Official files are read-only inputs and are not changed.

## Counts

| Item | Count |
|---|---:|
| Official train QA rows | 15773 |
| Excluded because video/group is in official val or test | 15540 |
| Eligible train QA rows | 233 |
| Derived train QA / groups | 184 / 17 |
| Derived validation QA / groups | 49 / 3 |

Train/validation group overlap: **0**.
Training paths shared with either official evaluation split: **0**.
Validation paths shared with official test: **0**.

## Limitation

The annotation schema has no source-sequence identifier, so normalized `video_path` is the minimum available grouping key. These files guarantee path-level isolation, not hidden common-source isolation for differently named clips. Obtain the original-to-clip mapping from the dataset authors to strengthen this guarantee. The official validation and test files remain untouched and must never be added to training.
