# ReVA Dataset Schema Audit

**Audit Date**: 2026-11-06  
**Auditor**: ReVA Universal Evaluation Framework Infrastructure Agent  
**Environment**: ChronoMind Workspace (`data/annotations/`)  

---

## 1. Executive Summary

An exhaustive inspection of the actual annotation files located in `data/annotations/` (`val.json`, `test.json`, `train.json`) was conducted.

### Key Audit Findings:
1. **Schema Format**: All three annotation files adhere to the **current top-level `QA` list format** (`{"metadata": {...}, "QA": [...]}`). None of the files use the legacy `videos` mapping format.
2. **Field Consistency**: Exactly 100% of records in all splits contain the identical set of 9 fields:
   - `qa_id` (string, unique within split)
   - `video_path` (string, relative path to video file)
   - `subdir` (string, video sub-directory or scene classification)
   - `dataset_name` (string, source drone dataset name)
   - `category` (string, one of 4 high-level taxonomy categories)
   - `task` (string, one of 11 lower-level subtask categories)
   - `question` (string, prompt text)
   - `options` (dictionary with exact keys `["A", "B", "C", "D"]`)
   - `correct_answer` (string: `"A"`, `"B"`, `"C"`, `"D"` for train/val; empty string `""` for test)
3. **No `video_id` Field**: As predicted in the specification, there is no top-level `video_id` field in the QA items. `video_path` serves as the primary video identifier.
4. **Test Split is Unlabeled**: All 4,000 QA pairs in `test.json` have `correct_answer: ""`. The evaluation framework must correctly flag this split as `unlabeled_ground_truth` and refuse to calculate accuracy metrics.
5. **Answer Balance in Validation**: The validation set (`val.json`, 2,000 items) is perfectly balanced with exactly 500 questions for each choice: 500 `A`, 500 `B`, 500 `C`, 500 `D`.

---

## 2. File-by-File Audit Breakdown

### 2.1 Validation Split (`data/annotations/val.json`)
- **File Size**: 1,293,436 bytes (~1.29 MB)
- **Top-Level Keys**: `["metadata", "QA"]`
- **Total Questions (`QA` length)**: 2,000
- **Unique `qa_id` count**: 2,000 (0 duplicates)
- **Metadata**:
  - `generated_at`: "2026-04-25"
  - `total_videos`: 379
  - `total_questions`: 2,000
- **Sample Item**:
  ```json
  {
    "qa_id": "val_000001",
    "video_path": "Hawk_UAV/NJ/DJI_0157_d4_01.mp4",
    "subdir": "NJ",
    "dataset_name": "Hawk_UAV",
    "category": "Temporal Understanding",
    "task": "Temporal Grounding",
    "question": "At what point in the video does the road with moving vehicles first appear?",
    "options": {
      "A": "At 5.0 seconds",
      "B": "At 4.2 seconds",
      "C": "At 2.5 seconds",
      "D": "At 3.8 seconds"
    },
    "correct_answer": "B"
  }
  ```
- **Category Distribution**:
  - `Factual Perception`: 670 (33.5%)
  - `Spatial Reasoning`: 610 (30.5%)
  - `Temporal Understanding`: 500 (25.0%)
  - `Causal Reasoning`: 220 (11.0%)
- **Task Distribution (11 Subtasks)**:
  - `Object and Land Cover Recognition`: 330
  - `Temporal Grounding`: 320
  - `Change Detection`: 250
  - `Perspective and Viewpoint`: 240
  - `Geometric Relation`: 200
  - `Trend and Pattern`: 180
  - `Structural Layout`: 170
  - `General Understanding`: 90
  - `Causation Reasoning`: 90
  - `Hypothetical Reasoning`: 80
  - `Consequence Reasoning`: 50
- **Answer Label Distribution**:
  - `A`: 500 (25.0%)
  - `B`: 500 (25.0%)
  - `C`: 500 (25.0%)
  - `D`: 500 (25.0%)

---

### 2.2 Test Split (`data/annotations/test.json`)
- **File Size**: 2,606,665 bytes (~2.61 MB)
- **Top-Level Keys**: `["metadata", "QA"]`
- **Total Questions (`QA` length)**: 4,000
- **Unique `qa_id` count**: 4,000 (0 duplicates)
- **Metadata**:
  - `generated_at`: "2026-04-25"
  - `total_videos`: 1,014
  - `total_questions`: 4,000
- **Sample Item**:
  ```json
  {
    "qa_id": "test_000001",
    "video_path": "VisDrone/uav0000009_03358_v_01.mp4",
    "subdir": "VisDrone",
    "dataset_name": "VisDrone",
    "category": "Factual Perception",
    "task": "Object and Land Cover Recognition",
    "question": "How many cars are visible in the parking lot at the 00:00 mark of the video?",
    "options": {
      "A": "40-50",
      "B": "10-20",
      "C": "20-30",
      "D": "30-40"
    },
    "correct_answer": ""
  }
  ```
- **Category Distribution**:
  - `Factual Perception`: 1,340 (33.5%)
  - `Spatial Reasoning`: 1,220 (30.5%)
  - `Temporal Understanding`: 1,000 (25.0%)
  - `Causal Reasoning`: 440 (11.0%)
- **Task Distribution (11 Subtasks)**:
  - `Object and Land Cover Recognition`: 660
  - `Temporal Grounding`: 640
  - `Change Detection`: 500
  - `Perspective and Viewpoint`: 480
  - `Geometric Relation`: 400
  - `Trend and Pattern`: 360
  - `Structural Layout`: 340
  - `General Understanding`: 180
  - `Causation Reasoning`: 180
  - `Hypothetical Reasoning`: 160
  - `Consequence Reasoning`: 100
- **Answer Label Distribution**:
  - `""` (empty string): 4,000 (100.0%)

---

### 2.3 Training Split (`data/annotations/train.json`)
- **File Size**: 10,380,722 bytes (~10.38 MB)
- **Top-Level Keys**: `["metadata", "QA"]`
- **Total Questions (`QA` length)**: 15,773
- **Unique `qa_id` count**: 15,773 (0 duplicates)
- **Sample Item**:
  ```json
  {
    "qa_id": "train_000001",
    "video_path": "VisDrone/uav0000009_03358_v_01.mp4",
    "subdir": "VisDrone",
    "dataset_name": "VisDrone",
    "category": "Factual Perception",
    "task": "General Understanding",
    "question": "Which of the following activities could be the best description of this video?",
    "options": {
      "A": "Fence building",
      "B": "Group walking",
      "C": "Car racing",
      "D": "Tree planting"
    },
    "correct_answer": "B"
  }
  ```
- **Answer Label Distribution**:
  - `D`: 3,944
  - `B`: 3,943
  - `A`: 3,943
  - `C`: 3,943

---

## 3. Observed Taxonomy Mapping

The metadata in the annotation files defines the exact official taxonomy hierarchy:

| Category | Associated Tasks |
| :--- | :--- |
| **Factual Perception** | General Understanding, Object and Land Cover Recognition, Change Detection |
| **Temporal Understanding** | Temporal Grounding, Trend and Pattern |
| **Spatial Reasoning** | Geometric Relation, Structural Layout, Perspective and Viewpoint |
| **Causal Reasoning** | Causation Reasoning, Consequence Reasoning, Hypothetical Reasoning |

The evaluator must dynamically discover categories and tasks from the loaded ground-truth samples, but this observed list confirms that exactly 4 categories and 11 tasks exist across all splits without exception.

---

## 4. Evaluation Framework Design Implications

1. **Loader Design**:
   - Primary path: Parse `QA` array from top-level dictionary containing `metadata` and `QA`.
   - Legacy path: Provide compatibility for legacy `videos` dictionary (`{video_id: {qa_list: [...]}}`) as specified, tested via synthetic fixtures.
   - Ground Truth Record mapping:
     - `qa_id` -> `sample_id`
     - `video_path` -> `video_path`
     - `subdir` -> `subdir`
     - `dataset_name` -> `dataset_name`
     - `question` -> `question`
     - `options` -> `options` (dictionary mapping A, B, C, D to option text)
     - `category` -> `category`
     - `task` -> `task`
     - `correct_answer`: If non-empty (`"A"`, `"B"`, `"C"`, `"D"`), set `ground_truth`. If empty `""` or null, set `ground_truth: null` and mark the canonical dataset as `labeled: false`.
2. **Strict Validation**:
   - Assert all `sample_id` values in ground truth are unique.
   - Assert options keys are strictly `["A", "B", "C", "D"]`.
3. **No Network Access**:
   - The framework operates entirely on local JSON annotations and predictions. No video downloads or external dependencies are required.

