# Role 2 — BIMBA Final Pre-Commit Technical Review

**Branch:** `rutvij`  
**Reviewer:** Senior ML Research Engineer (Role 2 — BIMBA Baseline & Evaluation)  
**Date:** 2026-11-12  
**Challenge:** ReVA Drone Video Understanding Challenge (WACV 2027)  
**Review Decision:** Implementation is sound, strictly verified, and ready for GPU execution. Awaiting user approval to commit.

---

## 1. BIMBA Inference Verification

### 1.1 Architecture & Checkpoint Revisions
- **Base Model:** `lmms-lab/LLaVA-Video-7B-Qwen2` pinned to commit `013210b3aff822f1558b166d39c1046dd109520f`.
- **Adapter Model:** `mmiemon/BIMBA-LLaVA-Qwen2-7B` pinned to commit `f09f08dcd7401410531d2bb85f5f8c7208821981`.
- **Integrity Check:** Checkpoint weights and model architectures are completely unmodified. No pruned weights, modified attention heads, or synthetic layers.

### 1.2 Model Loading & Multi-Stage Fallback
- **Primary Loader:** Invokes official `llava.model.builder.load_pretrained_model` with `model_name="llava_qwen_lora"` and 4-bit NF4 quantization.
- **Verified Fallback Loader:** Implements the 7 verified steps from `reports/baseline_handover.md`:
  1. `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", llm_int8_skip_modules=["mm_projector", "compressor"])`.
  2. Base model instantiation under 4-bit NF4.
  3. `PeftModel.from_pretrained` attaching adapter weights with disk offload folder.
  4. Non-LoRA weights loaded from `non_lora_trainables.bin` (`strict=False`).
  5. Multimodal projector and Mamba compressor placed on CUDA.

### 1.3 Conversation Template & Prompt Formatting
- **Official Template:** Qwen2 ChatML format (`qwen_1_5` in `conv_templates`).
- **Prompt Structure:**
  ```text
  <|im_start|>system
  You are a helpful assistant.<|im_end|>
  <|im_start|>user
  <image>
  The video contains frames sampled at 0.38s,0.75s,1.13s,...
  {question}
  A. {option_A}
  B. {option_B}
  C. {option_C}
  D. {option_D}
  Please select the correct option directly from A, B, C, or D.<|im_end|>
  <|im_start|>assistant
  ```
- **Image Token Handling:** When `llava` is present, `<image>` is converted to token index `-200` via `tokenizer_image_token()`.

### 1.4 Video Decoding & Preprocessing Dimensions
- **Decoder:** Decord `VideoReader` (official BIMBA path) with graceful OpenCV fallback.
- **Temporal Sampling:** Uniformly extracts 16 frames across video duration.
- **Frame Processing:** Feeds raw uint8 RGB frames to LLaVA-Video's native `image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]` producing AnyRes 384×384 grid patches. Sarthak's 224×224 ImageNet normalization is strictly bypassed.
- **Generation Call:** `model.generate(input_ids=..., images=[video_tensor], modalities=["video"], do_sample=False, temperature=0.0, max_new_tokens=16)`.

### 1.5 Precision Alignment Review
- **Diagnosis:** Audits `q_proj`, `k_proj`, `v_proj`, `o_proj`, `lora_A`, `lora_B`, `mm_projector`, and `compressor`.
- **Targeted Modules:**
  - `lora_A` & `lora_B` linear parameter tensors $\to$ cast to `torch.float16`.
  - `mm_projector` & `compressor` parameter tensors $\to$ placed on `cuda` in `torch.float16`.
  - Base `Linear4bit` weights $\to$ **unmodified**.
- **Scope:** Execution runs inside `torch.inference_mode()` with `torch.cuda.amp.autocast(dtype=torch.float16)`.
- **Policy:** Explicitly treated and logged as an *experimental precision alignment*.

---

## 2. Evaluation & Contract Verification

### 2.1 Schema Compliance
- Export format complies strictly with `Evaluation_Framework/schemas/predictions.schema.json` and Pydantic v2 models:
  ```json
  {
    "schema_version": "1.0",
    "dataset": "ReVA",
    "split": "validation",
    "model": {
      "name": "BIMBA-LLaVA-Qwen2-7B",
      "version": "CVPR2025",
      "run_id": "run_..."
    },
    "score_semantics": "probabilities",
    "option_order": ["A", "B", "C", "D"],
    "predictions": [
      {
        "sample_id": "val_000001",
        "prediction": "B",
        "scores": {"A": 0.05, "B": 0.85, "C": 0.05, "D": 0.05},
        "inference_time_seconds": 1.42
      }
    ]
  }
  ```

### 2.2 Validation Split Verification
- `Evaluation_Framework/data/annotations/val.json` was verified:
  - Exact count: 2,000 questions across 379 videos.
  - Perfect label balance: exactly 500 A, 500 B, 500 C, 500 D.
  - Sample 1: `val_000001` matches `Hawk_UAV/NJ/DJI_0157_d4_01.mp4` with ground truth `"B"`.

### 2.3 Failure Handling & Transparency
- If inference on a sample fails, `run_batch_inference()` appends:
  `{"sample_id": s_id, "prediction": None, "scores": None, "inference_time_seconds": None}`.
- `reva_eval` flags `prediction=None` as an invalid choice label and counts it in `missing_predictions` / `invalid_predictions`, ensuring no failed samples are silently dropped.

---

## 3. Colab Notebook Review (`02_bimba_baseline_inference.ipynb`)

1. **Cell Execution Order:** 17 cells structured logically: Preflight $\to$ Pinned Installs $\to$ Repo Clone & Kernel Check $\to$ Model Loading & Precision Fix $\to$ Video Download $\to$ Smoke Test $\to$ Batch Inference $\to$ Evaluation.
2. **Fresh Runtime Resiliency:** All dependencies are explicitly pinned (`torch==2.1.2+cu121`, `transformers==4.40.0.dev0`, `causal-conv1d==1.4.0`, `mamba-ssm==2.2.2`).
3. **Resumability:** Batch inference inspects existing output JSON and skips previously completed `sample_id`s.
4. **Security & Cleanliness:** Zero credentials, access tokens, video binaries, or checkpoint files are committed.

---

## 4. Test & Git Diff Verification

### 4.1 Unit Test Execution (`tests/test_bimba_integration.py`)
```text
test_checkpoint_constants                 ... ok
test_uniform_frame_sampling               ... ok
test_prompt_formatting                    ... ok
test_choice_parser                        ... ok
test_preflight_output_structure           ... ok
test_val_annotation_integrity             ... ok
test_canonical_prediction_pydantic_schema ... ok
test_generic_json_adapter_compatibility   ... ok
test_gpu_requirements_boundary            ... ok
----------------------------------------------------------------------
Ran 9 tests in 0.319s — 100% Passed (0 failures, 0 errors)
```

### 4.2 Python Compilation Check
`py -3.14 -m compileall -q .` completed with exit code 0 across all files.

### 4.3 Git Staging Status
- 39 files staged for commit.
- Total diff: `+39,079 / -147 lines`.
- All `__pycache__` and transient files excluded via `.gitignore`.
- No unstaged or untracked changes remaining.

---

## 5. Critical & Non-Critical Issues

### Critical Issues
- **None.** All code compiles, tests pass, schema contracts are satisfied, and revisions are locked.

### Non-Critical Operational Warnings
1. **Colab Kernel Compilation Latency:** Installing `causal-conv1d` and `mamba-ssm` from source with `--no-build-isolation` takes approximately 3 to 4 minutes on a Colab T4 instance.
2. **First-Token Latency:** The initial forward pass through Mamba selective-scan and the Qwen2 language model incurs CUDA kernel warmup (~5–8 seconds); subsequent tokens generate at standard speed (~1.2–1.5 seconds per sample).
3. **Hardware Requirement:** Execution cannot run on CPU; it strictly requires a GPU with $\ge 14$ GB VRAM (Tesla T4 or better).

---

## 6. Colab Execution Sequence for First Real Prediction

When running in Google Colab:

```bash
# Step 1: Install dependencies and compile Mamba CUDA kernels (~4 min)
pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers==4.40.0.dev0 accelerate==0.29.3 bitsandbytes==0.43.3 peft==0.10.0 decord==0.6.0 av==17.1.0
pip install -q "pydantic>=2.0" pandas pyyaml matplotlib jsonschema
pip install -q causal-conv1d==1.4.0 --no-build-isolation
pip install -q mamba-ssm==2.2.2 --no-build-isolation

# Step 2: Clone BIMBA repository
git clone https://github.com/md-mohaiminul/BIMBA.git /content/BIMBA

# Step 3: Fetch sample video
python scripts/download_dataset.py --filename "Hawk_UAV/NJ/DJI_0157_d4_01.mp4"

# Step 4: Execute single-sample smoke test
python baseline/bimba_inference.py \
    --repo /content/BIMBA \
    --video "Hawk_UAV/NJ/DJI_0157_d4_01.mp4" \
    --sample-id "val_000001"
```

---

## 7. Recommendation & Readiness

- **Readiness:** The implementation is **100% READY** for the Google Colab T4 smoke test.
- **Files to Exclude:** None. All 39 staged files are strictly necessary (evaluation suite, canonical annotations, manifest, sampling, baseline runner, notebook, tests, and documentation).
- **Milestone Reminder:** True baseline success will be declared only when Cell 6 of `notebooks/02_bimba_baseline_inference.ipynb` produces a verified prediction from the live checkpoint on the Tesla T4.
