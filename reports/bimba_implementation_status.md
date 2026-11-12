# BIMBA Baseline Implementation & Verification Status (Role 2)

**Branch:** `rutvij`  
**Engineer:** Senior ML Research Engineer (Role 2 — BIMBA Baseline & Evaluation)  
**Date:** 2026-11-12  
**Challenge:** ReVA Drone Video Understanding Challenge (WACV 2027)  
**Status:** Implementation complete; all integration unit tests passing; ready for Google Colab T4 validation inference.  

---

## 1. Executive Summary

In accordance with approved Phase 1–4 plans, the BIMBA baseline implementation on branch `rutvij` has been fully upgraded from a preflight stub into an end-to-end inference and evaluation harness without modifying unrelated branch code or importing unapproved models.

### Key Milestones Delivered:
1. **Selective Integration Completed**:
   - `Evaluation_Framework/src/reva_eval/`, `schemas/`, and `configs/default.yaml` imported from `origin/shubham`.
   - Canonical `val.json` (2,000 QA pairs across 379 videos; balanced at 500 A, 500 B, 500 C, 500 D) verified and imported.
   - Deterministic uniform frame sampling (`preprocessing/frame_sampling.py`), dataset manifest (`manifests/reva_manifest.json`), and dataset download utility (`scripts/download_dataset.py`) imported from `origin/sarthak`.
   - **Boundary Enforced**: Role 3's CLIP-based QA pipeline (`advanced_models/`, `role3_*.ipynb`, CLIP scripts) was **strictly excluded**.
2. **Complete BIMBA Inference Runner (`baseline/bimba_inference.py`)**:
   - Pinned base model (`lmms-lab/LLaVA-Video-7B-Qwen2` @ `013210b3aff822f1558b166d39c1046dd109520f`) and adapter (`mmiemon/BIMBA-LLaVA-Qwen2-7B` @ `f09f08dcd7401410531d2bb85f5f8c7208821981`).
   - Official Decord-based uniform 16-frame sampling with OpenCV fallback.
   - Official ChatML / Qwen2 prompt formatting (`qwen_1_5`) with `<image>`, sampled timestamps, and direct choice constraint.
   - Detailed diagnosis and smallest justified precision alignment fix for the Qwen2 attention dtype mismatch.
   - Strict `CanonicalPredictions` export for automated scoring via `reva_eval`.
3. **Colab Execution Notebook (`notebooks/02_bimba_baseline_inference.ipynb`)**:
   - 17 structured cells covering pinned environment installation, Mamba CUDA kernel verification, 4-bit NF4 quantized loading, single-sample smoke test (`val_000001`), resumable validation inference, and `reva_eval` scoring.
4. **Verification**:
   - All 9 integration and schema compatibility unit tests passed with 0 errors.

---

## 2. Component Integration Matrix

| Component | Source | Integrated into `rutvij`? | Rationale |
| :--- | :--- | :---: | :--- |
| `Evaluation_Framework/src/reva_eval/` | `origin/shubham` | **YES** | Core evaluation engine required for scoring predictions. |
| `Evaluation_Framework/schemas/` | `origin/shubham` | **YES** | Pydantic and JSON schemas defining canonical prediction format. |
| `Evaluation_Framework/data/annotations/val.json` | `origin/shubham` | **YES** | Official validation split (2,000 balanced samples; verified). |
| `Evaluation_Framework/data/annotations/train.json` | `origin/shubham` | **NO** | 10.4 MB file; unneeded for zero-shot baseline validation. |
| `preprocessing/frame_sampling.py` | `origin/sarthak` | **YES** | Deterministic temporal frame sampling (`uniform_indices`). |
| `manifests/reva_manifest.json` | `origin/sarthak` | **YES** | Pinned Hugging Face revision and dataset index. |
| `scripts/download_dataset.py` | `origin/sarthak` | **YES** | Single-sample video download utility from Hugging Face Hub. |
| `advanced_models/` (CLIP pipeline) | `origin/sarthak` | **STRICTLY NO** | Role 3 foundation model; unrelated to BIMBA. |
| `preprocessing/transformations.py` | `origin/sarthak` | **NO** | Incompatible 224×224 ImageNet normalization; BIMBA uses native AnyRes. |

---

## 3. Technical Diagnosis: Qwen2 Attention Dtype Clash & Precision Fix

### Root Cause Analysis
Under 4-bit NF4 quantization (`bitsandbytes`) on a 15 GB Tesla T4:
1. Base linear layers execute compute in `torch.float16`.
2. When the PEFT adapter (`mmiemon/BIMBA-LLaVA-Qwen2-7B`) is attached, adapter linear projections (`lora_A`, `lora_B`) default to `torch.float32`.
3. When non-LoRA trainables (`mm_projector`, `compressor`) are loaded from `non_lora_trainables.bin`, weights default to `torch.float32`.
4. The output addition $\mathbf{y} = \text{base}(\mathbf{x}) + \frac{\alpha}{r} \text{lora\_B}(\text{lora\_A}(\mathbf{x}))$ converts adapted attention projection outputs (`k_proj`, `v_proj`) to `float32`, while unadapted or unquantized inputs/projections remain `float16`.
5. During rotary embedding and attention dot-product:
   $$\mathbf{Q} \in \mathbb{R}^{\text{float16}}, \quad \mathbf{K} \in \mathbb{R}^{\text{float32}}, \quad \mathbf{V} \in \mathbb{R}^{\text{float32}}$$
   PyTorch throws: `RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float`.

### The Smallest Justified Fix
Implemented in `baseline/bimba_inference.py`:
1. `diagnose_model_dtypes(model)`: Scans all submodules, logs parameter dtypes, and flags float32/float16 divergences.
2. `apply_precision_alignment(model)`:
   - Traverses all LoRA modules (`lora_A`, `lora_B`) and casts their weights/biases to `torch.float16`.
   - Moves and casts `mm_projector` and Mamba `compressor` to `cuda` in `torch.float16`.
   - Leaves 4-bit quantized base linear weights untouched (`Linear4bit`).
3. Wraps forward pass in `torch.inference_mode()` and `torch.cuda.amp.autocast(dtype=torch.float16)`.
4. **Transparency Statement**: This precision alignment is treated as an *experimental fix* and logged explicitly during model loading.

---

## 4. Verification Test Results (`tests/test_bimba_integration.py`)

Run Environment: Python 3.14 / Pydantic 2.13.4  
Result: **9 passed, 0 failures, 0 errors (0.20s)**

```
test_checkpoint_constants                 ... ok (Verified base model and adapter revisions)
test_uniform_frame_sampling               ... ok (Verified 16-frame sampling and edge cases)
test_prompt_formatting                    ... ok (Verified ChatML format, image tokens, timestamps)
test_choice_parser                        ... ok (Verified exact letters, statements, punctuation)
test_preflight_output_structure           ... ok (Verified preflight schema)
test_val_annotation_integrity             ... ok (Verified 2,000 QA pairs, 379 videos, 500 A/B/C/D)
test_canonical_prediction_pydantic_schema ... ok (Verified CanonicalPredictions Pydantic model)
test_generic_json_adapter_compatibility   ... ok (Verified GenericJSONAdapter ingestion)
test_gpu_requirements_boundary            ... ok (Asserted hardware/GPU dependency boundaries)
```

### CPU-Verifiable vs. GPU-Dependent Tasks
- **CPU-Verifiable (Complete)**:
  - Dataset schema and annotation integrity verification.
  - Uniform temporal sampling algorithm.
  - ChatML prompt construction and options formatting.
  - Choice regex parser and score normalization.
  - Pydantic schema validation for model predictions.
  - `reva_eval` adapter compatibility.
- **Requires Google Colab Tesla T4 GPU (Next Step)**:
  - Compilation of CUDA extensions (`causal_conv1d_cuda`, `selective_scan_cuda`).
  - 4-bit NF4 quantized base model loading.
  - Live execution of `model.generate()`.
  - Producing the first real, verified prediction for `val_000001` (`DJI_0157_d4_01.mp4`).

---

## 5. Next Steps for Colab Execution

1. Open `notebooks/02_bimba_baseline_inference.ipynb` in Google Colab connected to a T4 GPU.
2. Execute Cells 1–3 to install dependencies and compile Mamba CUDA kernels.
3. Execute Cell 4 to load the 4-bit model and verify precision alignment.
4. Execute Cell 5–6 to run the smoke test on `val_000001`:
   - Sample: `DJI_0157_d4_01.mp4`
   - Question: *"At what point in the video does the road with moving vehicles first appear?"*
   - Ground Truth: `B` (4.2 seconds)
   - Record first real prediction, latency, and peak VRAM.
5. Execute Cells 7–8 to run validation inference and compute official evaluation metrics via `reva_eval`.
