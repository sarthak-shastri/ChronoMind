# Role 2 — BIMBA Baseline Handover

**Project:** ChronoMind — ReVA Drone Video Understanding Challenge (WACV 2027)
**Branch:** `rutvij`
**Role:** Lead Baseline Model & Video-Language Model Research Engineer
**Status:** Experimental; model loading was achieved in a prior session, but end-to-end inference remains unresolved.

## 1. Scope

Role 2 is responsible for reproducing the BIMBA video-language baseline, attempting inference on a ReVA sample, and handing over a documented baseline harness to the reasoning/fusion workstream.

This handover does not claim that a valid prediction has been produced.

## 2. Model references

- Base model: `lmms-lab/LLaVA-Video-7B-Qwen2`
- Adapter: `mmiemon/BIMBA-LLaVA-Qwen2-7B`
- Adapter revision previously accessed: `f09f08dcd7401410531d2bb85f5f8c7208821981`
- Base revision previously accessed: `013210b3aff822f1558b166d39c1046dd109520f`

No credentials or access tokens are included in this repository.

## 3. Previously verified environment

The earlier Colab session used:

| Component | Version / detail |
|---|---|
| GPU | NVIDIA Tesla T4, 14.56 GB VRAM |
| Python | 3.10.22 |
| PyTorch | 2.1.2+cu121 |
| Transformers | 4.40.0.dev0 |
| bitsandbytes | 0.43.3 |
| Accelerate | 0.29.3 |
| PEFT | 0.10.0 |
| NumPy | 1.26.4 |
| Decord | 0.6.0 |
| PyAV | 17.1.0 |
| causal-conv1d | 1.4.0 CUDA build |
| mamba-ssm | 2.2.2 CUDA build |

`causal_conv1d_cuda` and `selective_scan_cuda` imported successfully, and a Mamba forward test passed. An OpenCLIP-not-installed warning was observed; it was not the cause of the inference failure.

These versions describe the previous environment and have not been recreated in the current runtime.

## 4. Loading work verified in the previous session

The following loading changes were applied and checked:

1. 4-bit NF4 quantization was configured with double quantization and float16 compute.
2. `mm_projector` and `compressor` were excluded from 4-bit quantization.
3. The adapter configuration was reconciled with the base model vocabulary size.
4. `non_lora_trainables.bin` was loaded with `strict=False` and `assign=True`.
5. The PEFT adapter was loaded with an offload directory.
6. LoRA merging was skipped in the relevant LLaVA/Qwen loading branch.
7. The projector and compressor were explicitly moved to CUDA after model loading.

The resulting model was a `PeftModelForCausalLM`. A parameter inspection found zero meta parameters. Vision tower, projector, compressor, and language model parameters were placed on CUDA after the manual placement step.

A warning about newly initialized compressor weights appeared during initial model construction; the adapter's non-LoRA weights were loaded afterward. This warning alone was not treated as proof of missing trained compressor weights.

## 5. Inference attempt and blocker

A ReVA sample video was decoded into 16 frames and preprocessed at 384 × 384 resolution. The video duration was approximately 6.01 seconds.

Generation first failed because image features were on CUDA while the multimodal projector was on CPU. Moving the projector and compressor to CUDA resolved that device mismatch.

The next generation attempt failed in Qwen2 attention with a dtype mismatch:

- Query dtype: float16
- Key dtype: float32
- Value dtype: float32

A forward pre-hook that cast the `lm_head` input did not resolve the attention mismatch because the failure occurred earlier in the attention path.

**Result: no valid prediction was produced.** There is no measured answer accuracy, confidence, or successful inference latency to report.

## 6. Sample question attempted

- Video: `DJI_0157_d4_01.mp4`
- Question: “At what point in the video does the road with moving vehicles first appear?”
- Choices: A 5.0s; B 4.2s; C 2.5s; D 3.8s
- Sampling: 16 approximately uniformly spaced frames across the 6.01-second clip
- Prediction: Not available; generation failed before returning a valid answer.

Do not infer or fill in a prediction from the answer choices.

## 7. Current runtime status

The latest Colab runtime recovery check showed:

- `/content` exists.
- `/content/BIMBA` is missing.
- `/content/bimba-env` is missing.
- `/content/ChronoMind` was initially missing and was recloned on branch `rutvij`.
- Hugging Face model cache is missing, including both base and adapter snapshots.
- Tesla T4 GPU is available.

Therefore, the prior environment and checkpoints must be downloaded and rebuilt before another inference attempt. The current ChronoMind checkout contains the project README and these handover artifacts.

## 8. Recommended next debugging steps

1. Recreate the BIMBA environment and pin the versions listed above.
2. Clone the exact BIMBA repository revision used for the earlier test, if available.
3. Download the same base and adapter revisions.
4. Reapply and review the documented `builder.py` changes rather than blindly applying duplicate patches.
5. Load the model and inspect the first Qwen2 attention layer's `q_proj`, `k_proj`, `v_proj`, and `o_proj` weights, devices, and dtypes.
6. Trace the tensors entering Qwen2 attention to locate the exact point where key/value become float32 while query remains float16.
7. Apply a targeted fix only after identifying the dtype conversion source.
8. Rerun the single ReVA sample and record prediction, latency, and peak GPU memory.

Do not expand into advanced temporal reasoning, spatial reasoning, or question-aware fusion as part of this baseline handover; those belong to the downstream reasoning/fusion workstream.

## 9. Files in this handover

- `baseline/bimba_inference.py` — experimental preflight harness; not a completed inference runner.
- `notebooks/02_bimba_baseline_inference.ipynb` — reproduction notes and completion checklist.
- `reports/baseline_handover.md` — status, verified setup, blocker, and next steps.

## 10. Handover statement

The previous session demonstrated that the BIMBA model could be loaded and that the multimodal modules could be placed on the T4 GPU. However, generation did not complete because of a Qwen2 attention dtype mismatch. This handover preserves the verified work and the exact unresolved blocker without presenting an unverified prediction as a result.
