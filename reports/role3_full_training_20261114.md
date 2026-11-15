# ChronoMind Role 3 — Full Transformer Training and Validation

## Scope and provenance

This report covers the local five-epoch Transformer run and a separate canonical FP32 evaluation of the saved best checkpoint. Training/evaluation used the internally derived split made only from pinned official ReVA training annotations. No official validation or test annotation was used for training or tuning.

- Dataset: `ReVA-Benchmark/ReVA@06224fa1b42e5e35677a8b7f2145e06307a9a4ee`
- Preprocessing: `1a63b5009267a348b5fdd550b063869332d4ce0a`
- CLIP: `openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`
- QA source base recorded at training: `74c417fb52fc21f923d788714875cbdb025d28aa`; experiment metadata also records the exact source-file hashes used by the run.
- Split: 12,618 training / 3,155 validation QA rows; zero normalized video-path overlap. SHA-256 audit covered 1,045 local video files and found no cross-split identical-file overlap. Original source-sequence independence is unverified because source IDs/mappings are unavailable.
- Model: frozen CLIP ViT-B/32 vision/text towers; trainable two-layer temporal Transformer (512 dimensions, 8 heads), fusion, and four-choice scorer.
- Training: 4 uniform frames, 224×224 ImageNet normalization, batch size 1, validation batch size 1, `num_workers=0`, seed 17, AdamW `1e-4`, 5 epochs, CUDA mixed precision for training and FP32 validation.
- Hardware: NVIDIA RTX 3050 6GB Laptop GPU. Training peak PyTorch allocated memory was 777,777,152 bytes (~0.72 GiB).

## Training history

| Epoch | Train loss | Train accuracy | Validation loss | Trainer validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 1.170178 | 46.97% | 1.094241 | 52.81% |
| 2 | 1.065702 | 54.19% | 1.062583 | 54.74% |
| 3 | 0.998056 | 56.98% | 1.019171 | 57.05% |
| 4 | 0.937047 | 60.40% | 0.997594 | 57.69% |
| 5 | 0.888056 | 62.85% | 0.965849 | 59.21% |

## Canonical standalone validation

The best checkpoint was loaded with the pinned local CLIP checkpoint in offline mode. All 3,155 validation rows were evaluated in FP32 using four uniform frames, canonical A/B/C/D option ordering, and no optimizer updates. Every expected QA ID appeared exactly once; no sample failed.

- Correct: 1,867 / 3,155
- Accuracy: 59.1759%
- Cross-entropy loss: 0.9658558
- Macro F1: 0.591744; weighted F1: 0.591726
- Prediction counts: A=808, B=789, C=776, D=782
- Evaluation FrameCache: 3,155 hits, 0 misses
- Inference duration: 137.52 seconds; mean per-sample latency 0.021717 seconds
- Evaluation peak CUDA allocated/reserved: 666,030,080 / 723,517,440 bytes
- Evaluation process peak RSS: 2,166,829,056 bytes
- Failed samples: none

### Confusion matrix

Rows are ground truth and columns are predictions, both ordered A/B/C/D.

```text
[[475, 101, 101, 111],
 [109, 470, 101, 104],
 [109, 115, 461, 106],
 [115, 103, 113, 461]]
```

### Per-class metrics

| Answer | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| A | 0.5879 | 0.6028 | 0.5952 | 788 |
| B | 0.5957 | 0.5995 | 0.5976 | 784 |
| C | 0.5941 | 0.5828 | 0.5884 | 791 |
| D | 0.5895 | 0.5821 | 0.5858 | 792 |

### Category accuracy

| Category | Correct / total | Accuracy |
|---|---:|---:|
| Causal Reasoning | 153 / 254 | 60.24% |
| Factual Perception | 402 / 749 | 53.67% |
| Spatial Reasoning | 830 / 1,373 | 60.45% |
| Temporal Understanding | 482 / 779 | 61.87% |

Task-wise metrics and full prediction records are in `post_training_audit_metrics.json` and `post_training_audit_predictions.json` in the external run directory.

## Trainer/evaluator reconciliation

The trainer recorded 1,868 correct from aggregate epoch-5 accuracy; standalone evaluation recorded 1,867. A repeated canonical pass produced the same predictions as the earlier standalone pass. The best and latest checkpoint payloads compare equal, including all model and optimizer tensors. Validation ordering, batch size, four-frame sampling, preprocessing, and label mapping match. Rounding and prediction serialization do not explain the one-answer difference.

The trainer did not persist per-QA validation predictions, so the exact discrepant QA ID cannot be recovered. Inspection found the concrete precision-path difference: text embeddings first computed during mixed-precision training could be cached as FP32 tensors after reduced-precision computation and then reused during FP32 validation. The text cache now separates entries by device and autocast precision. This is a post-run implementation correction; the saved checkpoint and historical metrics were not changed or retrained. The precise QA-level causal effect remains unverified.

## Checkpoints

| File | Size | SHA-256 | Epoch | Saved best validation accuracy |
|---|---:|---|---:|---:|
| `best.pt` | 695,287,462 bytes | `6491e8eea5626ff0d34bff4ccd15c40f5b5a68494eec972bf12f7fa435cd8aeb` | 5 | 0.592076 |
| `latest.pt` | 695,314,216 bytes | `af40c18e0f26b57a7b7fcb5ac27f3c7bc2558d414f4d4c93edad45e5cb12bed2` | 5 | 0.592076 |

Their file hashes differ, but every deserialized checkpoint field compares equal, including model state, optimizer state, scaler state, history, and RNG state.

## Qualifications and artifacts

These results are from an internal validation split, not the official hidden test. Zero normalized path overlap and available-file hash checks support path/content isolation; they do not prove source-sequence independence. No model-quality improvement claim is made from this single run.

Full weights and predictions remain outside Git in the local `full_training_20261114` run directory. The committed source does not include model weights, dataset files, video files, caches, or the prediction dump.
