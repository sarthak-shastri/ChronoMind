# Preprocessing comparison

Three small real videos were fetched from the pinned Hub revision and decoded successfully. Each yielded eight sampled frames. Their measured FPS, frame counts, durations, resolutions, file sizes, and RGB frame shapes are in `video_sample_analysis.json`.

The complete E1–E7 extraction-time, memory, storage, and cache-hit comparisons have not yet been run. No QA accuracy claim is made. Uniform sampling supports 4, 8, 12, 16, and 32 frames. The adaptive helper accepts precomputed motion scores and falls back to uniform indices if scores are unavailable.

Tensor contract: `video_frames` is `torch.float32 [T, 3, H, W]`, RGB, resized to 224x224 by default, scaled to [0,1] then ImageNet-normalized. Real train and validation samples were each loaded through the Hub resolver and returned `[8,3,224,224]` without errors.
