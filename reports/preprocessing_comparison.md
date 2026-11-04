# Preprocessing comparison

No processing-time, memory, storage, or QA accuracy measurements are reported yet; those require accessible real video files and the downstream evaluation owned by other members. Uniform sampling supports 4, 8, 12, 16, and 32 frames. Adaptive sampling currently accepts precomputed motion scores and uses deterministic uniform sampling if scores are unavailable. The adaptive score extraction and experiment recorder remain pending.

Tensor contract: `video_frames` is a `torch.float32` tensor shaped `[T, 3, H, W]` in RGB channel order, resized to configured dimensions (224x224 by default), scaled to [0,1], and ImageNet-normalized by default.
