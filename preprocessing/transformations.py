import numpy as np
import torch
import torch.nn.functional as F


def transform_frames(frames, image_size=(224, 224), normalize=True,
                     mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """RGB uint8 HWC list/array -> float32 tensor [T,C,H,W], values normalized if requested."""
    if isinstance(frames, np.ndarray):
        arr = frames
    else:
        arr = np.stack(frames, axis=0)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError("frames must have shape [T,H,W,3] in RGB order")
    x = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).float().div_(255.0)
    x = F.interpolate(x, size=tuple(image_size), mode="bilinear", align_corners=False)
    if normalize:
        m = torch.tensor(mean, dtype=x.dtype).view(1, 3, 1, 1)
        s = torch.tensor(std, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - m) / s
    return x.contiguous()
