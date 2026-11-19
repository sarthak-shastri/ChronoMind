"""Batch collation for ReVA samples."""
import torch
from torch.utils.data import DataLoader

def collate_reva(batch):
    """Stack frame tensors and retain metadata/choices as ordered Python lists."""
    if not batch:
        return {}
    result={key:[item.get(key) for item in batch] for key in batch[0]}
    try:
        result["video_frames"]=torch.stack([item["video_frames"] for item in batch])
    except (KeyError,TypeError,RuntimeError):
        pass
    return result

def make_dataloader(dataset,batch_size=1,shuffle=False,num_workers=0,**kwargs):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,
                      collate_fn=collate_reva,**kwargs)
