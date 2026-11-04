import torch
from preprocessing.collate import collate_reva

def test_collate_stacks_frames_and_keeps_metadata():
    a={"video_frames":torch.zeros(8,3,224,224),"sample_id":"a","choices":{"A":"a","B":"b","C":"c","D":"d"}}
    b={"video_frames":torch.ones(8,3,224,224),"sample_id":"b","choices":{"A":"w","B":"x","C":"y","D":"z"}}
    batch=collate_reva([a,b])
    assert tuple(batch["video_frames"].shape)==(2,8,3,224,224)
    assert batch["sample_id"]==["a","b"]
    assert batch["choices"][0]["A"]=="a"
