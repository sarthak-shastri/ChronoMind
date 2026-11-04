import json
import numpy as np
import torch
from preprocessing.dataset import read_annotations, ReVADataset
from preprocessing.frame_sampling import uniform_indices, adaptive_indices
from preprocessing.transformations import transform_frames

def test_read_qa_object(tmp_path):
    f=tmp_path/'a.json'; f.write_text(json.dumps({'metadata':{},'QA':[{'qa_id':'x'}]}),encoding='utf-8'); assert read_annotations(f)[0]['qa_id']=='x'
def test_read_jsonl(tmp_path):
    f=tmp_path/'a.jsonl'; f.write_text('{"qa_id":"x"}\n{"qa_id":"y"}\n',encoding='utf-8'); assert len(read_annotations(f))==2
def test_uniform_sampling():
    a=uniform_indices(101,8); assert len(a)==8 and np.array_equal(a,uniform_indices(101,8)) and a[0]==0 and a[-1]==100
def test_adaptive_fallback(): assert np.array_equal(adaptive_indices(10,4),uniform_indices(10,4))
def test_transform_contract():
    x=transform_frames(np.zeros((4,20,30,3),dtype=np.uint8),(16,16),normalize=False); assert tuple(x.shape)==(4,3,16,16) and x.dtype==torch.float32
def test_test_split_without_answer(tmp_path):
    f=tmp_path/'test.json'; row={'qa_id':'test_1','video_path':'missing.mp4','question':'q','options':dict(A='a',B='b',C='c',D='d'),'correct_answer':''}; f.write_text(json.dumps({'QA':[row]}),encoding='utf-8')
    s=ReVADataset(f,tmp_path,missing_video='zeros')[0]; assert s['sample_id']=='test_1' and s['answer'] is None and list(s['choices'])==list('ABCD') and tuple(s['video_frames'].shape)==(8,3,224,224)
