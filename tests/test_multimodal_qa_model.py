import pytest
import torch
from torch import nn
from advanced_models.qa.multimodal_qa_model import MultimodalVideoQAModel, answer_index
from advanced_models.qa.text_encoder import CLIPTextFeatureExtractor
from advanced_models.temporal.temporal_encoder import TransformerTemporalEncoder
from advanced_models.vision.feature_extractor import CLIPVisionFeatureExtractor

class V(nn.Module):
    def __init__(self): super().__init__(); self.p=nn.Parameter(torch.ones(()))
    def forward(self,pixel_values): return type('Out',(),{'image_embeds':torch.ones(len(pixel_values),512,device=pixel_values.device)})()
class Tok:
    def __call__(self,texts,**kw): return {'input_ids':torch.ones(len(texts),3,dtype=torch.long)}
class T(nn.Module):
    def __init__(self): super().__init__(); self.p=nn.Parameter(torch.ones(()))
    def forward(self,input_ids): return type('Out',(),{'text_embeds':torch.ones(len(input_ids),512,device=input_ids.device)})()

def make():
    return MultimodalVideoQAModel(CLIPVisionFeatureExtractor(V(),input_normalization='zero_one'),CLIPTextFeatureExtractor(T(),Tok()),TransformerTemporalEncoder(512,num_heads=8,max_seq_len=16,dropout=0))

@pytest.mark.parametrize('t',[4,8,16])
def test_forward_gradients_and_freezing(t):
    m=make().train(); out=m(torch.rand(2,t,3,224,224),['q','q'],[dict(A='a',B='b',C='c',D='d')]*2)
    assert out['answer_logits'].shape==(2,4)
    assert out['frame_features'].shape==(2,t,512)
    assert out['prediction'][0] in 'ABCD'
    nn.functional.cross_entropy(out['answer_logits'],torch.tensor([0,3])).backward()
    assert any(p.grad is not None for p in m.temporal.parameters())
    assert any(p.grad is not None for p in m.fusion.parameters())
    assert all(p.grad is None and not p.requires_grad for p in m.vision.parameters())
    assert all(p.grad is None and not p.requires_grad for p in m.text.parameters())
    assert not m.vision.training and not m.text.training and m.temporal.training

def test_labels_and_bad_inputs():
    assert [answer_index(x) for x in 'ABCD']==[0,1,2,3]
    with pytest.raises(ValueError): answer_index('E')
    with pytest.raises(ValueError): make()(torch.rand(1,3,3,224,224),['q'],[dict(A='a',B='b',C='c',D='d')])
