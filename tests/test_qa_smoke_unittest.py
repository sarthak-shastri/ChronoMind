"""Dependency-light QA smoke tests runnable with python -m unittest."""
import json, tempfile, unittest
from pathlib import Path
import torch
from torch import nn
from advanced_models.qa.multimodal_qa_model import MultimodalVideoQAModel
from advanced_models.qa.text_encoder import CLIPTextFeatureExtractor
from advanced_models.qa.clip_similarity import clip_similarity_logits
from advanced_models.temporal.temporal_encoder import TransformerTemporalEncoder
from advanced_models.vision.feature_extractor import CLIPVisionFeatureExtractor
from scripts.evaluate_reva_qa import calculate_metrics, LETTERS

class Vision(nn.Module):
    def __init__(self): super().__init__(); self.scale=nn.Parameter(torch.ones(()))
    def forward(self,pixel_values): return type('O',(),{'image_embeds':torch.ones(len(pixel_values),512)*self.scale})()
class Tokenizer:
    def __call__(self,texts,**kw): return {'input_ids':torch.ones(len(texts),3,dtype=torch.long)}
class Text(nn.Module):
    def __init__(self): super().__init__(); self.scale=nn.Parameter(torch.ones(()))
    def forward(self,input_ids): return type('O',(),{'text_embeds':torch.ones(len(input_ids),512)*self.scale})()

class QASmoke(unittest.TestCase):
    def model(self):
        return MultimodalVideoQAModel(CLIPVisionFeatureExtractor(Vision(),input_normalization='zero_one'),CLIPTextFeatureExtractor(Text(),Tokenizer()),TransformerTemporalEncoder(512,num_heads=8,max_seq_len=16,dropout=0))
    def test_train_checkpoint_prediction_metrics(self):
        torch.manual_seed(3); m=self.model().train(); opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3)
        before=m.scorer[-1].weight.detach().clone()
        x=torch.rand(2,4,3,224,224); choices=[dict(zip(LETTERS,['a','b','c','d']))]*2
        out=m(x,['q1','q2'],choices); logits=out['answer_logits']; y=torch.tensor([0,3]); loss=nn.functional.cross_entropy(logits,y)
        self.assertTrue(torch.isfinite(loss)); opt.zero_grad(); loss.backward(); opt.step()
        self.assertFalse(torch.equal(before,m.scorer[-1].weight.detach()))
        self.assertTrue(any(p.grad is not None for p in m.temporal.parameters()))
        self.assertTrue(any(p.grad is not None for p in m.fusion.parameters()))
        self.assertTrue(all(p.grad is None for p in m.vision.parameters()))
        self.assertTrue(all(p.grad is None for p in m.text.parameters()))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ckpt.pt'; torch.save({'model_state_dict':m.state_dict()},path)
            m2=self.model(); m2.load_state_dict(torch.load(path,weights_only=True)['model_state_dict'])
            pred=m2.eval()(x,['q1','q2'],choices)['prediction']; self.assertEqual(len(pred),2)
            rows=[{'prediction':'A','ground_truth':'A','category':'Temporal','inference_time_seconds':.1},{'prediction':'C','ground_truth':'B','category':'Temporal','inference_time_seconds':.3}]
            metrics=calculate_metrics(rows); self.assertEqual(metrics['accuracy'],.5); self.assertEqual(metrics['confusion_matrix'][1][2],1)
            payload={'schema_version':'1.0','dataset':'ReVA','split':'validation','model':{'name':'mock'},'predictions':[{'sample_id':'v_1','prediction':'A','scores':dict(zip(LETTERS,[.7,.1,.1,.1])),'inference_time_seconds':.1}]}
            fp=Path(d)/'predictions.json'; fp.write_text(json.dumps(payload)); loaded=json.loads(fp.read_text())
            self.assertEqual(set(loaded['predictions'][0]['scores']),set(LETTERS))
    def test_frame_counts_and_baseline_shape(self):
        m=self.model().eval(); choices=[dict(A='a',B='b',C='c',D='d')]
        for t in (4,8,16): self.assertEqual(m(torch.rand(1,t,3,224,224),['q'],choices)['answer_logits'].shape,(1,4))
        self.assertEqual(clip_similarity_logits(torch.rand(2,512),torch.rand(2,512),torch.rand(2,4,512)).shape,(2,4))

if __name__=='__main__': unittest.main()
