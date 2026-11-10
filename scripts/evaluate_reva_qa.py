"""Evaluate a saved Role 3 QA checkpoint and export framework-compatible JSON."""
import argparse,json,os,sys,time,statistics
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
LETTERS=('A','B','C','D')

def make_prediction_records(logits, sample_ids, answers, categories, elapsed_seconds):
    """Convert every batch row to a prediction while retaining canonical answers for metrics."""
    if logits.ndim != 2 or logits.shape != (len(sample_ids), 4):
        raise ValueError('logits must have shape [batch,4] matching sample_ids')
    if not (len(answers) == len(categories) == len(sample_ids)):
        raise ValueError('sample IDs, answers, and categories must have equal lengths')
    probs=torch.softmax(logits.float().cpu(),dim=-1)
    records=[]; metric_rows=[]
    for i,sample_id in enumerate(sample_ids):
        prediction=LETTERS[int(logits[i].argmax())]
        record={'sample_id':sample_id,'prediction':prediction,'scores':{k:float(probs[i,j]) for j,k in enumerate(LETTERS)},'inference_time_seconds':elapsed_seconds}
        records.append(record); metric_rows.append({**record,'ground_truth':answers[i],'category':categories[i]})
    return records,metric_rows

def calculate_metrics(rows):
    labeled=[r for r in rows if r.get('ground_truth') in LETTERS]
    matrix=[[0]*4 for _ in range(4)]; cats=defaultdict(lambda:[0,0])
    for r in labeled:
        y=LETTERS.index(r['ground_truth']); p=LETTERS.index(r['prediction']); matrix[y][p]+=1
        cats[r.get('category','Unknown')][0]+=int(y==p); cats[r.get('category','Unknown')][1]+=1
    return {'sample_count':len(rows),'labeled_count':len(labeled),'accuracy':(sum(matrix[i][i] for i in range(4))/len(labeled) if labeled else None),
            'category_accuracy':{k:{'correct':v[0],'total':v[1],'accuracy':v[0]/v[1]} for k,v in sorted(cats.items())},'confusion_matrix':matrix,'confusion_matrix_labels':list(LETTERS),
            'mean_inference_time_seconds':statistics.mean([r['inference_time_seconds'] for r in rows]) if rows else None,
            'median_inference_time_seconds':statistics.median([r['inference_time_seconds'] for r in rows]) if rows else None}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--annotations',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--preprocessing-root',default=os.environ.get('CHRONOMIND_PREPROCESSING_ROOT')); p.add_argument('--video-root'); p.add_argument('--dataset-repo-id',default='ReVA-Benchmark/ReVA'); p.add_argument('--revision',required=True); p.add_argument('--output-dir',default='artifacts/role3_qa/evaluation'); p.add_argument('--limit-samples',type=int); p.add_argument('--batch-size',type=int); a=p.parse_args()
 if a.preprocessing_root: sys.path.insert(0,str(Path(a.preprocessing_root).resolve()))
 from preprocessing import ReVADataset
 from advanced_models.qa.data import ReVAQADatasetAdapter,collate_qa
 from advanced_models.qa.text_encoder import CLIPTextFeatureExtractor
 from advanced_models.qa.multimodal_qa_model import MultimodalVideoQAModel
 from advanced_models.temporal.temporal_encoder import TransformerTemporalEncoder
 from advanced_models.vision.feature_extractor import CLIPVisionFeatureExtractor
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 if ckpt.get('dataset_revision') and ckpt['dataset_revision'] != a.revision:
  raise ValueError(f"Dataset revision mismatch: checkpoint used {ckpt['dataset_revision']}, evaluator requested {a.revision}")
 clip_model=ckpt.get('clip_model','openai/clip-vit-base-patch32')
 vision=CLIPVisionFeatureExtractor.from_pretrained(model_name=clip_model,input_normalization=ckpt.get('clip_input_normalization','imagenet'),frame_chunk_size=ckpt.get('clip_frame_chunk_size',4),cuda_mixed_precision=False,revision=ckpt.get('clip_revision'))
 text=CLIPTextFeatureExtractor.from_pretrained(model_name=clip_model,revision=ckpt.get('clip_revision'))
 cfg=ckpt.get('model_config',{}); model=MultimodalVideoQAModel(vision,text,TransformerTemporalEncoder(cfg.get('embedding_dim',512),num_heads=cfg.get('num_heads',8),num_layers=cfg.get('num_layers',2),max_seq_len=cfg.get('max_seq_len',16)),dropout=cfg.get('dropout',0.1)).to(device)
 model.load_state_dict(ckpt.get('model_state_dict',ckpt.get('model'))); model.eval()
 dataset=ReVAQADatasetAdapter(ReVADataset(a.annotations,video_root=a.video_root,num_frames=ckpt.get('frame_count',8),sampling=ckpt.get('sampling','uniform'),image_size=tuple(ckpt.get('image_size',[224,224])),normalize=True,split='val',missing_video='raise',dataset_repo_id=a.dataset_repo_id,revision=a.revision))
 if a.limit_samples is not None:
  from torch.utils.data import Subset
  dataset=Subset(dataset,range(min(a.limit_samples,len(dataset))))
 batch_size=a.batch_size or ckpt.get('validation_batch_size',ckpt.get('batch_size',1))
 model.vision.cuda_mixed_precision=False
 loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_qa); predictions=[]; metric_rows=[]
 with torch.inference_mode():
  for b in loader:
   if device.type=='cuda': torch.cuda.synchronize()
   start=time.perf_counter(); out=model(b['video_frames'].to(device),b['questions'],b['choices']);
   if device.type=='cuda': torch.cuda.synchronize()
   elapsed=(time.perf_counter()-start)/len(b['sample_ids'])
   batch_predictions,batch_metrics=make_prediction_records(out['answer_logits'],b['sample_ids'],b['answers'],b['categories'],elapsed)
   predictions.extend(batch_predictions); metric_rows.extend(batch_metrics)
 outdir=Path(a.output_dir); outdir.mkdir(parents=True,exist_ok=True)
 payload={'schema_version':'1.0','dataset':'ReVA','split':'validation','model':{'name':'ChronoMind temporal multimodal QA','version':str(ckpt.get('epoch')),'metadata':{'checkpoint':str(Path(a.checkpoint).name),'frame_count':ckpt.get('frame_count')}},'score_semantics':'probabilities','option_order':list(LETTERS),'predictions':predictions}
 (outdir/'predictions.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); (outdir/'metrics.json').write_text(json.dumps(calculate_metrics(metric_rows),indent=2),encoding='utf-8')
if __name__=='__main__': main()
