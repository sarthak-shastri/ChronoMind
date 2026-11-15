"""Evaluate a saved Role 3 QA checkpoint and export framework-compatible JSON."""
import argparse,json,os,sys,time,statistics,hashlib
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
LETTERS=('A','B','C','D')

def make_prediction_records(logits, sample_ids, answers, categories, elapsed_seconds, *,
                            qa_ids=None, video_paths=None, tasks=None, questions=None, choices=None):
    """Serialize a complete batch in canonical A/B/C/D order; labels never enter the model."""
    if logits.ndim != 2 or logits.shape != (len(sample_ids), 4):
        raise ValueError('logits must have shape [batch,4] matching sample_ids')
    if not (len(answers) == len(categories) == len(sample_ids)):
        raise ValueError('sample IDs, answers, and categories must have equal lengths')
    optional={'qa_ids':qa_ids,'video_paths':video_paths,'tasks':tasks,'questions':questions,'choices':choices}
    if any(values is not None and len(values)!=len(sample_ids) for values in optional.values()):
        raise ValueError('optional prediction metadata must match batch size')
    probs=torch.softmax(logits.float().cpu(),dim=-1)
    records=[]; metric_rows=[]
    for i,sample_id in enumerate(sample_ids):
        prediction=LETTERS[int(logits[i].argmax())]
        answer=answers[i]
        if answer in (None,''):
            ground_truth=None
        elif answer in LETTERS:
            ground_truth=answer
        else:
            raise ValueError(f'ground-truth answer must be canonical A/B/C/D or empty, got {answer!r}')
        option_map=None
        if choices is not None and choices[i] is not None:
            if not isinstance(choices[i],dict) or set(choices[i])!=set(LETTERS):
                raise ValueError('options must contain exactly A/B/C/D')
            option_map={key:choices[i][key] for key in LETTERS}
        correct=(prediction==ground_truth) if ground_truth is not None else None
        scores={k:float(probs[i,j]) for j,k in enumerate(LETTERS)}
        record={'qa_id':(qa_ids[i] if qa_ids is not None else str(sample_id)),
                'sample_id':str(sample_id),
                'video_path':video_paths[i] if video_paths is not None else None,
                'category':categories[i], 'task':tasks[i] if tasks is not None else None,
                'question':questions[i] if questions is not None else None,
                'options':option_map, 'ground_truth':ground_truth,
                'predicted_answer':prediction, 'prediction':prediction,
                'scores':scores, 'prediction_confidence':max(scores.values()),
                'is_correct':correct,
                'correctness':('correct' if correct else 'incorrect') if correct is not None else 'unlabeled',
                'inference_time_seconds':elapsed_seconds}
        records.append(record); metric_rows.append(record)
    return records,metric_rows

def calculate_metrics(rows):
    labeled=[r for r in rows if r.get('ground_truth') in LETTERS]
    matrix=[[0]*4 for _ in range(4)]; cats=defaultdict(lambda:[0,0]); tasks=defaultdict(lambda:[0,0])
    for r in labeled:
        y=LETTERS.index(r['ground_truth']); p=LETTERS.index(r['prediction']); matrix[y][p]+=1
        cats[r.get('category','Unknown')][0]+=int(y==p); cats[r.get('category','Unknown')][1]+=1
        tasks[r.get('task') or 'Unknown'][0]+=int(y==p); tasks[r.get('task') or 'Unknown'][1]+=1
    return {'sample_count':len(rows),'labeled_count':len(labeled),'accuracy':(sum(matrix[i][i] for i in range(4))/len(labeled) if labeled else None),
            'category_accuracy':{k:{'correct':v[0],'total':v[1],'accuracy':v[0]/v[1]} for k,v in sorted(cats.items())},'confusion_matrix':matrix,'confusion_matrix_labels':list(LETTERS),
            'task_accuracy':{k:{'correct':v[0],'total':v[1],'accuracy':v[0]/v[1]} for k,v in sorted(tasks.items())},
            'prediction_distribution':{k:sum(r.get('predicted_answer',r.get('prediction'))==k for r in rows) for k in LETTERS},
            'mean_inference_time_seconds':statistics.mean([r['inference_time_seconds'] for r in rows]) if rows else None,
            'median_inference_time_seconds':statistics.median([r['inference_time_seconds'] for r in rows]) if rows else None}

def sha256_file(path):
 digest=hashlib.sha256()
 with open(path,'rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
 return digest.hexdigest()

def load_clip_feature_extractors(model_name, revision, input_normalization, frame_chunk_size,
                                 local_files_only=False, cache_dir=None):
 from advanced_models.qa.text_encoder import CLIPTextFeatureExtractor
 from advanced_models.vision.feature_extractor import CLIPVisionFeatureExtractor
 vision=CLIPVisionFeatureExtractor.from_pretrained(
  model_name=model_name,input_normalization=input_normalization,
  frame_chunk_size=frame_chunk_size,cuda_mixed_precision=False,
  revision=revision,local_files_only=local_files_only,cache_dir=cache_dir)
 text=CLIPTextFeatureExtractor.from_pretrained(
  model_name=model_name,revision=revision,
  local_files_only=local_files_only,cache_dir=cache_dir)
 return vision,text

def main():
 p=argparse.ArgumentParser(); p.add_argument('--annotations',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--train-annotations',help='Optional training annotations to audit validation overlap.'); p.add_argument('--test-annotations',help='Optional official test annotation for a three-way audit.'); p.add_argument('--split-integrity-mode',choices=('warn','strict','path_only_strict','off'),default='warn'); p.add_argument('--preprocessing-root',default=os.environ.get('CHRONOMIND_PREPROCESSING_ROOT')); p.add_argument('--video-root'); p.add_argument('--dataset-repo-id',default='ReVA-Benchmark/ReVA'); p.add_argument('--revision',required=True); p.add_argument('--clip-model-name',help='Optional local CLIP directory or model ID; defaults to the checkpoint model.'); p.add_argument('--clip-cache-dir',help='Optional Hugging Face cache directory.'); p.add_argument('--local-files-only',action='store_true',help='Forbid Hugging Face network access when loading CLIP.'); p.add_argument('--output-dir',default='artifacts/role3_qa/evaluation'); p.add_argument('--frame-cache-dir',help='Optional directory to cache decoded video frames on disk.'); p.add_argument('--limit-samples',type=int); p.add_argument('--batch-size',type=int); a=p.parse_args()
 if a.preprocessing_root: sys.path.insert(0,str(Path(a.preprocessing_root).resolve()))
 from preprocessing import ReVADataset
 from advanced_models.qa.data import ReVAQADatasetAdapter,collate_qa
 from advanced_models.qa.text_encoder import CLIPTextFeatureExtractor
 from advanced_models.qa.multimodal_qa_model import MultimodalVideoQAModel
 from advanced_models.temporal.temporal_encoder import build_temporal_encoder_from_config
 from advanced_models.vision.feature_extractor import CLIPVisionFeatureExtractor
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 split_audit={'status':'not_run','issues':[],'audited_splits':[]}
 if a.split_integrity_mode in ('strict','path_only_strict') and not a.train_annotations:
  raise SystemExit("Strict split integrity modes require --train-annotations so validation overlap can be audited.")
 if a.split_integrity_mode!='off' and a.train_annotations:
  from scripts.reva_split_preflight import preflight_splits
  split_audit=preflight_splits(a.train_annotations,a.annotations,a.test_annotations,video_root=a.video_root)
  if a.split_integrity_mode=='path_only_strict':
   if split_audit['path_only_issues']:
    raise SystemExit('Path-only strict split check failed: '+ '; '.join(split_audit['path_only_issues'][:8]))
   print('Path-only strict split check passed: normalized paths and available local hashes are isolated; source-sequence independence is not asserted.')
  elif split_audit['issues']:
   message='ReVA split preflight found path/source/hash overlaps or annotation issues (these do not alone establish answer leakage): '+ '; '.join(split_audit['issues'][:8])
   if a.split_integrity_mode=='strict': raise SystemExit(message)
   print('WARNING:',message)
  source_warning=split_audit.get('unverifiable_source_sequence_relationships',{}).get('warning')
  if source_warning:
   print('WARNING:',source_warning)
   if a.split_integrity_mode=='strict':
    raise SystemExit('Strict split mode cannot certify source-sequence isolation without explicit source IDs or an original-to-clip mapping.')
  if a.split_integrity_mode=='path_only_strict' and split_audit['source_sequence_warnings']:
   print('WARNING: explicit source-sequence overlap(s) observed; path-only strict mode does not claim source-sequence isolation:', '; '.join(split_audit['source_sequence_warnings'][:8]))
 elif a.split_integrity_mode!='off':
  split_audit={'status':'not_run','issues':['Supply --train-annotations to audit train/validation overlap.'],'audited_splits':['val']}
 if ckpt.get('dataset_revision') and ckpt['dataset_revision'] != a.revision:
  raise ValueError(f"Dataset revision mismatch: checkpoint used {ckpt['dataset_revision']}, evaluator requested {a.revision}")
 clip_model=a.clip_model_name or ckpt.get('clip_model','openai/clip-vit-base-patch32')
 vision,text=load_clip_feature_extractors(
  clip_model,ckpt.get('clip_revision'),ckpt.get('clip_input_normalization','imagenet'),
  ckpt.get('clip_frame_chunk_size',4),local_files_only=a.local_files_only,cache_dir=a.clip_cache_dir)
 cfg=ckpt.get('model_config',{}); temporal=build_temporal_encoder_from_config(cfg); model=MultimodalVideoQAModel(vision,text,temporal,dropout=cfg.get('dropout',0.1)).to(device)
 model.load_state_dict(ckpt.get('model_state_dict',ckpt.get('model'))); model.eval()
 dataset=ReVAQADatasetAdapter(ReVADataset(a.annotations,video_root=a.video_root,num_frames=ckpt.get('frame_count',8),sampling=ckpt.get('sampling','uniform'),image_size=tuple(ckpt.get('image_size',[224,224])),normalize=True,split=None,cache_dir=a.frame_cache_dir,missing_video='raise',dataset_repo_id=a.dataset_repo_id,revision=a.revision))
 if a.limit_samples is not None:
  from torch.utils.data import Subset
  dataset=Subset(dataset,range(min(a.limit_samples,len(dataset))))
 batch_size=a.batch_size or ckpt.get('validation_batch_size',ckpt.get('batch_size',1))
 model.vision.cuda_mixed_precision=False
 loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_qa); predictions=[]; metric_rows=[]
 with torch.inference_mode():
  for b in loader:
   if device.type=='cuda': torch.cuda.synchronize()
   start=time.perf_counter(); out=model(b['video_frames'].to(device,non_blocking=device.type=='cuda'),b['questions'],b['choices'],video_cache_keys=b.get('video_paths'));
   if device.type=='cuda': torch.cuda.synchronize()
   elapsed=(time.perf_counter()-start)/len(b['sample_ids'])
   batch_predictions,batch_metrics=make_prediction_records(out['answer_logits'],b['sample_ids'],b['answers'],b['categories'],elapsed,
    qa_ids=b.get('qa_ids'),video_paths=b.get('video_paths'),tasks=b.get('tasks'),questions=b.get('questions'),choices=b.get('choices'))
   predictions.extend(batch_predictions); metric_rows.extend(batch_metrics)
 outdir=Path(a.output_dir); outdir.mkdir(parents=True,exist_ok=True)
 checkpoint_sha256=sha256_file(a.checkpoint)
 payload={'schema_version':'1.1','dataset':'ReVA','split':'validation','model':{'name':'ChronoMind temporal multimodal QA','version':str(ckpt.get('epoch')),'metadata':{'checkpoint':str(Path(a.checkpoint).name),'checkpoint_sha256':checkpoint_sha256,'frame_count':ckpt.get('frame_count')}},'score_semantics':'probabilities','option_order':list(LETTERS),'predictions':predictions}
 (outdir/'predictions.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); report=calculate_metrics(metric_rows); report['checkpoint_sha256']=checkpoint_sha256; report['split_integrity_audit']=split_audit; report['generalization_warning']='Path/source overlap audit is required before interpreting validation accuracy as unseen-video generalization.'; (outdir/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 metadata_path=outdir/'experiment_metadata.json'
 metadata=json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
 metadata.update({'checkpoint_sha256':checkpoint_sha256,'evaluated_checkpoint':str(Path(a.checkpoint).resolve()),
  'evaluated_dataset_revision':a.revision,'evaluated_sample_count':len(predictions),
  'evaluated_qa_ids':[record['qa_id'] for record in predictions]})
 metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
if __name__=='__main__': main()
