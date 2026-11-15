"""Reproducible trainer for the frozen-CLIP temporal ReVA QA baseline."""
import argparse, hashlib, json, os, random, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-annotations',required=True); p.add_argument('--val-annotations',required=True)
    p.add_argument('--test-annotations',help='Optional official test annotation for three-way split audit (labels are not consumed).')
    p.add_argument('--video-root'); p.add_argument('--dataset-repo-id',default='ReVA-Benchmark/ReVA'); p.add_argument('--revision',required=True)
    p.add_argument('--split-integrity-mode',choices=('warn','strict','path_only_strict','off'),default='warn')
    p.add_argument('--preprocessing-root',default=os.environ.get('CHRONOMIND_PREPROCESSING_ROOT'))
    p.add_argument('--frames',type=int,default=8,choices=(4,8,16)); p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--temporal-encoder',choices=('mean_pooling','transformer'),default='transformer')
    p.add_argument('--val-batch-size',type=int)
    p.add_argument('--epochs',type=int,default=1); p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--output',default='artifacts/role3_qa'); p.add_argument('--seed',type=int,default=17)
    p.add_argument('--mixed-precision',action='store_true'); p.add_argument('--clip-revision')
    p.add_argument('--clip-model-name',default='openai/clip-vit-base-patch32')
    p.add_argument('--clip-cache-dir')
    p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--resume',help='Path to checkpoint to resume training from.')
    p.add_argument('--frame-cache-dir',help='Optional directory to cache decoded video frames on disk.')
    p.add_argument('--limit-train',type=int); p.add_argument('--limit-val',type=int)
    args=p.parse_args()
    if args.num_workers < 0: p.error('--num-workers must be non-negative')
    split_audit={'status':'not_run','issues':[],'audited_splits':[]}
    if args.split_integrity_mode!='off':
        from scripts.reva_split_preflight import preflight_splits
        split_audit=preflight_splits(args.train_annotations,args.val_annotations,args.test_annotations,video_root=args.video_root)
        if args.split_integrity_mode=='path_only_strict':
            if split_audit['path_only_issues']:
                raise SystemExit('Path-only strict split check failed: '+ '; '.join(split_audit['path_only_issues'][:8]))
            print('Path-only strict split check passed: normalized paths and available local hashes are isolated; source-sequence independence is not asserted.')
        elif split_audit['issues']:
            message='ReVA split preflight found path/source/hash overlaps or annotation issues (these do not alone establish answer leakage): '+ '; '.join(split_audit['issues'][:8])
            if args.split_integrity_mode=='strict': raise SystemExit(message)
            print('WARNING:',message)
        source_warning=split_audit.get('unverifiable_source_sequence_relationships',{}).get('warning')
        if source_warning:
            print('WARNING:',source_warning)
            if args.split_integrity_mode=='strict':
                raise SystemExit('Strict split mode cannot certify source-sequence isolation without explicit source IDs or an original-to-clip mapping.')
        if args.split_integrity_mode=='path_only_strict' and split_audit['source_sequence_warnings']:
            print('WARNING: explicit source-sequence overlap(s) observed; path-only strict mode does not claim source-sequence isolation:', '; '.join(split_audit['source_sequence_warnings'][:8]))
    if args.preprocessing_root:
        sys.path.insert(0,str(Path(args.preprocessing_root).resolve()))
    try: from preprocessing import ReVADataset
    except ImportError as exc:
        raise SystemExit("Cannot import Sarthak preprocessing. Clone/check out the pinned sarthak revision and pass --preprocessing-root <checkout> (the directory containing preprocessing/), or set CHRONOMIND_PREPROCESSING_ROOT.") from exc
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    from advanced_models.qa.data import ReVAQADatasetAdapter,collate_qa
    from advanced_models.qa.text_encoder import CLIPTextFeatureExtractor
    from advanced_models.qa.multimodal_qa_model import MultimodalVideoQAModel
    from advanced_models.temporal.temporal_encoder import build_temporal_encoder
    from advanced_models.vision.feature_extractor import CLIPVisionFeatureExtractor
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    repo_root=Path(__file__).resolve().parents[1]
    try: code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo_root,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): code_revision=None
    try: preprocessing_revision=subprocess.check_output(['git','-C',str(Path(args.preprocessing_root).resolve()),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip() if args.preprocessing_root else None
    except (OSError,subprocess.CalledProcessError): preprocessing_revision=None
    cache_config={'dataset_repo_id':args.dataset_repo_id,'dataset_revision':args.revision,
        'preprocessing_revision':preprocessing_revision,'clip_model':'openai/clip-vit-base-patch32',
        'clip_revision':args.clip_revision,'frames':args.frames,'sampling':'uniform',
        'image_size':[224,224],'normalization':'imagenet'}
    feature_cache_namespace=hashlib.sha256(json.dumps(cache_config,sort_keys=True).encode()).hexdigest()
    clip_load={'model_name':args.clip_model_name,'revision':args.clip_revision,
               'local_files_only':args.local_files_only,'cache_dir':args.clip_cache_dir}
    vision=CLIPVisionFeatureExtractor.from_pretrained(input_normalization='imagenet',frame_chunk_size=4,cuda_mixed_precision=args.mixed_precision,feature_cache_namespace=feature_cache_namespace,**clip_load)
    text=CLIPTextFeatureExtractor.from_pretrained(**clip_load)
    temporal=build_temporal_encoder(args.temporal_encoder,embedding_dim=512,num_heads=8,num_layers=2,max_seq_len=16)
    model=MultimodalVideoQAModel(vision,text,temporal).to(device)
    def ds(path, split):
        return ReVAQADatasetAdapter(ReVADataset(
            path, video_root=args.video_root, num_frames=args.frames,
            sampling='uniform', image_size=(224,224), normalize=True, split=None,
            cache_dir=args.frame_cache_dir,
            missing_video='raise', dataset_repo_id=args.dataset_repo_id,
            revision=args.revision,
        ))
    from torch.utils.data import Subset
    train_ds=ds(args.train_annotations, 'train'); val_ds=ds(args.val_annotations, 'val')
    if args.limit_train: train_ds=Subset(train_ds,range(min(args.limit_train,len(train_ds))))
    if args.limit_val: val_ds=Subset(val_ds,range(min(args.limit_val,len(val_ds))))
    loader_options={'num_workers':args.num_workers,'pin_memory':device.type=='cuda'}
    if args.num_workers>0:
        loader_options.update({'persistent_workers':True,'prefetch_factor':2})
    train=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,collate_fn=collate_qa,**loader_options)
    val_batch_size=args.val_batch_size or args.batch_size
    val=DataLoader(val_ds,batch_size=val_batch_size,shuffle=False,collate_fn=collate_qa,**loader_options)
    trainable=[x for x in model.parameters() if x.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=args.lr); scaler=torch.amp.GradScaler('cuda',enabled=args.mixed_precision and device.type=='cuda')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); best=-1.; history=[]; start_epoch=0
    if args.resume:
        resume_file=Path(args.resume)
        if not resume_file.is_file(): raise FileNotFoundError(f"Resume checkpoint not found: {resume_file}")
        ckpt=torch.load(resume_file,map_location=device,weights_only=False)
        model.load_state_dict(ckpt.get('model_state_dict',ckpt.get('model')))
        if 'optimizer_state_dict' in ckpt: opt.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict'] is not None and scaler.is_enabled():
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch=ckpt.get('epoch',0)
        best=ckpt.get('validation_accuracy',ckpt.get('best_validation_accuracy',-1.))
        history=ckpt.get('history',[])
        if 'torch_rng_state' in ckpt and ckpt['torch_rng_state'] is not None:
            torch.set_rng_state(ckpt['torch_rng_state'].cpu() if isinstance(ckpt['torch_rng_state'],torch.Tensor) else ckpt['torch_rng_state'])
        if 'cuda_rng_state' in ckpt and ckpt['cuda_rng_state'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
        if 'numpy_rng_state' in ckpt and ckpt['numpy_rng_state'] is not None:
            np.random.set_state(ckpt['numpy_rng_state'])
        if 'random_state' in ckpt and ckpt['random_state'] is not None:
            random.setstate(ckpt['random_state'])
        print(f"Resumed from epoch {start_epoch} (best val accuracy: {best})")
    code_files=[repo_root/'advanced_models/temporal/temporal_encoder.py',repo_root/'advanced_models/qa/multimodal_qa_model.py',repo_root/'advanced_models/qa/data.py',repo_root/'advanced_models/qa/text_encoder.py',repo_root/'advanced_models/vision/feature_extractor.py',repo_root/'scripts/train_reva_qa.py',repo_root/'scripts/evaluate_reva_qa.py']
    metadata={'arguments':vars(args),'code_base_revision':code_revision,
        'code_file_sha256':{str(path.relative_to(repo_root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files if path.is_file()},
        'dataset_repo_id':args.dataset_repo_id,'dataset_revision':args.revision,
        'preprocessing_root':args.preprocessing_root,'preprocessing_revision':preprocessing_revision,
        'train_question_count':len(train_ds),'validation_question_count':len(val_ds),
        'clip_model':'openai/clip-vit-base-patch32','clip_revision':args.clip_revision,
        'model_config':{'temporal_encoder_type':args.temporal_encoder,'embedding_dim':512,'num_heads':8,'num_layers':2,'max_seq_len':16,'dropout':0.1},
        'frame_count':args.frames,'sampling':'uniform','image_size':[224,224],
        'clip_input_normalization':'imagenet','validation_precision':'float32',
        'mixed_precision_training':args.mixed_precision,'split_integrity_audit':split_audit,
        'feature_cache':{'enabled':True,'namespace_sha256':feature_cache_namespace,'storage':'process-local CPU FP32 tensors'},
        'dataloader':{'num_workers':args.num_workers,'pin_memory':device.type=='cuda','persistent_workers':args.num_workers>0},
        'runtime':{'device':str(device),'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None}}
    (out/'experiment_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    for epoch in range(start_epoch, args.epochs):
        row={'epoch':epoch+1}
        for name,loader,training in [('train',train,True),('val',val,False)]:
            model.train(training); loss_sum=correct=n=0
            with torch.set_grad_enabled(training):
                for batch in loader:
                    if batch['labels'] is None: raise ValueError('labeled train/validation examples are required')
                    labels=batch['labels'].to(device)
                    if training: opt.zero_grad(set_to_none=True)
                    amp_enabled=training and args.mixed_precision and device.type=='cuda'
                    model.vision.cuda_mixed_precision=amp_enabled
                    with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp_enabled):
                        result=model(batch['video_frames'].to(device,non_blocking=device.type=='cuda'),batch['questions'],batch['choices'],video_cache_keys=batch['video_paths'])
                        logits=result['answer_logits']; loss=torch.nn.functional.cross_entropy(logits,labels)
                    if not torch.isfinite(logits).all() or not torch.isfinite(loss): raise FloatingPointError('non-finite logits or loss')
                    if training:
                        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    loss_sum+=loss.item()*len(labels); correct+=(logits.argmax(-1)==labels).sum().item(); n+=len(labels)
            row[name+'_loss']=loss_sum/max(n,1); row[name+'_accuracy']=correct/max(n,1)
        row['split_integrity_status']=split_audit['status']
        row['visual_cache']=vision.feature_cache_stats(); row['text_cache']=text.feature_cache_stats()
        history.append(row); print(json.dumps(row))
        ckpt_payload={'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),
            'scaler_state_dict':scaler.state_dict() if scaler.is_enabled() else None,
            'model_config':{'temporal_encoder_type':args.temporal_encoder,'embedding_dim':512,'num_heads':8,'num_layers':2,'max_seq_len':16,'dropout':0.1},
            'epoch':epoch+1,'validation_accuracy':row['val_accuracy'],'best_validation_accuracy':max(best,row['val_accuracy']),
            'history':history,'torch_rng_state':torch.get_rng_state(),
            'cuda_rng_state':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy_rng_state':np.random.get_state(),'random_state':random.getstate(),
            'seed':args.seed,'frame_count':args.frames,'dataset_repo_id':args.dataset_repo_id,'dataset_revision':args.revision,
            'clip_model':'openai/clip-vit-base-patch32','clip_revision':args.clip_revision,'clip_input_normalization':'imagenet',
            'clip_frame_chunk_size':4,'clip_mixed_precision_training':args.mixed_precision,'validation_precision':'float32',
            'batch_size':args.batch_size,'validation_batch_size':val_batch_size,'learning_rate':args.lr,'sampling':'uniform',
            'image_size':[224,224],'split_integrity_audit':split_audit}
        torch.save(ckpt_payload,out/'latest.pt')
        if row['val_accuracy']>best:
            best=row['val_accuracy']; ckpt_payload['validation_accuracy']=best
            torch.save(ckpt_payload,out/'best.pt')
    best_file=out/'best.pt'
    if not best_file.exists() and (out/'latest.pt').exists():
        import shutil; shutil.copy(out/'latest.pt',best_file)
    if best_file.exists():
        digest=hashlib.sha256()
        with best_file.open('rb') as checkpoint_file:
            for chunk in iter(lambda:checkpoint_file.read(1024*1024),b''): digest.update(chunk)
        metadata['checkpoint_sha256']=digest.hexdigest()
    metadata['visual_feature_cache']=vision.feature_cache_stats(); metadata['text_feature_cache']=text.feature_cache_stats()
    if device.type=='cuda': metadata['peak_cuda_memory_bytes']=torch.cuda.max_memory_allocated(device)
    (out/'experiment_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
if __name__=='__main__': main()
