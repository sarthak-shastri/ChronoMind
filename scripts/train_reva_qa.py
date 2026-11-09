"""Reproducible trainer for the frozen-CLIP temporal ReVA QA baseline."""
import argparse, json, os, random, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-annotations',required=True); p.add_argument('--val-annotations',required=True)
    p.add_argument('--video-root'); p.add_argument('--dataset-repo-id',default='ReVA-Benchmark/ReVA'); p.add_argument('--revision',required=True)
    p.add_argument('--preprocessing-root',default=os.environ.get('CHRONOMIND_PREPROCESSING_ROOT'))
    p.add_argument('--frames',type=int,default=8,choices=(4,8,16)); p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--val-batch-size',type=int)
    p.add_argument('--epochs',type=int,default=1); p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--output',default='artifacts/role3_qa'); p.add_argument('--seed',type=int,default=17)
    p.add_argument('--mixed-precision',action='store_true'); p.add_argument('--clip-revision')
    p.add_argument('--limit-train',type=int); p.add_argument('--limit-val',type=int)
    args=p.parse_args()
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
    from advanced_models.temporal.temporal_encoder import TransformerTemporalEncoder
    from advanced_models.vision.feature_extractor import CLIPVisionFeatureExtractor
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vision=CLIPVisionFeatureExtractor.from_pretrained(input_normalization='imagenet',frame_chunk_size=4,cuda_mixed_precision=args.mixed_precision,revision=args.clip_revision)
    text=CLIPTextFeatureExtractor.from_pretrained(revision=args.clip_revision)
    model=MultimodalVideoQAModel(vision,text,TransformerTemporalEncoder(512,num_heads=8,num_layers=2,max_seq_len=16)).to(device)
    def ds(path, split):
        return ReVAQADatasetAdapter(ReVADataset(
            path, video_root=args.video_root, num_frames=args.frames,
            sampling='uniform', image_size=(224,224), normalize=True, split=split,
            missing_video='raise', dataset_repo_id=args.dataset_repo_id,
            revision=args.revision,
        ))
    from torch.utils.data import Subset
    train_ds=ds(args.train_annotations, 'train'); val_ds=ds(args.val_annotations, 'val')
    if args.limit_train: train_ds=Subset(train_ds,range(min(args.limit_train,len(train_ds))))
    if args.limit_val: val_ds=Subset(val_ds,range(min(args.limit_val,len(val_ds))))
    train=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,collate_fn=collate_qa)
    val_batch_size=args.val_batch_size or args.batch_size
    val=DataLoader(val_ds,batch_size=val_batch_size,shuffle=False,collate_fn=collate_qa)
    trainable=[x for x in model.parameters() if x.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=args.lr); scaler=torch.amp.GradScaler('cuda',enabled=args.mixed_precision and device.type=='cuda')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); best=-1.; history=[]
    for epoch in range(args.epochs):
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
                        result=model(batch['video_frames'].to(device),batch['questions'],batch['choices'])
                        logits=result['answer_logits']; loss=torch.nn.functional.cross_entropy(logits,labels)
                    if not torch.isfinite(logits).all() or not torch.isfinite(loss): raise FloatingPointError('non-finite logits or loss')
                    if training:
                        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    loss_sum+=loss.item()*len(labels); correct+=(logits.argmax(-1)==labels).sum().item(); n+=len(labels)
            row[name+'_loss']=loss_sum/max(n,1); row[name+'_accuracy']=correct/max(n,1)
        history.append(row); print(json.dumps(row))
        if row['val_accuracy']>best:
            best=row['val_accuracy']; torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'model_config':{'embedding_dim':512,'num_heads':8,'num_layers':2,'max_seq_len':16,'dropout':0.1},'epoch':epoch+1,'validation_accuracy':best,'seed':args.seed,'frame_count':args.frames,'dataset_repo_id':args.dataset_repo_id,'dataset_revision':args.revision,'clip_model':'openai/clip-vit-base-patch32','clip_revision':args.clip_revision,'clip_input_normalization':'imagenet','clip_frame_chunk_size':4,'clip_mixed_precision_training':args.mixed_precision,'validation_precision':'float32','batch_size':args.batch_size,'validation_batch_size':val_batch_size,'learning_rate':args.lr,'sampling':'uniform','image_size':[224,224]},out/'best.pt')
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
if __name__=='__main__': main()
