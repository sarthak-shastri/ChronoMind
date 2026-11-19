from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

SPLITS=("train","val","test")
def norm(p): return re.sub(r"\\+","/",str(p or "").strip()).lstrip("./").casefold()
def rows(path):
 if isinstance(path, list): return path
 d=json.loads(Path(path).read_text(encoding="utf-8-sig")); return d.get("QA",[]) if isinstance(d,dict) else d
def audit(split_files, video_root=None):
 data={s:rows(split_files[s]) for s in SPLITS if s in split_files}; issues=[]; paths={}; sources={}; ids={}
 for split, records in data.items():
  for i,r in enumerate(records):
   qid=r.get("qa_id")
   if qid:
    if qid in ids: issues.append(f"duplicate qa_id {qid}: {ids[qid]} and {split}")
    ids[qid]=split
   p=r.get("video_path",r.get("file_path")); k=norm(p)
   if not k: issues.append(f"{split}[{i}] missing video path")
   else:
    paths.setdefault(k,set()).add(split)
    if video_root and not Path(video_root).joinpath(*str(p).replace('\\','/').split('/')).is_file(): issues.append(f"missing video: {split}: {p}")
   for f in ("source_sequence_id","original_video_id","source_id","sequence_id"):
    if r.get(f) not in (None,""): sources.setdefault(str(r[f]).casefold(),set()).add(split); break
   if split in ("train","val"):
    if r.get("correct_answer") not in {"A","B","C","D"}: issues.append(f"invalid/missing answer: {split} {qid}")
    opts=r.get("options")
    if not isinstance(opts,dict) or set(opts)!={"A","B","C","D"}: issues.append(f"invalid/missing options: {split} {qid}")
 for k,ss in paths.items():
  if len(ss)>1: issues.append(f"video-path overlap {k}: {','.join(sorted(ss))}")
 for k,ss in sources.items():
  if len(ss)>1: issues.append(f"source-sequence overlap {k}: {','.join(sorted(ss))}")
 return issues

def main():
 p=argparse.ArgumentParser(); p.add_argument('--train',required=True); p.add_argument('--val',required=True); p.add_argument('--test'); p.add_argument('--video-root'); a=p.parse_args()
 fs={'train':a.train,'val':a.val};
 if a.test: fs['test']=a.test
 issues=audit(fs,a.video_root)
 if issues:
  print('\n'.join(issues)); return 1
 print('Split integrity OK: no detected path/source overlap, duplicate QA IDs, missing videos, or invalid train/val labels.')
 return 0
if __name__=='__main__': sys.exit(main())
