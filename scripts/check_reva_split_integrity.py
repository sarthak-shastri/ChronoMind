from __future__ import annotations
import argparse, hashlib, json, posixpath, re, sys
from pathlib import Path

SPLITS=("train","val","test")
def norm(p):
    value = re.sub(r"\\+", "/", str(p or "").strip())
    while value.startswith("./"):
        value = value[2:]
    return posixpath.normpath(value).casefold() if value else ""
def rows(path):
 if isinstance(path, list): return path
 d=json.loads(Path(path).read_text(encoding="utf-8-sig")); return d.get("QA",[]) if isinstance(d,dict) else d
def audit_report(split_files, video_root=None):
 data={s:rows(split_files[s]) for s in SPLITS if s in split_files}; issues=[]; paths={}; sources={}; ids={}; missing_source={s:0 for s in data}; hash_paths={}; hash_cache={}
 root = Path(video_root).expanduser().resolve() if video_root else None
 def local_video(path):
  if root is None: return None
  candidate=(root / str(path).replace('\\','/')).resolve()
  if candidate != root and root not in candidate.parents: return None
  return candidate if candidate.is_file() else None
 def sha256(path):
  if path not in hash_cache:
   h=hashlib.sha256()
   with path.open('rb') as stream:
    for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
   hash_cache[path]=h.hexdigest()
  return hash_cache[path]
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
    if root:
     file=local_video(p)
     if file is None: issues.append(f"missing video or out-of-root path: {split}: {p}")
     else: hash_paths.setdefault(sha256(file),[]).append((split,k,str(p)))
   found_source=False
   for f in ("source_sequence_id","original_video_id","source_id","sequence_id"):
    if r.get(f) not in (None,""): sources.setdefault(f"{f}:{str(r[f]).strip().casefold()}",set()).add(split); found_source=True; break
   if not found_source: missing_source[split]+=1
   if split in ("train","val"):
    if r.get("correct_answer") not in {"A","B","C","D"}: issues.append(f"invalid/missing answer: {split} {qid}")
    opts=r.get("options")
    if not isinstance(opts,dict) or set(opts)!={"A","B","C","D"}: issues.append(f"invalid/missing options: {split} {qid}")
 path_overlaps=[{'normalized_path':k,'splits':sorted(ss)} for k,ss in sorted(paths.items()) if len(ss)>1]
 source_overlaps=[{'source_sequence_id':k,'splits':sorted(ss)} for k,ss in sorted(sources.items()) if len(ss)>1]
 identical_hash_overlaps=[]
 for digest, occurrences in sorted(hash_paths.items()):
  split_names=sorted({item[0] for item in occurrences})
  distinct_paths=sorted({item[1] for item in occurrences})
  if len(split_names)>1:
   identical_hash_overlaps.append({'sha256':digest,'splits':split_names,'paths':distinct_paths})
 for item in path_overlaps: issues.append(f"video-path overlap {item['normalized_path']}: {','.join(item['splits'])}")
 for item in source_overlaps: issues.append(f"source-sequence overlap {item['source_sequence_id']}: {','.join(item['splits'])}")
 for item in identical_hash_overlaps: issues.append(f"identical-file SHA-256 overlap {item['sha256']}: {','.join(item['splits'])}")
 # Keep the absence of a mapping as a limitation/warning, not a path overlap or answer-leakage claim.
 unknown_source={'status':'unverifiable' if any(missing_source.values()) else 'explicit_ids_present',
                 'rows_without_source_sequence_id':missing_source,
                 'warning':('Source-sequence relationships for rows without explicit source IDs cannot be established; dataset_name and subdir are not used as sequence IDs.' if any(missing_source.values()) else None)}
 return {'issues':issues,'path_overlaps':path_overlaps,'source_sequence_overlaps':source_overlaps,
         'identical_content_hash_overlaps':identical_hash_overlaps,
         'unverifiable_source_sequence_relationships':unknown_source,
         'video_hashing':{'enabled':root is not None,'video_root':str(root) if root else None,
                          'hashed_files':len(hash_cache)}}

def audit(split_files, video_root=None):
 """Compatibility API returning actionable issues as the legacy list."""
 return audit_report(split_files, video_root)['issues']

def main():
 p=argparse.ArgumentParser(); p.add_argument('--train',required=True); p.add_argument('--val',required=True); p.add_argument('--test'); p.add_argument('--video-root'); a=p.parse_args()
 fs={'train':a.train,'val':a.val};
 if a.test: fs['test']=a.test
 report=audit_report(fs,a.video_root); issues=report['issues']
 if issues:
  print('\n'.join(issues))
 if report['unverifiable_source_sequence_relationships']['warning']:
  print('WARNING:',report['unverifiable_source_sequence_relationships']['warning'])
 if issues: return 1
 print('No exact path, explicit source-ID, duplicate QA-ID, file-hash, missing-video, or annotation issue was detected.')
 print('This does not establish hidden source-sequence independence or answer leakage.')
 return 0
if __name__=='__main__': sys.exit(main())
