from __future__ import annotations
import argparse, json, random, re
from collections import defaultdict
from pathlib import Path

SOURCE_FIELDS = ("source_sequence_id", "original_video_id", "source_id", "sequence_id")

def norm_path(value: str) -> str:
    return re.sub(r"\\+", "/", str(value).strip()).lstrip("./").casefold()

def group_id(row: dict) -> str:
    for field in SOURCE_FIELDS:
        value = row.get(field)
        if value not in (None, ""):
            return f"source:{field}:{str(value).strip().casefold()}"
    value = row.get("video_path", row.get("file_path"))
    if not value:
        raise ValueError(f"QA row missing video_path/file_path: {row.get('qa_id')}")
    return "path:" + norm_path(value)

def read_json(path: Path):
    obj=json.loads(path.read_text(encoding="utf-8-sig"))
    rows=obj.get("QA") if isinstance(obj,dict) else obj
    if not isinstance(rows,list): raise ValueError(f"Expected QA list in {path}")
    return obj, rows

def write_split(path: Path, template, rows, metadata):
    data=dict(template) if isinstance(template,dict) else {"metadata":{}}
    data["QA"]=rows
    data.setdefault("metadata",{})["derived_split"] = metadata
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def main():
    p=argparse.ArgumentParser(description="Make internal train/validation copies that exclude all official validation/test video groups.")
    p.add_argument("--train",type=Path,required=True); p.add_argument("--official-val",type=Path,required=True); p.add_argument("--official-test",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--seed",type=int,default=42); p.add_argument("--val-fraction",type=float,default=.2)
    a=p.parse_args()
    if not 0<a.val_fraction<1: p.error("--val-fraction must be between 0 and 1")
    train_meta, train_rows=read_json(a.train); _, official_val=read_json(a.official_val); _, official_test=read_json(a.official_test)
    eval_rows=official_val+official_test
    eval_groups={group_id(r) for r in eval_rows}
    eval_paths={norm_path(r.get("video_path",r.get("file_path",""))) for r in eval_rows}
    excluded=[]; eligible=[]
    for r in train_rows:
        gid=group_id(r); path=norm_path(r.get("video_path",r.get("file_path","")))
        if gid in eval_groups or path in eval_paths: excluded.append(r)
        else: eligible.append(r)
    if not eligible: raise ValueError("No official training QA remain after removing official validation/test video groups")
    buckets=defaultdict(list)
    for r in eligible: buckets[group_id(r)].append(r)
    rng=random.Random(a.seed); keys=list(buckets); rng.shuffle(keys); keys.sort(key=lambda k:len(buckets[k]),reverse=True)
    target=round(len(eligible)*a.val_fraction); val_groups=set(); val_count=0
    for k in keys:
        size=len(buckets[k])
        if abs(val_count+size-target)<abs(val_count-target): val_groups.add(k); val_count+=size
    val=[r for k in keys if k in val_groups for r in buckets[k]]
    train=[r for k in keys if k not in val_groups for r in buckets[k]]
    train_g={group_id(r) for r in train}; val_g={group_id(r) for r in val}
    if train_g&val_g: raise AssertionError("Generated train/validation group overlap")
    for part in (train,val):
        if any(group_id(r) in eval_groups or norm_path(r.get("video_path",r.get("file_path",""))) in eval_paths for r in part):
            raise AssertionError("Generated training-derived split overlaps official validation/test")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    provenance={"method":"exclude official val/test groups from official train, then deterministic grouped split","seed":a.seed,"grouping":"source-sequence ID if present, otherwise normalized video_path","official_val_source":a.official_val.name,"official_test_source":a.official_test.name}
    write_split(a.output_dir/"train_grouped.json",train_meta,train,{**provenance,"name":"train_grouped"})
    write_split(a.output_dir/"val_grouped.json",train_meta,val,{**provenance,"name":"val_grouped"})
    stats={"seed":a.seed,"val_fraction_requested":a.val_fraction,"grouping":provenance["grouping"],"official_train_qa":len(train_rows),"excluded_train_qa_overlapping_official_val_or_test":len(excluded),"eligible_train_qa":len(eligible),"official_val_unique_groups":len({group_id(r) for r in official_val}),"official_test_unique_groups":len({group_id(r) for r in official_test}),"train_qa":len(train),"train_groups":len(train_g),"val_qa":len(val),"val_groups":len(val_g),"train_val_group_overlap":len(train_g&val_g),"train_paths_overlapping_official_val_test":0,"val_paths_overlapping_official_test":0}
    (a.output_dir/"split_statistics.json").write_text(json.dumps(stats,indent=2)+"\n",encoding="utf-8")
    report=["# ReVA leakage-safe internal split", "", "Only official `train.json` QA rows are eligible for these derived splits. Before splitting, all train rows whose source-sequence ID or normalized video path occurs in either official validation or test are excluded. Remaining QA rows are grouped and assigned deterministically using seed 42 (or `--seed`). Official files are read-only inputs and are not changed.", "", "## Counts", "", "| Item | Count |", "|---|---:|"]
    report += [f"| Official train QA rows | {len(train_rows)} |",f"| Excluded because video/group is in official val or test | {len(excluded)} |",f"| Eligible train QA rows | {len(eligible)} |",f"| Derived train QA / groups | {len(train)} / {len(train_g)} |",f"| Derived validation QA / groups | {len(val)} / {len(val_g)} |", "",f"Train/validation group overlap: **{len(train_g&val_g)}**.","Training paths shared with either official evaluation split: **0**.","Validation paths shared with official test: **0**.","", "## Limitation", "", "The annotation schema has no source-sequence identifier, so normalized `video_path` is the minimum available grouping key. These files guarantee path-level isolation, not hidden common-source isolation for differently named clips. Obtain the original-to-clip mapping from the dataset authors to strengthen this guarantee. The official validation and test files remain untouched and must never be added to training.", ""]
    (a.output_dir/"split_statistics.md").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps(stats,indent=2))
if __name__=="__main__": main()
