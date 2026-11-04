from pathlib import Path
import argparse, json
from collections import Counter
from preprocessing.dataset import read_annotations

def main():
    p=argparse.ArgumentParser(); p.add_argument('annotation_file',type=Path); p.add_argument('--video-root',type=Path); p.add_argument('--output',type=Path); a=p.parse_args()
    rows=read_annotations(a.annotation_file); ids=[str(x.get('qa_id',x.get('sample_id',x.get('id')))) for x in rows]
    paths=[x.get('video_path',x.get('file_path')) for x in rows]
    report={'annotation_file':str(a.annotation_file),'records':len(rows),'top_level_keys':sorted(rows[0]) if rows else [],'duplicate_sample_ids':[k for k,v in Counter(ids).items() if v>1],'category_counts':dict(Counter(x.get('category') for x in rows)),'task_counts':dict(Counter(x.get('task') for x in rows)),'missing_video_reference':sum(not x for x in paths)}
    if a.video_root: report['missing_video_files']=[str(x) for x in paths if x and not (a.video_root/x).is_file()]
    out=json.dumps(report,indent=2,ensure_ascii=False); print(out)
    if a.output: a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(out+'\n',encoding='utf-8')
if __name__=='__main__': main()
